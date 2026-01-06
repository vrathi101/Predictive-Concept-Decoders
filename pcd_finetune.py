import os
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from pcd_pretrain import (
    set_hf_cache_env,
    init_distributed,
    maybe_hf_login_from_env,
    load_model_and_tokenizer,
    Encoder,
    get_resid_stream_vector_efficient,
    patch_state,
    patch_resid_stream_hook,
)

# Dataset helpers

SYSTEM_PREFIX = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n
{system}\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"""


def filter_qa_for_length(tokenizer, rpath, wpath, min_l=16):
    out = []
    with open(rpath, "r") as f:
        for line in f:
            ex = json.loads(line)

            ids = tokenizer(
                ex["user"],
                add_special_tokens=False
            )["input_ids"]

            if len(ids) < min_l:
                continue

            out.append(ex)

    with open(wpath, "w") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("filtered len: ", len(out))


def jsonl_to_list(path: str):
    out = []
    with open(path, "r") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def format_answer_choices(choices):
    # choices is a list of dicts like {"label": "A", "text": "..."}
    # Produces "A. ...; B. ...; C. ...; D. ..."
    res = ""
    for c in choices:
        res += c["label"] + ". " + c["text"] + ("; " if c != choices[-1] else "")
    return res

    
class MCQJsonlDataset(Dataset):
    def __init__(self, tokenizer, jsonl_path, system_prefix=SYSTEM_PREFIX, dummy_token_len=16):
        self.tokenizer = tokenizer
        self.data = jsonl_to_list(jsonl_path)
        self.system_prefix = system_prefix
        self.dummy_token_len = int(dummy_token_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        system, user, q = ex["system"], ex["user"], ex["decoder_question"]
        choices, ans = ex["choices"], ex["correct_label"]
        formatted_choices = format_answer_choices(choices)

        pre_q = self.system_prefix.format(system=system) + user + "\n"
        post_q = q + "\n" + formatted_choices + "\nAnswer: " + ans

        dummy = " X" * self.dummy_token_len

        pre_ids = self.tokenizer(pre_q, return_tensors=None, add_special_tokens=False)["input_ids"]
        post_ids = self.tokenizer(dummy + post_q, return_tensors=None, add_special_tokens=False)["input_ids"]

        return torch.tensor(pre_ids, dtype=torch.long), torch.tensor(post_ids, dtype=torch.long)


def collate_mcq(batch, tokenizer):
    pre_list  = [x[0].tolist() for x in batch]
    post_list = [x[1].tolist() for x in batch]

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    pre = tokenizer.pad({"input_ids": pre_list}, padding=True, return_tensors="pt")

    tokenizer.padding_side = "right"
    post = tokenizer.pad({"input_ids": post_list}, padding=True, return_tensors="pt")

    post_ids  = post["input_ids"]
    post_mask = post["attention_mask"]

    labels = torch.full_like(post_ids, -100)
    last_pos = post_mask.sum(dim=1) - 1
    rows = torch.arange(post_ids.size(0))
    labels[rows, last_pos] = post_ids[rows, last_pos]

    return {
        "pre_input_ids": pre["input_ids"],
        "pre_attention_mask": pre["attention_mask"],
        "post_input_ids": post_ids,
        "post_attention_mask": post_mask,
        "labels": labels,
    }

# Decoder builder & ckpt loader

def build_decoder_with_lora(
    device,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
):
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=None,
        attn_implementation="sdpa",
    ).to(device)

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    dec = get_peft_model(base, lora_cfg).to(device)
    return dec


def load_enc_dec(pretrain_ckpt_path, device, model_name="meta-llama/Llama-3.2-3B-Instruct"):
    ckpt = torch.load(pretrain_ckpt_path, map_location="cpu")

    decoder = build_decoder_with_lora(device=device, model_name=model_name).to(device).bfloat16()

    for name, p in decoder.named_parameters():
        if "lora_" in name:
            p.data = p.data.float()

    d_model = decoder.config.hidden_size
    d_model_multiplier = 8
    encoder = Encoder(d_in=d_model, multiplier=d_model_multiplier, top_k=16).to(device).eval()

    encoder.load_state_dict(ckpt["encoder"], strict=True)
    decoder.load_state_dict(ckpt["decoder"], strict=True)

    encoder.eval()
    decoder.train()
    return encoder, decoder

# Training

def train_step(
    subject_model,
    encoder,
    decoder,
    batch,
    layer,
    dummy_token_len,
    device,
):
    pre_ids  = batch["pre_input_ids"].to(device, non_blocking=True)
    pre_mask = batch["pre_attention_mask"].to(device, non_blocking=True)

    with torch.no_grad():
        end_pos = pre_ids.size(1) - 1
        start_pos = end_pos - dummy_token_len
        encoder_in = get_resid_stream_vector_efficient(
            subject_model,
            pre_ids,
            layer,
            start_pos,
            end_pos,
            attention_mask=pre_mask,
        )
        encoder_out, _ = encoder(encoder_in.float())

    patch_state["vecs"] = encoder_out

    post_ids = batch["post_input_ids"].to(device, non_blocking=True)
    post_mask = batch["post_attention_mask"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)

    out = decoder(
        input_ids=post_ids,
        attention_mask=post_mask,
        labels=labels,
        use_cache=False,
    )
    return out.loss


def run_validation(subject_model, encoder, decoder, val_loader, layer, dummy_token_len, device):
    decoder.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="val", leave=False):
            loss = train_step(
                subject_model=subject_model,
                encoder=encoder,
                decoder=decoder,
                batch=batch,
                layer=layer,
                dummy_token_len=dummy_token_len,
                device=device,
            )
            total += float(loss.item())
            n += 1

    t = torch.tensor([total, n], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)

    decoder.train()
    return (t[0] / t[1]).item()


def do_train_full(
    subject_model,
    encoder,
    decoder,
    optim,
    train_loader,
    val_loader,
    device,
    rank,
    layer=15,
    dummy_token_len=16,
    num_epochs=5,
    val_every_n_steps=20,
    val_patience=3,
    save_path="decoder_finetuned.pt",
):
    best_val = float("inf")
    curr_bad = 0
    stop_training = False

    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)

        pbar = tqdm(enumerate(train_loader, start=1), desc=f"epoch {epoch+1}/{num_epochs}") if rank == 0 else enumerate(train_loader, start=1)
        for step, batch in pbar:
            loss = train_step(
                subject_model=subject_model,
                encoder=encoder,
                decoder=decoder,
                batch=batch,
                layer=layer,
                dummy_token_len=dummy_token_len,
                device=device,
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            if rank == 0:
                pbar.set_postfix(loss=float(loss.item()))

            if (step % val_every_n_steps) == 0:
                val_mean = run_validation(
                    subject_model=subject_model,
                    encoder=encoder,
                    decoder=decoder,
                    val_loader=val_loader,
                    layer=layer,
                    dummy_token_len=dummy_token_len,
                    device=device,
                )

                if rank == 0:
                    
                    if val_mean < best_val:
                        best_val = val_mean
                        curr_bad = 0
                        torch.save(
                            {
                                "decoder": decoder.module.state_dict(),
                                "optim": optim.state_dict(),
                                "epoch": epoch,
                                "step": step,
                                "best_val": best_val,
                            },
                            save_path,
                        )
                    else:
                        curr_bad += 1
                        if curr_bad >= val_patience:
                            stop_training = True

                    pbar.set_postfix(val=float(val_mean), best_val=float(best_val))

                stop_flag = torch.tensor([1 if stop_training else 0], device=device)
                dist.broadcast(stop_flag, src=0)
                stop_training = bool(stop_flag.item())

                if stop_training:
                    break

        if stop_training:
            break

    return best_val

# Main

def main():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"

    pretrain_ckpt_path = "pcd_3B_layer15_all-lora-modules_rank16.pt"
    train_read_jsonl_path = "synthsys_train.jsonl"
    val_read_jsonl_path = "synthsys_val.jsonl"
    train_jsonl_path = "synthsys_train_filtered.jsonl"
    val_jsonl_path = "synthsys_val_filtered.jsonl"

    layer = 15
    dummy_token_len = 16

    batch_size = 64
    num_epochs = 20

    val_every_n_steps = 40
    val_patience = 3

    save_path = "decoder_finetuned.pt"

    set_hf_cache_env()
    local_rank, device, rank, world_size = init_distributed()
    maybe_hf_login_from_env()

    subject_model, tokenizer, _ = load_model_and_tokenizer(model_name=model_name, mode="eval")
    subject_model = subject_model.to(device)
    for p in subject_model.parameters():
        p.requires_grad = False

    tokenizer.pad_token = tokenizer.eos_token

    # filter_qa_for_length(tokenizer, train_read_jsonl_path, train_jsonl_path)
    # filter_qa_for_length(tokenizer, val_read_jsonl_path, val_jsonl_path)

    train_ds = MCQJsonlDataset(tokenizer, train_jsonl_path, SYSTEM_PREFIX, dummy_token_len=dummy_token_len)
    val_ds = MCQJsonlDataset(tokenizer, val_jsonl_path,   SYSTEM_PREFIX, dummy_token_len=dummy_token_len)

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_ds,
        sampler=train_sampler,
        shuffle=False,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda b: collate_mcq(b, tokenizer),
    )
    val_loader = DataLoader(
        val_ds,
        sampler=val_sampler,
        shuffle=False,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=lambda b: collate_mcq(b, tokenizer),
    )


    encoder, decoder = load_enc_dec(pretrain_ckpt_path, device=device, model_name=model_name)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    decoder = DDP(decoder, device_ids=[local_rank], output_device=local_rank)

    patch_idx = torch.arange(dummy_token_len, device=device)
    handle = decoder.module.base_model.model.model.embed_tokens.register_forward_hook(
        patch_resid_stream_hook(patch_idx)
    )

    optim = torch.optim.AdamW(
        [p for p in decoder.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=0.01,
    )

    try:
        best_val = do_train_full(
            subject_model=subject_model,
            encoder=encoder,
            decoder=decoder,
            optim=optim,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            rank=rank,
            layer=layer,
            dummy_token_len=dummy_token_len,
            num_epochs=num_epochs,
            val_every_n_steps=val_every_n_steps,
            val_patience=val_patience,
            save_path=save_path,
        )
        if rank == 0:
            print("done; best_val =", best_val)
    finally:
        handle.remove()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
