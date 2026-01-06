import os
import json
import re
import random
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
# from nnsight import LanguageModel
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from itertools import islice
from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download, login
from transformers import AutoTokenizer, AutoModelForCausalLM


# Environment stuff

def set_hf_cache_env():
    os.environ["HF_HOME"] = "/root/.cache/huggingface"
    os.environ["HF_HUB_CACHE"] = "/root/.cache/huggingface/hub"
    os.environ["HF_DATASETS_CACHE"] = "/root/.cache/huggingface/datasets"
    os.environ["TRANSFORMERS_CACHE"] = "/root/.cache/huggingface/transformers"
    os.environ["PIP_CACHE_DIR"] = "/root/.cache/pip"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def init_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return local_rank, device, rank, world_size


def maybe_hf_login_from_env():
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)


# FineWeb sample/10BT has 15 shards: 000..014  (total ~30.6GB)
FINEWEB_REV = "9bb295ddab0e05d785b879661af7260fed5140fc"
FILE_IDS = list(range(4))


def fineweb_local_parquets(file_ids, local_only: bool):
    return [
        hf_hub_download(
            repo_id="HuggingFaceFW/fineweb",
            repo_type="dataset",
            filename=f"sample/10BT/{i:03d}_00000.parquet",
            revision=FINEWEB_REV,
            local_files_only=local_only,
        )
        for i in file_ids
    ]


def ensure_fineweb_parquets_available(rank, local_rank):
    if rank == 0:
        _ = fineweb_local_parquets(FILE_IDS, local_only=False)

    dist.barrier(device_ids=[local_rank])
    return fineweb_local_parquets(FILE_IDS, local_only=True)


# Model & data helpers

def load_model_and_tokenizer(model_name="meta-llama/Llama-3.2-3B-Instruct", attn_implementation="sdpa", mode="eval", **kwargs):
    """Load model and tokenizer with standard setup.

    Returns:
        tuple: (model, tokenizer, config_dict) where config_dict has num_layers, num_heads, head_dim
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map=None,
        attn_implementation=attn_implementation,
        **kwargs
    )

    if mode == "eval":
        model.eval()
    else:
        model.train()

    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    num_layers = model.config.num_hidden_layers

    config = {
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
    }

    return model, tokenizer, config


def load_data(dataset_name="HuggingFaceFW/fineweb", split="train", streaming=True, first_k=int(1e5), buffer_frac=0.1, val_frac=0.05):
    ds_stream = load_dataset(
        dataset_name,
        name="sample-10BT",
        split=split,
        streaming=streaming
    )
    ds_stream = ds_stream.shuffle(buffer_size=int(first_k * buffer_frac), seed=0)
    ds_train = ds_stream.take(first_k)
    ds_val = ds_stream.skip(first_k).take(int(first_k * val_frac))
    return ds_train, ds_val


def get_cropped_text_ids(dataset, tokenizer, prefix_ids, cropped_len=48):
    for item in dataset:
        text = item["text"]
        text_ids = tokenizer(
            text,
            return_tensors=None,
            add_special_tokens=False
        )["input_ids"]

        if len(text_ids) >= cropped_len:
            start = rng.randint(0, len(text_ids) - cropped_len)
            selected_ids = text_ids[start:start + cropped_len]
            yield prefix_ids + selected_ids


class CroppedTokenDataset(IterableDataset):
    def __init__(
        self,
        data_files,
        tokenizer,
        prefix_ids,
        cropped_len=48,
        mode="train",
        total_samples=100_000,
        skip_samples=0,
        seed=42,
        shuffle_buffer=50_000,
    ):
        self.data_files = data_files
        self.tokenizer = tokenizer
        self.prefix = torch.tensor(prefix_ids, dtype=torch.long)
        self.cropped_len = cropped_len
        self.mode = mode
        self.total_samples = total_samples
        self.skip_samples = skip_samples
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        # DDP rank/world
        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        
        epoch = getattr(self, "epoch", 0)
        seed = self.seed + 10_000 * epoch + rank

        take_local = self.total_samples // world
        skip_local = self.skip_samples // world

        local_files = self.data_files[(rank + epoch) % world :: world]
        if len(local_files) == 0:
            return

        ds = load_dataset(
            "parquet",
            data_files=local_files,
            split="train",
            streaming=True,
        )

        if self.mode == "train":
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=seed)

        stream = islice(ds, skip_local, None)

        g = torch.Generator()
        g.manual_seed(seed)

        emitted = 0
        for item in stream:
            text_ids = self.tokenizer(
                item["text"],
                return_tensors=None,
                add_special_tokens=False
            )["input_ids"]

            if len(text_ids) < self.cropped_len:
                continue

            if self.mode == "train":
                start = int(torch.randint(
                    0, len(text_ids) - self.cropped_len + 1, (1,), generator=g
                ).item())
            else:
                start = 0

            cropped = torch.tensor(text_ids[start:start + self.cropped_len], dtype=torch.long)
            yield torch.cat([self.prefix, cropped], dim=0)

            emitted += 1
            if emitted >= take_local:
                break


class Encoder(nn.Module):
    def __init__(self, d_in=2048, multiplier=8, top_k=16):
        super().__init__()
        self.top_k = top_k
        self.w_enc = nn.Linear(d_in, d_in * multiplier, bias=True)
        self.w_emb = nn.Linear(d_in * multiplier, d_in, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            W = torch.randn_like(self.w_enc.weight)
            W /= W.norm(dim=1, keepdim=True)
            self.w_enc.weight.copy_(W)
            self.w_enc.bias.zero_()
            self.w_emb.weight.copy_(self.w_enc.weight.T)

    def forward(self, x):  # (B, 16, d_in)
        y = self.w_enc(x)  # (B, 16, d_in*mult)

        idx = torch.topk(y, self.top_k, dim=-1).indices
        mask = torch.zeros_like(y, dtype=torch.bool)
        mask.scatter_(-1, idx, True)
        masked_y = y * mask.to(y.dtype)

        out = self.w_emb(masked_y)  # (B, 16, d_in)
        return out, idx


# def get_resid_stream_vector_slow(layer, input_ids, prefix_ids, cropped_len):
#     with nnsight_model.trace(input_ids):
#         resid = nnsight_model.model.layers[layer].output[:]
#         start = len(prefix_ids) + cropped_len // 3
#         end = start + cropped_len // 3
#         out = resid[:, start:end, :].save()
#         return out

# def get_resid_stream_vector(model, input_ids, layer, start, end, attention_mask=None):
#     out = model(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         output_hidden_states=True,
#         return_dict=True,
#         use_cache=False
#     )
#     resid = out.hidden_states[layer + 1]
#     return resid[:, start:end, :]


def get_resid_stream_vector_efficient(model, input_ids, layer, start, end, attention_mask=None):
    saved = {}
    def hook(module, inp, out):
        saved["slice"] = out[:, start:end, :].detach()

    h = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=False
            )
        return saved["slice"]
    finally:
        h.remove()


patch_state = {"vecs": None}


def patch_resid_stream_hook(idx):
    def hook(module, inp, out):
        h = out.clone()
        h[:, idx, :] = patch_state["vecs"].to(h.dtype)
        return h
    return hook


class TrainCtx:
    def __init__(
        self,
        local_rank,
        device,
        rank,
        world_size,
        subject,
        tokenizer,
        encoder,
        decoder,
        optim,
        dummy,
        patch_idx,
        d_model,
        d_model_multiplier,
        concepts_last_occ_by_seen_tokens,
        seen_tokens,
        inactive_concepts_tracker,
        best_val,
        curr_bad,
        stop_training,
    ):
        self.local_rank = local_rank
        self.device = device
        self.rank = rank
        self.world_size = world_size

        self.subject = subject
        self.tokenizer = tokenizer

        self.encoder = encoder
        self.decoder = decoder
        self.optim = optim

        self.dummy = dummy
        self.patch_idx = patch_idx

        self.d_model = d_model
        self.d_model_multiplier = d_model_multiplier

        self.concepts_last_occ_by_seen_tokens = concepts_last_occ_by_seen_tokens
        self.seen_tokens = seen_tokens
        self.inactive_concepts_tracker = inactive_concepts_tracker

        self.best_val = best_val
        self.curr_bad = curr_bad
        self.stop_training = stop_training


def train_step(ctx, batch, layer, start_pos, end_pos,
               include_aux_loss=True, update_last_occ=True, aux_thresh=2e5, eps_aux=1e-4, k_aux=250):

    with torch.no_grad():
        encoder_in = get_resid_stream_vector_efficient(
            ctx.subject, batch, layer, start_pos, end_pos
        ).to(ctx.device)  # (B, 16, d_model)

    encoder_in = encoder_in.float()
    encoder_out, idx = ctx.encoder(encoder_in)
    suffix = batch[:, -16:]
    decoder_in = torch.cat([ctx.dummy, suffix], dim=-1)

    patch_state["vecs"] = encoder_out

    label_ids = decoder_in.clone()
    label_ids[:, :ctx.dummy.size(-1)] = -100

    out = ctx.decoder(
        input_ids=decoder_in,
        labels=label_ids,
        use_cache=False
    )
    ce_loss = out.loss

    recent_concepts = torch.unique(idx.reshape(-1))

    if update_last_occ:
        recent_mask = torch.zeros((ctx.d_model * ctx.d_model_multiplier), device=ctx.device, dtype=torch.uint8)  # (2048*8,)
        recent_mask[recent_concepts] = 1
        dist.all_reduce(recent_mask, op=dist.ReduceOp.MAX)
        global_recent = recent_mask.nonzero().squeeze(1)  # (2048*8,) --> nonzero --> (2048*8,1) --> squeeze --> (2048*8,)
        ctx.concepts_last_occ_by_seen_tokens[global_recent] = ctx.seen_tokens

    window_start = max(0, ctx.seen_tokens - aux_thresh)
    inactive = ctx.concepts_last_occ_by_seen_tokens < window_start

    num_inactive = inactive.sum().item()
    aux_loss = 0.0

    if include_aux_loss:

        W_inactive = ctx.encoder.module.w_enc.weight[inactive]  # (#inactive, d_model)
        num_for_aux = W_inactive.size(0)

        if num_for_aux > 0:
            x_flat = encoder_in.reshape(-1, encoder_in.size(-1))  # (B*16, d_model)
            dot = x_flat @ W_inactive.T  # (B*16, #inactive)

            k_eff = min(num_for_aux, k_aux)
            top_vals = torch.topk(dot, k_eff, dim=1).values

            aux_loss = -(eps_aux / k_eff) * top_vals.sum(dim=1).mean()

    return ce_loss + aux_loss, num_inactive


def run_validation(ctx, val_loader, layer, start_cropped_pos, end_cropped_pos):
    ctx.encoder.eval()
    ctx.decoder.eval()

    total = 0.0
    n = 0
    with torch.no_grad():
        for val_batch in tqdm(val_loader, desc="val", leave=False):
            val_batch = val_batch.to(ctx.device, non_blocking=True)
            val_loss, _ = train_step(
                ctx, val_batch, layer, start_cropped_pos, end_cropped_pos,
                include_aux_loss=False, update_last_occ=False)

            total += val_loss.item()
            n += 1

    t = torch.tensor([total, n], device=ctx.device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)

    ctx.encoder.train()
    ctx.decoder.train()

    return (t[0] / t[1]).item()


def load_ckpt_if_exists(ctx, ckpt_path):
    if ctx.rank == 0 and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
    else:
        ckpt = None
        
    obj_list = [ckpt]
    dist.broadcast_object_list(obj_list, src=0)
    ckpt = obj_list[0]

    if ckpt is None:
        return 0, 1

    # models
    ctx.encoder.module.load_state_dict(ckpt["encoder"])
    ctx.decoder.module.load_state_dict(ckpt["decoder"])

    ctx.optim.load_state_dict(ckpt["optim"])
    for st in ctx.optim.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(ctx.device)

    ctx.best_val = ckpt["best_val"]
    ctx.curr_bad = ckpt["curr_bad"]
    ctx.seen_tokens = ckpt["seen_tokens"]
    ctx.inactive_concepts_tracker = ckpt["inactive_concepts_tracker"]

    ctx.concepts_last_occ_by_seen_tokens.copy_(ckpt["concepts_last_occ_by_seen_tokens"].to(ctx.device))

    start_epoch = int(ckpt["epoch"])
    start_step  = int(ckpt["step"]) + 1
    return start_epoch, start_step


def do_train_full(
    ctx,
    train_loader,
    val_loader,
    *,
    start_epoch=0,
    start_step=1,
    save_path="pcd.pt",
    ckpt_save_path="pcd_3B_layer15_all-lora-modules.pt",
    num_epochs=100,
    patience=10,
    every_n_steps=25,
    inactive_concepts_n_steps=50,
    layer=15,
    start_cropped_pos=0,
    end_cropped_pos=0,
    cropped_len=48,
):
    total_inactive_concepts = 0
    global_step = 0
    count_steps = 0

    for epoch in range(start_epoch, num_epochs):
        train_loader.dataset.set_epoch(epoch)
    
        pbar = tqdm(enumerate(train_loader, start=1), desc=f"epoch {epoch+1}/{num_epochs}")
        for step, train_batch in pbar:
            if epoch == start_epoch and step < start_step:
                continue
                
            global_step += 1
    
            train_batch = train_batch.to(ctx.device, non_blocking=True)
    
            loss, num_inact_concepts = train_step(
                ctx, train_batch, layer, start_cropped_pos, end_cropped_pos,
                include_aux_loss=True, update_last_occ=True
            )
            count_steps += 1
            total_inactive_concepts += num_inact_concepts
            ctx.seen_tokens += train_batch.size(0) * (cropped_len // 3) * ctx.world_size
    
            ctx.optim.zero_grad(set_to_none=True)
            loss.backward()
            ctx.optim.step()
            pbar.set_postfix(loss=float(loss.item()))
    
            if global_step % inactive_concepts_n_steps == 0:
                ctx.inactive_concepts_tracker.append((ctx.seen_tokens, total_inactive_concepts / count_steps))
                total_inactive_concepts = 0
                count_steps = 0
    
            if step % every_n_steps == 0:
                val_mean = run_validation(ctx, val_loader, layer, start_cropped_pos, end_cropped_pos)
    
                if ctx.rank == 0:
                    
                    if val_mean < ctx.best_val:
                        ctx.best_val = val_mean
                        ctx.curr_bad = 0
                        torch.save(
                            {
                                "encoder": ctx.encoder.module.state_dict(),
                                "decoder": ctx.decoder.module.state_dict(),
                                "optim": ctx.optim.state_dict(),
                                "epoch": epoch,
                                "step": step,
                                "best_val": ctx.best_val,
                                "curr_bad": ctx.curr_bad,
                                "concepts_last_occ_by_seen_tokens": ctx.concepts_last_occ_by_seen_tokens,
                                "seen_tokens": ctx.seen_tokens,
                                "inactive_concepts_tracker": ctx.inactive_concepts_tracker
                            },
                            ckpt_save_path,
                        )
        
                    else:
                        ctx.curr_bad += 1
                        if ctx.curr_bad >= patience:
                            ctx.stop_training = True

                    pbar.set_postfix(loss=float(loss.item()), val=float(val_mean), best_val=float(ctx.best_val))
                else:
                    pbar.set_postfix(loss=float(loss.item()))
    
            stop_flag = torch.tensor([1 if ctx.stop_training else 0], device=ctx.device)
            dist.broadcast(stop_flag, src=0)
            ctx.stop_training = bool(stop_flag.item())
            
            if ctx.stop_training:
                break
    
        if ctx.stop_training:
            break

    return ctx.best_val


# Build everything

def main():
    set_hf_cache_env()

    local_rank, device, rank, world_size = init_distributed()
    LOCAL_PARQUETS = ensure_fineweb_parquets_available(rank, local_rank)

    maybe_hf_login_from_env()

    subject, tokenizer, config = load_model_and_tokenizer()
    subject = subject.to(device)

    for p in subject.parameters():
        p.requires_grad = False

    # ds_train, ds_val = load_data(first_k=500000)

    # nnsight_model = LanguageModel(subject, tokenizer)

    INSTRUCT_PREFIX = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n
Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n
<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"""

    prefix_ids = tokenizer(
        INSTRUCT_PREFIX,
        return_tensors=None,
        add_special_tokens=False
    )["input_ids"]
    len_prefix = len(prefix_ids)

    cropped_len = 48

    rng = random.Random(42)

    decoder_base, _, _ = load_model_and_tokenizer(mode="train")
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
    )
    decoder = get_peft_model(decoder_base, lora_cfg).to(device).bfloat16().train()

    for name, p in decoder.named_parameters():
        if "lora_" in name:
            p.data = p.data.float()

    d_model = decoder.config.hidden_size
    d_model_multiplier = 8
    encoder = Encoder(d_in=d_model, multiplier=d_model_multiplier, top_k=16).to(device).train()

    decoder = DDP(decoder, device_ids=[local_rank], output_device=local_rank)
    encoder = DDP(encoder, device_ids=[local_rank], output_device=local_rank)

    optim = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=1e-4,
        weight_decay=0.01
    )

    start_cropped_pos = len_prefix + cropped_len // 3
    end_cropped_pos = start_cropped_pos + cropped_len // 3
    layer = 15
    batch_size = 64
    dummy = tokenizer(
        " X" * (cropped_len // 3),
        return_tensors="pt",
        add_special_tokens=False
    )["input_ids"].expand(batch_size, -1).to(device)
    patch_idx = torch.arange(16, device=device)

    concepts_last_occ_by_seen_tokens = torch.full(
        (d_model * d_model_multiplier,),
        -1,
        dtype=torch.long,
        device=device
    )
    seen_tokens = 0
    inactive_concepts_tracker = []

    VAL_SAMPLES = 10_000
    TRAIN_SAMPLES = 500_000

    pcd_train_ds = CroppedTokenDataset(
        data_files=LOCAL_PARQUETS,
        tokenizer=tokenizer,
        prefix_ids=prefix_ids,
        cropped_len=48,
        mode="train",
        total_samples=TRAIN_SAMPLES,
        skip_samples=VAL_SAMPLES,
    )

    pcd_val_ds = CroppedTokenDataset(
        data_files=LOCAL_PARQUETS,
        tokenizer=tokenizer,
        prefix_ids=prefix_ids,
        cropped_len=48,
        mode="val",
        total_samples=VAL_SAMPLES,
        skip_samples=0,
    )

    train_loader = DataLoader(pcd_train_ds, batch_size=64, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(pcd_val_ds,   batch_size=64, num_workers=0, pin_memory=True, drop_last=True)

    num_epochs = 100
    patience = 10
    curr_bad = 0
    best_val = float("inf")

    handle = decoder.module.base_model.model.model.embed_tokens.register_forward_hook(
        patch_resid_stream_hook(patch_idx)
    )

    every_n_steps = 25
    inactive_concepts_n_steps = 50

    stop_training = False

    CKPT_PATH = "chk-lora-modules.pt"
    save_path = "pcd-trained.pt"

    ctx = TrainCtx(
        local_rank=local_rank,
        device=device,
        rank=rank,
        world_size=world_size,
        subject=subject,
        tokenizer=tokenizer,
        encoder=encoder,
        decoder=decoder,
        optim=optim,
        dummy=dummy,
        patch_idx=patch_idx,
        d_model=d_model,
        d_model_multiplier=d_model_multiplier,
        concepts_last_occ_by_seen_tokens=concepts_last_occ_by_seen_tokens,
        seen_tokens=seen_tokens,
        inactive_concepts_tracker=inactive_concepts_tracker,
        best_val=best_val,
        curr_bad=curr_bad,
        stop_training=stop_training,
    )

    try:
        start_epoch, start_step = load_ckpt_if_exists(ctx, CKPT_PATH)
        dist.barrier(device_ids=[local_rank])
        do_train_full(
            ctx,
            train_loader,
            val_loader,
            start_epoch=start_epoch,
            start_step=start_step,
            save_path=save_path,
            ckpt_save_path="pcd_3B_layer15_all-lora-modules.pt",
            num_epochs=num_epochs,
            patience=patience,
            every_n_steps=every_n_steps,
            inactive_concepts_n_steps=inactive_concepts_n_steps,
            layer=layer,
            start_cropped_pos=start_cropped_pos,
            end_cropped_pos=end_cropped_pos,
            cropped_len=cropped_len,
        )
    finally:
        handle.remove()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
