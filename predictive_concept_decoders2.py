import os
os.environ["PIP_CACHE_DIR"] = "/workspace/.cache/pip"
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"

import sys
sys.path.append('.')

# !pip install --no-cache-dir -U "transformers>=4.51.0" accelerate datasets torch pandas tqdm nnsight huggingface_hub peft
# !pip install --no-cache-dir typing-extensions --upgrade
# !pip uninstall -y torchvision

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
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM  # other option: Qwen/Qwen2.5-0.5B-Instruct

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

# from huggingface_hub import notebook_login
# notebook_login()
from huggingface_hub import login
import os

token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)

def load_model_and_tokenizer(model_name="meta-llama/Llama-3.2-1B-Instruct", attn_implementation="sdpa", mode="eval", **kwargs):
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

# text = INSTRUCT_PREFIX + next(iter(ds_val))["text"][:200]
# enc = tokenizer(
#     text,
#     return_tensors=None,
#     add_special_tokens=False
# )
# input_ids = enc["input_ids"]
# tokens = tokenizer.convert_ids_to_tokens(input_ids)
# for a, b in zip(tokens, input_ids):
#     print(a, "    ", b)

# text = INSTRUCT_PREFIX + next(iter(ds_val))["text"][:20]
# enc = tokenizer(
#     text,
#     return_tensors=None,
#     add_special_tokens=False
# )
# input_ids = enc["input_ids"]
# tokens = tokenizer.convert_ids_to_tokens(input_ids)
# for a, b in zip(tokens, input_ids):
#     print(a, "    ", b)

# cropped_data_ids = get_cropped_text_ids(ds_val, tokenizer, prefix_ids)

# li = next(iter(cropped_data_ids))
# print(li)
# tokens = tokenizer.convert_ids_to_tokens(li)
# len(tokens)

# tokens

# tokenizer.decode(li[30:], skip_special_tokens=False)

# enc = tokenizer(
#     " X" * 16,
#     return_tensors=None,
#     add_special_tokens=True
# )["input_ids"]
# enc

# prompt = li
# batch_ids = torch.tensor([prefix_ids, prefix_ids], device=subject.device)
# with nnsight_model.trace(batch_ids) as tracer:
#     resid = nnsight_model.model.layers[10].output[:].save()
# print(resid.shape)

# resid[0].float().cpu().numpy()[:10]

class CroppedTokenDataset(IterableDataset):
    def __init__(
        self,
        dataset_name,
        dataset_config,
        tokenizer,
        prefix_ids,
        cropped_len=48,
        mode="train",
        total_samples=100_000,
        skip_samples=0,
        seed=42,
        shuffle_buffer=50_000,
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.prefix = torch.tensor(prefix_ids, dtype=torch.long)
        self.cropped_len = cropped_len
        self.mode = mode
        self.total_samples = total_samples
        self.skip_samples = skip_samples
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self):
        info = get_worker_info()
        wid = 0 if info is None else info.id
        nw  = 1 if info is None else info.num_workers

        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])

        num_shards = world * nw
        shard_index = rank * nw + wid

        ds = load_dataset(
            self.dataset_name,
            name=self.dataset_config,
            split="train",
            streaming=True,
        )

        # shard FIRST so each worker/process reads different data
        ds = ds.shard(num_shards=num_shards, index=shard_index)

        # shuffle within shard (cheap + avoids everyone touching the same broken files)
        ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)

        # local counts (simple + stable)
        skip_local = self.skip_samples // num_shards
        take_local = self.total_samples // num_shards

        it = iter(ds)
        for _ in range(skip_local):
            next(it)

        g = torch.Generator()
        g.manual_seed(self.seed + shard_index)

        emitted = 0
        while emitted < take_local:
            item = next(it)
            text_ids = self.tokenizer(item["text"], return_tensors=None, add_special_tokens=False)["input_ids"]
            if len(text_ids) < self.cropped_len:
                continue

            if self.mode == "train":
                start = int(torch.randint(0, len(text_ids) - self.cropped_len + 1, (1,), generator=g).item())
            else:
                start = 0

            cropped = torch.tensor(text_ids[start:start + self.cropped_len], dtype=torch.long)
            emitted += 1
            yield torch.cat([self.prefix, cropped], dim=0)


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

# def get_resid_stream_vector(layer, input_ids, prefix_ids, cropped_len):
#     with nnsight_model.trace(input_ids):
#         resid = nnsight_model.model.layers[layer].output[:]
#         start = len(prefix_ids) + cropped_len // 3
#         end = start + cropped_len // 3
#         out = resid[:, start:end, :].save()
#         return out

def get_resid_stream_vector(model, input_ids, layer, start, end, attention_mask=None):
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False
    )
    resid = out.hidden_states[layer + 1]
    return resid[:, start:end, :]

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

# test = get_resid_stream_vector(subject, torch.tensor([[2040,3520]], device="cuda"),3,0,5 )

decoder_base, _, _ = load_model_and_tokenizer(mode="train")
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj"]
)
decoder = get_peft_model(decoder_base, lora_cfg).to(device).train()

d_model = decoder.config.hidden_size
d_model_multiplier = 8
encoder = Encoder(d_in=d_model, multiplier=d_model_multiplier, top_k=16).to(device).to(torch.bfloat16).train()

decoder = DDP(decoder, device_ids=[local_rank], output_device=local_rank)
encoder = DDP(encoder, device_ids=[local_rank], output_device=local_rank)

# A = decoder.base_model.model.model.layers[0].self_attn.q_proj.lora_A["default"].weight.detach()
# B = decoder.base_model.model.model.layers[0].self_attn.q_proj.lora_B["default"].weight.detach()

# print("A mean/std:", A.mean().item(), A.std().item())
# print("B mean/std:", B.mean().item(), B.std().item())
# print("B all zero:", (B == 0).all().item())

optim = torch.optim.AdamW(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=1e-4,
    weight_decay=0.01
)
start_cropped_pos = len_prefix + cropped_len // 3
end_cropped_pos = start_cropped_pos + cropped_len // 3
layer = 8
batch_size=64
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



TRAIN_SAMPLES = 500_000
VAL_SAMPLES   = 10_000

DATASET = "HuggingFaceFW/fineweb"
CONFIG  = "sample-10BT" 

pcd_train_ds = CroppedTokenDataset(
    dataset_name=DATASET,
    dataset_config=CONFIG,
    tokenizer=tokenizer,
    prefix_ids=prefix_ids,
    cropped_len=48,
    mode="train",
    total_samples=TRAIN_SAMPLES,
    skip_samples=VAL_SAMPLES,
    seed=42,
)

pcd_val_ds = CroppedTokenDataset(
    dataset_name=DATASET,
    dataset_config=CONFIG,
    tokenizer=tokenizer,
    prefix_ids=prefix_ids,
    cropped_len=48,
    mode="val",
    total_samples=VAL_SAMPLES,
    skip_samples=0,
    seed=42,
)

train_loader = DataLoader(pcd_train_ds, batch_size=64, num_workers=4, pin_memory=True, drop_last=True)
val_loader   = DataLoader(pcd_val_ds,   batch_size=64, num_workers=2, pin_memory=True, drop_last=True)

patch_state = {"vecs": None}
def patch_resid_stream_hook(idx):
    def hook(module, inp, out):
        h = out.clone()
        h[:, idx, :] = patch_state["vecs"].to(h.dtype)
        return h
    return hook

def train_step(subject_model, batch, layer, start_pos, end_pos,
               include_aux_loss=True, update_last_occ=True, aux_thresh=2e5, eps_aux=1e-4, k_aux=250):

    with torch.no_grad():
        encoder_in = get_resid_stream_vector_efficient(
            subject_model, batch, layer, start_pos, end_pos
        ).to(device)  # (B, 16, d_model)

    encoder_out, idx = encoder(encoder_in)
    suffix = batch[:, -16:]
    decoder_in = torch.cat([dummy, suffix], dim=-1)

    patch_state["vecs"] = encoder_out

    label_ids = decoder_in.clone()
    label_ids[:, :dummy.size(-1)] = -100

    out = decoder(
        input_ids=decoder_in,
        labels=label_ids,
        use_cache=False
    )
    ce_loss = out.loss

    recent_concepts = torch.unique(idx.reshape(-1))

    if update_last_occ:
        recent_mask = torch.zeros((d_model * d_model_multiplier), device=device, dtype=torch.uint8)  # (2048*8,)
        recent_mask[recent_concepts] = 1
        dist.all_reduce(recent_mask, op=dist.ReduceOp.MAX)
        global_recent = recent_mask.nonzero().squeeze(1)  # (2048*8,) --> nonzero --> (2048*8,1) --> squeeze --> (2048*8,)
        concepts_last_occ_by_seen_tokens[global_recent] = seen_tokens

    window_start = max(0, seen_tokens - aux_thresh)
    inactive = concepts_last_occ_by_seen_tokens < window_start

    num_inactive = inactive.sum().item()
    aux_loss = 0.0

    if include_aux_loss:

        W_inactive = encoder.module.w_enc.weight[inactive]  # (#inactive, d_model)
        num_for_aux = W_inactive.size(0)

        if num_for_aux > 0:
            x_flat = encoder_in.reshape(-1, encoder_in.size(-1))  # (B*16, d_model)
            dot = x_flat @ W_inactive.T  # (B*16, #inactive)

            k_eff = min(num_for_aux, k_aux)
            top_vals = torch.topk(dot, k_eff, dim=1).values

            aux_loss = -(eps_aux / k_eff) * top_vals.sum(dim=1).mean()

    return ce_loss + aux_loss, num_inactive

num_epochs = 100
patience = 5
curr_bad = 0
best_val = float("inf")

handle = decoder.module.base_model.model.model.embed_tokens.register_forward_hook(
    patch_resid_stream_hook(patch_idx)
)

every_n_steps = 25
inactive_concepts_n_steps = 50
total_inactive_concepts = 0
global_step = 0
count_steps = 0

stop_training = False

def do_train_full():
    global global_step, count_steps, total_inactive_concepts
    global seen_tokens, best_val, curr_bad, stop_training
    
    for epoch in range(num_epochs):
    
        pbar = tqdm(enumerate(train_loader, start=1), desc=f"epoch {epoch+1}/{num_epochs}")
        for step, train_batch in pbar:
            global_step += 1
    
            train_batch = train_batch.to(device, non_blocking=True)
    
            loss, num_inact_concepts = train_step(
                subject, train_batch, layer, start_cropped_pos, end_cropped_pos,
                include_aux_loss=True, update_last_occ=True
            )
            count_steps += 1
            total_inactive_concepts += num_inact_concepts
            seen_tokens += train_batch.size(0) * (cropped_len // 3) * world_size
    
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=float(loss.item()))
    
            if global_step % inactive_concepts_n_steps == 0:
                inactive_concepts_tracker.append((seen_tokens, total_inactive_concepts / count_steps))
                total_inactive_concepts = 0
                count_steps = 0
    
            if step % every_n_steps == 0:
    
                encoder.eval()
                decoder.eval()
    
                total = 0.0
                n = 0
                with torch.no_grad():
                    for val_batch in tqdm(val_loader, desc="val", leave=False):
                        val_batch = val_batch.to(device, non_blocking=True)
                        val_loss, _ = train_step(
                            subject, val_batch, layer, start_cropped_pos, end_cropped_pos,
                            include_aux_loss=False, update_last_occ=False)
    
                        total += val_loss.item()
                        n += 1
    
                t = torch.tensor([total, n], device=device, dtype=torch.float64)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                
                val_mean = (t[0] / t[1]).item()
    
                if rank == 0:
                    
                    if val_mean < best_val:
                        best_val = val_mean
                        curr_bad = 0
                        torch.save(
                            {
                                "encoder": encoder.module.state_dict(),
                                "decoder": decoder.module.state_dict(),
                                "optim": optim.state_dict(),
                                "epoch": epoch,
                                "step": step,
                                "best_val": best_val,
                                "curr_bad": curr_bad,
                                "concepts_last_occ_by_seen_tokens": concepts_last_occ_by_seen_tokens,
                                "seen_tokens": seen_tokens,
                                "inactive_concepts_tracker": inactive_concepts_tracker
                            },
                            "best_checkpoint_aux.pt",
                        )
        
                    else:
                        curr_bad += 1
                        if curr_bad >= patience:
                            stop_training = True
    
                    pbar.set_postfix(loss=float(loss.item()), val=float(val_mean), best_val=float(best_val))
                else:
                    pbar.set_postfix(loss=float(loss.item()))
                
                encoder.train()
                decoder.train()
    
            stop_flag = torch.tensor([1 if stop_training else 0], device=device)
            dist.broadcast(stop_flag, src=0)
            stop_training = bool(stop_flag.item())
            
            if stop_training:
                break
    
        if stop_training:
            break

    handle.remove()
    dist.destroy_process_group()

if __name__ == "__main__":
    do_train_full()