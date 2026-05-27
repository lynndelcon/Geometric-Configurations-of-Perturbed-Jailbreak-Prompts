#Last-Layer-Last-Token Embedding retrieval along with the top-50 next-token probabilities and ids.

#For the adv queries

# ============================
# ONE-PASS: last-token embedding + top-50 next-token probabilities
# For prompts stored in adv_prompts_models_filtered.py as a list of dicts
# ============================

import os
import time
import gc
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional: helps fragmentation on some setups
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# -----------------------------
# USER PARAMETERS
# -----------------------------
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


TOP_K = 50

# For 1.5B you can try 4
BATCH_SIZE = 4
# For 7B use:
# BATCH_SIZE = 1

DTYPE = torch.float16
LOCAL_FILES_ONLY = False
TRUST_REMOTE_CODE = True

OUT_DIR = Path("llama3b")
OUT_EMB = OUT_DIR / "adv_prompts_lasttok_emb.npy"
OUT_TOPK_IDS = OUT_DIR / "adv_prompts_nexttok_topk_ids.npy"
OUT_TOPK_PROBS = OUT_DIR / "adv_prompts_nexttok_topk_probs.npy"

# -----------------------------
# Clean up (best effort)
# -----------------------------
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    free_b, total_b = torch.cuda.mem_get_info(0)
    print(f"GPU free/total (GiB): {free_b/1024**3:.2f} / {total_b/1024**3:.2f}")

# -----------------------------
# Load prompts from .py file
# -----------------------------
from adv_queries import prompts

prompts_only = [str(entry["prompt"]) for entry in prompts]
N = len(prompts_only)

print(f"Loaded {N} prompts from adv_prompts_models_filtered.py")
if N != 96:
    print(f"[WARN] Expected 96 prompts, got N={N}")

# -----------------------------
# Load tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=LOCAL_FILES_ONLY,
    trust_remote_code=TRUST_REMOTE_CODE,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# -----------------------------
# Load model
# -----------------------------
assert torch.cuda.is_available(), "CUDA not available."

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    local_files_only=LOCAL_FILES_ONLY,
    trust_remote_code=TRUST_REMOTE_CODE,
    low_cpu_mem_usage=True,
)
model.to("cuda:0")
model.eval()

device = next(model.parameters()).device
print("Model loaded on:", device)
print("Model dtype:", next(model.parameters()).dtype)

base_model = model.model
lm_head = model.lm_head


@torch.inference_mode()
def compute_emb_and_topk_flat(
    texts,
    *,
    batch_size: int,
    top_k: int,
    verbose: bool = True,
):
    """
    Returns:
      emb_cpu:        (N_total, H) float16 on CPU
      topk_ids_cpu:   (N_total, K) int32 on CPU
      topk_probs_cpu: (N_total, K) float32 on CPU
    """
    N_total = len(texts)
    emb_cpu = None
    topk_ids_cpu = None
    topk_probs_cpu = None

    for bi, start in enumerate(range(0, N_total, batch_size)):
        end = min(start + batch_size, N_total)
        batch_texts = texts[start:end]

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        if verbose and bi == 0:
            torch.cuda.synchronize()
            t0 = time.time()

        out = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        last_hidden = out.last_hidden_state

        lengths = attention_mask.sum(dim=1)
        last_idx = (lengths - 1).clamp(min=0)
        b_idx = torch.arange(last_hidden.size(0), device=device)

        last_h = last_hidden[b_idx, last_idx, :]
        next_logits = lm_head(last_h)

        # exact top-k probabilities
        topk_logits, topk_ids = torch.topk(next_logits, k=top_k, dim=-1)
        log_denom = torch.logsumexp(next_logits, dim=-1, keepdim=True)
        topk_probs = torch.exp(topk_logits - log_denom)

        if verbose and bi == 0:
            torch.cuda.synchronize()
            print(f"[timing] first batch seconds: {time.time() - t0:.3f}")

        B, H = last_h.shape

        if emb_cpu is None:
            emb_cpu = torch.empty((N_total, H), dtype=torch.float16, device="cpu")
            topk_ids_cpu = torch.empty((N_total, top_k), dtype=torch.int32, device="cpu")
            topk_probs_cpu = torch.empty((N_total, top_k), dtype=torch.float32, device="cpu")

        emb_cpu[start:end] = last_h.detach().to("cpu", dtype=torch.float16)
        topk_ids_cpu[start:end] = topk_ids.detach().to("cpu", dtype=torch.int32)
        topk_probs_cpu[start:end] = topk_probs.detach().to("cpu", dtype=torch.float32)

        if verbose:
            print(f"[batch {bi+1}] rows {start}:{end} | max_tokens={int(lengths.max().item())}")

        del enc, input_ids, attention_mask
        del out, last_hidden, last_h, next_logits
        del topk_logits, topk_ids, topk_probs, log_denom
        del lengths, last_idx, b_idx

        torch.cuda.empty_cache()

    return emb_cpu, topk_ids_cpu, topk_probs_cpu


# -----------------------------
# Run compute
# -----------------------------
emb_cpu, topk_ids_cpu, topk_probs_cpu = compute_emb_and_topk_flat(
    prompts_only,
    batch_size=BATCH_SIZE,
    top_k=TOP_K,
    verbose=True,
)

# -----------------------------
# Convert / save
# -----------------------------
emb = emb_cpu.numpy().astype(np.float32, copy=False)               # (N, H)
topk_ids = topk_ids_cpu.numpy().astype(np.int32, copy=False)       # (N, K)
topk_probs = topk_probs_cpu.numpy().astype(np.float32, copy=False) # (N, K)

OUT_DIR.mkdir(parents=True, exist_ok=True)
np.save(OUT_EMB, emb)
np.save(OUT_TOPK_IDS, topk_ids)
np.save(OUT_TOPK_PROBS, topk_probs)

print(f"Saved embeddings: {OUT_EMB} | shape={emb.shape} dtype={emb.dtype}")
print(f"Saved topk ids:   {OUT_TOPK_IDS} | shape={topk_ids.shape} dtype={topk_ids.dtype}")
print(f"Saved topk probs: {OUT_TOPK_PROBS} | shape={topk_probs.shape} dtype={topk_probs.dtype}")

#for the adv symbol paraphrases
# ============================
# ONE-PASS: last-token embedding + top-50 next-token probabilities
# For nested prompts (N, M) stored in a .npy (dtype=object).
# Qwen2.5-1.5B-Instruct, CUDA, low-memory (no (B,L,V) logits).
# ============================

import os
import time
import gc
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional: helps fragmentation on some setups
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# -----------------------------
# USER PARAMETERS
# -----------------------------
#Qwen/Qwen2.5-7B-Instruct
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
PARA_NPY = "para_symbol_025_adv.npy"      # nested (N, M)
TOP_K = 50

# Start conservative on 8GB and increase if stable
BATCH_SIZE = 4                                # try 4 first; then 8

DTYPE = torch.float16
LOCAL_FILES_ONLY = True
TRUST_REMOTE_CODE = True

OUT_DIR = Path("qwen3b")
OUT_EMB = OUT_DIR / "symbol_adv_lasttok_emb.npy"
OUT_TOPK_IDS = OUT_DIR / "para_symbol_adv_nexttok_topk_ids.npy"
OUT_TOPK_PROBS = OUT_DIR / "para_symbol_adv_nexttok_topk_probs.npy"

# -----------------------------
# Clean up (best effort)
# -----------------------------
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    free_b, total_b = torch.cuda.mem_get_info(0)
    print("GPU free/total (GiB):", free_b/1024**3, "/", total_b/1024**3)

# -----------------------------
# Load nested prompts (N, M)
# -----------------------------
#para = np.load(PARA_NPY, allow_pickle=True)
#prompts_nested = [[str(x) for x in row] for row in para]

#from adv_prompts_models_filtered import prompts
prompts_nested =[str(x) for x in np.load('para_symbol_025_adv.npy')]

N = len(prompts_nested)
M = len(prompts_nested[0]) if N > 0 else 0
row_lens = {len(r) for r in prompts_nested}
if len(row_lens) != 1:
    raise ValueError(f"Inconsistent number of variants per prompt: {sorted(row_lens)}")

print(f"Loaded nested prompts from {PARA_NPY}: N={N}, M={M}, total={N*M}")

prompts_flat = [s for row in prompts_nested for s in row]  # length N*M

# -----------------------------
# Load tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=LOCAL_FILES_ONLY,
    trust_remote_code=TRUST_REMOTE_CODE,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# -----------------------------
# Load model (safe path: no device_map warmup)
# -----------------------------
assert torch.cuda.is_available(), "CUDA not available."

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    local_files_only=LOCAL_FILES_ONLY,
    trust_remote_code=TRUST_REMOTE_CODE,
)
model.to("cuda:0")
model.eval()

device = next(model.parameters()).device
print("Model loaded on:", device)

# Backbone + LM head
base_model = model.model
lm_head = model.lm_head


@torch.inference_mode()
def compute_emb_and_topk_flat(
    texts,
    *,
    batch_size: int,
    top_k: int,
    verbose: bool = True,
):
    """
    Returns:
      emb_cpu:        (N_total, H) float16 on CPU
      topk_ids_cpu:   (N_total, K) int32 on CPU
      topk_probs_cpu: (N_total, K) float32 on CPU
    """
    N_total = len(texts)
    emb_cpu = None
    topk_ids_cpu = None
    topk_probs_cpu = None

    for bi, start in enumerate(range(0, N_total, batch_size)):
        end = min(start + batch_size, N_total)
        batch_texts = texts[start:end]

        enc = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=False)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        if verbose and bi == 0:
            torch.cuda.synchronize()
            t0 = time.time()

        # Backbone forward: (B, L, H)
        out = base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        last_hidden = out.last_hidden_state

        # Last non-pad positions
        lengths = attention_mask.sum(dim=1)                 # (B,)
        last_idx = (lengths - 1).clamp(min=0)               # (B,)
        b_idx = torch.arange(last_hidden.size(0), device=device)

        # Last-token embedding: (B, H)
        last_h = last_hidden[b_idx, last_idx, :]

        # Next-token logits only at last position: (B, V)
        next_logits = lm_head(last_h)

        # Top-k
        probs = F.softmax(next_logits, dim=-1)
        topk_probs, topk_ids = torch.topk(probs, k=top_k, dim=-1)

        if verbose and bi == 0:
            torch.cuda.synchronize()
            print(f"[timing] first batch seconds: {time.time() - t0:.3f}")

        B, H = last_h.shape

        # Allocate once
        if emb_cpu is None:
            emb_cpu = torch.empty((N_total, H), dtype=torch.float16, device="cpu")
            topk_ids_cpu = torch.empty((N_total, top_k), dtype=torch.int32, device="cpu")
            topk_probs_cpu = torch.empty((N_total, top_k), dtype=torch.float32, device="cpu")

        # Store to CPU
        emb_cpu[start:end] = last_h.detach().to("cpu", dtype=torch.float16)
        topk_ids_cpu[start:end] = topk_ids.detach().to("cpu", dtype=torch.int32)
        topk_probs_cpu[start:end] = topk_probs.detach().to("cpu", dtype=torch.float32)

        if verbose:
            print(f"[batch {bi+1}] rows {start}:{end} | max_tokens={int(lengths.max().item())}")

        # Free intermediates
        del out, last_hidden, last_h, next_logits, probs, topk_probs, topk_ids

    return emb_cpu, topk_ids_cpu, topk_probs_cpu


# -----------------------------
# Run compute
# -----------------------------
emb_flat_cpu, topk_ids_flat_cpu, topk_probs_flat_cpu = compute_emb_and_topk_flat(
    prompts_flat,
    batch_size=BATCH_SIZE,
    top_k=TOP_K,
    verbose=True,
)

# -----------------------------
# Reshape back to (N, M, ...)
# -----------------------------
H = emb_flat_cpu.shape[1]
emb = emb_flat_cpu.reshape(N, M, H).numpy().astype(np.float32, copy=False)
topk_ids = topk_ids_flat_cpu.reshape(N, M, TOP_K).numpy().astype(np.int32, copy=False)
topk_probs = topk_probs_flat_cpu.reshape(N, M, TOP_K).numpy().astype(np.float32, copy=False)

# -----------------------------
# Save
# -----------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
np.save(OUT_EMB, emb)
np.save(OUT_TOPK_IDS, topk_ids)
np.save(OUT_TOPK_PROBS, topk_probs)

print(f"Saved embeddings: {OUT_EMB} | shape={emb.shape} dtype={emb.dtype}")
print(f"Saved topk ids:   {OUT_TOPK_IDS} | shape={topk_ids.shape} dtype={topk_ids.dtype}")
print(f"Saved topk probs: {OUT_TOPK_PROBS} | shape={topk_probs.shape} dtype={topk_probs.dtype}")

