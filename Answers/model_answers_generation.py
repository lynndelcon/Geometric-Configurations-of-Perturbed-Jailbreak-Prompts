#Model's answers

#For query prompts
#module to load: accelerate, transformers, huggingfface, torch

import os
import time
import tempfile
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# USER SETTINGS
# ============================================================
SYSTEM = (
    "You are a helpful assistant. "
    "Answer the user's prompt directly, clearly, and accurately."
)

MODEL_NAME = ""
OUT_PATH = ""

TEMPERATURE = 0.7
TOP_P = 0.95
MAX_TOKENS = 700
MAX_ATTEMPTS = 5
DO_SAMPLE = False   # False = deterministic decoding

# If the model is gated/private, run:
# huggingface-cli login
# or in notebook:
# from huggingface_hub import login
# login("your_token")


# ============================================================
# ATOMIC SAVE (safe checkpointing)
# ============================================================
def atomic_save_npy(path: str, arr_obj: np.ndarray) -> None:
    path_dir = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=path_dir, prefix=".tmp_", suffix=".npy")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            np.save(f, arr_obj, allow_pickle=True)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def save_partial(out_path: str, answers: List[str]) -> None:
    arr = np.array(answers, dtype=object)
    atomic_save_npy(out_path, arr)
    print(f"[checkpoint] Saved {out_path} | shape={arr.shape}")


# ============================================================
# MODEL / TOKENIZER LOADING
# ============================================================
def load_hf_model(model_name: str):
    print(f"[model] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Some chat models do not define pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Safer dtype choice
    if torch.cuda.is_available():
        # bfloat16 is often preferable if supported, otherwise float16
        major, _ = torch.cuda.get_device_capability()
        torch_dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        torch_dtype = torch.float32

    print(f"[model] Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if not torch.cuda.is_available():
        model = model.to("cpu")

    device = next(model.parameters()).device
    print(f"[model] Loaded on device: {device}")
    print(f"[model] torch_dtype: {torch_dtype}")

    # Avoid generation warnings in deterministic mode
    try:
        model.generation_config.top_k = None
        model.generation_config.top_p = None
        model.generation_config.temperature = None
    except Exception:
        pass

    return tokenizer, model


# ============================================================
# SINGLE ANSWER GENERATION
# ============================================================
def one_answer(
    prompt: str,
    *,
    tokenizer,
    model,
    seed: int,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 512,
    do_sample: bool = True,
) -> str:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    model_device = next(model.parameters()).device
    model_inputs = {k: v.to(model_device) for k, v in model_inputs.items()}

    generate_kwargs = dict(
        **model_inputs,
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    with torch.no_grad():
        generated_ids = model.generate(**generate_kwargs)

    input_len = model_inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, input_len:]

    txt = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    if (txt.startswith('"') and txt.endswith('"')) or (txt.startswith("'") and txt.endswith("'")):
        txt = txt[1:-1].strip()

    return txt


# ============================================================
# RETRY WRAPPER
# ============================================================
def generate_answer_with_retry(
    prompt: str,
    *,
    tokenizer,
    model,
    base_seed: int,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 512,
    max_attempts: int = 5,
    do_sample: bool = True,
) -> str:
    fail_streak = 0

    for j in range(max_attempts):
        seed = base_seed + j
        try:
            answer = one_answer(
                prompt,
                tokenizer=tokenizer,
                model=model,
                seed=seed,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                do_sample=do_sample,
            )
            if answer:
                return answer

        except RuntimeError as e:
            fail_streak += 1
            err = str(e).lower()

            if "out of memory" in err and torch.cuda.is_available():
                torch.cuda.empty_cache()

            cooldown = min(60.0, 2.0 * (2 ** min(fail_streak, 5)))
            print(
                f"  [warn] runtime error (attempt {j+1}/{max_attempts}, seed={seed}): "
                f"{repr(e)} | cooldown={cooldown:.1f}s"
            )
            time.sleep(cooldown)

        except Exception as e:
            fail_streak += 1
            cooldown = min(60.0, 2.0 * (2 ** min(fail_streak, 5)))
            print(
                f"  [warn] error (attempt {j+1}/{max_attempts}, seed={seed}): "
                f"{repr(e)} | cooldown={cooldown:.1f}s"
            )
            time.sleep(cooldown)

    return "[ERROR] No answer generated."


# ============================================================
# MAIN NOTEBOOK FUNCTION
# ============================================================
def run_generation(
    prompts_only: List[str],
    model_name: str = MODEL_NAME,
    out_path: str = OUT_PATH,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_tokens: int = MAX_TOKENS,
    max_attempts: int = MAX_ATTEMPTS,
    do_sample: bool = DO_SAMPLE,
):
    tokenizer, model = load_hf_model(model_name)

    answers: List[str] = []
    start_idx = 0

    if os.path.exists(out_path):
        try:
            existing = np.load(out_path, allow_pickle=True)
            answers = existing.tolist()
            start_idx = len(answers)
            print(f"[resume] Loaded {start_idx} completed prompts from {out_path}")
        except Exception as e:
            print(f"[resume] Found {out_path} but failed to load it: {repr(e)}")
            print("[resume] Starting from scratch to avoid mixing partial/corrupt state.")
            answers = []
            start_idx = 0
    else:
        print("[resume] No checkpoint found, starting from scratch")

    total = len(prompts_only)

    for i in range(start_idx, total):
        p = prompts_only[i]
        print(f"Prompt {i+1}/{total} | chars={len(p)}")

        answer = generate_answer_with_retry(
            p,
            tokenizer=tokenizer,
            model=model,
            base_seed=2026 + i * 10000,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_attempts=max_attempts,
            do_sample=do_sample,
        )

        answers.append(answer)
        save_partial(out_path, answers)
        time.sleep(0.05)

    arr = np.array(answers, dtype=object)
    atomic_save_npy(out_path, arr)
    print(f"[done] Saved {out_path} | shape={arr.shape}")

    return answers

#For query prompts	
from adv_queries import prompts

prompts_only = [entry["prompt"] for entry in prompts]

answers = run_generation(
    prompts_only,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    out_path="answers_llama3b_adv_queries.npy",
    max_tokens=700,
    max_attempts=5,
    do_sample=False,
)

#For paraphrases

def run_paraphrase_generation(
    para_path: str = PARA_PATH,
    out_path: str = OUT_PATH,
    model_name: str = MODEL_NAME,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_tokens: int = MAX_TOKENS,
    max_attempts: int = MAX_ATTEMPTS,
    do_sample: bool = DO_SAMPLE,
):
    # -------------------------
    # Load paraphrases
    # -------------------------
    para = np.load(para_path, allow_pickle=True)

    if para.ndim != 2:
        raise ValueError(
            f"Expected a 2D paraphrase array, got shape={para.shape} from {para_path}"
        )

    n_rows, n_cols = para.shape   # e.g. 96 x 50
    flat_prompts = para.reshape(-1).tolist()
    total = len(flat_prompts)

    print(f"[input] Loaded {para_path} with shape={para.shape}")
    print(f"[input] Flattened to {total} prompts")

    tokenizer, model = load_hf_model(model_name)

    answers: List[str] = []
    start_idx = 0

    # -------------------------
    # Resume logic
    # -------------------------
    if os.path.exists(out_path):
        try:
            existing = np.load(out_path, allow_pickle=True)

            if existing.ndim == 1:
                answers = existing.tolist()
                start_idx = len(answers)

            elif existing.ndim == 2:
                answers = existing.reshape(-1).tolist()
                start_idx = len(answers)

            else:
                raise ValueError(f"Unsupported checkpoint shape: {existing.shape}")

            if start_idx > total:
                raise ValueError(
                    f"Checkpoint has {start_idx} answers but input has only {total} prompts"
                )

            print(f"[resume] Loaded {start_idx}/{total} completed prompts from {out_path}")

        except Exception as e:
            print(f"[resume] Found {out_path} but failed to load it: {repr(e)}")
            print("[resume] Starting from scratch to avoid mixing partial/corrupt state.")
            answers = []
            start_idx = 0
    else:
        print("[resume] No checkpoint found, starting from scratch")

    # -------------------------
    # Generation loop
    # -------------------------
    for i in range(start_idx, total):
        p = flat_prompts[i]

        row = i // n_cols
        col = i % n_cols

        print(
            f"Prompt {i+1}/{total} | group={row+1}/{n_rows} | paraphrase={col+1}/{n_cols} | chars={len(str(p))}"
        )

        answer = generate_answer_with_retry(
            p,
            tokenizer=tokenizer,
            model=model,
            base_seed=2026 + i * 10000,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_attempts=max_attempts,
            do_sample=do_sample,
        )

        answers.append(answer)

        # save flat checkpoint while running
        save_partial(out_path, answers)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        time.sleep(0.05)

    # -------------------------
    # Final save in original 2D shape
    # -------------------------
    arr = np.array(answers, dtype=object).reshape(n_rows, n_cols)
    atomic_save_npy(out_path, arr)
    print(f"[done] Saved {out_path} | shape={arr.shape}")

    return arr

def run_paraphrase_generation(
    para_path: str = PARA_PATH,
    out_path: str = OUT_PATH,
    model_name: str = MODEL_NAME,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_tokens: int = MAX_TOKENS,
    max_attempts: int = MAX_ATTEMPTS,
    do_sample: bool = DO_SAMPLE,
):
    # -------------------------
    # Load paraphrases
    # -------------------------
    para = np.load(para_path, allow_pickle=True)

    if para.ndim != 2:
        raise ValueError(
            f"Expected a 2D paraphrase array, got shape={para.shape} from {para_path}"
        )

    n_rows, n_cols = para.shape   # e.g. 96 x 50
    flat_prompts = para.reshape(-1).tolist()
    total = len(flat_prompts)

    print(f"[input] Loaded {para_path} with shape={para.shape}")
    print(f"[input] Flattened to {total} prompts")

    tokenizer, model = load_hf_model(model_name)

    answers: List[str] = []
    start_idx = 0

    # -------------------------
    # Resume logic
    # -------------------------
    if os.path.exists(out_path):
        try:
            existing = np.load(out_path, allow_pickle=True)

            if existing.ndim == 1:
                answers = existing.tolist()
                start_idx = len(answers)

            elif existing.ndim == 2:
                answers = existing.reshape(-1).tolist()
                start_idx = len(answers)

            else:
                raise ValueError(f"Unsupported checkpoint shape: {existing.shape}")

            if start_idx > total:
                raise ValueError(
                    f"Checkpoint has {start_idx} answers but input has only {total} prompts"
                )

            print(f"[resume] Loaded {start_idx}/{total} completed prompts from {out_path}")

        except Exception as e:
            print(f"[resume] Found {out_path} but failed to load it: {repr(e)}")
            print("[resume] Starting from scratch to avoid mixing partial/corrupt state.")
            answers = []
            start_idx = 0
    else:
        print("[resume] No checkpoint found, starting from scratch")

    # -------------------------
    # Generation loop
    # -------------------------
    for i in range(start_idx, total):
        p = flat_prompts[i]

        row = i // n_cols
        col = i % n_cols

        print(
            f"Prompt {i+1}/{total} | group={row+1}/{n_rows} | paraphrase={col+1}/{n_cols} | chars={len(str(p))}"
        )

        answer = generate_answer_with_retry(
            p,
            tokenizer=tokenizer,
            model=model,
            base_seed=2026 + i * 10000,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_attempts=max_attempts,
            do_sample=do_sample,
        )

        answers.append(answer)

        # save flat checkpoint while running
        save_partial(out_path, answers)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        time.sleep(0.05)

    # -------------------------
    # Final save in original 2D shape
    # -------------------------
    arr = np.array(answers, dtype=object).reshape(n_rows, n_cols)
    atomic_save_npy(out_path, arr)
    print(f"[done] Saved {out_path} | shape={arr.shape}")

    return arr
    
#load accelerate
LABEL="swap"
answers_2d = run_paraphrase_generation(
    para_path=f"para_{LABEL}_adv.npy",
    out_path=f"answers_llama3b_para_{LABEL}_adv.npy",
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    max_tokens=700,
    max_attempts=5,
    do_sample=False,
)