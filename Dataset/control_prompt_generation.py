from datasets import load_dataset

ds = load_dataset(
    "pegah-a/small-natural-instructions",
    split="train",
    verification_mode="no_checks",
)

import numpy as np
from adv_queries import prompts as adv_queries_prompts

LEN_ADV_N = 96
if len(adv_queries_prompts) < LEN_ADV_N:
    raise ValueError(
        f"Expected at least {LEN_ADV_N} adversarial queries, found {len(adv_queries_prompts)}."
    )
len_adv = np.asarray(
    [len(str(item.get("prompt", "")).strip()) for item in adv_queries_prompts[:LEN_ADV_N]],
    dtype=int,
)

"""
Length-matched selection from HuggingFace dataset `ds` (Natural Instructions-style)

Goal:
- len_adv is a 1D array/list of desired character lengths (one per target).
- For each target length len_adv[j], select EXACTLY ONE definition from ds such that:
    abs(len_chars(definition_text) - len_adv[j]) <= 20
  using a dynamic tolerance schedule.
- If still missing, use a fallback: pick the closest available definition anywhere (min abs diff),
  while still enforcing uniqueness (no reused ds index, no duplicate text).

Outputs:
- result: list of length len(len_adv) with dict entries (selection metadata) or None (should be none after fallback if pool is sufficient)
- selected_definitions: list of raw definitions (str or list[str]) for successful selections
- selected_meta: list of metadata dicts for successful selections
"""

import re
from collections import defaultdict, deque
import numpy as np


# ------------------------
# CONFIG
# ------------------------
N_TASKS = 967
TASK_PREFIX_RE = re.compile(r"^(task\d{3})_")

# Use np.arange for tolerance schedule: 10, 20, ..., 200
TOL_SCHEDULE = tuple(np.arange(10, 201, 10))  # dynamic widening
STRICT_TOL = 20                              # used for reporting only


# ------------------------
# HELPERS
# ------------------------
def def_to_text(defn):
    """Normalize definition to a single string for char-length counting."""
    if isinstance(defn, list):
        return " ".join(s.strip() for s in defn if isinstance(s, str)).strip()
    if isinstance(defn, str):
        return defn.strip()
    return ""


def task_id_from_task_name(task_name: str):
    """Return 'taskNNN' if task_name begins with taskNNN_ and NNN in [001..N_TASKS]. Else None."""
    m = TASK_PREFIX_RE.match(task_name or "")
    if not m:
        return None
    tid = m.group(1)  # task001
    k = int(tid[4:])
    if 1 <= k <= N_TASKS:
        return tid
    return None


# ------------------------
# MAIN (expects `ds` and `len_adv` already defined in your notebook/session)
# ------------------------
len_adv = np.asarray(len_adv).astype(int)

# buckets[L] = deque of candidates with char length L
# candidate = (ds_index, task_id, raw_def, text, char_len)
buckets = defaultdict(deque)

for idx, ex in enumerate(ds):
    tid = task_id_from_task_name(ex.get("task_name", ""))
    if tid is None:
        continue

    raw_def = ex.get("definition", None)
    text = def_to_text(raw_def)
    if not text:
        continue

    L = len(text)
    buckets[L].append((idx, tid, raw_def, text, L))

# Uniqueness tracking
used_ds = set()
used_texts = set()


def pick_one_dynamic(t: int, tolerances=TOL_SCHEDULE):
    """
    Try increasingly wide tolerances; pick best candidate within current tolerance.

    Returns:
      best: (diff, char_len, cand) or None
      tol_used: tolerance that succeeded or None
    """
    for tol_local in tolerances:
        best = None  # (diff, char_len, cand)
        # scan candidate lengths in [t - tol_local, t + tol_local]
        for L in range(t - tol_local, t + tol_local + 1):
            if L not in buckets:
                continue
            q = buckets[L]
            # try up to len(q) elements (rotate for fairness)
            for _ in range(len(q)):
                cand = q[0]
                q.rotate(-1)
                idx, tid, raw_def, text, char_len = cand

                if idx in used_ds:
                    continue
                if text in used_texts:
                    continue

                diff = abs(char_len - t)
                key = (diff, char_len)
                if best is None or key < (best[0], best[1]):
                    best = (diff, char_len, cand)
        if best is not None:
            return best, tol_local
    return None, None


def pick_closest_anywhere(t: int):
    """
    Fallback: pick the closest unused candidate anywhere in the pool (no tolerance bound).
    Returns:
      best: (diff, char_len, cand) or None
    """
    best = None
    # iterate all buckets (length -> candidates)
    for L, q in buckets.items():
        for cand in q:
            idx, tid, raw_def, text, char_len = cand
            if idx in used_ds:
                continue
            if text in used_texts:
                continue
            diff = abs(char_len - t)
            key = (diff, char_len)
            if best is None or key < (best[0], best[1]):
                best = (diff, char_len, cand)
    return best


# Pass 1: dynamic tolerance
result = [None] * len(len_adv)
failures = []  # list of (j, target_len)

for j, t in enumerate(len_adv):
    best, tol_used = pick_one_dynamic(int(t))
    if best is None:
        failures.append((j, int(t)))
        continue

    diff, L, cand = best
    idx, tid, raw_def, text, char_len = cand

    used_ds.add(idx)
    used_texts.add(text)

    result[j] = {
        "target": int(t),
        "diff": int(diff),
        "tol_used": int(tol_used),
        "fallback": False,
        "within_strict_tol": (abs(char_len - int(t)) <= STRICT_TOL),
        "ds_index": int(idx),
        "task_id": tid,
        "char_len": int(char_len),
        "definition": raw_def,
    }

print(f"[Pass1] Targets: {len(len_adv)} | Selected: {sum(r is not None for r in result)} | Missing: {len(failures)}")


# Pass 2: fallback for missing (closest anywhere)
still_missing = []
for j, t in failures:
    best = pick_closest_anywhere(int(t))
    if best is None:
        still_missing.append((j, int(t)))
        continue

    diff, L, cand = best
    idx, tid, raw_def, text, char_len = cand

    used_ds.add(idx)
    used_texts.add(text)

    result[j] = {
        "target": int(t),
        "diff": int(diff),
        "tol_used": None,
        "fallback": True,
        "within_strict_tol": (abs(char_len - int(t)) <= STRICT_TOL),
        "ds_index": int(idx),
        "task_id": tid,
        "char_len": int(char_len),
        "definition": raw_def,
    }

print(f"[Pass2] Filled: {len(failures) - len(still_missing)} | Still missing: {len(still_missing)}")
if still_missing:
    print("First still-missing (up to 10):", still_missing[:10])


# Flatten outputs
selected_meta = [r for r in result if r is not None]
selected_definitions = [r["definition"] for r in selected_meta]

print(f"[Final] Total selected: {len(selected_definitions)} / {len(len_adv)}")

# Optional sanity checks
# - duplicates should be none

# Plot histogram comparing len_adv and selected_definitions lengths
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
selected_lengths = [len(def_to_text(d)) for d in selected_definitions]
plt.hist(len_adv, bins=30, alpha=0.5, label='len_adv')
plt.hist(selected_lengths, bins=30, alpha=0.5, label='selected_definitions')
plt.legend()
plt.xlabel('Character Length')
plt.ylabel('Frequency')
plt.title('Length Distribution: len_adv vs. selected_definitions')
plt.savefig('length_distribution_hist.png')
print('Histogram saved as length_distribution_hist.png')

# --- CSV OUTPUT: 10 control prompts (from still_missing) and 10 random adversarial prompts ---
import csv
import random
import ast

# 1. Get 10 control prompts: definitions for the first 10 still-missing indexes (if any), else just first 10 selected definitions
control_prompts = []
if still_missing:
    for j, t in still_missing[:10]:
        # Try to get the definition for this index from ds
        ex = ds[j]
        defn = ex.get("definition", "")
        # Normalize to string
        if isinstance(defn, list):
            defn = " ".join(str(s).strip() for s in defn if isinstance(s, str)).strip()
        elif not isinstance(defn, str):
            defn = str(defn)
        control_prompts.append(defn)
else:
    # fallback: just use first 10 selected definitions
    for defn in selected_definitions[:10]:
        if isinstance(defn, list):
            defn = " ".join(str(s).strip() for s in defn if isinstance(s, str)).strip()
        elif not isinstance(defn, str):
            defn = str(defn)
        control_prompts.append(defn)

np.save('control_prompts.npy', control_prompts)
# 2. Get 10 random adversarial prompts from adv_queries.py

# Robust extraction: find all 'prompt': ... occurrences using regex
import re
adv_prompts = []
with open('adv_queries.py', 'r', encoding='utf-8') as f:
    content = f.read()
    # Regex to find 'prompt': "..." or 'prompt': '...'
    pattern = re.compile(r"['\"]prompt['\"]\s*:\s*(['\"])(.*?)\1", re.DOTALL)
    adv_prompts = [m.group(2).strip() for m in pattern.finditer(content)]

if len(adv_prompts) >= 10:
    adv_sample = random.sample(adv_prompts, 10)
else:
    adv_sample = adv_prompts[:10]

# 3. Write to CSV
csv_filename = 'control_vs_adversarial_prompts.csv'
with open(csv_filename, 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['control prompt', 'adversarial prompt'])
    for c, a in zip(control_prompts, adv_sample):
        writer.writerow([c, a])
print(f'CSV file saved as {csv_filename}')
