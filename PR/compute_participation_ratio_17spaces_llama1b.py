from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PR_DIR = Path(__file__).resolve().parent
ROOT = PR_DIR.parent
SVM_DIR = ROOT / "svm"

REGION_CSV = SVM_DIR / "all_embeddings_region_labels_llama1b.csv"
OUT_TEX = PR_DIR / "participation_ratio_17spaces_llama1b.tex"
OUT_JSON = PR_DIR / "participation_ratio_17spaces_llama1b.json"


@dataclass(frozen=True)
class SpaceResult:
    name: str
    pr: float
    n: int


def _load_query(path: Path) -> np.ndarray:
    x = np.asarray(np.load(path, allow_pickle=True), dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D query embedding array at {path}, got {x.shape}")
    return x


def _load_para(path: Path) -> np.ndarray:
    x = np.asarray(np.load(path, allow_pickle=True), dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D paraphrase embedding array at {path}, got {x.shape}")
    q, m, d = x.shape
    return x.reshape(q * m, d)


def _participation_ratio(x: np.ndarray) -> float:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {x.shape}")
    n = x.shape[0]
    if n < 2:
        return float("nan")
    xc = x - np.mean(x, axis=0, keepdims=True)
    s = np.linalg.svd(xc, full_matrices=False, compute_uv=False)
    lam = (s * s) / (n - 1)
    denom = float(np.sum(lam * lam))
    if denom <= 0.0:
        return float("nan")
    return float((np.sum(lam) ** 2) / denom)


def _build_10_family_spaces() -> list[tuple[str, np.ndarray]]:
    spaces: list[tuple[str, np.ndarray]] = []

    x_q_ctrl = _load_query(ROOT / "control_prompts_lasttok_emb.npy")
    x_q_adv = _load_query(ROOT / "adv_prompts_lasttok_emb.npy")
    spaces.append(("Control Query", x_q_ctrl))
    spaces.append(("Jailbreak Query", x_q_adv))

    for fam in ("synonyms", "swap", "symbol", "leetspeak"):
        x_ctrl = _load_para(ROOT / f"para_{fam}_control_lasttok_emb.npy")
        x_adv = _load_para(ROOT / f"para_{fam}_adv_lasttok_emb.npy")
        fam_title = "Numbers" if fam == "symbol" else fam.capitalize()
        spaces.append((f"Control {fam_title} Para", x_ctrl))
        spaces.append((f"Jailbreak {fam_title} Para", x_adv))

    if len(spaces) != 10:
        raise RuntimeError(f"Expected 10 family spaces, got {len(spaces)}")
    return spaces


def _build_all_embeddings() -> tuple[np.ndarray, np.ndarray]:
    blocks_ctrl = [_load_query(ROOT / "control_prompts_lasttok_emb.npy")]
    blocks_adv = [_load_query(ROOT / "adv_prompts_lasttok_emb.npy")]
    for fam in ("synonyms", "swap", "symbol", "leetspeak"):
        blocks_ctrl.append(_load_para(ROOT / f"para_{fam}_control_lasttok_emb.npy"))
        blocks_adv.append(_load_para(ROOT / f"para_{fam}_adv_lasttok_emb.npy"))
    x_ctrl = np.vstack(blocks_ctrl)
    x_adv = np.vstack(blocks_adv)
    if x_ctrl.shape[0] != 19296 or x_adv.shape[0] != 19296:
        raise ValueError(f"Expected 19296 per condition, got control={x_ctrl.shape[0]}, jailbreak={x_adv.shape[0]}")
    return x_ctrl, x_adv


def _build_region_spaces() -> list[tuple[str, np.ndarray]]:
    x_ctrl, x_adv = _build_all_embeddings()
    x_all = np.vstack([x_ctrl, x_adv])
    if x_all.shape[0] != 38592:
        raise ValueError(f"Expected 38592 rows in all embeddings, got {x_all.shape[0]}")

    rows: list[dict[str, str]] = []
    with REGION_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if len(rows) != 38592:
        raise ValueError(f"Expected 38592 rows in region csv, got {len(rows)}")

    idx_by_region: dict[str, list[int]] = {
        "usual_tokens": [],
        "unusual_tokens": [],
        "compliance_jailbreak": [],
        "refusal_jailbreak": [],
    }
    for i, r in enumerate(rows):
        reg = r["region"].strip()
        if reg in idx_by_region:
            idx_by_region[reg].append(i)

    label_map = {
        "usual_tokens": "Usual Tokens Region",
        "unusual_tokens": "Unusual Tokens Region",
        "compliance_jailbreak": "Compliance Region",
        "refusal_jailbreak": "Refusal Region",
    }

    out: list[tuple[str, np.ndarray]] = []
    for reg in ("usual_tokens", "unusual_tokens", "compliance_jailbreak", "refusal_jailbreak"):
        idx = np.asarray(idx_by_region[reg], dtype=np.int64)
        out.append((label_map[reg], x_all[idx]))
    return out


def _to_latex(results_sorted: list[SpaceResult]) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Participation Ratio across 17 embedding spaces (llama1b), sorted in ascending PR.}",
        r"\begin{tabular}{l c c}",
        r"\toprule",
        r"Embedding Space & PR & $n$ \\",
        r"\midrule",
    ]
    for r in results_sorted:
        lines.append(f"{r.name} & {r.pr:.4f} & {r.n} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    print("[1/4] Building the 10 family spaces...")
    family_spaces = _build_10_family_spaces()

    print("[2/4] Building global spaces (all, control, jailbreak)...")
    x_ctrl, x_adv = _build_all_embeddings()
    global_spaces = [
        ("All Embeddings", np.vstack([x_ctrl, x_adv])),
        ("All Control Embeddings", x_ctrl),
        ("All Jailbreak Embeddings", x_adv),
    ]

    print("[3/4] Building the 4 region spaces from CSV...")
    region_spaces = _build_region_spaces()

    all_spaces = global_spaces + family_spaces + region_spaces
    if len(all_spaces) != 17:
        raise RuntimeError(f"Expected exactly 17 spaces, got {len(all_spaces)}")

    print("[4/4] Computing PR and writing outputs...")
    results = [SpaceResult(name=n, pr=_participation_ratio(x), n=int(x.shape[0])) for n, x in all_spaces]
    results_sorted = sorted(results, key=lambda z: z.pr)

    tex = _to_latex(results_sorted)
    OUT_TEX.write_text(tex, encoding="utf-8")

    json_rows = [
        {"embedding_space": r.name, "pr": r.pr, "n": r.n}
        for r in results_sorted
    ]
    import json
    OUT_JSON.write_text(json.dumps(json_rows, indent=2), encoding="utf-8")

    print(f"saved_tex={OUT_TEX}")
    print(f"saved_json={OUT_JSON}")


if __name__ == "__main__":
    main()
