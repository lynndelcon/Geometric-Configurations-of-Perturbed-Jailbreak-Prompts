from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


SEED = 42
N_TREES = 400
N_SPLITS = 5

PROBA_DIR = Path(__file__).resolve().parent
ROOT = PROBA_DIR.parent
SVM_DIR = ROOT / "svm"
ANSWERS_DIR = ROOT / "answers"

OUT_JSON = PROBA_DIR / "rf_5models_firstprob_llama1b_results.json"
OUT_TEX = PROBA_DIR / "rf_5models_firstprob_llama1b_summary.tex"

REGION_CSV = SVM_DIR / "all_embeddings_region_labels_llama1b.csv"


@dataclass(frozen=True)
class Key:
    embedding_type: str
    condition: str
    family: str
    query_index: int
    paraphrase_index: int


def _parse_para_idx(s: str) -> int:
    s = s.strip()
    return -1 if s == "" else int(s)


def _load_region_lookup(path: Path) -> dict[Key, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing region CSV: {path}")
    out: dict[Key, str] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            k = Key(
                embedding_type=r["embedding_type"].strip(),
                condition=r["condition"].strip(),
                family=r["family"].strip(),
                query_index=int(r["query_index"]),
                paraphrase_index=_parse_para_idx(r["paraphrase_index"]),
            )
            out[k] = r["region"].strip()
    return out


def _load_np(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.asarray(np.load(path, allow_pickle=True))


def _load_top25_query_tokens() -> tuple[list[int], list[int]]:
    ctrl_ids = _load_np(ROOT / "control_prompts_nexttok_topk_ids.npy")
    adv_ids = _load_np(ROOT / "adv_prompts_nexttok_topk_ids.npy")
    if ctrl_ids.ndim != 2 or adv_ids.ndim != 2:
        raise ValueError("Expected query topk_ids as 2D arrays")
    ctrl_top1 = ctrl_ids[:, 0].astype(int).tolist()
    adv_top1 = adv_ids[:, 0].astype(int).tolist()

    def topk(vals: list[int], k: int) -> list[int]:
        c: dict[int, int] = {}
        for v in vals:
            c[v] = c.get(v, 0) + 1
        items = sorted(c.items(), key=lambda t: (-t[1], t[0]))
        return [tid for tid, _ in items[:k]]

    return topk(ctrl_top1, 25), topk(adv_top1, 25)


def _best_balanced_accuracy_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    # Deterministic candidate thresholds from score quantiles.
    qs = np.linspace(0.0, 1.0, 401)
    cands = np.unique(np.quantile(y_score, qs))
    best_thr = float(cands[0])
    best_bal = -1.0
    for thr in cands:
        pred = (y_score >= thr).astype(int)
        bal = float(balanced_accuracy_score(y_true, pred))
        if (bal > best_bal) or (bal == best_bal and thr < best_thr):
            best_bal = bal
            best_thr = float(thr)
    return best_thr, best_bal


def _adjusted_r2(r2: float, n: int, p: int) -> float:
    denom = n - p - 1
    if denom <= 0:
        return float("nan")
    return float(1.0 - (1.0 - r2) * ((n - 1) / denom))


def _build_dataset() -> tuple[np.ndarray, list[str], list[str], list[str], list[str], np.ndarray]:
    print("[checkpoint] Loading region labels and metadata...")
    region_lookup = _load_region_lookup(REGION_CSV)

    q_adv_lg = _load_np(ANSWERS_DIR / "llamaguard_labels_answers_llama1b_adv_queries.npy").reshape(-1).astype(int)
    para_lg = {
        fam: _load_np(ANSWERS_DIR / f"llamaguard_labels_answers_llama1b_para_{fam}_adv.npy").astype(int)
        for fam in ("synonyms", "swap", "symbol", "leetspeak")
    }

    ctrl_top25, adv_top25 = _load_top25_query_tokens()
    token_vocab = ctrl_top25 + adv_top25  # 50 total by construction
    token_index = {tid: i for i, tid in enumerate(token_vocab)}

    y_vals: list[float] = []
    top1_ids: list[int] = []
    family10: list[str] = []
    region4: list[str] = []
    lg3: list[str] = []
    y_bin_adv: list[int] = []

    def add_block(
        *,
        ids_arr: np.ndarray,
        probs_arr: np.ndarray,
        embedding_type: str,
        condition: str,
        family_key: str,
        lg_arr: np.ndarray | None,
    ) -> None:
        if ids_arr.shape[:-1] != probs_arr.shape[:-1]:
            raise ValueError(f"Shape mismatch ids/probs: {ids_arr.shape} vs {probs_arr.shape}")
        if ids_arr.shape[-1] < 1 or probs_arr.shape[-1] < 1:
            raise ValueError("topk arrays must have at least 1 entry in last dimension")

        if ids_arr.ndim == 2:
            for qi in range(ids_arr.shape[0]):
                tid = int(ids_arr[qi, 0])
                y = float(probs_arr[qi, 0])
                key = Key(embedding_type, condition, family_key, qi, -1)
                if key not in region_lookup:
                    raise KeyError(f"Missing region for key: {key}")
                top1_ids.append(tid)
                y_vals.append(y)
                family10.append(("query_control" if condition == "control" else "query_adv"))
                region4.append(region_lookup[key])
                if lg_arr is None:
                    lg3.append("missing")
                else:
                    lg3.append("compliance_1" if int(lg_arr[qi]) == 1 else "refusal_0")
                y_bin_adv.append(1 if condition == "adv" else 0)
        elif ids_arr.ndim == 3:
            for qi in range(ids_arr.shape[0]):
                for pj in range(ids_arr.shape[1]):
                    tid = int(ids_arr[qi, pj, 0])
                    y = float(probs_arr[qi, pj, 0])
                    key = Key(embedding_type, condition, family_key, qi, pj)
                    if key not in region_lookup:
                        raise KeyError(f"Missing region for key: {key}")
                    top1_ids.append(tid)
                    y_vals.append(y)
                    family10.append(f"{family_key}_{condition}")
                    region4.append(region_lookup[key])
                    if lg_arr is None:
                        lg3.append("missing")
                    else:
                        lg3.append("compliance_1" if int(lg_arr[qi, pj]) == 1 else "refusal_0")
                    y_bin_adv.append(1 if condition == "adv" else 0)
        else:
            raise ValueError(f"Unsupported ndim: {ids_arr.ndim}")

    print("[checkpoint] Building query blocks...")
    # Query blocks
    add_block(
        ids_arr=_load_np(ROOT / "control_prompts_nexttok_topk_ids.npy"),
        probs_arr=_load_np(ROOT / "control_prompts_nexttok_topk_probs.npy"),
        embedding_type="query",
        condition="control",
        family_key="none",
        lg_arr=None,
    )
    add_block(
        ids_arr=_load_np(ROOT / "adv_prompts_nexttok_topk_ids.npy"),
        probs_arr=_load_np(ROOT / "adv_prompts_nexttok_topk_probs.npy"),
        embedding_type="query",
        condition="adv",
        family_key="none",
        lg_arr=q_adv_lg,
    )

    print("[checkpoint] Building paraphrase blocks...")
    # Paraphrase blocks
    for fam in ("synonyms", "swap", "symbol", "leetspeak"):
        add_block(
            ids_arr=_load_np(ROOT / f"para_{fam}_control_nexttok_topk_ids.npy"),
            probs_arr=_load_np(ROOT / f"para_{fam}_control_nexttok_topk_probs.npy"),
            embedding_type="paraphrase",
            condition="control",
            family_key=fam,
            lg_arr=None,
        )
        add_block(
            ids_arr=_load_np(ROOT / f"para_{fam}_adv_nexttok_topk_ids.npy"),
            probs_arr=_load_np(ROOT / f"para_{fam}_adv_nexttok_topk_probs.npy"),
            embedding_type="paraphrase",
            condition="adv",
            family_key=fam,
            lg_arr=para_lg[fam],
        )

    n = len(y_vals)
    if n != 38592:
        raise ValueError(f"Expected 38592 rows, got {n}")
    print(f"[checkpoint] Built unified dataset rows: {n}")

    # Token feature matrix: 50 binary indicators
    x_tok = np.zeros((n, 50), dtype=np.float64)
    for i, tid in enumerate(top1_ids):
        idx = token_index.get(tid, None)
        if idx is not None:
            x_tok[i, idx] = 1.0

    return x_tok, family10, region4, lg3, token_vocab, np.asarray(y_vals, dtype=np.float64), np.asarray(y_bin_adv, dtype=int)


def _run_model(
    *,
    name: str,
    x_tok: np.ndarray,
    family10: list[str],
    region4: list[str],
    lg3: list[str],
    y: np.ndarray,
    y_bin_adv: np.ndarray,
    use_family: bool,
    use_region: bool,
    use_lg: bool,
) -> dict[str, float | str | int]:
    print(
        f"[checkpoint] Starting model: {name} | "
        f"use_family={use_family}, use_region={use_region}, use_lg={use_lg}"
    )
    n = y.shape[0]
    cols = []
    blocks = [x_tok]
    if use_family:
        cols.append(np.asarray(family10, dtype=object).reshape(-1, 1))
    if use_region:
        cols.append(np.asarray(region4, dtype=object).reshape(-1, 1))
    if use_lg:
        cols.append(np.asarray(lg3, dtype=object).reshape(-1, 1))

    # Build full feature table as object for ColumnTransformer.
    if cols:
        x_cat = np.hstack(cols)
        x_full = np.hstack([x_tok, x_cat])
    else:
        x_full = x_tok.copy()

    n_tok = x_tok.shape[1]
    cat_indices = list(range(n_tok, x_full.shape[1]))

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", list(range(n_tok))),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_indices),
        ],
        remainder="drop",
    )
    model = RandomForestRegressor(
        n_estimators=N_TREES,
        bootstrap=True,
        random_state=SEED,
        n_jobs=1,
    )
    pipe = Pipeline([("pre", pre), ("rf", model)])

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    y_oof = np.zeros_like(y, dtype=np.float64)
    for fold_idx, (tr, te) in enumerate(cv.split(x_full, y_bin_adv), start=1):
        print(f"[checkpoint]   {name} fold {fold_idx}/{N_SPLITS}: fit/predict...")
        pipe.fit(x_full[tr], y[tr])
        y_oof[te] = pipe.predict(x_full[te])

    r2 = float(r2_score(y, y_oof))
    # p for adjusted R2 from transformed design size after full fit preprocessor.
    pre.fit(x_full)
    p_eff = int(pre.transform(x_full[:1]).shape[1])
    adj_r2 = _adjusted_r2(r2, n=n, p=p_eff)

    thr, bal = _best_balanced_accuracy_threshold(y_bin_adv, y_oof)
    print(
        f"[checkpoint] Completed model: {name} | "
        f"adj_r2={adj_r2:.6f}, bal_acc={bal:.6f}, best_thr={thr:.6f}"
    )
    return {
        "model": name,
        "n_samples": int(n),
        "n_features_after_encoding": p_eff,
        "r2_oof": r2,
        "adjusted_r2_oof": float(adj_r2),
        "balanced_accuracy_oof": float(bal),
        "best_threshold_for_adv_vs_control": float(thr),
    }


def _fmt(x: float) -> str:
    return f"{x:.3f}" if np.isfinite(x) else "--"


def main() -> None:
    print("[checkpoint] Step 1/4: Build dataset and features...")
    x_tok, family10, region4, lg3, token_vocab, y, y_bin_adv = _build_dataset()
    print("[checkpoint] Step 2/4: Run 5 RF regression models...")

    specs = [
        ("Top-50 Tokens", False, False, False),
        ("Top-50 Tokens + Family", True, False, False),
        ("Top-50 Tokens + Region", False, True, False),
        ("Top-50 Tokens + Family + Region", True, True, False),
        ("Top-50 Tokens + Family + Region + LlamaGuard", True, True, True),
    ]

    results = []
    for name, uf, ur, ulg in specs:
        results.append(
            _run_model(
                name=name,
                x_tok=x_tok,
                family10=family10,
                region4=region4,
                lg3=lg3,
                y=y,
                y_bin_adv=y_bin_adv,
                use_family=uf,
                use_region=ur,
                use_lg=ulg,
            )
        )

    print("[checkpoint] Step 3/4: Write JSON and LaTeX outputs...")
    payload = {
        "seed": SEED,
        "n_trees": N_TREES,
        "n_splits": N_SPLITS,
        "target": "first probability value (top1 prob)",
        "balanced_accuracy_note": "Computed on adv-vs-control labels using OOF regression predictions thresholded at threshold maximizing OOF balanced accuracy.",
        "top50_token_vocab": token_vocab,
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Random-forest regression results on first-probability prediction (llama1b).}",
        r"\begin{tabular}{l c c}",
        r"\toprule",
        r"Model & Adjusted $R^2$ & Balanced Accuracy \\",
        r"\midrule",
    ]
    for r in results:
        lines.append(
            f"{r['model']} & {_fmt(float(r['adjusted_r2_oof']))} & {_fmt(float(r['balanced_accuracy_oof']))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")

    print("[checkpoint] Step 4/4: Done.")
    print(f"saved_json={OUT_JSON}")
    print(f"saved_tex={OUT_TEX}")
    for r in results:
        print(
            f"{r['model']}: adjusted_r2={float(r['adjusted_r2_oof']):.6f}, "
            f"balanced_accuracy={float(r['balanced_accuracy_oof']):.6f}"
        )


if __name__ == "__main__":
    main()
