from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


PROBA_DIR = Path(__file__).resolve().parent
ROOT = PROBA_DIR.parent
ANSWERS_DIR = ROOT / "answers"

PART1_RESULTS_JSON = PROBA_DIR / "rf_5models_firstprob_llama1b_results.json"
THRESHOLD_MODEL_NAME = "Top-50 Tokens + Family"
MIN_SELECTED_TOKEN_COUNT = 20
OUT_JSON = PROBA_DIR / "gee_label_vs_token_pcat_family_top3_queries.json"
OUT_TXT = PROBA_DIR / "gee_label_vs_token_pcat_family_top3_queries_summary.txt"
OUT_TEX = PROBA_DIR / "gee_label_vs_token_pcat_family_top3_queries.tex"
TOKENIZER_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"


def _resolve_file(filename: str) -> Path:
    p_proba = PROBA_DIR / filename
    if p_proba.exists():
        return p_proba
    p_root = ROOT / filename
    if p_root.exists():
        return p_root
    raise FileNotFoundError(f"Missing required file: {filename}")


def _extract_query_top1(arr: np.ndarray, *, name: str) -> np.ndarray:
    a = np.asarray(arr)
    # Query file expected as (96, 50): take first top-k entry => (96,)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D query top-k array for {name}, got {a.shape}")
    return np.asarray(a[:, 0], dtype=int).reshape(-1)


def _extract_paraphrase_top1_ids(arr: np.ndarray, *, name: str) -> np.ndarray:
    a = np.asarray(arr)
    # Paraphrase files expected as (96, 50, 50):
    # take first top-k entry => (96, 50), then flatten => (4800,)
    if a.ndim == 3:
        return np.asarray(a[:, :, 0], dtype=int).reshape(-1)
    if a.ndim == 2:
        return np.asarray(a[:, 0], dtype=int).reshape(-1)
    if a.ndim == 1:
        return np.asarray(a, dtype=int).reshape(-1)
    raise ValueError(f"Unsupported paraphrase top-k array shape for {name}: {a.shape}")


def _extract_paraphrase_top1_probs(arr: np.ndarray, *, name: str) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 3:
        return np.asarray(a[:, :, 0], dtype=float).reshape(-1)
    if a.ndim == 2:
        return np.asarray(a[:, 0], dtype=float).reshape(-1)
    if a.ndim == 1:
        return np.asarray(a, dtype=float).reshape(-1)
    raise ValueError(f"Unsupported paraphrase top-k array shape for {name}: {a.shape}")


def _top6_adv_query_tokens() -> list[int]:
    arr = np.asarray(np.load(_resolve_file("adv_prompts_nexttok_topk_ids.npy"), allow_pickle=True))
    top1 = _extract_query_top1(arr, name="adv_prompts_nexttok_topk_ids.npy")
    vc = pd.Series(top1).value_counts()
    return [int(x) for x in vc.head(6).index.tolist()]


def _build_para_rows(family_name: str, ids_file: str, probs_file: str, labels_file: str) -> pd.DataFrame:
    ids_path = _resolve_file(ids_file)
    probs_path = _resolve_file(probs_file)

    ids = np.load(ids_path, allow_pickle=True)
    probs = np.load(probs_path, allow_pickle=True)
    labels = np.asarray(np.load(ANSWERS_DIR / labels_file, allow_pickle=True), dtype=int)

    ids = _extract_paraphrase_top1_ids(ids, name=ids_file)
    probs = _extract_paraphrase_top1_probs(probs, name=probs_file)

    # labels: (96, 50) -> one label per paraphrase with cluster = base query.
    if labels.ndim != 2:
        raise ValueError(f"Expected (96,50) labels for {family_name}, got {labels.shape}")
    n_query, n_var = labels.shape
    y = labels.reshape(-1)
    cluster = np.repeat(np.arange(n_query), n_var)

    n = min(len(ids), len(probs), len(y), len(cluster))
    return pd.DataFrame(
        {
            "family": family_name,
            "prompt_index": cluster[:n].astype(int),
            "top1_prob": probs[:n],
            "top1_token_id": ids[:n],
            "label": y[:n],
        }
    )


def _build_dataset() -> pd.DataFrame:
    frames = []
    frames.append(
        _build_para_rows(
            "leetspeak",
            "para_leetspeak_adv_nexttok_topk_ids.npy",
            "para_leetspeak_adv_nexttok_topk_probs.npy",
            "llamaguard_labels_answers_llama1b_para_leetspeak_adv.npy",
        )
    )
    frames.append(
        _build_para_rows(
            "swap",
            "para_swap_adv_nexttok_topk_ids.npy",
            "para_swap_adv_nexttok_topk_probs.npy",
            "llamaguard_labels_answers_llama1b_para_swap_adv.npy",
        )
    )
    frames.append(
        _build_para_rows(
            "numbers",
            "para_symbol_adv_nexttok_topk_ids.npy",
            "para_symbol_adv_nexttok_topk_probs.npy",
            "llamaguard_labels_answers_llama1b_para_symbol_adv.npy",
        )
    )
    frames.append(
        _build_para_rows(
            "synonyms",
            "para_synonyms_adv_nexttok_topk_ids.npy",
            "para_synonyms_adv_nexttok_topk_probs.npy",
            "llamaguard_labels_answers_llama1b_para_synonyms_adv.npy",
        )
    )
    return pd.concat(frames, ignore_index=True)


def _load_threshold_from_part1(path: Path, model_name: str) -> float:
    if not path.exists():
        raise FileNotFoundError(f"Missing threshold source JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results") or []
    for r in rows:
        if str(r.get("model")) == model_name:
            return float(r["best_threshold_for_adv_vs_control"])
    raise KeyError(f"Model '{model_name}' not found in {path}")


def _latex_escape(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def _load_tokenizer(model_id: str) -> Any | None:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
    except Exception:
        return None


def _token_label(tokenizer: Any | None, token_id: int) -> str:
    if tokenizer is None:
        return str(token_id)
    try:
        piece = tokenizer.convert_ids_to_tokens(int(token_id))
        return str(piece) if piece is not None else str(token_id)
    except Exception:
        return str(token_id)


def _humanize_term(term: str, tokenizer: Any | None) -> str:
    # Example statsmodels term:
    # C(token_group, Treatment(reference='other_tokens'))[T.12345]
    marker = "C(token_group, Treatment(reference='other_tokens'))[T."
    if term.startswith(marker) and term.endswith("]"):
        tok_id_str = term[len(marker):-1]
        try:
            tok_id = int(tok_id_str)
            tok_piece = _token_label(tokenizer, tok_id)
            return f"token_group[{tok_id}: {tok_piece}]"
        except Exception:
            return term
    return term


def main() -> None:
    print("[checkpoint] Step 0/6: Loading threshold from part1 results...")
    threshold = _load_threshold_from_part1(PART1_RESULTS_JSON, THRESHOLD_MODEL_NAME)
    print(
        f"[checkpoint] Using threshold={threshold:.6f} from "
        f"{PART1_RESULTS_JSON.name} model='{THRESHOLD_MODEL_NAME}'"
    )
    print("[checkpoint] Loading tokenizer for token-id term decoding...")
    tokenizer = _load_tokenizer(TOKENIZER_MODEL_ID)
    if tokenizer is None:
        print("[checkpoint] Tokenizer unavailable; LaTeX terms will keep token IDs.")
    else:
        print(f"[checkpoint] Tokenizer loaded: {TOKENIZER_MODEL_ID}")

    print("[checkpoint] Step 1/6: Selecting top-6 tokens from adversarial queries...")
    selected = _top6_adv_query_tokens()

    print("[checkpoint] Step 2/6: Building jailbreak paraphrase dataset...")
    df = _build_dataset()
    # Cluster on base prompt index across jailbreak paraphrase families (96 clusters).
    df["cluster_id"] = df["prompt_index"].astype(int).map(lambda i: f"prompt_{i}")
    df["p_cat"] = (df["top1_prob"] >= threshold).astype(int)
    print(f"[checkpoint] Dataset rows={len(df)}, clusters={df['cluster_id'].nunique()}")

    print("[checkpoint] Step 3/6: Applying selected-token count filter...")
    token_counts = df["top1_token_id"].value_counts()
    selected_nontrivial = [int(t) for t in selected if int(token_counts.get(int(t), 0)) >= MIN_SELECTED_TOKEN_COUNT]
    if len(selected_nontrivial) == 0:
        selected_nontrivial = [int(t) for t in selected]
    print(f"[checkpoint] Selected tokens kept={selected_nontrivial}")
    df["token_group"] = df["top1_token_id"].apply(lambda x: str(int(x)) if int(x) in selected_nontrivial else "other_tokens")

    print("[checkpoint] Step 4/6: Fitting GEE logistic regression...")
    formula = (
        "label ~ "
        "C(p_cat, Treatment(reference=0)) + "
        "C(family, Treatment(reference='synonyms')) + "
        "C(token_group, Treatment(reference='other_tokens'))"
    )
    model = smf.gee(
        formula,
        groups="cluster_id",
        data=df,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    res = model.fit()
    print(f"[checkpoint] GEE fit complete. converged={bool(res.converged)}")

    print("[checkpoint] Step 5/6: Building parameter summary...")
    conf = res.conf_int()
    params = []
    for name in res.params.index:
        beta = float(res.params[name])
        lo = float(conf.loc[name, 0])
        hi = float(conf.loc[name, 1])
        params.append(
            {
                "term": name,
                "coef": beta,
                "se_robust": float(res.bse[name]),
                "z": float(res.tvalues[name]),
                "p_value": float(res.pvalues[name]),
                "coef_ci95_low": lo,
                "coef_ci95_high": hi,
                "odds_ratio": float(np.exp(beta)),
                "odds_ratio_ci95_low": float(np.exp(lo)),
                "odds_ratio_ci95_high": float(np.exp(hi)),
            }
        )

    out = {
        "documentation": {
            "goal": "Association between LlamaGuard label and predictors (top1 token id group, probability category, family) on jailbreak data.",
            "token_selection_rule": "Use top-6 most frequent top1 token IDs from adversarial (jailbreak) queries.",
            "nontrivial_token_filter": f"Keep selected tokens with count >= {MIN_SELECTED_TOKEN_COUNT} in the jailbreak analysis set.",
            "response_variable": "label (0/1 LlamaGuard)",
            "predictors": "p_cat, C(family), C(token_group)",
            "p_cat_definition": f"1[top1_prob >= {threshold}] else 0",
            "threshold_source": {
                "path": str(PART1_RESULTS_JSON),
                "model_name": THRESHOLD_MODEL_NAME,
            },
            "cluster_definition": "cluster_id = base prompt index shared across jailbreak paraphrase families (96 clusters).",
            "model": "GEE logistic regression with Exchangeable working correlation and robust SE.",
            "reference_levels_note": "References fixed to family='synonyms', p_cat=0 (low probability), token_group='other_tokens'.",
        },
        "selected_tokens": {
            "top6_adv_queries": selected,
            "nontrivial_selected_tokens": selected_nontrivial,
        },
        "data_summary": {
            "n_observations": int(len(df)),
            "n_clusters": int(df["cluster_id"].nunique()),
            "family_counts": {k: int(v) for k, v in df["family"].value_counts().to_dict().items()},
            "token_group_counts": {k: int(v) for k, v in df["token_group"].value_counts().to_dict().items()},
            "proportion_label_1": float(df["label"].mean()),
            "proportion_p_cat_1": float(df["p_cat"].mean()),
        },
        "gee": {
            "formula": formula,
            "covariance_structure": "Exchangeable",
            "family": "Binomial",
            "converged": bool(res.converged),
            "scale": float(res.scale),
            "params": params,
        },
    }

    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # concise human summary
    lines = []
    lines.append("GEE label ~ p_cat + C(family) + C(token_group)")
    lines.append(f"n={len(df)}, clusters={df['cluster_id'].nunique()}, converged={res.converged}")
    lines.append(f"Top6 adversarial query token IDs: {selected}")
    lines.append("")
    lines.append("Terms (coef, OR, p-value):")
    for p in params:
        lines.append(
            f"- {p['term']}: coef={p['coef']:.4f}, OR={p['odds_ratio']:.4f}, p={p['p_value']:.4g}"
        )
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tex_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{GEE logistic regression: token group, probability category, and family (llama1b).}",
        r"\begin{tabular}{l c c c}",
        r"\toprule",
        r"Term & Coefficient & Odds Ratio & p-value \\",
        r"\midrule",
    ]
    for p in params:
        term_h = _humanize_term(str(p["term"]), tokenizer)
        tex_lines.append(
            f"{_latex_escape(term_h)} & "
            f"{float(p['coef']):.4f} & "
            f"{float(p['odds_ratio']):.4f} & "
            f"{float(p['p_value']):.4g} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    OUT_TEX.write_text("\n".join(tex_lines), encoding="utf-8")

    print("[checkpoint] Step 6/6: Outputs saved.")
    print(f"Saved JSON: {OUT_JSON}")
    print(f"Saved summary: {OUT_TXT}")
    print(f"Saved LaTeX: {OUT_TEX}")


if __name__ == "__main__":
    main()
