from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for plotting. Install it, then rerun:\n"
        "pip install matplotlib"
    ) from exc


PROBA_DIR = Path(__file__).resolve().parent
ROOT = PROBA_DIR.parent

DEFAULT_DATASET_PNG = PROBA_DIR / "top1_token_string_hist_datasets_shared_y_5x2_llama1b.png"
DEFAULT_COUNTS_CSV = PROBA_DIR / "top1_token_string_counts_by_group_llama1b.csv"
DEFAULT_SUMMARY_JSON = PROBA_DIR / "top1_token_string_histograms_summary_llama1b.json"

DATASET_SPECS = {
    "control_prompts": ROOT / "control_prompts_nexttok_topk_ids.npy",
    "adv_prompts": ROOT / "adv_prompts_nexttok_topk_ids.npy",
    "para_synonyms_control": ROOT / "para_synonyms_control_nexttok_topk_ids.npy",
    "para_synonyms_adv": ROOT / "para_synonyms_adv_nexttok_topk_ids.npy",
    "para_swap_control": ROOT / "para_swap_control_nexttok_topk_ids.npy",
    "para_swap_adv": ROOT / "para_swap_adv_nexttok_topk_ids.npy",
    "para_symbol_control": ROOT / "para_symbol_control_nexttok_topk_ids.npy",
    "para_symbol_adv": ROOT / "para_symbol_adv_nexttok_topk_ids.npy",
    "para_leetspeak_control": ROOT / "para_leetspeak_control_nexttok_topk_ids.npy",
    "para_leetspeak_adv": ROOT / "para_leetspeak_adv_nexttok_topk_ids.npy",
}
DATASET_PROB_SPECS = {
    "control_prompts": ROOT / "control_prompts_nexttok_topk_probs.npy",
    "adv_prompts": ROOT / "adv_prompts_nexttok_topk_probs.npy",
    "para_synonyms_control": ROOT / "para_synonyms_control_nexttok_topk_probs.npy",
    "para_synonyms_adv": ROOT / "para_synonyms_adv_nexttok_topk_probs.npy",
    "para_swap_control": ROOT / "para_swap_control_nexttok_topk_probs.npy",
    "para_swap_adv": ROOT / "para_swap_adv_nexttok_topk_probs.npy",
    "para_symbol_control": ROOT / "para_symbol_control_nexttok_topk_probs.npy",
    "para_symbol_adv": ROOT / "para_symbol_adv_nexttok_topk_probs.npy",
    "para_leetspeak_control": ROOT / "para_leetspeak_control_nexttok_topk_probs.npy",
    "para_leetspeak_adv": ROOT / "para_leetspeak_adv_nexttok_topk_probs.npy",
}

DATASET_GRID_ROWS = [
    ("control_prompts", "adv_prompts", "Query"),
    ("para_synonyms_control", "para_synonyms_adv", "Synonyms"),
    ("para_swap_control", "para_swap_adv", "Swap"),
    ("para_symbol_control", "para_symbol_adv", "Numbers"),
    ("para_leetspeak_control", "para_leetspeak_adv", "Leetspeak"),
]

def _load_top1_token_ids(path: Path) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(f"Missing top-k ID file: {path}")
    arr = np.asarray(np.load(path, allow_pickle=True), dtype=np.int64)
    if arr.ndim == 2:
        # Query shape: [n_query, topk]
        top1 = arr[:, 0]
    elif arr.ndim == 3:
        # Paraphrase shape: [n_query, n_para, topk]
        top1 = arr[:, :, 0].reshape(-1)
    else:
        raise ValueError(f"Expected 2D or 3D array at {path}, got shape {arr.shape}")
    return top1.astype(int).tolist()


def _load_top1_probs(path: Path) -> list[float]:
    if not path.exists():
        raise FileNotFoundError(f"Missing top-k prob file: {path}")
    arr = np.asarray(np.load(path, allow_pickle=True), dtype=np.float64)
    if arr.ndim == 2:
        top1 = arr[:, 0]
    elif arr.ndim == 3:
        top1 = arr[:, :, 0].reshape(-1)
    else:
        raise ValueError(f"Expected 2D or 3D array at {path}, got shape {arr.shape}")
    return top1.astype(float).tolist()


def _build_group_rows(token_ids: list[int], group_name: str) -> list[dict[str, Any]]:
    c = Counter(token_ids)
    n = len(token_ids)
    rows: list[dict[str, Any]] = []
    for tid, cnt in c.items():
        rows.append(
            {
                "group": group_name,
                "top1_token_id": int(tid),
                "count": int(cnt),
                "prop": float(cnt / n) if n else 0.0,
                "n_points": int(n),
            }
        )
    rows.sort(key=lambda r: (-int(r["count"]), int(r["top1_token_id"])))
    return rows


def _load_tokenizer(model_id: str) -> Any | None:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
    except Exception as exc:  # pragma: no cover
        print(f"Warning: tokenizer load failed ({exc}). Falling back to token IDs.")
        return None


def _token_label(tokenizer: Any | None, token_id: int) -> str:
    if tokenizer is None:
        return str(token_id)
    try:
        piece = tokenizer.convert_ids_to_tokens(int(token_id))
        return str(piece) if piece is not None else str(token_id)
    except Exception:
        return str(token_id)


def _plot_datasets_5x2(
    dataset_rows: dict[str, list[dict[str, Any]]],
    dataset_token_probs: dict[str, dict[int, list[float]]],
    token_label_map: dict[int, str],
    control_top3_ids: set[int],
    adv_top3_ids: set[int],
    out_png: Path,
    top_n: int,
) -> None:
    selected_map: dict[str, list[dict[str, Any]]] = {}
    for c_key, a_key, _ in DATASET_GRID_ROWS:
        selected_map[c_key] = dataset_rows[c_key][: max(1, int(top_n))]
        selected_map[a_key] = dataset_rows[a_key][: max(1, int(top_n))]

    max_y = max(float(r["prop"]) for rows in selected_map.values() for r in rows)
    y_top = max_y * 1.10 if max_y > 0 else 1.0

    fig, axes = plt.subplots(5, 2, figsize=(16.0, 19.5), sharey=True, squeeze=False)
    cmap = plt.get_cmap("plasma")
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    prob_edges = np.linspace(0.0, 1.0, num=7)  # 6 bins
    bin_centers = 0.5 * (prob_edges[:-1] + prob_edges[1:])
    for i, (control_key, adv_key, row_label) in enumerate(DATASET_GRID_ROWS):
        for j, ds_key in enumerate((control_key, adv_key)):
            ax = axes[i][j]
            rows = selected_map[ds_key]
            x = np.arange(len(rows), dtype=np.float64)
            y = np.asarray([float(r["prop"]) for r in rows], dtype=np.float64)
            tids = [int(r["top1_token_id"]) for r in rows]
            labels = [token_label_map.get(tid, str(tid)) for tid in tids]

            for k, (tid, tot_prop) in enumerate(zip(tids, y)):
                probs = np.asarray(dataset_token_probs.get(ds_key, {}).get(int(tid), []), dtype=np.float64)
                if probs.size == 0:
                    continue
                counts, _ = np.histogram(probs, bins=prob_edges)
                n_points_ds = int(rows[k]["n_points"])
                heights = counts.astype(np.float64) / float(n_points_ds) if n_points_ds > 0 else np.zeros_like(counts)
                bottom = 0.0
                for b_idx, h_seg in enumerate(heights.tolist()):
                    if h_seg <= 0:
                        continue
                    ax.bar(
                        [x[k]],
                        [h_seg],
                        bottom=[bottom],
                        color=cmap(norm(bin_centers[b_idx])),
                        alpha=0.92,
                        width=0.82,
                    )
                    bottom += float(h_seg)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
            for tick, tid in zip(ax.get_xticklabels(), tids):
                if (tid in control_top3_ids) or (tid in adv_top3_ids):
                    tick.set_fontweight("bold")
                if tid in control_top3_ids:
                    tick.set_color("tab:blue")
                elif tid in adv_top3_ids:
                    tick.set_color("tab:orange")
            ax.grid(alpha=0.2, axis="y")
            ax.set_ylim(0.0, y_top)
            if i == 0:
                ax.set_title("Control" if j == 0 else "Jailbreak")
            if j == 0:
                ax.set_ylabel(row_label)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.08, hspace=0.40, wspace=0.16)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.36, 0.968, 0.28, 0.015])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Top-1 Probability", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot 5x2 top-1 token histograms from llama1b *topk_ids.npy files."
    )
    parser.add_argument("--dataset-png", type=Path, default=DEFAULT_DATASET_PNG)
    parser.add_argument("--counts-csv", type=Path, default=DEFAULT_COUNTS_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--tokenizer-model-id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    args = parser.parse_args()

    dataset_token_ids: dict[str, list[int]] = {}
    dataset_first_probs: dict[str, list[float]] = {}
    dataset_rows: dict[str, list[dict[str, Any]]] = {}
    dataset_token_probs: dict[str, dict[int, list[float]]] = {}
    for ds_key, path in DATASET_SPECS.items():
        tids = _load_top1_token_ids(path)
        probs = _load_top1_probs(DATASET_PROB_SPECS[ds_key])
        if len(tids) != len(probs):
            raise ValueError(f"Mismatched lengths for {ds_key}: ids={len(tids)} probs={len(probs)}")
        dataset_token_ids[ds_key] = tids
        dataset_first_probs[ds_key] = probs
        dataset_rows[ds_key] = _build_group_rows(tids, ds_key)
        tok_probs: dict[int, list[float]] = {}
        for tid, p in zip(tids, probs):
            tok_probs.setdefault(int(tid), []).append(float(p))
        dataset_token_probs[ds_key] = tok_probs

    control_top3_ids = {
        int(r["top1_token_id"]) for r in dataset_rows["control_prompts"][:3]
    }
    adv_top3_ids = {
        int(r["top1_token_id"]) for r in dataset_rows["adv_prompts"][:3]
    }

    tokenizer = _load_tokenizer(args.tokenizer_model_id)
    all_ids = set()
    for ds_key in DATASET_SPECS:
        all_ids.update(int(r["top1_token_id"]) for r in dataset_rows[ds_key][: max(1, int(args.top_n))])
    token_label_map = {int(tid): _token_label(tokenizer, int(tid)) for tid in sorted(all_ids)}

    _plot_datasets_5x2(
        dataset_rows=dataset_rows,
        dataset_token_probs=dataset_token_probs,
        token_label_map=token_label_map,
        control_top3_ids=control_top3_ids,
        adv_top3_ids=adv_top3_ids,
        out_png=args.dataset_png,
        top_n=int(args.top_n),
    )

    args.counts_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.counts_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["group", "top1_token_id", "token_string", "count", "prop", "n_points"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for ds_key in DATASET_SPECS:
            for r in dataset_rows[ds_key]:
                tid = int(r["top1_token_id"])
                w.writerow(
                    {
                        "group": ds_key,
                        "top1_token_id": tid,
                        "token_string": token_label_map.get(tid, str(tid)),
                        "count": int(r["count"]),
                        "prop": float(r["prop"]),
                        "n_points": int(r["n_points"]),
                    }
                )

    summary = {
        "dataset_png_path": str(args.dataset_png),
        "counts_csv_path": str(args.counts_csv),
        "n_points_total": int(sum(len(v) for v in dataset_token_ids.values())),
        "top_n_tokens_per_panel": int(args.top_n),
        "tokenizer_model_id": str(args.tokenizer_model_id),
        "datasets": list(DATASET_SPECS.keys()),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved dataset top1 token-string histogram: {args.dataset_png}")
    print(f"Saved counts CSV: {args.counts_csv}")
    print(f"Saved summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
