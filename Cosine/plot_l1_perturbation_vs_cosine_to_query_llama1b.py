from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent.parent
COSINE_DIR = ROOT / "cosine"
MODEL_NAME = ROOT.name

CONTROL_EMBEDDINGS_PATH = ROOT / "control_prompts_lasttok_emb.npy"
ADV_EMBEDDINGS_PATH = ROOT / "adv_prompts_lasttok_emb.npy"

CONTROL_PARA_PATHS = {
    "synonyms": ROOT / "para_synonyms_control_lasttok_emb.npy",
    "swap": ROOT / "para_swap_control_lasttok_emb.npy",
    "numbers": ROOT / "para_symbol_control_lasttok_emb.npy",
    "leetspeak": ROOT / "para_leetspeak_control_lasttok_emb.npy",
}
ADV_PARA_PATHS = {
    "synonyms": ROOT / "para_synonyms_adv_lasttok_emb.npy",
    "swap": ROOT / "para_swap_adv_lasttok_emb.npy",
    "numbers": ROOT / "para_symbol_adv_lasttok_emb.npy",
    "leetspeak": ROOT / "para_leetspeak_adv_lasttok_emb.npy",
}

FAMILY_COLORS = {
    "synonyms": "#ff8c00",
    "swap": "#e75480",
    "numbers": "#1f77ff",
    "leetspeak": "#2ca02c",
}
SIDE_MARKERS = {
    "control": "o",
    "jailbreak": "^",
}

OUT_PNG = COSINE_DIR / f"scatter_l1_perturbation_vs_cosine_to_query_{MODEL_NAME}.png"
OUT_JSON = COSINE_DIR / f"scatter_l1_perturbation_vs_cosine_to_query_summary_{MODEL_NAME}.json"


def _load_control_embeddings(path: Path) -> np.ndarray:
    if path.suffix == ".pt":
        x = torch.load(path, map_location="cpu")
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    else:
        x = np.load(path, allow_pickle=True)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D control embeddings, got {x.shape}")
    return x


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def _compute_pair_metrics(para_path: Path, query_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    para = np.asarray(np.load(para_path, allow_pickle=True), dtype=np.float64)
    if para.ndim != 3:
        raise ValueError(f"Expected 3D paraphrase tensor at {para_path.name}, got {para.shape}")
    if para.shape[0] != query_embeddings.shape[0] or para.shape[2] != query_embeddings.shape[1]:
        raise ValueError(
            f"Shape mismatch for {para_path.name}: para={para.shape}, query={query_embeddings.shape}"
        )

    # L1 perturbation norm to source query: ||para_ij - query_i||_1
    l1 = np.sum(np.abs(para - query_embeddings[:, None, :]), axis=2).reshape(-1)

    # Cosine similarity to source query.
    qn = _normalize_rows(query_embeddings)  # (n, d)
    pn = para / np.maximum(np.linalg.norm(para, axis=2, keepdims=True), 1e-12)  # (n, m, d)
    cos = np.einsum("ijd,id->ij", pn, qn).reshape(-1)
    return l1, cos


def main() -> None:
    COSINE_DIR.mkdir(parents=True, exist_ok=True)

    x_control = _load_control_embeddings(CONTROL_EMBEDDINGS_PATH)
    x_adv = np.asarray(np.load(ADV_EMBEDDINGS_PATH, allow_pickle=True), dtype=np.float64)
    if x_adv.ndim != 2:
        raise ValueError(f"Expected 2D adversarial query embeddings, got {x_adv.shape}")

    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    summary: dict[str, dict[str, float | int]] = {}

    for family, path in CONTROL_PARA_PATHS.items():
        l1, cos = _compute_pair_metrics(path, x_control)
        ax.scatter(
            l1,
            cos,
            s=8,
            alpha=0.26,
            c=FAMILY_COLORS[family],
            marker=SIDE_MARKERS["control"],
            edgecolors="none",
        )
        summary[f"{family}_control"] = {
            "n_points": int(l1.size),
            "l1_min": float(np.min(l1)),
            "l1_max": float(np.max(l1)),
            "cos_min": float(np.min(cos)),
            "cos_max": float(np.max(cos)),
            "cos_mean": float(np.mean(cos)),
        }

    for family, path in ADV_PARA_PATHS.items():
        l1, cos = _compute_pair_metrics(path, x_adv)
        ax.scatter(
            l1,
            cos,
            s=8,
            alpha=0.26,
            c=FAMILY_COLORS[family],
            marker=SIDE_MARKERS["jailbreak"],
            edgecolors="none",
        )
        summary[f"{family}_jailbreak"] = {
            "n_points": int(l1.size),
            "l1_min": float(np.min(l1)),
            "l1_max": float(np.max(l1)),
            "cos_min": float(np.min(cos)),
            "cos_max": float(np.max(cos)),
            "cos_mean": float(np.mean(cos)),
        }

    ax.set_xlabel("Perturbation Norm L1 to Query")
    ax.set_ylabel("Cosine Similarity to Query")

    max_l1 = max(v["l1_max"] for v in summary.values()) if summary else 0.0
    ax.set_xticks([0.0, float(max_l1)])
    ax.set_yticks([0.0, 1.0])
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#999999", markeredgecolor="black", markeredgewidth=0.4, label="Control", markersize=7),
        Line2D([0], [0], marker="^", linestyle="None", markerfacecolor="#999999", markeredgecolor="black", markeredgewidth=0.4, label="Jailbreak", markersize=7),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=FAMILY_COLORS["synonyms"], markeredgecolor="black", markeredgewidth=0.4, label="Synonyms", markersize=7),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=FAMILY_COLORS["swap"], markeredgecolor="black", markeredgewidth=0.4, label="Swap", markersize=7),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=FAMILY_COLORS["numbers"], markeredgecolor="black", markeredgewidth=0.4, label="Numbers", markersize=7),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=FAMILY_COLORS["leetspeak"], markeredgecolor="black", markeredgewidth=0.4, label="Leetspeak", markersize=7),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=8,
        frameon=True,
        facecolor="white",
        edgecolor="#808080",
    )

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=220)
    plt.close(fig)

    out = {
        "description": "Scatter of paraphrase perturbation L1 norm to query (x) vs cosine similarity to query (y).",
        "figure_path": str(OUT_PNG),
        "groups": summary,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"saved_figure={OUT_PNG}")
    print(f"saved_summary={OUT_JSON}")


if __name__ == "__main__":
    main()
