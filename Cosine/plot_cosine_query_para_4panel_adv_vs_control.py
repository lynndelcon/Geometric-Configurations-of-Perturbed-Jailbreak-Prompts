from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
COSINE_DIR = ROOT / "cosine"

CONTROL_EMBEDDINGS_PATH = ROOT / "control_prompts_lasttok_emb.npy"
ADV_EMBEDDINGS_PATH = ROOT / "adv_prompts_lasttok_emb.npy"

CONTROL_PARA_PATHS = {
    "synonyms": ROOT / "para_synonyms_control_lasttok_emb.npy",
    "swap": ROOT / "para_swap_control_lasttok_emb.npy",
    "symbol_025": ROOT / "para_symbol_control_lasttok_emb.npy",
    "leetspeak": ROOT / "para_leetspeak_control_lasttok_emb.npy",
}
ADV_PARA_PATHS = {
    "synonyms": ROOT / "para_synonyms_adv_lasttok_emb.npy",
    "swap": ROOT / "para_swap_adv_lasttok_emb.npy",
    "symbol_025": ROOT / "para_symbol_adv_lasttok_emb.npy",
    "leetspeak": ROOT / "para_leetspeak_adv_lasttok_emb.npy",
}

QUERY_CONTROL = "#0072b2"
QUERY_ADV = "#d55e00"

TITLE_MAP = {
    "synonyms": "Synonyms",
    "swap": "Swap",
    "symbol_025": "Numbers",
    "leetspeak": "Leetspeak",
}

OUT_PNG = COSINE_DIR / "cosine_query_para_4panel_adv_vs_control_violin_box_shared_y.png"
OUT_JSON = COSINE_DIR / "cosine_query_para_4panel_adv_vs_control_violin_box_shared_y_summary.json"


def load_query_embeddings(path: Path) -> np.ndarray:
    x = np.asarray(np.load(path, allow_pickle=True), dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D query embeddings for {path.name}, got {x.shape}")
    return x


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def query_to_para_cosines(para_path: Path, query_embeddings: np.ndarray) -> np.ndarray:
    para = np.asarray(np.load(para_path, allow_pickle=True), dtype=np.float64)
    if para.ndim != 3:
        raise ValueError(f"Expected 3D para tensor for {para_path.name}, got {para.shape}")
    if para.shape[0] != query_embeddings.shape[0] or para.shape[2] != query_embeddings.shape[1]:
        raise ValueError(
            f"Shape mismatch for {para_path.name}: para={para.shape}, query={query_embeddings.shape}"
        )
    qn = normalize_rows(query_embeddings)
    pn = para / np.maximum(np.linalg.norm(para, axis=2, keepdims=True), 1e-12)
    cos = np.einsum("ijd,id->ij", pn, qn)
    return cos.reshape(-1)


def draw_panel(ax: plt.Axes, adv: np.ndarray, control: np.ndarray, title: str, rng: np.random.Generator) -> None:
    vals = [adv, control]
    positions = [0.8, 2.2]
    colors = [QUERY_ADV, QUERY_CONTROL]

    parts = ax.violinplot(
        vals,
        positions=positions,
        showmeans=False,
        showmedians=True,
        showextrema=True,
        widths=0.62,
    )
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.30)
    if "cmedians" in parts:
        parts["cmedians"].set_color(colors)
        parts["cmedians"].set_linewidth(1.2)
    for key in ("cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color(colors)
            parts[key].set_linewidth(1.0)

    for pos, arr, color in zip(positions, vals, colors):
        n = min(arr.size, 1300)
        if n < arr.size:
            idx = rng.choice(arr.size, size=n, replace=False)
            arr_plot = arr[idx]
        else:
            arr_plot = arr
        x = pos + rng.uniform(-0.08, 0.08, size=arr_plot.size)
        ax.scatter(x, arr_plot, s=8, color=color, alpha=0.22, edgecolors="white", linewidths=0.2)

    ax.set_xticks(positions, ["Jailbreak", "Control"])
    ax.set_xlim(0.2, 2.8)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.15)


def main() -> None:
    COSINE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    x_control = load_query_embeddings(CONTROL_EMBEDDINGS_PATH)
    x_adv = load_query_embeddings(ADV_EMBEDDINGS_PATH)

    families = ["synonyms", "swap", "symbol_025", "leetspeak"]
    control_vals = {f: query_to_para_cosines(CONTROL_PARA_PATHS[f], x_control) for f in families}
    adv_vals = {f: query_to_para_cosines(ADV_PARA_PATHS[f], x_adv) for f in families}

    for family in families:
        if adv_vals[family].size != 4800 or control_vals[family].size != 4800:
            raise ValueError(
                f"Expected 4800 cosine values for {family}; "
                f"got adv={adv_vals[family].size}, control={control_vals[family].size}"
            )

    all_vals = np.concatenate([control_vals[f] for f in families] + [adv_vals[f] for f in families])
    y_min = float(np.min(all_vals))
    y_max = float(np.max(all_vals))
    pad = max(0.01, 0.03 * (y_max - y_min))

    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.8), sharey=True)
    for ax, family in zip(axes, families):
        draw_panel(ax, adv_vals[family], control_vals[family], TITLE_MAP[family], rng)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_xlabel("")

    axes[0].set_ylabel("Cosine similarity")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=220)
    plt.close(fig)

    summary = {
        "description": "4-panel family plot with jailbreak vs control cosine(query, paraphrase) distributions",
        "y_limits": [y_min - pad, y_max + pad],
        "panels": {
            TITLE_MAP[f]: {
                "adv_n": int(adv_vals[f].size),
                "adv_mean": float(adv_vals[f].mean()),
                "adv_std": float(adv_vals[f].std(ddof=1)),
                "control_n": int(control_vals[f].size),
                "control_mean": float(control_vals[f].mean()),
                "control_std": float(control_vals[f].std(ddof=1)),
            }
            for f in families
        },
        "figure_path": str(OUT_PNG),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"saved_figure={OUT_PNG}")
    print(f"saved_summary={OUT_JSON}")


if __name__ == "__main__":
    main()
