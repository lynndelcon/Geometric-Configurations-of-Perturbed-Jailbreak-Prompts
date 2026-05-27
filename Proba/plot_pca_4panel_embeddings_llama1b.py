from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required. Install it with: pip install matplotlib") from exc

try:
    from sklearn.decomposition import PCA
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scikit-learn is required. Install it with: pip install scikit-learn") from exc


PROBA_DIR = Path(__file__).resolve().parent
ROOT = PROBA_DIR.parent
SVM_DIR = ROOT / "svm"
ANSWERS_DIR = ROOT / "answers"

REGION_CSV = SVM_DIR / "all_embeddings_region_labels_llama1b.csv"
OUT_PNG = PROBA_DIR / "pca_4panel_embeddings_llama1b.png"

FAMILIES = ("query", "synonyms", "swap", "symbol", "leetspeak")

FAMILY_COLORS = {
    "query": "#3a3a3a",
    "synonyms": "#ff8c00",
    "swap": "#e75480",
    "symbol": "#1f77ff",
    "leetspeak": "#2ca02c",
}
SHAPE_CODE_MARKERS = {"control": "o", "adv": "^"}
REGION_COLORS = {
    "usual_tokens": "#0000ff",
    "unusual_tokens": "#ff0000",
    "compliance_jailbreak": "#008000",
    "refusal_jailbreak": "#800080",
}
REGION_LABELS = {
    "usual_tokens": "Usual Tokens",
    "unusual_tokens": "Unusual Tokens",
    "compliance_jailbreak": "Compliance",
    "refusal_jailbreak": "Refusal",
}


@dataclass(frozen=True)
class Key:
    embedding_type: str
    condition: str
    family: str
    query_index: int
    paraphrase_index: int


def _key_to_csv_fields(k: Key) -> tuple[str, str, str, str, str]:
    para_idx = "" if k.paraphrase_index < 0 else str(k.paraphrase_index)
    return (k.embedding_type, k.condition, k.family, str(k.query_index), para_idx)


def _load_region_lookup() -> dict[Key, str]:
    if not REGION_CSV.exists():
        raise FileNotFoundError(f"Missing region CSV: {REGION_CSV}")
    out: dict[Key, str] = {}
    with REGION_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = Key(
                embedding_type=r["embedding_type"].strip(),
                condition=r["condition"].strip(),
                family=r["family"].strip(),
                query_index=int(r["query_index"]),
                paraphrase_index=(-1 if r["paraphrase_index"].strip() == "" else int(r["paraphrase_index"])),
            )
            out[key] = r["region"].strip()
    return out


def _append_query_block(
    *,
    emb_path: Path,
    prob_path: Path,
    condition: str,
    x_blocks: list[np.ndarray],
    prob_blocks: list[np.ndarray],
    prob0: list[float],
    keys: list[Key],
    family_labels: list[str],
    adv_llamaguard_labels: list[int | None],
) -> None:
    x = np.asarray(np.load(emb_path, allow_pickle=True), dtype=np.float32)
    p = np.asarray(np.load(prob_path, allow_pickle=True), dtype=np.float64)
    if x.ndim != 2 or p.ndim != 2 or x.shape[0] != p.shape[0]:
        raise ValueError(f"Shape mismatch query block: emb={x.shape}, probs={p.shape}")
    if p.shape[1] < 1:
        raise ValueError(f"Expected probs with at least 1 column at {prob_path}, got {p.shape}")

    x_blocks.append(x)
    prob_blocks.append(p)
    prob0.extend(p[:, 0].tolist())

    n = x.shape[0]
    for i in range(n):
        keys.append(Key("query", condition, "none", i, -1))
        family_labels.append("query")
        adv_llamaguard_labels.append(None)


def _append_para_block(
    *,
    emb_path: Path,
    prob_path: Path,
    condition: str,
    family: str,
    x_blocks: list[np.ndarray],
    prob_blocks: list[np.ndarray],
    prob0: list[float],
    keys: list[Key],
    family_labels: list[str],
    adv_llamaguard_labels: list[int | None],
) -> None:
    x3 = np.asarray(np.load(emb_path, allow_pickle=True), dtype=np.float32)
    p3 = np.asarray(np.load(prob_path, allow_pickle=True), dtype=np.float64)
    if x3.ndim != 3 or p3.ndim != 3 or x3.shape[:2] != p3.shape[:2]:
        raise ValueError(f"Shape mismatch paraphrase block: emb={x3.shape}, probs={p3.shape}")
    if p3.shape[2] < 1:
        raise ValueError(f"Expected probs with at least 1 last-dim entry at {prob_path}, got {p3.shape}")

    q, m, d = x3.shape
    x_blocks.append(x3.reshape(q * m, d))
    prob_blocks.append(p3.reshape(q * m, p3.shape[2]))
    prob0.extend(p3[:, :, 0].reshape(-1).tolist())

    for qi in range(q):
        for pj in range(m):
            keys.append(Key("paraphrase", condition, family, qi, pj))
            family_labels.append(family)
            adv_llamaguard_labels.append(None)


def main() -> None:
    print("[1/5] Loading region labels...")
    region_lookup = _load_region_lookup()

    print("[2/5] Loading embeddings + top-k probs and assembling aligned rows...")
    x_blocks: list[np.ndarray] = []
    prob_blocks: list[np.ndarray] = []
    prob0: list[float] = []
    keys: list[Key] = []
    family_labels: list[str] = []
    adv_llamaguard_labels: list[int | None] = []

    _append_query_block(
        emb_path=ROOT / "control_prompts_lasttok_emb.npy",
        prob_path=ROOT / "control_prompts_nexttok_topk_probs.npy",
        condition="control",
        x_blocks=x_blocks,
        prob_blocks=prob_blocks,
        prob0=prob0,
        keys=keys,
        family_labels=family_labels,
        adv_llamaguard_labels=adv_llamaguard_labels,
    )
    _append_query_block(
        emb_path=ROOT / "adv_prompts_lasttok_emb.npy",
        prob_path=ROOT / "adv_prompts_nexttok_topk_probs.npy",
        condition="adv",
        x_blocks=x_blocks,
        prob_blocks=prob_blocks,
        prob0=prob0,
        keys=keys,
        family_labels=family_labels,
        adv_llamaguard_labels=adv_llamaguard_labels,
    )

    for fam in ("synonyms", "swap", "symbol", "leetspeak"):
        _append_para_block(
            emb_path=ROOT / f"para_{fam}_control_lasttok_emb.npy",
            prob_path=ROOT / f"para_{fam}_control_nexttok_topk_probs.npy",
            condition="control",
            family=fam,
            x_blocks=x_blocks,
            prob_blocks=prob_blocks,
            prob0=prob0,
            keys=keys,
            family_labels=family_labels,
            adv_llamaguard_labels=adv_llamaguard_labels,
        )
        _append_para_block(
            emb_path=ROOT / f"para_{fam}_adv_lasttok_emb.npy",
            prob_path=ROOT / f"para_{fam}_adv_nexttok_topk_probs.npy",
            condition="adv",
            family=fam,
            x_blocks=x_blocks,
            prob_blocks=prob_blocks,
            prob0=prob0,
            keys=keys,
            family_labels=family_labels,
            adv_llamaguard_labels=adv_llamaguard_labels,
        )

    x_all = np.vstack(x_blocks)
    p_all = np.vstack(prob_blocks)
    prob0_arr = np.asarray(prob0, dtype=np.float64)
    if (
        x_all.shape[0] != 38592
        or p_all.shape != (38592, 50)
        or prob0_arr.shape[0] != 38592
        or len(keys) != 38592
    ):
        raise ValueError(
            "Expected shapes emb=(38592,d), probs=(38592,50), prob0=(38592,), keys=38592. "
            f"Got emb={x_all.shape}, probs={p_all.shape}, prob0={prob0_arr.shape}, keys={len(keys)}"
        )

    print("[3/5] Loading LlamaGuard labels for jailbreak points...")
    q_adv_labels = np.asarray(
        np.load(ANSWERS_DIR / "llamaguard_labels_answers_llama1b_adv_queries.npy", allow_pickle=True), dtype=np.int8
    ).reshape(-1)
    if q_adv_labels.shape[0] != 96:
        raise ValueError(f"Expected 96 adv query labels, got {q_adv_labels.shape}")
    adv_query_ptr = 0
    adv_para_labels: dict[str, np.ndarray] = {}
    for fam in ("synonyms", "swap", "symbol", "leetspeak"):
        lbl = np.asarray(
            np.load(ANSWERS_DIR / f"llamaguard_labels_answers_llama1b_para_{fam}_adv.npy", allow_pickle=True),
            dtype=np.int8,
        )
        if lbl.shape != (96, 50):
            raise ValueError(f"Expected (96,50) for family {fam}, got {lbl.shape}")
        adv_para_labels[fam] = lbl

    for i, k in enumerate(keys):
        if k.condition != "adv":
            continue
        if k.embedding_type == "query":
            adv_llamaguard_labels[i] = int(q_adv_labels[adv_query_ptr])
            adv_query_ptr += 1
        else:
            adv_llamaguard_labels[i] = int(adv_para_labels[k.family][k.query_index, k.paraphrase_index])
    if adv_query_ptr != 96:
        raise RuntimeError(f"Unexpected adv query label pointer end: {adv_query_ptr}")

    print("[4/5] Running PCA (2D) on embeddings and top-k probabilities...")
    pca = PCA(n_components=2, svd_solver="randomized", random_state=42)
    _xy_embeddings = pca.fit_transform(x_all)
    pca_probs = PCA(n_components=2, svd_solver="full", random_state=42)
    xy = pca_probs.fit_transform(p_all)

    regions = np.array([region_lookup.get(k, "unknown") for k in keys], dtype=object)
    families = np.array(family_labels, dtype=object)
    adv_lg = np.array(adv_llamaguard_labels, dtype=object)
    conditions = np.array([k.condition for k in keys], dtype=object)

    print("[5/5] Rendering 4-panel figure...")
    fig, axs = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)

    # Panel 1: color by first top-k probability entry
    for cond in ("control", "adv"):
        mask = conditions == cond
        sc0 = axs[0, 0].scatter(
            xy[mask, 0],
            xy[mask, 1],
            c=prob0_arr[mask],
            cmap="plasma",
            s=8,
            alpha=0.75,
            marker=SHAPE_CODE_MARKERS[cond],
            linewidths=0,
        )
    cbar0 = fig.colorbar(sc0, ax=axs[0, 0], fraction=0.046, pad=0.04)
    cbar0.set_label("Top-1 Probability")

    # Panel 2: color by family + shape by control/jailbreak
    for fam in FAMILIES:
        for cond in ("control", "adv"):
            mask = (families == fam) & (conditions == cond)
            if not np.any(mask):
                continue
            axs[0, 1].scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=8,
                alpha=0.75,
                c=FAMILY_COLORS[fam],
                marker=SHAPE_CODE_MARKERS[cond],
                linewidths=0.1,
                edgecolors="black",
            )
    family_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=6, markerfacecolor=FAMILY_COLORS["query"], markeredgecolor="#808080", markeredgewidth=0.5, label="Query"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=6, markerfacecolor=FAMILY_COLORS["synonyms"], markeredgecolor="#808080", markeredgewidth=0.5, label="Synonyms"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=6, markerfacecolor=FAMILY_COLORS["swap"], markeredgecolor="#808080", markeredgewidth=0.5, label="Swap"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=6, markerfacecolor=FAMILY_COLORS["symbol"], markeredgecolor="#808080", markeredgewidth=0.5, label="Numbers"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=6, markerfacecolor=FAMILY_COLORS["leetspeak"], markeredgecolor="#808080", markeredgewidth=0.5, label="Leetspeak"),
    ]
    shape_handles = [
        Line2D([0], [0], marker=SHAPE_CODE_MARKERS["control"], linestyle="None", markersize=6, markerfacecolor="#9a9a9a", markeredgecolor="#808080", markeredgewidth=0.5, label="Control"),
        Line2D([0], [0], marker=SHAPE_CODE_MARKERS["adv"], linestyle="None", markersize=6, markerfacecolor="#9a9a9a", markeredgecolor="#808080", markeredgewidth=0.5, label="Jailbreak"),
    ]
    axs[0, 1].legend(
        handles=shape_handles + family_handles,
        loc="upper left",
        fontsize=8,
        frameon=True,
        facecolor="white",
        edgecolor="#808080",
    )

    # Panel 3: color by region label
    for region, color in REGION_COLORS.items():
        mask = regions == region
        if not np.any(mask):
            continue
        for cond in ("control", "adv"):
            submask = mask & (conditions == cond)
            if not np.any(submask):
                continue
            axs[1, 0].scatter(
                xy[submask, 0],
                xy[submask, 1],
                s=8,
                alpha=0.75,
                c=color,
                marker=SHAPE_CODE_MARKERS[cond],
                label=region,
                linewidths=0,
            )
    unknown_mask = regions == "unknown"
    if np.any(unknown_mask):
        axs[1, 0].scatter(
            xy[unknown_mask, 0], xy[unknown_mask, 1], s=8, alpha=0.75, c="#000000", marker="x", label="unknown"
        )
    axs[1, 0].legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="None", markersize=6, markerfacecolor=REGION_COLORS["usual_tokens"], markeredgecolor="#808080", markeredgewidth=0.5, label=REGION_LABELS["usual_tokens"]),
            Line2D([0], [0], marker="o", linestyle="None", markersize=6, markerfacecolor=REGION_COLORS["unusual_tokens"], markeredgecolor="#808080", markeredgewidth=0.5, label=REGION_LABELS["unusual_tokens"]),
            Line2D([0], [0], marker="^", linestyle="None", markersize=6, markerfacecolor=REGION_COLORS["compliance_jailbreak"], markeredgecolor="#808080", markeredgewidth=0.5, label=REGION_LABELS["compliance_jailbreak"]),
            Line2D([0], [0], marker="^", linestyle="None", markersize=6, markerfacecolor=REGION_COLORS["refusal_jailbreak"], markeredgecolor="#808080", markeredgewidth=0.5, label=REGION_LABELS["refusal_jailbreak"]),
        ],
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#808080",
    )

    # Panel 4: jailbreak-only by LlamaGuard, others grey
    non_adv_mask = conditions != "adv"
    axs[1, 1].scatter(
        xy[non_adv_mask, 0],
        xy[non_adv_mask, 1],
        s=8,
        alpha=0.35,
        c="#bdbdbd",
        marker=SHAPE_CODE_MARKERS["control"],
        linewidths=0,
        label="Non-Jailbreak",
    )

    adv_refusal_mask = (conditions == "adv") & (adv_lg == 0)
    adv_comp_mask = (conditions == "adv") & (adv_lg == 1)
    axs[1, 1].scatter(
        xy[adv_refusal_mask, 0],
        xy[adv_refusal_mask, 1],
        s=9,
        alpha=0.82,
        c="#1f77b4",
        marker=SHAPE_CODE_MARKERS["adv"],
        linewidths=0,
        label="Refusal",
    )
    axs[1, 1].scatter(
        xy[adv_comp_mask, 0],
        xy[adv_comp_mask, 1],
        s=9,
        alpha=0.82,
        c="#d62728",
        marker=SHAPE_CODE_MARKERS["adv"],
        linewidths=0,
        label="Compliance",
    )
    axs[1, 1].legend(
        handles=[
            Line2D([0], [0], marker=SHAPE_CODE_MARKERS["control"], linestyle="None", markersize=6, markerfacecolor="#bdbdbd", markeredgecolor="#808080", markeredgewidth=0.5, label="Non-Jailbreak"),
            Line2D([0], [0], marker=SHAPE_CODE_MARKERS["adv"], linestyle="None", markersize=6, markerfacecolor="#d62728", markeredgecolor="#808080", markeredgewidth=0.5, label="Compliance"),
            Line2D([0], [0], marker=SHAPE_CODE_MARKERS["adv"], linestyle="None", markersize=6, markerfacecolor="#1f77b4", markeredgecolor="#808080", markeredgewidth=0.5, label="Refusal"),
        ],
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#808080",
    )

    for r in range(2):
        for c in range(2):
            ax = axs[r, c]
            if r == 0:
                ax.set_xlabel("")
            else:
                ax.set_xlabel("PC1")
            if r == 0 and c == 1:
                ax.set_ylabel("")
            elif r == 1 and c == 1:
                ax.set_ylabel("")
            else:
                ax.set_ylabel("PC2")
            ax.set_xticks([])
            ax.set_yticks([])
        ax.grid(True, alpha=0.20, linestyle="--", linewidth=0.6)

    fig.savefig(OUT_PNG, dpi=240)
    plt.close(fig)

    explained = pca.explained_variance_ratio_
    print(f"saved_plot={OUT_PNG}")
    print(f"n_points={x_all.shape[0]}")
    print(f"pca_explained_variance_ratio_pc1={explained[0]:.6f}")
    print(f"pca_explained_variance_ratio_pc2={explained[1]:.6f}")


if __name__ == "__main__":
    main()

