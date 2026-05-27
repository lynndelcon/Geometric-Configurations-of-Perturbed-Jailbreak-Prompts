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


SVM_DIR = Path(__file__).resolve().parent
ROOT = SVM_DIR.parent

# Signed-distance CSVs (all embeddings projected on each hyperplane)
CSV_CJ_PATH = SVM_DIR / "all_embeddings_signed_distances_control_adv_llama1b.csv"
CSV_CP_PATH = SVM_DIR / "all_embeddings_signed_distances_control_adv_llama1b_cross_para.csv"
CSV_CR_PATH = SVM_DIR / "all_embeddings_signed_distances_comp-ref_llama1b.csv"

# Hyperplanes (for margins + pairwise angles)
W_CJ_PATH = SVM_DIR / "w_control_adv_llama1b.npy"
W_CP_PATH = SVM_DIR / "w_control_adv_llama1b_cross_para.npy"
W_CR_PATH = SVM_DIR / "w_comp-ref_llama1b.npy"

OUT_PNG = SVM_DIR / "hyperplanes_3x2_signed_distance_panels_llama1b.png"

FAMILY_COLORS = {
    "query": "#3a3a3a",
    "synonyms": "#ff8c00",
    "swap": "#e75480",
    "symbol": "#1f77ff",
    "leetspeak": "#2ca02c",
}
MARKERS = {
    "control": "o",
    "adv": "^",
}


@dataclass(frozen=True)
class Key:
    embedding_type: str
    condition: str
    family: str
    query_index: int
    paraphrase_index: int


@dataclass
class PointRow:
    key: Key
    d_cj: float
    d_cp: float
    d_cr: float
    l1: float


def _parse_para_idx(x: str) -> int:
    x = x.strip()
    return -1 if x == "" else int(x)


def _read_distance_csv(path: Path) -> dict[Key, float]:
    out: dict[Key, float] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = Key(
                embedding_type=r["embedding_type"].strip(),
                condition=r["condition"].strip(),
                family=r["family"].strip(),
                query_index=int(r["query_index"]),
                paraphrase_index=_parse_para_idx(r["paraphrase_index"]),
            )
            out[key] = float(r["signed_distance"])
    return out


def _load_l1_lookup() -> dict[Key, float]:
    lookup: dict[Key, float] = {}

    q_adv = np.asarray(np.load(ROOT / "adv_prompts_lasttok_emb.npy", allow_pickle=True), dtype=np.float64)
    q_control = np.asarray(np.load(ROOT / "control_prompts_lasttok_emb.npy", allow_pickle=True), dtype=np.float64)

    for i in range(q_adv.shape[0]):
        lookup[Key("query", "adv", "none", i, -1)] = float(np.sum(np.abs(q_adv[i])))
    for i in range(q_control.shape[0]):
        lookup[Key("query", "control", "none", i, -1)] = float(np.sum(np.abs(q_control[i])))

    for fam in ("synonyms", "swap", "symbol", "leetspeak"):
        p_adv = np.asarray(np.load(ROOT / f"para_{fam}_adv_lasttok_emb.npy", allow_pickle=True), dtype=np.float64)
        p_control = np.asarray(np.load(ROOT / f"para_{fam}_control_lasttok_emb.npy", allow_pickle=True), dtype=np.float64)
        for i in range(p_adv.shape[0]):
            for j in range(p_adv.shape[1]):
                lookup[Key("paraphrase", "adv", fam, i, j)] = float(np.sum(np.abs(p_adv[i, j])))
                lookup[Key("paraphrase", "control", fam, i, j)] = float(np.sum(np.abs(p_control[i, j])))
    return lookup


def _line_normal(line_angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(line_angle_deg)
    return np.array([np.sin(theta), -np.cos(theta)], dtype=np.float64)


def _line_angle_from_between(desired_between_deg: float) -> float:
    return 90.0 - desired_between_deg


def _reconstruct_points(d_a: np.ndarray, d_b: np.ndarray, angle_b_deg: float) -> np.ndarray:
    x = d_a.copy()
    n_b = _line_normal(angle_b_deg)
    y = (d_b - n_b[0] * x) / n_b[1]
    return np.column_stack((x, y))


def _draw_two_hyperplanes(
    ax: plt.Axes,
    angle_b_deg: float,
    margin_a: float,
    margin_b: float,
    xlim: tuple[float, float],
) -> None:
    x_line = np.linspace(xlim[0], xlim[1], 500)
    n_b = _line_normal(angle_b_deg)

    # Hyperplane A + margins (vertical)
    ax.axvline(0.0, color="black", linewidth=1.7)
    ax.axvline(-margin_a, color="gray", linestyle="--", linewidth=1.0)
    ax.axvline(margin_a, color="gray", linestyle="--", linewidth=1.0)

    # Hyperplane B + margins (oblique)
    y_b = np.tan(np.deg2rad(angle_b_deg)) * x_line
    ax.plot(x_line, y_b, color="black", linewidth=1.7)
    for sign in (-1.0, 1.0):
        y_off = y_b + sign * (margin_b / abs(n_b[1]))
        ax.plot(x_line, y_off, color="gray", linestyle="--", linewidth=1.0)


def _scatter_usual(ax: plt.Axes, rows: list[PointRow], xy: np.ndarray) -> None:
    # Batch per (family, condition) to avoid one-scatter-call-per-point overhead.
    fam_order = ["synonyms", "swap", "symbol", "leetspeak", "query"]  # query last for visibility
    cond_order = ["control", "adv"]
    idx = np.arange(len(rows))

    families = np.array(
        ["query" if r.key.embedding_type == "query" else r.key.family for r in rows],
        dtype=object,
    )
    conditions = np.array([r.key.condition for r in rows], dtype=object)

    for fam in fam_order:
        for cond in cond_order:
            mask = (families == fam) & (conditions == cond)
            if not np.any(mask):
                continue
            ids = idx[mask]
            ax.scatter(
                xy[ids, 0],
                xy[ids, 1],
                s=8,
                alpha=0.75,
                c=FAMILY_COLORS.get(fam, "#666666"),
                marker=MARKERS["adv" if cond == "adv" else "control"],
                linewidths=0.15,
                edgecolors="black",
            )


def _scatter_l1(ax: plt.Axes, rows: list[PointRow], xy: np.ndarray, norm: plt.Normalize) -> plt.cm.ScalarMappable:
    # Batch by marker type while coloring continuously by L1.
    idx = np.arange(len(rows))
    conditions = np.array([r.key.condition for r in rows], dtype=object)
    l1_vals = np.array([r.l1 for r in rows], dtype=np.float64)

    for cond in ("control", "adv"):
        mask = conditions == cond
        if not np.any(mask):
            continue
        ids = idx[mask]
        ax.scatter(
            xy[ids, 0],
            xy[ids, 1],
            s=8,
            alpha=0.80,
            c=l1_vals[ids],
            cmap="plasma",
            norm=norm,
            marker=MARKERS["adv" if cond == "adv" else "control"],
            linewidths=0.12,
            edgecolors="black",
        )
    sm = plt.cm.ScalarMappable(norm=norm, cmap="plasma")
    sm.set_array([])
    return sm


def _angle_between_hyperplanes_deg(w_a: np.ndarray, w_b: np.ndarray) -> float:
    cosang = float(np.clip(np.dot(w_a, w_b) / (np.linalg.norm(w_a) * np.linalg.norm(w_b)), -1.0, 1.0))
    return float(np.degrees(np.arccos(abs(cosang))))


def main() -> None:
    d_cj_map = _read_distance_csv(CSV_CJ_PATH)
    d_cp_map = _read_distance_csv(CSV_CP_PATH)
    d_cr_map = _read_distance_csv(CSV_CR_PATH)

    common = sorted(
        set(d_cj_map.keys()) & set(d_cp_map.keys()) & set(d_cr_map.keys()),
        key=lambda k: (k.embedding_type, k.condition, k.family, k.query_index, k.paraphrase_index),
    )
    if not common:
        raise ValueError("No common embedding keys across the three distance CSV files.")

    l1_lookup = _load_l1_lookup()
    rows: list[PointRow] = []
    for k in common:
        if k not in l1_lookup:
            raise KeyError(f"Missing L1 lookup for {k}")
        rows.append(
            PointRow(
                key=k,
                d_cj=d_cj_map[k],
                d_cp=d_cp_map[k],
                d_cr=d_cr_map[k],
                l1=l1_lookup[k],
            )
        )

    # Hyperplane vectors for angles/margins
    w_cj = np.asarray(np.load(W_CJ_PATH, allow_pickle=True), dtype=np.float64).reshape(-1)
    w_cp = np.asarray(np.load(W_CP_PATH, allow_pickle=True), dtype=np.float64).reshape(-1)
    w_cr = np.asarray(np.load(W_CR_PATH, allow_pickle=True), dtype=np.float64).reshape(-1)

    m_cj = 1.0 / float(np.linalg.norm(w_cj))
    m_cp = 1.0 / float(np.linalg.norm(w_cp))
    m_cr = 1.0 / float(np.linalg.norm(w_cr))

    # Compute pairwise angles (no hardcoded values).
    ang_cj_cp = _angle_between_hyperplanes_deg(w_cj, w_cp)
    ang_cj_cr = _angle_between_hyperplanes_deg(w_cj, w_cr)
    ang_cp_cr = _angle_between_hyperplanes_deg(w_cp, w_cr)

    d_cj = np.asarray([r.d_cj for r in rows], dtype=np.float64)
    d_cp = np.asarray([r.d_cp for r in rows], dtype=np.float64)
    d_cr = np.asarray([r.d_cr for r in rows], dtype=np.float64)
    l1_vals = np.asarray([r.l1 for r in rows], dtype=np.float64)
    l1_norm = plt.Normalize(vmin=float(np.min(l1_vals)), vmax=float(np.max(l1_vals)))

    row_specs = [
        ("Control-Jailbreak vs Crossed Paraphrases", d_cj, d_cp, m_cj, m_cp, ang_cj_cp, _line_angle_from_between(ang_cj_cp)),
        ("Control-Jailbreak vs Compliance-Refusal", d_cj, d_cr, m_cj, m_cr, ang_cj_cr, _line_angle_from_between(ang_cj_cr)),
        ("Crossed Paraphrases vs Compliance-Refusal", d_cp, d_cr, m_cp, m_cr, ang_cp_cr, _line_angle_from_between(ang_cp_cr)),
    ]

    fig, axs = plt.subplots(3, 2, figsize=(18, 20), constrained_layout=True)
    sm_for_cbar = None

    for r, (title, d_a, d_b, m_a, m_b, between_deg, line_angle_deg) in enumerate(row_specs):
        xy = _reconstruct_points(d_a, d_b, line_angle_deg)
        xpad, ypad = 1.2, 1.2
        xlim = (float(np.min(xy[:, 0]) - xpad), float(np.max(xy[:, 0]) + xpad))
        ylim = (float(np.min(xy[:, 1]) - ypad), float(np.max(xy[:, 1]) + ypad))

        # Left: usual style
        _scatter_usual(axs[r, 0], rows, xy)
        _draw_two_hyperplanes(axs[r, 0], line_angle_deg, m_a, m_b, xlim)
        axs[r, 0].set_xlim(*xlim)
        axs[r, 0].set_ylim(*ylim)
        axs[r, 0].grid(True, alpha=0.20, linestyle="--", linewidth=0.6)

        # Right: L1 style
        sm = _scatter_l1(axs[r, 1], rows, xy, l1_norm)
        _draw_two_hyperplanes(axs[r, 1], line_angle_deg, m_a, m_b, xlim)
        axs[r, 1].set_xlim(*xlim)
        axs[r, 1].set_ylim(*ylim)
        axs[r, 1].grid(True, alpha=0.20, linestyle="--", linewidth=0.6)
        sm_for_cbar = sm

        for c in (0, 1):
            axs[r, c].set_xlabel("")
            axs[r, c].set_ylabel("")
            axs[r, c].set_xticks([])
            axs[r, c].set_yticks([])

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#999999", markeredgecolor="black", markeredgewidth=0.4, label="Control", markersize=7),
        Line2D([0], [0], marker="^", linestyle="None", markerfacecolor="#999999", markeredgecolor="black", markeredgewidth=0.4, label="Jailbreak", markersize=7),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=FAMILY_COLORS["query"], markeredgecolor="black", markeredgewidth=0.4, label="Query", markersize=7),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=FAMILY_COLORS["synonyms"], markeredgecolor="black", markeredgewidth=0.4, label="Synonyms", markersize=7),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=FAMILY_COLORS["swap"], markeredgecolor="black", markeredgewidth=0.4, label="Swap", markersize=7),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=FAMILY_COLORS["symbol"], markeredgecolor="black", markeredgewidth=0.4, label="Numbers", markersize=7),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=FAMILY_COLORS["leetspeak"], markeredgecolor="black", markeredgewidth=0.4, label="Leetspeak", markersize=7),
    ]
    axs[0, 0].legend(handles=legend_handles, loc="upper left", frameon=True, facecolor="white", edgecolor="#808080")

    if sm_for_cbar is not None:
        cbar = fig.colorbar(sm_for_cbar, ax=axs[:, 1], fraction=0.020, pad=0.015)
        cbar.set_label("L1 Norm to Origin")

    fig.savefig(OUT_PNG, dpi=220)
    plt.close(fig)

    print(f"saved_plot={OUT_PNG}")
    print(f"n_points={len(rows)}")
    print(f"pairwise_angles_deg={{'cj_cp': {ang_cj_cp:.6f}, 'cj_cr': {ang_cj_cr:.6f}, 'cp_cr': {ang_cp_cr:.6f}}}")


if __name__ == "__main__":
    main()
