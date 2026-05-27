from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required. Install it with: pip install matplotlib") from exc

try:
    from sklearn.decomposition import PCA
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scikit-learn is required. Install it with: pip install scikit-learn") from exc


ROOT = Path(__file__).resolve().parent.parent
PROBA_DIR = Path(__file__).resolve().parent

TOPK_PROB_FILES = [
    ROOT / "control_prompts_nexttok_topk_probs.npy",
    ROOT / "adv_prompts_nexttok_topk_probs.npy",
    ROOT / "para_synonyms_control_nexttok_topk_probs.npy",
    ROOT / "para_synonyms_adv_nexttok_topk_probs.npy",
    ROOT / "para_swap_control_nexttok_topk_probs.npy",
    ROOT / "para_swap_adv_nexttok_topk_probs.npy",
    ROOT / "para_symbol_control_nexttok_topk_probs.npy",
    ROOT / "para_symbol_adv_nexttok_topk_probs.npy",
    ROOT / "para_leetspeak_control_nexttok_topk_probs.npy",
    ROOT / "para_leetspeak_adv_nexttok_topk_probs.npy",
]

OUT_JSON = PROBA_DIR / "participation_ratio_topk_probs_llama1b.json"
OUT_PCA_CUM_PNG = PROBA_DIR / "pca_cumulative_explained_variance_topk_probs_llama1b.png"

EXPECTED_N = 38592
EXPECTED_D = 50


def _load_as_2d(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    x = np.asarray(np.load(path, allow_pickle=True), dtype=np.float64)
    if x.ndim == 2:
        if x.shape[1] != EXPECTED_D:
            raise ValueError(f"Expected second dim {EXPECTED_D} for {path}, got {x.shape}")
        return x
    if x.ndim == 3:
        if x.shape[2] != EXPECTED_D:
            raise ValueError(f"Expected last dim {EXPECTED_D} for {path}, got {x.shape}")
        return x.reshape(-1, x.shape[2])
    raise ValueError(f"Expected 2D or 3D array for {path}, got shape {x.shape}")


def participation_ratio(x: np.ndarray) -> tuple[float, np.ndarray]:
    x_centered = x - np.mean(x, axis=0, keepdims=True)
    cov = (x_centered.T @ x_centered) / (x_centered.shape[0] - 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    s1 = float(np.sum(eigvals))
    s2 = float(np.sum(eigvals**2))
    if s2 <= 0.0:
        raise ValueError("Degenerate covariance: sum of squared eigenvalues is non-positive.")
    pr = (s1 * s1) / s2
    return float(pr), eigvals


def main() -> None:
    blocks = []
    counts_by_file: dict[str, int] = {}
    for p in TOPK_PROB_FILES:
        arr = _load_as_2d(p)
        blocks.append(arr)
        counts_by_file[p.name] = int(arr.shape[0])

    x = np.vstack(blocks)
    if x.shape != (EXPECTED_N, EXPECTED_D):
        raise ValueError(f"Expected stacked shape ({EXPECTED_N}, {EXPECTED_D}), got {x.shape}")

    pr, eigvals = participation_ratio(x)
    pca = PCA(n_components=EXPECTED_D, svd_solver="full")
    pca.fit(x)
    explained_ratio = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    cumulative_ratio = np.cumsum(explained_ratio)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    dims = np.arange(1, EXPECTED_D + 1)
    ax.plot(dims, cumulative_ratio, marker="o", markersize=3, linewidth=1.6, color="#1f77b4")
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance Ratio")
    ax.set_xlim(1, EXPECTED_D)
    ax.set_ylim(0.0, 1.01)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.7)
    fig.savefig(OUT_PCA_CUM_PNG, dpi=220)
    plt.close(fig)

    payload = {
        "dataset": "llama1b top-k probabilities",
        "n_datapoints": int(x.shape[0]),
        "n_dimensions": int(x.shape[1]),
        "participation_ratio": pr,
        "formula": "(sum(lambda_i)^2) / sum(lambda_i^2), where lambda_i are covariance eigenvalues",
        "counts_by_file": counts_by_file,
        "eigenvalue_summary": {
            "min": float(np.min(eigvals)),
            "max": float(np.max(eigvals)),
            "sum": float(np.sum(eigvals)),
            "sum_sq": float(np.sum(eigvals**2)),
        },
        "pca_explained_variance_ratio": explained_ratio.tolist(),
        "pca_cumulative_explained_variance_ratio": cumulative_ratio.tolist(),
        "outputs": {
            "pca_cumulative_curve_png": str(OUT_PCA_CUM_PNG),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"n_datapoints={x.shape[0]}")
    print(f"n_dimensions={x.shape[1]}")
    print(f"participation_ratio={pr:.12f}")
    print(f"saved_json={OUT_JSON}")
    print(f"saved_pca_cumulative_png={OUT_PCA_CUM_PNG}")


if __name__ == "__main__":
    main()
