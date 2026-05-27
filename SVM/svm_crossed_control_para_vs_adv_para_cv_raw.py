from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC


SVM_DIR = Path(__file__).resolve().parent
ROOT = SVM_DIR.parent

CROSSED_CSV_PATH = SVM_DIR / "control_paraphrases_crossed_to_jailbreak_side_llama1b.csv"

CONTROL_PARA_PATHS = {
    "synonyms": ROOT / "para_synonyms_control_lasttok_emb.npy",
    "swap": ROOT / "para_swap_control_lasttok_emb.npy",
    "symbol": ROOT / "para_symbol_control_lasttok_emb.npy",
    "leetspeak": ROOT / "para_leetspeak_control_lasttok_emb.npy",
}
ADV_PARA_PATHS = {
    "synonyms": ROOT / "para_synonyms_adv_lasttok_emb.npy",
    "swap": ROOT / "para_swap_adv_lasttok_emb.npy",
    "symbol": ROOT / "para_symbol_adv_lasttok_emb.npy",
    "leetspeak": ROOT / "para_leetspeak_adv_lasttok_emb.npy",
}

JSON_OUT_PATH = SVM_DIR / "svm_crossed_control_para_vs_adv_para_cv_raw_results_llama1b.json"
TXT_OUT_PATH = SVM_DIR / "svm_crossed_control_para_vs_adv_para_cv_raw_summary_llama1b.txt"

BEST_W_PATH = SVM_DIR / "w_control_adv_llama1b_cross_para.npy"
BEST_B_PATH = SVM_DIR / "b_control_adv_llama1b_cross_para.npy"
LABELS_BINARY_PATH = SVM_DIR / "binary_labels_crossed_control_para_vs_adv_para_raw_llama1b.npy"
INDICES_MAP_PATH = SVM_DIR / "indices_mapping_crossed_control_para_vs_adv_para_raw_llama1b.json"

SEED = 42
MAX_OUTER_FOLDS = 5
INNER_SPLITS = 4
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def build_estimator() -> LinearSVC:
    return LinearSVC(
        class_weight="balanced",
        dual="auto",
        fit_intercept=True,
        max_iter=20000,
        random_state=SEED,
    )


def select_final_c_from_outer(best_cs: list[float]) -> float:
    counts: dict[float, int] = {}
    for c in best_cs:
        counts[c] = counts.get(c, 0) + 1
    max_count = max(counts.values())
    winners = [c for c, ct in counts.items() if ct == max_count]
    return float(min(winners))


def load_crossed_control_from_csv(path: Path) -> tuple[np.ndarray, list[dict[str, int | str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing crossed-control CSV: {path}")

    # Load all control paraphrase tensors once.
    control_arrays = {}
    for fam, p in CONTROL_PARA_PATHS.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing control paraphrase embeddings: {p}")
        arr = np.asarray(np.load(p, allow_pickle=True), dtype=np.float64)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D control paraphrase array for {fam}, got {arr.shape}")
        control_arrays[fam] = arr

    vectors: list[np.ndarray] = []
    mapping: list[dict[str, int | str]] = []
    seen = set()

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("embedding_type") != "paraphrase" or row.get("condition") != "control":
                continue
            fam = str(row["family"]).strip().lower()
            if fam not in control_arrays:
                continue
            qi = int(row["query_index"])
            pj = int(row["paraphrase_index"])
            key = (fam, qi, pj)
            if key in seen:
                continue
            seen.add(key)

            arr = control_arrays[fam]
            if qi < 0 or qi >= arr.shape[0] or pj < 0 or pj >= arr.shape[1]:
                raise IndexError(f"Out-of-range index in CSV row: family={fam}, query_index={qi}, paraphrase_index={pj}")
            vectors.append(arr[qi, pj, :])
            mapping.append({"family": fam, "query_index": qi, "paraphrase_index": pj})

    if not vectors:
        raise ValueError("No crossed control paraphrase embeddings found from CSV.")
    return np.vstack(vectors), mapping


def load_all_adv_paraphrases() -> tuple[np.ndarray, list[dict[str, int | str]]]:
    vectors: list[np.ndarray] = []
    mapping: list[dict[str, int | str]] = []
    for fam, p in ADV_PARA_PATHS.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing adversarial paraphrase embeddings: {p}")
        arr = np.asarray(np.load(p, allow_pickle=True), dtype=np.float64)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D adversarial paraphrase array for {fam}, got {arr.shape}")
        for qi in range(arr.shape[0]):
            for pj in range(arr.shape[1]):
                vectors.append(arr[qi, pj, :])
                mapping.append({"family": fam, "query_index": qi, "paraphrase_index": pj})
    return np.vstack(vectors), mapping


def main() -> None:
    print(f"Loading crossed control paraphrases from {CROSSED_CSV_PATH}...")
    x_control, control_mapping = load_crossed_control_from_csv(CROSSED_CSV_PATH)
    print(f"Loaded crossed control paraphrases shape: {x_control.shape}")

    print("Loading all adversarial paraphrases...")
    x_adv, adv_mapping = load_all_adv_paraphrases()
    print(f"Loaded adversarial paraphrases shape: {x_adv.shape}")

    if x_control.shape[1] != x_adv.shape[1]:
        raise ValueError(f"Feature mismatch: control {x_control.shape[1]} vs adv {x_adv.shape[1]}")

    x = np.vstack([x_control, x_adv])
    y = np.array([0] * len(x_control) + [1] * len(x_adv), dtype=np.int64)

    SVM_DIR.mkdir(parents=True, exist_ok=True)
    np.save(LABELS_BINARY_PATH, y)
    INDICES_MAP_PATH.write_text(
        json.dumps(
            {
                "label_definition": {"0": "crossed_control_paraphrase", "1": "adversarial_paraphrase"},
                "crossed_control_indices_in_joint_matrix": list(range(len(x_control))),
                "adversarial_indices_in_joint_matrix": list(range(len(x_control), len(x_control) + len(x_adv))),
                "crossed_control_source_mapping": control_mapping,
                "adversarial_source_mapping": adv_mapping,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nTraining subset (Crossed Control Paraphrases vs All Adversarial Paraphrases):")
    print(f"  Crossed control (0): {(y == 0).sum()}")
    print(f"  Adversarial paraphrase (1): {(y == 1).sum()}")
    print(f"  Total: {len(y)}")

    outer_cv = StratifiedKFold(n_splits=MAX_OUTER_FOLDS, shuffle=True, random_state=SEED)
    inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)
    param_grid = {"C": C_GRID}

    fold_metrics = []
    outer_details = []
    selected_cs: list[float] = []
    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    y_score_all: list[float] = []

    for fold_id, (train_idx, test_idx) in enumerate(outer_cv.split(x, y), start=1):
        if len(selected_cs) >= MAX_OUTER_FOLDS:
            break
        search = GridSearchCV(
            estimator=build_estimator(),
            param_grid=param_grid,
            scoring="balanced_accuracy",
            cv=inner_cv,
            refit=True,
            n_jobs=-1,
        )
        search.fit(x[train_idx], y[train_idx])

        best_model = search.best_estimator_
        pred = best_model.predict(x[test_idx])
        scores = best_model.decision_function(x[test_idx])

        acc = accuracy_score(y[test_idx], pred)
        bal_acc = balanced_accuracy_score(y[test_idx], pred)
        auc = roc_auc_score(y[test_idx], scores)
        best_c = float(search.best_params_["C"])
        selected_cs.append(best_c)

        fold_metrics.append(
            {
                "outer_fold_id": fold_id,
                "accuracy": float(acc),
                "balanced_accuracy": float(bal_acc),
                "roc_auc": float(auc),
                "best_c": best_c,
                "n_train": int(train_idx.size),
                "n_test": int(test_idx.size),
            }
        )
        outer_details.append(
            {
                "outer_fold_id": fold_id,
                "best_c": best_c,
                "train_indices": train_idx.tolist(),
                "test_indices": test_idx.tolist(),
                "accuracy": float(acc),
                "balanced_accuracy": float(bal_acc),
                "roc_auc": float(auc),
                "inner_cv_mean_balanced_accuracy_by_c": {
                    str(params["C"]): float(score)
                    for params, score in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"])
                },
            }
        )

        y_true_all.extend(y[test_idx].tolist())
        y_pred_all.extend(pred.tolist())
        y_score_all.extend(scores.tolist())
        print(f"  Outer fold {fold_id}/{MAX_OUTER_FOLDS}: bal_acc={bal_acc:.4f}, auc={auc:.4f}, best_C={best_c:g}")
        if len(selected_cs) >= MAX_OUTER_FOLDS:
            break

    if len(selected_cs) != MAX_OUTER_FOLDS:
        raise RuntimeError(f"Expected exactly {MAX_OUTER_FOLDS} best-C values, got {len(selected_cs)}")

    final_c = select_final_c_from_outer(selected_cs)
    final_model = build_estimator()
    final_model.set_params(C=final_c)
    final_model.fit(x, y)
    final_w = final_model.coef_.ravel().astype(np.float64)
    final_b = float(final_model.intercept_[0])

    np.save(BEST_W_PATH, final_w)
    np.save(BEST_B_PATH, np.asarray([final_b], dtype=np.float64))

    y_true_all_np = np.asarray(y_true_all, dtype=np.int64)
    y_pred_all_np = np.asarray(y_pred_all, dtype=np.int64)
    y_score_all_np = np.asarray(y_score_all, dtype=np.float64)

    class_counts = {
        "0_crossed_control_paraphrase": int((y == 0).sum()),
        "1_adversarial_paraphrase": int((y == 1).sum()),
    }
    class_proportions = {
        "0_crossed_control_paraphrase": float((y == 0).mean()),
        "1_adversarial_paraphrase": float((y == 1).mean()),
    }

    c_frequency: dict[str, int] = {}
    for c in selected_cs:
        key = str(c)
        c_frequency[key] = c_frequency.get(key, 0) + 1

    results = {
        "model_name": "llama1b",
        "analysis_name": "crossed_control_paraphrases_vs_all_adversarial_paraphrases",
        "crossed_control_csv_path": str(CROSSED_CSV_PATH),
        "control_paraphrase_paths": {k: str(v) for k, v in CONTROL_PARA_PATHS.items()},
        "adversarial_paraphrase_paths": {k: str(v) for k, v in ADV_PARA_PATHS.items()},
        "subset_definition": "Binary classification on raw embeddings: crossed-control paraphrases vs all adversarial paraphrases.",
        "label_definition": {"0": "crossed_control_paraphrase", "1": "adversarial_paraphrase"},
        "preprocessing": {
            "description": "none; raw embeddings used as stored",
            "note": "The saved w_control_adv_llama1b_cross_para and b_control_adv_llama1b_cross_para operate directly on raw embeddings.",
        },
        "class_imbalance_strategy": {
            "class_weighting": "LinearSVC(class_weight='balanced')",
            "primary_selection_metric": "balanced_accuracy",
        },
        "model_selection": {
            "estimator": "sklearn LinearSVC",
            "C_grid": C_GRID,
            "outer_cv": {
                "type": "StratifiedKFold",
                "n_splits": MAX_OUTER_FOLDS,
                "shuffle": True,
                "random_state": SEED,
                "policy_note": "Stopped best-C search after exactly 5 outer folds.",
            },
            "inner_cv": {
                "type": "StratifiedKFold",
                "n_splits": INNER_SPLITS,
                "shuffle": True,
                "random_state": SEED,
            },
            "selection_scoring": "balanced_accuracy",
            "final_c_selection": "Mode across 5 outer-fold best C values; deterministic tie-breaker picks smaller C.",
        },
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "label_distribution": {
            "counts": class_counts,
            "proportions": class_proportions,
        },
        "fold_metrics": fold_metrics,
        "summary_metrics": {
            "accuracy": summarize([item["accuracy"] for item in fold_metrics]),
            "balanced_accuracy": summarize([item["balanced_accuracy"] for item in fold_metrics]),
            "roc_auc": summarize([item["roc_auc"] for item in fold_metrics]),
        },
        "pooled_out_of_fold_metrics": {
            "accuracy": float(accuracy_score(y_true_all_np, y_pred_all_np)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true_all_np, y_pred_all_np)),
            "roc_auc": float(roc_auc_score(y_true_all_np, y_score_all_np)),
        },
        "selected_c_values_from_outer_folds": selected_cs,
        "selected_c_frequency_across_outer_folds": c_frequency,
        "final_refit_model": {
            "selected_c_from_outer_mode": final_c,
            "saved_w_path": str(BEST_W_PATH),
            "saved_b_path": str(BEST_B_PATH),
            "w_shape": list(final_w.shape),
            "b_shape": [1],
        },
        "outer_cv_details": outer_details,
    }
    JSON_OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "Linear SVM on llama1b paraphrase embeddings",
        "Binary Classification: Crossed Control Paraphrases vs All Adversarial Paraphrases",
        f"Crossed-control source CSV: {CROSSED_CSV_PATH}",
        "",
        "Raw-embedding setup",
        "- Uses raw embeddings with no normalization or centering.",
        "- Uses class_weight='balanced' and balanced accuracy for model selection.",
        f"- Tunes C by nested CV with exactly {MAX_OUTER_FOLDS} outer folds and {INNER_SPLITS}-fold inner CV.",
        "- Selects final C as mode(best_C across outer folds), tie-breaker = smaller C.",
        "- Refits a final hyperplane on all samples with selected C and saves llama1b_cross_para-named w/b files.",
        "",
        f"Samples: {x.shape[0]}",
        f"Features: {x.shape[1]}",
        "",
        "Label distribution",
        f"0 (crossed control paraphrase): {class_counts['0_crossed_control_paraphrase']} / {x.shape[0]} = {class_proportions['0_crossed_control_paraphrase']:.4f}",
        f"1 (adversarial paraphrase): {class_counts['1_adversarial_paraphrase']} / {x.shape[0]} = {class_proportions['1_adversarial_paraphrase']:.4f}",
        "",
        "Outer-fold summary (5 folds)",
        f"Accuracy mean +- std: {results['summary_metrics']['accuracy']['mean']:.4f} +- {results['summary_metrics']['accuracy']['std']:.4f}",
        f"Balanced accuracy mean +- std: {results['summary_metrics']['balanced_accuracy']['mean']:.4f} +- {results['summary_metrics']['balanced_accuracy']['std']:.4f}",
        f"ROC-AUC mean +- std: {results['summary_metrics']['roc_auc']['mean']:.4f} +- {results['summary_metrics']['roc_auc']['std']:.4f}",
        "",
        "Pooled out-of-fold metrics",
        f"Accuracy: {results['pooled_out_of_fold_metrics']['accuracy']:.4f}",
        f"Balanced accuracy: {results['pooled_out_of_fold_metrics']['balanced_accuracy']:.4f}",
        f"ROC-AUC: {results['pooled_out_of_fold_metrics']['roc_auc']:.4f}",
        "",
        "Saved final refit hyperplane",
        f"Selected final C from 5 outer folds: {final_c:g}",
        f"Saved w_control_adv_llama1b_cross_para: {BEST_W_PATH}",
        f"Saved b_control_adv_llama1b_cross_para: {BEST_B_PATH}",
    ]
    TXT_OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved JSON results to {JSON_OUT_PATH}")
    print(f"Saved text summary to {TXT_OUT_PATH}")
    print(f"Saved binary labels to {LABELS_BINARY_PATH}")
    print(f"Saved index mapping to {INDICES_MAP_PATH}")
    print(f"Saved final hyperplane w_control_adv_llama1b_cross_para to {BEST_W_PATH}")
    print(f"Saved final hyperplane b_control_adv_llama1b_cross_para to {BEST_B_PATH}")


if __name__ == "__main__":
    main()
