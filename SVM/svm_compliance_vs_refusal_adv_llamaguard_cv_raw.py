from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC


SVM_DIR = Path(__file__).resolve().parent
ROOT = SVM_DIR.parent
ANSWERS_DIR = ROOT / "answers"

ADV_QUERY_EMBED_PATH = ROOT / "adv_prompts_lasttok_emb.npy"
PARA_EMBED_PATHS = {
    "synonyms": ROOT / "para_synonyms_adv_lasttok_emb.npy",
    "swap": ROOT / "para_swap_adv_lasttok_emb.npy",
    "symbol": ROOT / "para_symbol_adv_lasttok_emb.npy",
    "leetspeak": ROOT / "para_leetspeak_adv_lasttok_emb.npy",
}

ADV_QUERY_LABELS_PATH = ANSWERS_DIR / "llamaguard_labels_answers_llama1b_adv_queries.npy"
PARA_LABEL_PATHS = {
    "synonyms": ANSWERS_DIR / "llamaguard_labels_answers_llama1b_para_synonyms_adv.npy",
    "swap": ANSWERS_DIR / "llamaguard_labels_answers_llama1b_para_swap_adv.npy",
    "symbol": ANSWERS_DIR / "llamaguard_labels_answers_llama1b_para_symbol_adv.npy",
    "leetspeak": ANSWERS_DIR / "llamaguard_labels_answers_llama1b_para_leetspeak_adv.npy",
}

COMPLIANCE_POOL_PATH = SVM_DIR / "compliance_embeddings_comp-ref_llama1b.npy"
REFUSAL_POOL_PATH = SVM_DIR / "refusal_embeddings_comp-ref_llama1b.npy"
JSON_OUT_PATH = SVM_DIR / "svm_comp-ref_llama1b_cv_raw_results.json"
TXT_OUT_PATH = SVM_DIR / "svm_comp-ref_llama1b_cv_raw_summary.txt"
LABELS_BINARY_PATH = SVM_DIR / "binary_labels_comp-ref_llama1b.npy"
INDICES_MAP_PATH = SVM_DIR / "indices_mapping_comp-ref_llama1b.json"
BEST_W_PATH = SVM_DIR / "w_comp-ref_llama1b.npy"
BEST_B_PATH = SVM_DIR / "b_comp-ref_llama1b.npy"
CHECKPOINT_PATH = SVM_DIR / "svm_comp-ref_llama1b_cv_progress_checkpoint.json"

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


def save_checkpoint(
    *,
    completed: bool,
    fold_metrics: list[dict[str, object]],
    outer_details: list[dict[str, object]],
    selected_cs: list[float],
    y_true_all: list[int],
    y_pred_all: list[int],
    y_score_all: list[float],
) -> None:
    payload = {
        "completed": completed,
        "progress": {
            "completed_outer_folds": len(selected_cs),
            "target_outer_folds": MAX_OUTER_FOLDS,
        },
        "selected_c_values_from_outer_folds": selected_cs,
        "fold_metrics": fold_metrics,
        "outer_cv_details": outer_details,
        "pooled_so_far": {
            "n_true": len(y_true_all),
            "n_pred": len(y_pred_all),
            "n_score": len(y_score_all),
        },
    }
    CHECKPOINT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_labels_1d(path: Path, expected_len: int) -> np.ndarray:
    x = np.asarray(np.load(path, allow_pickle=True), dtype=np.int8).reshape(-1)
    if x.shape != (expected_len,):
        raise ValueError(f"Expected shape ({expected_len},) at {path}, got {x.shape}")
    if not np.all((x == 0) | (x == 1)):
        raise ValueError(f"Labels at {path} are not strictly binary 0/1")
    return x


def load_embeddings_2d(path: Path, expected_rows: int, expected_dim: int | None = None) -> np.ndarray:
    x = np.asarray(np.load(path, allow_pickle=True), dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D embeddings at {path}, got {x.shape}")
    if x.shape[0] != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows at {path}, got {x.shape[0]}")
    if expected_dim is not None and x.shape[1] != expected_dim:
        raise ValueError(f"Expected feature dim {expected_dim} at {path}, got {x.shape[1]}")
    return x


def load_para_embeddings_3d(path: Path, expected_q: int, expected_p: int, expected_dim: int | None = None) -> np.ndarray:
    x = np.asarray(np.load(path, allow_pickle=True), dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D paraphrase embeddings at {path}, got {x.shape}")
    if x.shape[0] != expected_q or x.shape[1] != expected_p:
        raise ValueError(f"Expected ({expected_q}, {expected_p}, d) at {path}, got {x.shape}")
    if expected_dim is not None and x.shape[2] != expected_dim:
        raise ValueError(f"Expected feature dim {expected_dim} at {path}, got {x.shape[2]}")
    return x


def build_adv_compliance_refusal_pools() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    print("Loading adversarial query embeddings and labels...")
    adv_query_emb = load_embeddings_2d(ADV_QUERY_EMBED_PATH, expected_rows=96)
    feat_dim = int(adv_query_emb.shape[1])
    adv_query_labels = load_labels_1d(ADV_QUERY_LABELS_PATH, expected_len=96)

    # LlamaGuard mapping:
    # 0 = refusal (safe), 1 = compliance (unsafe)
    comp_query_mask = adv_query_labels == 1
    ref_query_mask = adv_query_labels == 0
    compliance_parts = [adv_query_emb[comp_query_mask]]
    refusal_parts = [adv_query_emb[ref_query_mask]]

    source_index_map: dict[str, object] = {
        "adv_queries_indices": {
            "compliance_label_1": np.where(comp_query_mask)[0].astype(int).tolist(),
            "refusal_label_0": np.where(ref_query_mask)[0].astype(int).tolist(),
        },
        "adv_paraphrases_indices": {
            "compliance_label_1": {},
            "refusal_label_0": {},
        },
    }

    total_adv_points = int(adv_query_emb.shape[0])
    print(f"Loaded adv queries: {adv_query_emb.shape}, feature_dim={feat_dim}")

    for family in ("synonyms", "swap", "symbol", "leetspeak"):
        print(f"Loading family '{family}' paraphrase embeddings and labels...")
        emb = load_para_embeddings_3d(PARA_EMBED_PATHS[family], expected_q=96, expected_p=50, expected_dim=feat_dim)
        labels = np.asarray(np.load(PARA_LABEL_PATHS[family], allow_pickle=True), dtype=np.int8)
        if labels.shape != (96, 50):
            raise ValueError(f"Expected (96, 50) labels at {PARA_LABEL_PATHS[family]}, got {labels.shape}")
        if not np.all((labels == 0) | (labels == 1)):
            raise ValueError(f"Labels at {PARA_LABEL_PATHS[family]} are not strictly binary 0/1")

        flat_emb = emb.reshape(-1, feat_dim)
        flat_labels = labels.reshape(-1)
        comp_mask = flat_labels == 1
        ref_mask = flat_labels == 0

        compliance_parts.append(flat_emb[comp_mask])
        refusal_parts.append(flat_emb[ref_mask])

        comp_idx = np.argwhere(labels == 1)
        ref_idx = np.argwhere(labels == 0)
        source_index_map["adv_paraphrases_indices"]["compliance_label_1"][family] = [
            {"query_index": int(i), "paraphrase_index": int(j)} for i, j in comp_idx
        ]
        source_index_map["adv_paraphrases_indices"]["refusal_label_0"][family] = [
            {"query_index": int(i), "paraphrase_index": int(j)} for i, j in ref_idx
        ]

        total_adv_points += int(flat_emb.shape[0])
        print(
            f"  family={family}: emb={emb.shape}, compliance={int(comp_mask.sum())}, refusal={int(ref_mask.sum())}"
        )

    if total_adv_points != 19296:
        raise ValueError(f"Expected total adversarial points = 19296, got {total_adv_points}")
    print(f"Total adversarial points (queries + paraphrases): {total_adv_points}")

    x_compliance = np.vstack(compliance_parts).astype(np.float64)
    x_refusal = np.vstack(refusal_parts).astype(np.float64)
    return x_compliance, x_refusal, source_index_map


def main() -> None:
    print("=== Llama1b Compliance-vs-Refusal SVM (LlamaGuard labels) ===")
    print(f"SVM directory: {SVM_DIR}")
    print(f"Answers directory: {ANSWERS_DIR}")
    print(f"Outer fold policy: collect exactly {MAX_OUTER_FOLDS} best C values, then stop search.")

    x_compliance, x_refusal, source_index_map = build_adv_compliance_refusal_pools()
    print(f"Compliance pool shape: {x_compliance.shape}")
    print(f"Refusal pool shape:    {x_refusal.shape}")

    np.save(COMPLIANCE_POOL_PATH, x_compliance)
    np.save(REFUSAL_POOL_PATH, x_refusal)
    print(f"Saved compliance pool: {COMPLIANCE_POOL_PATH}")
    print(f"Saved refusal pool:    {REFUSAL_POOL_PATH}")

    x = np.vstack([x_compliance, x_refusal])
    y = np.array([1] * len(x_compliance) + [0] * len(x_refusal), dtype=np.int64)
    # label 1 = compliance, label 0 = refusal (matches LlamaGuard semantics)

    np.save(LABELS_BINARY_PATH, y)
    INDICES_MAP_PATH.write_text(
        json.dumps(
            {
                "label_definition": {"0": "refusal_from_llamaguard_0", "1": "compliance_from_llamaguard_1"},
                "compliance_indices_in_joint_matrix": list(range(len(x_compliance))),
                "refusal_indices_in_joint_matrix": list(range(len(x_compliance), len(x_compliance) + len(x_refusal))),
                "source_indices": source_index_map,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved binary labels: {LABELS_BINARY_PATH}")
    print(f"Saved source index map: {INDICES_MAP_PATH}")

    print("\nTraining subset (Compliance vs Refusal from jailbreak embeddings):")
    print(f"  Compliance label=1: {(y == 1).sum()}")
    print(f"  Refusal    label=0: {(y == 0).sum()}")
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

    print("\nStarting nested CV...")
    for fold_id, (train_idx, test_idx) in enumerate(outer_cv.split(x, y), start=1):
        if len(selected_cs) >= MAX_OUTER_FOLDS:
            break
        print(f"\n[Outer fold {fold_id}/{MAX_OUTER_FOLDS}] train={train_idx.size}, test={test_idx.size}")
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

        print(f"  best_C={best_c:g} | acc={acc:.4f} | bal_acc={bal_acc:.4f} | auc={auc:.4f}")
        save_checkpoint(
            completed=False,
            fold_metrics=fold_metrics,
            outer_details=outer_details,
            selected_cs=selected_cs,
            y_true_all=y_true_all,
            y_pred_all=y_pred_all,
            y_score_all=y_score_all,
        )
        print(f"  checkpoint updated: {CHECKPOINT_PATH}")
        if len(selected_cs) >= MAX_OUTER_FOLDS:
            print(f"Reached {MAX_OUTER_FOLDS} best-C values. Stopping best-C search.")
            break

    if len(selected_cs) != MAX_OUTER_FOLDS:
        raise RuntimeError(f"Expected exactly {MAX_OUTER_FOLDS} best-C values, got {len(selected_cs)}")

    final_c = select_final_c_from_outer(selected_cs)
    print(f"\nSelected final C from 5-fold mode rule: {final_c:g}")

    print("Refitting final hyperplane on full compliance/refusal dataset...")
    final_model = build_estimator()
    final_model.set_params(C=final_c)
    final_model.fit(x, y)
    final_w = final_model.coef_.ravel().astype(np.float64)
    final_b = float(final_model.intercept_[0])
    np.save(BEST_W_PATH, final_w)
    np.save(BEST_B_PATH, np.asarray([final_b], dtype=np.float64))
    print(f"Saved final w: {BEST_W_PATH}")
    print(f"Saved final b: {BEST_B_PATH}")

    y_true_all_np = np.asarray(y_true_all, dtype=np.int64)
    y_pred_all_np = np.asarray(y_pred_all, dtype=np.int64)
    y_score_all_np = np.asarray(y_score_all, dtype=np.float64)

    c_frequency: dict[str, int] = {}
    for c in selected_cs:
        k = str(c)
        c_frequency[k] = c_frequency.get(k, 0) + 1

    results = {
        "model_name": "llama1b",
        "analysis_name": "compliance_vs_refusal_on_all_adversarial_embeddings_llamaguard",
        "label_definition": {"0": "refusal", "1": "compliance"},
        "llamaguard_mapping": {"0": "refusal/safe", "1": "compliance/unsafe"},
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
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
            "final_c_selection": "Mode across 5 outer-fold best C values; tie-breaker = smaller C.",
        },
        "label_distribution": {
            "counts": {
                "0_refusal": int((y == 0).sum()),
                "1_compliance": int((y == 1).sum()),
            },
            "proportions": {
                "0_refusal": float((y == 0).mean()),
                "1_compliance": float((y == 1).mean()),
            },
        },
        "fold_metrics": fold_metrics,
        "summary_metrics": {
            "accuracy": summarize([m["accuracy"] for m in fold_metrics]),
            "balanced_accuracy": summarize([m["balanced_accuracy"] for m in fold_metrics]),
            "roc_auc": summarize([m["roc_auc"] for m in fold_metrics]),
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
        "Linear SVM on llama1b adversarial embeddings",
        "Binary Classification: Compliance vs Refusal (LlamaGuard labels)",
        f"Samples: {x.shape[0]}",
        f"Features: {x.shape[1]}",
        "",
        f"Compliance (1): {(y == 1).sum()}",
        f"Refusal (0): {(y == 0).sum()}",
        "",
        f"Selected C values (5 folds): {selected_cs}",
        f"Final C (mode/tie->smaller): {final_c:g}",
        "",
        f"Pooled accuracy: {results['pooled_out_of_fold_metrics']['accuracy']:.4f}",
        f"Pooled balanced accuracy: {results['pooled_out_of_fold_metrics']['balanced_accuracy']:.4f}",
        f"Pooled ROC-AUC: {results['pooled_out_of_fold_metrics']['roc_auc']:.4f}",
        "",
        f"Saved w: {BEST_W_PATH}",
        f"Saved b: {BEST_B_PATH}",
    ]
    TXT_OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_checkpoint(
        completed=True,
        fold_metrics=fold_metrics,
        outer_details=outer_details,
        selected_cs=selected_cs,
        y_true_all=y_true_all,
        y_pred_all=y_pred_all,
        y_score_all=y_score_all,
    )

    print(f"\nSaved JSON results: {JSON_OUT_PATH}")
    print(f"Saved text summary: {TXT_OUT_PATH}")
    print(f"Saved checkpoint JSON: {CHECKPOINT_PATH}")
    print(f"Balanced accuracy (pooled OOF): {results['pooled_out_of_fold_metrics']['balanced_accuracy']:.6f}")


if __name__ == "__main__":
    main()
