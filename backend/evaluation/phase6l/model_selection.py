"""Phase 6L.2 — Stage 6 & Stage 7: Full Development Model Selection & Baseline Comparison Engine.

Executes repeated stratified 5-fold CV (5 folds x 3 repeats = 15 folds) on FULL DEV partition (N = 58,002)
for nominated candidates and baseline models. Computes threshold-free and threshold metrics,
Out-Of-Fold (OOF) predictions, and baseline comparisons.

Strict Data Firewall Rule:
    * DEV partition ONLY (N = 58,002).
    * HELD-OUT VAL partition (N = 12,483) is 100% SEALED and UNTOUCHED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.stats as scipy_stats
from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    auc as calc_auc,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import LinearSVC
import structlog

from evaluation.phase6l.config import PHASE6L_DIR, STRUCTURAL_FEATURE_COLUMNS

logger = structlog.get_logger(__name__)


def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """Compute Expected Calibration Error (ECE) and bin details."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins_info = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper) if i < n_bins - 1 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = float(np.mean(in_bin))

        if prop_in_bin > 0:
            accuracy_in_bin = float(np.mean(y_true[in_bin]))
            avg_confidence_in_bin = float(np.mean(y_prob[in_bin]))
            bin_error = abs(accuracy_in_bin - avg_confidence_in_bin)
            ece += bin_error * prop_in_bin

            bins_info.append({
                "bin_lower": round(bin_lower, 2),
                "bin_upper": round(bin_upper, 2),
                "count": int(np.sum(in_bin)),
                "accuracy": round(accuracy_in_bin, 4),
                "confidence": round(avg_confidence_in_bin, 4),
            })

    return {"ece": float(ece), "bins": bins_info}


def evaluate_model_cv(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: List[int],
    scaler_type: str,
    clf_factory: Any,
    clf_name: str,
    rskf: RepeatedStratifiedKFold,
) -> Dict[str, Any]:
    """Execute RepeatedStratifiedKFold evaluation for a model configuration on DEV.

    Returns:
        Dict containing fold metrics, aggregated mean/std metrics, and OOF probabilities.
    """
    X_sub = X[:, feature_indices]
    n_samples = X.shape[0]

    oof_probs = np.zeros(n_samples, dtype=float)
    oof_counts = np.zeros(n_samples, dtype=int)

    fold_metrics: List[Dict[str, float]] = []

    for fold_idx, (tr_idx, te_idx) in enumerate(rskf.split(X_sub, y), start=1):
        X_tr, y_tr = X_sub[tr_idx], y[tr_idx]
        X_te, y_te = X_sub[te_idx], y[te_idx]

        # Fit Scaler on TRAIN FOLD ONLY (No leakage)
        if scaler_type == "StandardScaler":
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)
        elif scaler_type == "RobustScaler":
            scaler = RobustScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)
        else:
            X_tr_sc, X_te_sc = X_tr, X_te

        # Instantiate fresh classifier
        clf = clf_factory()
        clf.fit(X_tr_sc, y_tr)

        if hasattr(clf, "predict_proba"):
            te_prob = clf.predict_proba(X_te_sc)[:, 1]
        elif hasattr(clf, "decision_function"):
            dec = clf.decision_function(X_te_sc)
            te_prob = expit(dec)
        else:
            te_prob = clf.predict(X_te_sc).astype(float)

        oof_probs[te_idx] += te_prob
        oof_counts[te_idx] += 1

        # Fold metrics (at 0.50 default threshold)
        roc_v = float(roc_auc_score(y_te, te_prob))
        prec, rec, _ = precision_recall_curve(y_te, te_prob)
        pr_v = float(calc_auc(rec, prec))
        brier_v = float(brier_score_loss(y_te, te_prob))
        log_v = float(log_loss(y_te, te_prob, labels=[0, 1]))

        te_pred_50 = (te_prob >= 0.50).astype(int)
        acc_v = float(accuracy_score(y_te, te_pred_50))
        mcc_v = float(matthews_corrcoef(y_te, te_pred_50))
        f1_v = float(f1_score(y_te, te_pred_50, zero_division=0))

        fold_metrics.append({
            "fold": fold_idx,
            "roc_auc": roc_v,
            "pr_auc": pr_v,
            "brier_score": brier_v,
            "log_loss": log_v,
            "accuracy": acc_v,
            "mcc": mcc_v,
            "f1": f1_v,
        })

    # Average OOF probabilities across repeats
    oof_probs_mean = oof_probs / np.maximum(1, oof_counts)

    # Compute overall OOF metrics
    oof_roc = float(roc_auc_score(y, oof_probs_mean))
    prec_oof, rec_oof, _ = precision_recall_curve(y, oof_probs_mean)
    oof_pr = float(calc_auc(rec_oof, prec_oof))
    oof_brier = float(brier_score_loss(y, oof_probs_mean))
    oof_log_loss = float(log_loss(y, oof_probs_mean, labels=[0, 1]))

    ece_res = compute_expected_calibration_error(y, oof_probs_mean)

    # Threshold analysis on OOF predictions
    best_mcc = -1.0
    best_thresh = 0.50
    threshold_curve = []

    for t in np.linspace(0.10, 0.90, 81):
        preds_t = (oof_probs_mean >= t).astype(int)
        mcc_t = float(matthews_corrcoef(y, preds_t))
        f1_t = float(f1_score(y, preds_t, zero_division=0))
        threshold_curve.append({"threshold": round(float(t), 2), "mcc": round(mcc_t, 4), "f1": round(f1_t, 4)})
        if mcc_t > best_mcc:
            best_mcc = mcc_t
            best_thresh = float(t)

    # Evaluate metrics at best MCC threshold
    preds_best = (oof_probs_mean >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds_best).ravel()

    summary_metrics = {
        "roc_auc_mean": float(np.mean([m["roc_auc"] for m in fold_metrics])),
        "roc_auc_std": float(np.std([m["roc_auc"] for m in fold_metrics])),
        "pr_auc_mean": float(np.mean([m["pr_auc"] for m in fold_metrics])),
        "pr_auc_std": float(np.std([m["pr_auc"] for m in fold_metrics])),
        "brier_score_mean": float(np.mean([m["brier_score"] for m in fold_metrics])),
        "log_loss_mean": float(np.mean([m["log_loss"] for m in fold_metrics])),
        "ece": round(ece_res["ece"], 4),
        "best_mcc_threshold": round(best_thresh, 2),
        "best_mcc": round(best_mcc, 4),
        "accuracy_at_best_thresh": round(float(accuracy_score(y, preds_best)), 4),
        "balanced_accuracy_at_best_thresh": round(float(balanced_accuracy_score(y, preds_best)), 4),
        "precision_at_best_thresh": round(float(precision_score(y, preds_best, zero_division=0)), 4),
        "recall_at_best_thresh": round(float(recall_score(y, preds_best, zero_division=0)), 4),
        "specificity_at_best_thresh": round(float(tn / max(1, tn + fp)), 4),
        "f1_at_best_thresh": round(float(f1_score(y, preds_best, zero_division=0)), 4),
        "confusion_matrix_at_best_thresh": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    return {
        "classifier_name": clf_name,
        "scaler_type": scaler_type,
        "summary_metrics": summary_metrics,
        "fold_metrics": fold_metrics,
        "oof_probabilities": oof_probs_mean,
        "ece_details": ece_res,
        "threshold_curve": threshold_curve,
    }


def run_development_model_selection(
    X: np.ndarray,
    y: np.ndarray,
    candidate_sets: Dict[str, Any],
    feature_names: List[str] = STRUCTURAL_FEATURE_COLUMNS,
    out_dir: Path = PHASE6L_DIR,
) -> Dict[str, Any]:
    """Execute repeated 5-fold cross-validation across nominated candidates & baselines on DEV.

    Returns:
        Dict containing full CV results, baseline comparisons, and winning model selection.
    """
    logger.info("stage6_model_selection_start", n_samples=X.shape[0])

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

    # 1. Nominated Candidates
    candidates_config = {
        "Candidate 1": {
            "name": "Candidate 1 (SET_A + StandardScaler + LogisticRegression)",
            "set_key": "SET_A_FULL_SCHEMA",
            "scaler": "StandardScaler",
            "clf_factory": lambda: LogisticRegression(solver="liblinear", penalty="l2", C=1.0, random_state=42, max_iter=1000),
        },
        "Candidate 2": {
            "name": "Candidate 2 (SET_B + RobustScaler + LogisticRegression-SAGA)",
            "set_key": "SET_B_LOW_CORRELATION",
            "scaler": "RobustScaler",
            "clf_factory": lambda: LogisticRegression(solver="saga", penalty="l2", C=1.0, random_state=42, max_iter=1000),
        },
        "Candidate 3": {
            "name": "Candidate 3 (SET_C + StandardScaler + LogisticRegression)",
            "set_key": "SET_C_LOW_VIF",
            "scaler": "StandardScaler",
            "clf_factory": lambda: LogisticRegression(solver="liblinear", penalty="l2", C=1.0, random_state=42, max_iter=1000),
        },
        "Candidate 4": {
            "name": "Candidate 4 (SET_D + RobustScaler + LogisticRegression)",
            "set_key": "SET_D_HIGH_INFORMATION",
            "scaler": "RobustScaler",
            "clf_factory": lambda: LogisticRegression(solver="liblinear", penalty="l2", C=1.0, random_state=42, max_iter=1000),
        },
        "Candidate 5": {
            "name": "Candidate 5 (SET_D + StandardScaler + RandomForest)",
            "set_key": "SET_D_HIGH_INFORMATION",
            "scaler": "StandardScaler",
            "clf_factory": lambda: RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
        },
        "Candidate 6": {
            "name": "Candidate 6 (SET_D + StandardScaler + Calibrated LinearSVC)",
            "set_key": "SET_D_HIGH_INFORMATION",
            "scaler": "StandardScaler",
            "clf_factory": lambda: CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42, max_iter=2000, dual="auto"), cv=3),
        },
    }

    eval_results: Dict[str, Any] = {}

    for cand_key, c_info in candidates_config.items():
        feat_sub_cols = candidate_sets[c_info["set_key"]]["features"]
        feat_indices = [feature_names.index(c) for c in feat_sub_cols]

        res = evaluate_model_cv(
            X=X,
            y=y,
            feature_indices=feat_indices,
            scaler_type=c_info["scaler"],
            clf_factory=c_info["clf_factory"],
            clf_name=c_info["name"],
            rskf=rskf,
        )
        res["feature_set"] = c_info["set_key"]
        res["features"] = feat_sub_cols
        eval_results[cand_key] = res

    # 2. Baseline Comparisons (Stage 7)
    # Baseline A: Majority Classifier (always 0.50 probability)
    prob_maj = np.full(X.shape[0], 0.50)
    baseline_a = {
        "name": "Baseline A (Majority)",
        "roc_auc": 0.5000,
        "pr_auc": float(np.mean(y)),
        "mcc": 0.0000,
        "accuracy": float(max(np.mean(y == 0), np.mean(y == 1))),
    }

    # Baseline B: Random Classifier
    np.random.seed(42)
    prob_rnd = np.random.uniform(0.0, 1.0, X.shape[0])
    baseline_b = {
        "name": "Baseline B (Random)",
        "roc_auc": float(roc_auc_score(y, prob_rnd)),
        "pr_auc": float(calc_auc(*precision_recall_curve(y, prob_rnd)[1::-1])),
        "mcc": float(matthews_corrcoef(y, (prob_rnd >= 0.50).astype(int))),
        "accuracy": float(accuracy_score(y, (prob_rnd >= 0.50).astype(int))),
    }

    # Baseline C: num_claims only
    idx_nc = [feature_names.index("num_claims")]
    res_base_c = evaluate_model_cv(X, y, idx_nc, "StandardScaler", lambda: LogisticRegression(solver="liblinear", random_state=42), "Baseline C (num_claims)", rskf)

    # Baseline D: strongest single feature (max_pairwise_contradiction)
    idx_max_c = [feature_names.index("max_pairwise_contradiction")]
    res_base_d = evaluate_model_cv(X, y, idx_max_c, "RobustScaler", lambda: LogisticRegression(solver="liblinear", random_state=42), "Baseline D (max_contradiction)", rskf)

    # Baseline E: Contradiction-centric subset
    idx_set_f = [feature_names.index(c) for c in candidate_sets["SET_F_CONTRADICTION_CENTRIC"]["features"]]
    res_base_e = evaluate_model_cv(X, y, idx_set_f, "RobustScaler", lambda: LogisticRegression(solver="liblinear", random_state=42), "Baseline E (Contradiction Subset)", rskf)

    baselines_payload = {
        "baseline_a_majority": baseline_a,
        "baseline_b_random": baseline_b,
        "baseline_c_num_claims": res_base_c["summary_metrics"],
        "baseline_d_max_contradiction": res_base_d["summary_metrics"],
        "baseline_e_contradiction_subset": res_base_e["summary_metrics"],
    }

    with open(out_dir / "baseline_comparison.json", "w", encoding="utf-8") as f:
        json.dump(baselines_payload, f, indent=2)

    # Determine Winning Development Candidate (Highest OOF ROC-AUC & MCC)
    best_cand_key = max(eval_results.keys(), key=lambda k: (eval_results[k]["summary_metrics"]["roc_auc_mean"], eval_results[k]["summary_metrics"]["best_mcc"]))
    winning_candidate = eval_results[best_cand_key]

    # Save summary JSON
    summary_export = {
        "winning_candidate_key": best_cand_key,
        "winning_candidate_name": winning_candidate["classifier_name"],
        "winning_candidate_metrics": winning_candidate["summary_metrics"],
        "all_candidates": {
            k: {
                "name": v["classifier_name"],
                "feature_set": v["feature_set"],
                "scaler": v["scaler_type"],
                "summary_metrics": v["summary_metrics"],
            }
            for k, v in eval_results.items()
        },
    }

    with open(out_dir / "full_dev_candidate_comparison.json", "w", encoding="utf-8") as f:
        json.dump(summary_export, f, indent=2)

    logger.info(
        "stage6_model_selection_complete",
        winner=best_cand_key,
        name=winning_candidate["classifier_name"],
        roc_auc=winning_candidate["summary_metrics"]["roc_auc_mean"],
        mcc=winning_candidate["summary_metrics"]["best_mcc"],
    )

    return {
        "eval_results": eval_results,
        "baselines": baselines_payload,
        "winning_candidate_key": best_cand_key,
        "winning_candidate": winning_candidate,
    }
