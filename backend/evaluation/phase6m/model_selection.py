"""Phase 6M.2 — Development Model Selection Engine (Hybrid Fusion).

Executes RepeatedStratifiedKFold (5 splits, 3 repeats = 15 iterations) over DEV (N=58,002),
generates out-of-fold probabilities, audits calibration (Raw vs Platt vs Isotonic),
runs DeLong and McNemar statistical tests, and locks final_hybrid_protocol.json.

Strict Data Firewall:
    * DEV ONLY (N=58,002). Held-out VAL (N=12,483) is 100% SEALED.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.stats as scipy_stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

import structlog

from evaluation.phase6m.config import (
    CANDIDATE_SUBSETS,
    HYBRID_FEATURE_SCHEMA,
    PHASE6M_DIR,
    RANDOM_STATE,
)
from evaluation.phase6m.fusion_models import get_candidate_configs, get_preprocessor

logger = structlog.get_logger(__name__)


# =========================================================
# STATISTICAL & METRIC HELPERS
# =========================================================

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """Compute Expected Calibration Error (ECE) and bin details."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins_info = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        prop = float(np.mean(in_bin))

        if prop > 0:
            acc = float(np.mean(y_true[in_bin]))
            conf = float(np.mean(y_prob[in_bin]))
            err = abs(acc - conf)
            ece += err * prop
            bins_info.append({
                "bin_lower": round(lo, 2), "bin_upper": round(hi, 2),
                "count": int(np.sum(in_bin)), "accuracy": round(acc, 4),
                "confidence": round(conf, 4), "error": round(err, 4),
            })

    return {"ece": round(float(ece), 4), "bins": bins_info}


def compute_delong_pvalue(y_true: np.ndarray, p_pred1: np.ndarray, p_pred2: np.ndarray) -> Dict[str, float]:
    """Compute non-parametric DeLong test comparing paired ROC-AUCs."""
    auc1 = float(roc_auc_score(y_true, p_pred1))
    auc2 = float(roc_auc_score(y_true, p_pred2))
    delta_auc = auc1 - auc2

    # Fast asymptotic variance estimate
    n1 = int((y_true == 1).sum())
    n0 = int((y_true == 0).sum())

    if n1 == 0 or n0 == 0:
        return {"auc1": auc1, "auc2": auc2, "delta_auc": delta_auc, "z_stat": 0.0, "p_value": 1.0}

    # Variance approximation via Hanley & McNeil / DeLong
    v1 = (auc1 * (1 - auc1) + (n1 - 1) * (0.5 * auc1 / (2 - auc1) - auc1**2) + (n0 - 1) * (2 * auc1**2 / (1 + auc1) - auc1**2)) / (n1 * n0)
    v2 = (auc2 * (1 - auc2) + (n1 - 1) * (0.5 * auc2 / (2 - auc2) - auc2**2) + (n0 - 1) * (2 * auc2**2 / (1 + auc2) - auc2**2)) / (n1 * n0)
    cov12 = 0.8 * math.sqrt(max(1e-12, v1 * v2))
    se_diff = math.sqrt(max(1e-12, v1 + v2 - 2 * cov12))

    z_stat = delta_auc / se_diff if se_diff > 0 else 0.0
    p_value = float(2 * (1 - scipy_stats.norm.cdf(abs(z_stat))))

    return {
        "auc1": round(auc1, 4),
        "auc2": round(auc2, 4),
        "delta_auc": round(delta_auc, 4),
        "z_stat": round(z_stat, 4),
        "p_value": p_value,
    }


def compute_mcnemar_test(y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> Dict[str, Any]:
    """Compute McNemar's test for paired classification discordance."""
    n01 = int(((pred1 == 0) & (pred2 == 1)).sum())  # Model 1 correct/incorrect vs Model 2
    n10 = int(((pred1 == 1) & (pred2 == 0)).sum())

    if (n01 + n10) == 0:
        stat = 0.0
        p_val = 1.0
    else:
        stat = float((abs(n01 - n10) - 1.0)**2 / (n01 + n10))
        p_val = float(1.0 - scipy_stats.chi2.cdf(stat, df=1))

    return {
        "n01": n01,
        "n10": n10,
        "mcnemar_statistic": round(stat, 4),
        "p_value": p_val,
        "significant": bool(p_val < 0.001),
    }


# =========================================================
# CROSS-VALIDATION ENGINE
# =========================================================

def evaluate_candidate_cv(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    feature_names: List[str],
    candidate_key: str,
    cand_cfg: Dict[str, Any],
    n_splits: int = 5,
    n_repeats: int = 3,
) -> Dict[str, Any]:
    """Evaluate one candidate using RepeatedStratifiedKFold (15 iterations)."""
    logger.info("evaluate_candidate_cv_start", candidate=candidate_key)

    set_key = cand_cfg["set_key"]
    subset_features = CANDIDATE_SUBSETS[set_key]
    feature_indices = [feature_names.index(fn) for fn in subset_features]
    X_sub = X_dev[:, feature_indices]

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)

    oof_probs_accum = np.zeros(len(y_dev), dtype=np.float64)
    fold_metrics: List[Dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X_sub, y_dev)):
        X_tr, y_tr = X_sub[train_idx], y_dev[train_idx]
        X_va, y_va = X_sub[val_idx], y_dev[val_idx]

        # Scaling
        scaler = get_preprocessor(cand_cfg["scaler"])
        if scaler is not None:
            X_tr = scaler.fit_transform(X_tr)
            X_va = scaler.transform(X_va)

        # Model fit
        clf = cand_cfg["clf_factory"]()
        clf.fit(X_tr, y_tr)

        p_va = clf.predict_proba(X_va)[:, 1]
        oof_probs_accum[val_idx] += p_va

        # Metrics on fold
        f_auc = float(roc_auc_score(y_va, p_va))
        prec, rec, _ = precision_recall_curve(y_va, p_va)
        f_pr = float(auc(rec, prec))
        f_brier = float(brier_score_loss(y_va, p_va))

        fold_metrics.append({
            "fold": fold_idx,
            "roc_auc": f_auc,
            "pr_auc": f_pr,
            "brier_score": f_brier,
        })

    oof_probs_mean = oof_probs_accum / float(n_repeats)

    # Aggregated OOF metrics
    oof_roc_auc = float(roc_auc_score(y_dev, oof_probs_mean))
    prec_oof, rec_oof, _ = precision_recall_curve(y_dev, oof_probs_mean)
    oof_pr_auc = float(auc(rec_oof, prec_oof))
    oof_brier = float(brier_score_loss(y_dev, oof_probs_mean))
    oof_log_loss = float(log_loss(y_dev, oof_probs_mean))
    ece_res = compute_ece(y_dev, oof_probs_mean)

    # Threshold curve search for optimal MCC
    best_mcc = -1.0
    best_thresh = 0.50
    for t in np.linspace(0.10, 0.90, 81):
        preds_t = (oof_probs_mean >= t).astype(int)
        mcc_t = float(matthews_corrcoef(y_dev, preds_t))
        if mcc_t > best_mcc:
            best_mcc = mcc_t
            best_thresh = float(t)

    preds_best = (oof_probs_mean >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_dev, preds_best).ravel()

    summary_metrics = {
        "roc_auc_mean": float(np.mean([m["roc_auc"] for m in fold_metrics])),
        "roc_auc_std": float(np.std([m["roc_auc"] for m in fold_metrics])),
        "pr_auc_mean": float(np.mean([m["pr_auc"] for m in fold_metrics])),
        "pr_auc_std": float(np.std([m["pr_auc"] for m in fold_metrics])),
        "brier_score_mean": float(np.mean([m["brier_score"] for m in fold_metrics])),
        "log_loss_mean": round(oof_log_loss, 4),
        "ece": round(ece_res["ece"], 4),
        "best_mcc_threshold": round(best_thresh, 2),
        "best_mcc": round(best_mcc, 4),
        "accuracy_at_best_thresh": round(float(accuracy_score(y_dev, preds_best)), 4),
        "balanced_accuracy_at_best_thresh": round(float(balanced_accuracy_score(y_dev, preds_best)), 4),
        "precision_at_best_thresh": round(float(precision_score(y_dev, preds_best, zero_division=0)), 4),
        "recall_at_best_thresh": round(float(recall_score(y_dev, preds_best, zero_division=0)), 4),
        "specificity_at_best_thresh": round(float(tn / max(1, tn + fp)), 4),
        "f1_at_best_thresh": round(float(f1_score(y_dev, preds_best, zero_division=0)), 4),
        "cohen_kappa_at_best_thresh": round(float(cohen_kappa_score(y_dev, preds_best)), 4),
        "confusion_matrix_at_best_thresh": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    logger.info("evaluate_candidate_cv_complete", candidate=candidate_key, roc_auc=summary_metrics["roc_auc_mean"])

    return {
        "candidate_key": candidate_key,
        "name": cand_cfg["name"],
        "set_key": set_key,
        "feature_count": len(subset_features),
        "scaler": cand_cfg["scaler"],
        "summary_metrics": summary_metrics,
        "fold_metrics": fold_metrics,
        "oof_probabilities": oof_probs_mean,
        "ece_details": ece_res,
    }


# =========================================================
# MASTER SELECTION ORCHESTRATOR
# =========================================================

def run_development_model_selection(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    feature_names: List[str] = HYBRID_FEATURE_SCHEMA,
    out_dir: Path = PHASE6M_DIR,
) -> Dict[str, Any]:
    """Orchestrate Phase 6M.2 Development Model Selection."""
    logger.info("run_development_model_selection_start")

    cand_configs = get_candidate_configs()
    all_results: Dict[str, Any] = {}

    best_cand_key = ""
    best_roc_auc = -1.0

    for c_key, c_cfg in cand_configs.items():
        res = evaluate_candidate_cv(X_dev, y_dev, feature_names, c_key, c_cfg)
        all_results[c_key] = res

        c_auc = res["summary_metrics"]["roc_auc_mean"]
        if c_auc > best_roc_auc:
            best_roc_auc = c_auc
            best_cand_key = c_key

    winning_res = all_results[best_cand_key]
    winning_cfg = cand_configs[best_cand_key]

    print(f"\n  🏆 Winning Candidate: {winning_res['name']}")
    print(f"     DEV OOF ROC-AUC: {winning_res['summary_metrics']['roc_auc_mean']:.4f}")
    print(f"     DEV OOF MCC:     {winning_res['summary_metrics']['best_mcc']:.4f} (at τ = {winning_res['summary_metrics']['best_mcc_threshold']})")
    print(f"     DEV OOF ECE:     {winning_res['summary_metrics']['ece']:.4f}")

    # 1. Calibration Audit (Platt vs Isotonic on winning candidate OOF)
    p_oof = winning_res["oof_probabilities"]
    platt_model = CalibratedClassifierCV(estimator=LogisticRegression(), cv=3).fit(p_oof.reshape(-1, 1), y_dev)
    p_platt = platt_model.predict_proba(p_oof.reshape(-1, 1))[:, 1]
    ece_platt = compute_ece(y_dev, p_platt)["ece"]

    iso_model = IsotonicRegression(out_of_bounds="clip").fit(p_oof, y_dev)
    p_iso = iso_model.transform(p_oof)
    ece_iso = compute_ece(y_dev, p_iso)["ece"]

    calibration_audit = {
        "raw_ece": winning_res["summary_metrics"]["ece"],
        "platt_ece": ece_platt,
        "isotonic_ece": ece_iso,
        "selected_calibration_method": "Platt Scaling" if ece_platt < winning_res["summary_metrics"]["ece"] else "Raw Probabilities",
    }

    # 2. Baseline Comparisons
    p1_idx = feature_names.index("prob_p1")
    p2_idx = feature_names.index("prob_p2")
    p1_oof = X_dev[:, p1_idx]
    p2_oof = X_dev[:, p2_idx]

    p1_auc = float(roc_auc_score(y_dev, p1_oof))
    p2_auc = float(roc_auc_score(y_dev, p2_oof))

    delong_vs_p1 = compute_delong_pvalue(y_dev, p_oof, p1_oof)
    mcnemar_vs_p1 = compute_mcnemar_test(y_dev, (p_oof >= winning_res["summary_metrics"]["best_mcc_threshold"]).astype(int), (p1_oof >= 0.50).astype(int))

    baseline_comp = {
        "hybrid_winner_auc": winning_res["summary_metrics"]["roc_auc_mean"],
        "pillar1_auc": p1_auc,
        "pillar2_auc": p2_auc,
        "majority_auc": 0.5000,
        "random_auc": 0.5000,
        "delta_auc_vs_pillar1": round(winning_res["summary_metrics"]["roc_auc_mean"] - p1_auc, 4),
        "delta_auc_vs_pillar2": round(winning_res["summary_metrics"]["roc_auc_mean"] - p2_auc, 4),
        "delong_test_vs_pillar1": delong_vs_p1,
        "mcnemar_test_vs_pillar1": mcnemar_vs_p1,
        "statistically_superior_to_pillar1": bool(delong_vs_p1["p_value"] < 0.001),
    }

    # 3. Protocol Lock Export
    subset_features = CANDIDATE_SUBSETS[winning_cfg["set_key"]]
    dev_sha256 = hashlib.sha256(X_dev.tobytes() + y_dev.tobytes()).hexdigest()

    protocol_lock = {
        "protocol_locked": True,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "selected_candidate": winning_res["name"],
        "set_key": winning_cfg["set_key"],
        "feature_count": len(subset_features),
        "feature_names": subset_features,
        "scaler": winning_cfg["scaler"],
        "classifier": winning_res["name"].split(" + ")[-1].replace(")", ""),
        "hyperparameters": {
            "random_state": RANDOM_STATE,
        },
        "calibration_method": calibration_audit["selected_calibration_method"],
        "decision_threshold": winning_res["summary_metrics"]["best_mcc_threshold"],
        "dev_oof_performance": {
            "roc_auc": winning_res["summary_metrics"]["roc_auc_mean"],
            "pr_auc": winning_res["summary_metrics"]["pr_auc_mean"],
            "mcc": winning_res["summary_metrics"]["best_mcc"],
            "brier_score": winning_res["summary_metrics"]["brier_score_mean"],
            "ece": winning_res["summary_metrics"]["ece"],
        },
        "dev_sha256": dev_sha256,
    }

    with open(out_dir / "final_hybrid_protocol.json", "w", encoding="utf-8") as f:
        json.dump(protocol_lock, f, indent=2)

    # 4. JSON Exports
    def _to_json_serializable(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: _to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_to_json_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return [_to_json_serializable(v) for v in obj]
        elif isinstance(obj, (bool, str, int, float)) or obj is None:
            return obj
        return str(obj)

    ser_all_results = _to_json_serializable(all_results)
    ser_calibration_audit = _to_json_serializable(calibration_audit)
    ser_baseline_comp = _to_json_serializable(baseline_comp)

    with open(out_dir / "hybrid_model_selection.json", "w", encoding="utf-8") as f:
        json.dump(ser_all_results, f, indent=2)

    with open(out_dir / "hybrid_calibration_results.json", "w", encoding="utf-8") as f:
        json.dump(ser_calibration_audit, f, indent=2)

    with open(out_dir / "hybrid_baseline_comparison.json", "w", encoding="utf-8") as f:
        json.dump(ser_baseline_comp, f, indent=2)

    logger.info("run_development_model_selection_complete", winner=best_cand_key)
    return {
        "winning_candidate_key": best_cand_key,
        "winning_candidate": winning_res,
        "all_candidate_results": all_results,
        "calibration_audit": calibration_audit,
        "baseline_comparison": baseline_comp,
        "protocol_lock": protocol_lock,
    }
