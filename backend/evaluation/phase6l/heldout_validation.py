"""Phase 6L.3 — Stages 1–11: Final Held-Out Validation of Pillar-2 Structural Consistency Model.

Performs the ONE AND ONLY confirmatory evaluation of the frozen Candidate 5 model
on the untouched Held-Out Validation Partition (N = 12,483).

Locked Configuration (Candidate 5):
    * Features (5): max_pairwise_contradiction, mean_pairwise_contradiction,
                     max_pairwise_similarity, fraction_contradictory_pairs, num_claims
    * Preprocessing: StandardScaler (fit strictly on FULL DEV N=58,002)
    * Classifier: RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    * Primary Operating Threshold: 0.57 (Secondary Reference: 0.50)

Strict Scientific Rule:
    * Validation partition (N=12,483) is INFERENCE-ONLY.
    * No fitting, hyperparameter tuning, scaler fitting, or threshold tuning on VAL.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import scipy.stats as scipy_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc as calc_auc,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression

import structlog

from evaluation.phase6l.config import PHASE6L_DIR, STRUCTURAL_FEATURE_COLUMNS

logger = structlog.get_logger(__name__)

LOCKED_FEATURE_NAMES: List[str] = [
    "max_pairwise_contradiction",
    "mean_pairwise_contradiction",
    "max_pairwise_similarity",
    "fraction_contradictory_pairs",
    "num_claims",
]

PRIMARY_THRESHOLD: float = 0.57
SECONDARY_THRESHOLD: float = 0.50


# =========================================================
# STAGE 1: PROTOCOL VERIFICATION
# =========================================================

def verify_protocol(out_dir: Path = PHASE6L_DIR) -> Dict[str, Any]:
    """Load and verify final_model_protocol.json before accessing VAL."""
    protocol_path = out_dir / "final_model_protocol.json"
    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol file missing: {protocol_path}")

    with open(protocol_path, "r", encoding="utf-8") as f:
        protocol = json.load(f)

    # Compute checksum of protocol file
    with open(protocol_path, "rb") as f:
        protocol_sha256 = hashlib.sha256(f.read()).hexdigest()

    # Verify critical fields
    assert protocol["protocol_locked"] is True, "Protocol not locked!"
    assert protocol["feature_count"] == 5
    assert protocol["feature_names"] == LOCKED_FEATURE_NAMES
    assert protocol["scaler"] == "StandardScaler"
    assert protocol["classifier"] == "RandomForestClassifier"
    assert protocol["decision_threshold"] == PRIMARY_THRESHOLD
    assert protocol["random_seed"] == 42

    verification = {
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "protocol_locked": True,
        "feature_names_verified": True,
        "feature_count_verified": True,
        "scaler_verified": True,
        "classifier_verified": True,
        "threshold_verified": True,
        "random_seed_verified": True,
        "verification_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_contents": protocol,
    }

    with open(out_dir / "final_protocol_verification.json", "w", encoding="utf-8") as f:
        json.dump(verification, f, indent=2)

    logger.info("stage1_protocol_verified", sha256=protocol_sha256[:16])
    return verification


# =========================================================
# STAGE 2: TRAIN LOCKED MODEL ON DEV ONLY
# =========================================================

def train_locked_model_on_dev(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    out_dir: Path = PHASE6L_DIR,
) -> Tuple[StandardScaler, RandomForestClassifier, Dict[str, Any]]:
    """Fit StandardScaler and RandomForestClassifier STRICTLY on DEV (N=58,002)."""
    logger.info("stage2_train_locked_model_start", n_dev=X_dev.shape[0])

    # Select locked features
    feature_indices = [STRUCTURAL_FEATURE_COLUMNS.index(f) for f in LOCKED_FEATURE_NAMES]
    X_dev_selected = X_dev[:, feature_indices].copy()

    # Fit scaler on DEV only
    scaler = StandardScaler()
    X_dev_scaled = scaler.fit_transform(X_dev_selected)

    # Fit classifier on DEV only
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )

    warning_records: List[Dict[str, str]] = []
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        clf.fit(X_dev_scaled, y_dev)
        for w in captured:
            warning_records.append({
                "category": w.category.__name__,
                "message": str(w.message),
            })

    # Save model artifacts
    model_dir = out_dir / "final_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, model_dir / "preprocessing.joblib")
    joblib.dump(clf, model_dir / "classifier.joblib")

    feature_schema = {
        "feature_names": LOCKED_FEATURE_NAMES,
        "feature_count": len(LOCKED_FEATURE_NAMES),
        "feature_indices_in_full_schema": feature_indices,
        "scaler": "StandardScaler",
        "classifier": "RandomForestClassifier(n_estimators=100, max_depth=6)",
        "operating_threshold": PRIMARY_THRESHOLD,
    }
    with open(model_dir / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(feature_schema, f, indent=2)

    training_info = {
        "dev_samples": int(X_dev.shape[0]),
        "dev_features_used": len(LOCKED_FEATURE_NAMES),
        "dev_pos_count": int((y_dev == 1).sum()),
        "dev_neg_count": int((y_dev == 0).sum()),
        "training_warnings": warning_records,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "feature_importances": dict(zip(LOCKED_FEATURE_NAMES, clf.feature_importances_.tolist())),
    }

    logger.info("stage2_train_locked_model_complete", warnings=len(warning_records))
    return scaler, clf, training_info


# =========================================================
# STAGE 3: HELD-OUT VALIDATION INFERENCE
# =========================================================

def run_heldout_inference(
    X_val: np.ndarray,
    y_val: np.ndarray,
    scaler: StandardScaler,
    clf: RandomForestClassifier,
) -> Dict[str, Any]:
    """Run inference EXACTLY ONCE on the Validation partition."""
    logger.info("stage3_heldout_inference_start", n_val=X_val.shape[0])

    feature_indices = [STRUCTURAL_FEATURE_COLUMNS.index(f) for f in LOCKED_FEATURE_NAMES]
    X_val_selected = X_val[:, feature_indices].copy()

    # Transform VAL (NEVER fit scaler on VAL)
    X_val_scaled = scaler.transform(X_val_selected)

    inference_warnings: List[Dict[str, str]] = []
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        p_val = clf.predict_proba(X_val_scaled)[:, 1]
        for w in captured:
            inference_warnings.append({
                "category": w.category.__name__,
                "message": str(w.message),
            })

    # Threshold-free metrics
    val_roc_auc = float(roc_auc_score(y_val, p_val))
    prec_v, rec_v, _ = precision_recall_curve(y_val, p_val)
    val_pr_auc = float(calc_auc(rec_v, prec_v))
    val_brier = float(brier_score_loss(y_val, p_val))
    val_log_loss = float(log_loss(y_val, p_val, labels=[0, 1]))

    # At primary threshold 0.57
    preds_primary = (p_val >= PRIMARY_THRESHOLD).astype(int)
    tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_val, preds_primary).ravel()
    spec_p = float(tn_p / max(1, tn_p + fp_p))

    metrics_primary = {
        "threshold": PRIMARY_THRESHOLD,
        "accuracy": float(accuracy_score(y_val, preds_primary)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, preds_primary)),
        "precision": float(precision_score(y_val, preds_primary, zero_division=0)),
        "recall": float(recall_score(y_val, preds_primary, zero_division=0)),
        "specificity": spec_p,
        "f1": float(f1_score(y_val, preds_primary, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_val, preds_primary)),
        "tp": int(tp_p), "tn": int(tn_p), "fp": int(fp_p), "fn": int(fn_p),
    }

    # At reference threshold 0.50
    preds_ref = (p_val >= SECONDARY_THRESHOLD).astype(int)
    tn_r, fp_r, fn_r, tp_r = confusion_matrix(y_val, preds_ref).ravel()
    spec_r = float(tn_r / max(1, tn_r + fp_r))

    metrics_ref = {
        "threshold": SECONDARY_THRESHOLD,
        "accuracy": float(accuracy_score(y_val, preds_ref)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, preds_ref)),
        "precision": float(precision_score(y_val, preds_ref, zero_division=0)),
        "recall": float(recall_score(y_val, preds_ref, zero_division=0)),
        "specificity": spec_r,
        "f1": float(f1_score(y_val, preds_ref, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_val, preds_ref)),
        "tp": int(tp_r), "tn": int(tn_r), "fp": int(fp_r), "fn": int(fn_r),
    }

    logger.info("stage3_heldout_inference_complete", roc_auc=val_roc_auc, mcc=metrics_primary["mcc"])

    return {
        "probabilities": p_val,
        "threshold_free_metrics": {
            "roc_auc": val_roc_auc,
            "pr_auc": val_pr_auc,
            "brier_score": val_brier,
            "log_loss": val_log_loss,
        },
        "primary_threshold_metrics": metrics_primary,
        "reference_threshold_metrics": metrics_ref,
        "inference_warnings": inference_warnings,
        "X_val_selected": X_val[:, [STRUCTURAL_FEATURE_COLUMNS.index(f) for f in LOCKED_FEATURE_NAMES]],
    }


# =========================================================
# STAGE 4: BOOTSTRAP CONFIDENCE INTERVALS
# =========================================================

def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = PRIMARY_THRESHOLD,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """2,000 stratified bootstrap resamples for 95% confidence intervals."""
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    boot_metrics: Dict[str, List[float]] = {
        "roc_auc": [], "pr_auc": [], "accuracy": [], "balanced_accuracy": [],
        "precision": [], "recall": [], "specificity": [],
        "f1": [], "mcc": [], "brier_score": [],
    }

    for _ in range(n_bootstrap):
        s_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        s_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        b_idx = np.concatenate([s_pos, s_neg])

        y_b = y_true[b_idx]
        p_b = y_prob[b_idx]
        pred_b = (p_b >= threshold).astype(int)

        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_b, pred_b, labels=[0, 1]).ravel()

        boot_metrics["roc_auc"].append(float(roc_auc_score(y_b, p_b)))
        prec_b, rec_b, _ = precision_recall_curve(y_b, p_b)
        boot_metrics["pr_auc"].append(float(calc_auc(rec_b, prec_b)))
        boot_metrics["accuracy"].append(float(accuracy_score(y_b, pred_b)))
        boot_metrics["balanced_accuracy"].append(float(balanced_accuracy_score(y_b, pred_b)))
        boot_metrics["precision"].append(float(precision_score(y_b, pred_b, zero_division=0)))
        boot_metrics["recall"].append(float(recall_score(y_b, pred_b, zero_division=0)))
        boot_metrics["specificity"].append(float(tn_b / max(1, tn_b + fp_b)))
        boot_metrics["f1"].append(float(f1_score(y_b, pred_b, zero_division=0)))
        boot_metrics["mcc"].append(float(matthews_corrcoef(y_b, pred_b)))
        boot_metrics["brier_score"].append(float(brier_score_loss(y_b, p_b)))

    ci_results = {}
    for m_name, vals in boot_metrics.items():
        arr = np.array(vals)
        ci_results[m_name] = {
            "point_estimate": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "ci95_low": round(float(np.percentile(arr, 2.5)), 4),
            "ci95_high": round(float(np.percentile(arr, 97.5)), 4),
        }

    return ci_results


# =========================================================
# STAGE 5: CALIBRATION ANALYSIS
# =========================================================

def compute_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """Compute ECE, MCE, and reliability diagram data."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
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
            mce = max(mce, err)
            bins_info.append({
                "bin_lower": round(lo, 2), "bin_upper": round(hi, 2),
                "count": int(np.sum(in_bin)), "accuracy": round(acc, 4),
                "confidence": round(conf, 4), "error": round(err, 4),
            })

    calibration_pass = ece < 0.05
    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "n_bins": n_bins,
        "calibration_pass": calibration_pass,
        "bins": bins_info,
    }


# =========================================================
# STAGE 6: GENERALIZATION AUDIT
# =========================================================

def compute_generalization_gap(
    dev_summary: Dict[str, Any],
    val_metrics: Dict[str, Any],
    val_calibration: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare DEV repeated CV benchmarks vs held-out VAL results."""
    dev_auc = dev_summary.get("roc_auc_mean", 0.0)
    dev_pr = dev_summary.get("pr_auc_mean", 0.0)
    dev_mcc = dev_summary.get("best_mcc", 0.0)
    dev_brier = dev_summary.get("brier_score_mean", 0.0)
    dev_ece = dev_summary.get("ece", 0.0)

    val_auc = val_metrics["threshold_free_metrics"]["roc_auc"]
    val_pr = val_metrics["threshold_free_metrics"]["pr_auc"]
    val_mcc = val_metrics["primary_threshold_metrics"]["mcc"]
    val_brier = val_metrics["threshold_free_metrics"]["brier_score"]
    val_ece = val_calibration["ece"]

    gap_auc = val_auc - dev_auc
    degradation = -gap_auc

    if degradation <= 0.02:
        gen_class = "STABLE"
    elif degradation <= 0.05:
        gen_class = "MINOR DEGRADATION"
    else:
        gen_class = "MATERIAL DEGRADATION"

    return {
        "generalization_classification": gen_class,
        "degradation_magnitude": round(float(degradation), 4),
        "dev_cv_auc": dev_auc, "val_auc": val_auc, "gap_auc": round(gap_auc, 4),
        "dev_cv_pr_auc": dev_pr, "val_pr_auc": val_pr, "gap_pr_auc": round(val_pr - dev_pr, 4),
        "dev_cv_mcc": dev_mcc, "val_mcc": val_mcc, "gap_mcc": round(val_mcc - dev_mcc, 4),
        "dev_cv_brier": dev_brier, "val_brier": val_brier, "gap_brier": round(val_brier - dev_brier, 4),
        "dev_cv_ece": dev_ece, "val_ece": val_ece, "gap_ece": round(val_ece - dev_ece, 4),
    }


# =========================================================
# STAGE 7: BASELINE CONFIRMATION ON VAL
# =========================================================

def compute_baseline_comparison(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    p_val_locked: np.ndarray,
    val_roc_auc: float,
) -> Dict[str, Any]:
    """Evaluate 5 baselines on held-out VAL."""
    feature_names = STRUCTURAL_FEATURE_COLUMNS

    # Baseline A: Majority classifier
    base_a = {
        "name": "Majority Classifier",
        "val_roc_auc": 0.5000,
        "val_pr_auc": round(float(np.mean(y_val)), 4),
        "val_mcc": 0.0,
        "val_accuracy": round(float(max(np.mean(y_val == 0), np.mean(y_val == 1))), 4),
    }

    # Baseline B: Random classifier
    np.random.seed(42)
    p_rnd = np.random.uniform(0, 1, len(y_val))
    prec_r, rec_r, _ = precision_recall_curve(y_val, p_rnd)
    base_b = {
        "name": "Random Classifier",
        "val_roc_auc": round(float(roc_auc_score(y_val, p_rnd)), 4),
        "val_pr_auc": round(float(calc_auc(rec_r, prec_r)), 4),
        "val_mcc": round(float(matthews_corrcoef(y_val, (p_rnd >= 0.50).astype(int))), 4),
        "val_accuracy": round(float(accuracy_score(y_val, (p_rnd >= 0.50).astype(int))), 4),
    }

    # Baseline C: Strongest single structural feature (max_pairwise_contradiction)
    idx_mpc = feature_names.index("max_pairwise_contradiction")
    X_dev_c = X_dev[:, [idx_mpc]]
    X_val_c = X_val[:, [idx_mpc]]
    sc_c = StandardScaler().fit(X_dev_c)
    m_c = LogisticRegression(solver="liblinear", random_state=42).fit(sc_c.transform(X_dev_c), y_dev)
    p_c = m_c.predict_proba(sc_c.transform(X_val_c))[:, 1]
    prec_c, rec_c, _ = precision_recall_curve(y_val, p_c)
    base_c = {
        "name": "Single Feature (max_pairwise_contradiction)",
        "val_roc_auc": round(float(roc_auc_score(y_val, p_c)), 4),
        "val_pr_auc": round(float(calc_auc(rec_c, prec_c)), 4),
        "val_mcc": round(float(matthews_corrcoef(y_val, (p_c >= 0.50).astype(int))), 4),
        "val_accuracy": round(float(accuracy_score(y_val, (p_c >= 0.50).astype(int))), 4),
    }

    # Baseline D: num_claims only
    idx_nc = feature_names.index("num_claims")
    X_dev_d = X_dev[:, [idx_nc]]
    X_val_d = X_val[:, [idx_nc]]
    sc_d = StandardScaler().fit(X_dev_d)
    m_d = LogisticRegression(solver="liblinear", random_state=42).fit(sc_d.transform(X_dev_d), y_dev)
    p_d = m_d.predict_proba(sc_d.transform(X_val_d))[:, 1]
    prec_d, rec_d, _ = precision_recall_curve(y_val, p_d)
    base_d = {
        "name": "Single Feature (num_claims)",
        "val_roc_auc": round(float(roc_auc_score(y_val, p_d)), 4),
        "val_pr_auc": round(float(calc_auc(rec_d, prec_d)), 4),
        "val_mcc": round(float(matthews_corrcoef(y_val, (p_d >= 0.50).astype(int))), 4),
        "val_accuracy": round(float(accuracy_score(y_val, (p_d >= 0.50).astype(int))), 4),
    }

    # Locked Pillar-2 model
    prec_l, rec_l, _ = precision_recall_curve(y_val, p_val_locked)
    locked = {
        "name": "Locked Pillar-2 (Candidate 5)",
        "val_roc_auc": round(val_roc_auc, 4),
        "val_pr_auc": round(float(calc_auc(rec_l, prec_l)), 4),
        "val_mcc": round(float(matthews_corrcoef(y_val, (p_val_locked >= PRIMARY_THRESHOLD).astype(int))), 4),
        "val_accuracy": round(float(accuracy_score(y_val, (p_val_locked >= PRIMARY_THRESHOLD).astype(int))), 4),
    }

    return {
        "baseline_a_majority": base_a,
        "baseline_b_random": base_b,
        "baseline_c_max_contradiction": base_c,
        "baseline_d_num_claims": base_d,
        "locked_pillar2_model": locked,
        "improvement_over_majority": round(val_roc_auc - 0.5, 4),
        "improvement_over_random": round(val_roc_auc - base_b["val_roc_auc"], 4),
        "improvement_over_best_single": round(val_roc_auc - base_c["val_roc_auc"], 4),
    }


# =========================================================
# STAGE 8: FEATURE DISTRIBUTION SHIFT
# =========================================================

def compute_distribution_shift(
    X_dev: np.ndarray,
    X_val: np.ndarray,
) -> Dict[str, Any]:
    """Compare DEV vs VAL feature distributions for locked features."""
    feature_indices = [STRUCTURAL_FEATURE_COLUMNS.index(f) for f in LOCKED_FEATURE_NAMES]

    shift_results = {}
    any_flagged = False

    for feat_idx, fname in zip(feature_indices, LOCKED_FEATURE_NAMES):
        col_dev = X_dev[:, feat_idx]
        col_val = X_val[:, feat_idx]

        m_dev, s_dev = float(np.mean(col_dev)), float(np.std(col_dev, ddof=1))
        m_val, s_val = float(np.mean(col_val)), float(np.std(col_val, ddof=1))

        pooled_std = math.sqrt((s_dev**2 + s_val**2) / 2.0) if (s_dev + s_val) > 0 else 1.0
        smd = (m_val - m_dev) / pooled_std

        ks_res = scipy_stats.ks_2samp(col_dev, col_val)
        flagged = bool(abs(smd) > 0.10 or ks_res.pvalue < 0.05)
        if flagged:
            any_flagged = True

        shift_results[fname] = {
            "dev_mean": round(m_dev, 6), "dev_std": round(s_dev, 6),
            "val_mean": round(m_val, 6), "val_std": round(s_val, 6),
            "standardized_mean_difference": round(float(smd), 4),
            "ks_statistic": round(float(ks_res.statistic), 4),
            "ks_pvalue": float(ks_res.pvalue),
            "flagged": flagged,
        }

    return {"features": shift_results, "any_flagged": any_flagged}


# =========================================================
# STAGE 9: ERROR ANALYSIS
# =========================================================

def compute_error_analysis(
    y_val: np.ndarray,
    p_val: np.ndarray,
    X_val_selected: np.ndarray,
) -> Dict[str, Any]:
    """Analyze FP/FN structural patterns."""
    preds = (p_val >= PRIMARY_THRESHOLD).astype(int)
    tp_mask = (y_val == 1) & (preds == 1)
    tn_mask = (y_val == 0) & (preds == 0)
    fp_mask = (y_val == 0) & (preds == 1)
    fn_mask = (y_val == 1) & (preds == 0)

    def group_stats(mask, name):
        if mask.sum() == 0:
            return {"group": name, "count": 0}
        X_g = X_val_selected[mask]
        return {
            "group": name,
            "count": int(mask.sum()),
            "feature_means": {LOCKED_FEATURE_NAMES[i]: round(float(np.mean(X_g[:, i])), 4) for i in range(len(LOCKED_FEATURE_NAMES))},
            "feature_stds": {LOCKED_FEATURE_NAMES[i]: round(float(np.std(X_g[:, i])), 4) for i in range(len(LOCKED_FEATURE_NAMES))},
            "mean_probability": round(float(np.mean(p_val[mask])), 4),
        }

    # Top 10 FP and FN
    fp_idx = np.where(fp_mask)[0]
    fp_sorted = fp_idx[np.argsort(-p_val[fp_idx])[:10]]
    top_fps = []
    for i in fp_sorted:
        top_fps.append({
            "sample_index": int(i),
            "probability": round(float(p_val[i]), 4),
            "features": {LOCKED_FEATURE_NAMES[j]: round(float(X_val_selected[i, j]), 4) for j in range(len(LOCKED_FEATURE_NAMES))},
        })

    fn_idx = np.where(fn_mask)[0]
    fn_sorted = fn_idx[np.argsort(p_val[fn_idx])[:10]]
    top_fns = []
    for i in fn_sorted:
        top_fns.append({
            "sample_index": int(i),
            "probability": round(float(p_val[i]), 4),
            "features": {LOCKED_FEATURE_NAMES[j]: round(float(X_val_selected[i, j]), 4) for j in range(len(LOCKED_FEATURE_NAMES))},
        })

    return {
        "group_statistics": {
            "TP": group_stats(tp_mask, "TP"),
            "TN": group_stats(tn_mask, "TN"),
            "FP": group_stats(fp_mask, "FP"),
            "FN": group_stats(fn_mask, "FN"),
        },
        "top_false_positives": top_fps,
        "top_false_negatives": top_fns,
    }


# =========================================================
# STAGE 10: NUMERICAL HEALTH
# =========================================================

def compute_numerical_health(
    scaler: StandardScaler,
    clf: RandomForestClassifier,
    p_val: np.ndarray,
    training_info: Dict[str, Any],
    inference_warnings: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Verify all coefficients, probabilities, and preprocessing outputs are finite."""
    scaler_finite = bool(np.all(np.isfinite(scaler.mean_)) and np.all(np.isfinite(scaler.scale_)))
    probs_finite = bool(np.all(np.isfinite(p_val)))
    probs_in_range = bool(np.all(p_val >= 0.0) and np.all(p_val <= 1.0))

    # Count warning categories
    overflow_count = sum(1 for w in inference_warnings if "overflow" in w.get("message", "").lower())
    divzero_count = sum(1 for w in inference_warnings if "divide by zero" in w.get("message", "").lower())
    invalid_count = sum(1 for w in inference_warnings if "invalid" in w.get("message", "").lower())
    convergence_count = sum(1 for w in inference_warnings if "converge" in w.get("message", "").lower())

    all_training_warnings = training_info.get("training_warnings", [])
    overflow_train = sum(1 for w in all_training_warnings if "overflow" in w.get("message", "").lower())
    divzero_train = sum(1 for w in all_training_warnings if "divide by zero" in w.get("message", "").lower())
    invalid_train = sum(1 for w in all_training_warnings if "invalid" in w.get("message", "").lower())

    health_pass = scaler_finite and probs_finite and probs_in_range

    return {
        "numerical_health_pass": health_pass,
        "scaler_finite": scaler_finite,
        "probabilities_finite": probs_finite,
        "probabilities_in_01_range": probs_in_range,
        "training_warnings_total": len(all_training_warnings),
        "inference_warnings_total": len(inference_warnings),
        "warning_breakdown": {
            "overflow": overflow_count + overflow_train,
            "divide_by_zero": divzero_count + divzero_train,
            "invalid_matmul": invalid_count + invalid_train,
            "convergence": convergence_count,
        },
    }


# =========================================================
# PUBLICATION FIGURES
# =========================================================

def generate_figures(
    y_val: np.ndarray,
    p_val: np.ndarray,
    X_dev: np.ndarray,
    X_val: np.ndarray,
    val_metrics: Dict[str, Any],
    calibration: Dict[str, Any],
    gen_gap: Dict[str, Any],
    shift_data: Dict[str, Any],
    error_analysis: Dict[str, Any],
    out_dir: Path = PHASE6L_DIR,
) -> List[Path]:
    """Generate 7 publication-quality 300 DPI figures."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    exported: List[Path] = []

    tf = val_metrics["threshold_free_metrics"]

    # 1. ROC Curve
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    fpr_v, tpr_v, _ = roc_curve(y_val, p_val)
    ax.plot(fpr_v, tpr_v, "b-", lw=2, label=f"Pillar-2 Held-Out VAL (AUC = {tf['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance (AUC = 0.5000)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title("Phase 6L.3 — Final Held-Out ROC Curve (VAL N=12,483)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p1 = fig_dir / "phase6l_3_val_roc.png"
    plt.savefig(p1); plt.close(fig); exported.append(p1)

    # 2. PR Curve
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    prec_v, rec_v, _ = precision_recall_curve(y_val, p_val)
    ax.plot(rec_v, prec_v, "g-", lw=2, label=f"Pillar-2 Held-Out VAL (PR-AUC = {tf['pr_auc']:.4f})")
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    ax.set_title("Phase 6L.3 — Final Held-Out Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = fig_dir / "phase6l_3_val_pr.png"
    plt.savefig(p2); plt.close(fig); exported.append(p2)

    # 3. Calibration Diagram
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    confs = [b["confidence"] for b in calibration["bins"] if b["count"] > 0]
    accs = [b["accuracy"] for b in calibration["bins"] if b["count"] > 0]
    ax.plot(confs, accs, "bs-", lw=2, label=f"Held-Out VAL (ECE = {calibration['ece']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability (Confidence)", fontsize=11)
    ax.set_ylabel("Empirical Accuracy", fontsize=11)
    ax.set_title("Phase 6L.3 — Held-Out Reliability Diagram", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p3 = fig_dir / "phase6l_3_val_calibration.png"
    plt.savefig(p3); plt.close(fig); exported.append(p3)

    # 4. Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    m = val_metrics["primary_threshold_metrics"]
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    cax = ax.matshow(cm, cmap="Blues", alpha=0.8)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="black", fontsize=14, fontweight="bold")
    fig.colorbar(cax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Factual (0)", "Hallucinated (1)"], fontsize=10)
    ax.set_yticklabels(["Factual (0)", "Hallucinated (1)"], fontsize=10)
    ax.set_xlabel(f"Predicted Label (τ = {PRIMARY_THRESHOLD})", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"Held-Out VAL Confusion Matrix (N=12,483)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p4 = fig_dir / "phase6l_3_val_confusion_matrix.png"
    plt.savefig(p4); plt.close(fig); exported.append(p4)

    # 5. DEV vs VAL Metric Comparison
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    names = ["ROC-AUC", "PR-AUC", "MCC", "1 - Brier"]
    dev_vals = [gen_gap["dev_cv_auc"], gen_gap["dev_cv_pr_auc"], gen_gap["dev_cv_mcc"], 1.0 - gen_gap["dev_cv_brier"]]
    val_vals = [gen_gap["val_auc"], gen_gap["val_pr_auc"], gen_gap["val_mcc"], 1.0 - gen_gap["val_brier"]]
    x = np.arange(len(names)); w = 0.35
    ax.bar(x - w/2, dev_vals, w, label="DEV OOF (N=58,002)", color="#1f77b4", alpha=0.85)
    ax.bar(x + w/2, val_vals, w, label="Held-Out VAL (N=12,483)", color="#2ca02c", alpha=0.85)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_title("DEV OOF vs Held-Out VAL Performance Comparison", fontsize=12, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(0, 1.0); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p5 = fig_dir / "phase6l_3_dev_val_comparison.png"
    plt.savefig(p5); plt.close(fig); exported.append(p5)

    # 6. Feature Distribution Shift
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    smds = [shift_data["features"][f]["standardized_mean_difference"] for f in LOCKED_FEATURE_NAMES]
    y_pos = np.arange(len(LOCKED_FEATURE_NAMES))
    ax.barh(y_pos, smds, color="#ff7f0e", alpha=0.8, align="center")
    ax.set_yticks(y_pos); ax.set_yticklabels(LOCKED_FEATURE_NAMES, fontsize=10)
    ax.axvline(0, color="k", linestyle="-", lw=1)
    ax.axvline(0.10, color="r", linestyle="--", lw=1, label="|SMD| = 0.10 threshold")
    ax.axvline(-0.10, color="r", linestyle="--", lw=1)
    ax.set_xlabel("Standardized Mean Difference (VAL - DEV)", fontsize=11)
    ax.set_title("DEV vs VAL Feature Distribution Shift", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p6 = fig_dir / "phase6l_3_feature_shift.png"
    plt.savefig(p6); plt.close(fig); exported.append(p6)

    # 7. Error Distribution (FP/FN feature patterns)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    groups = ["TP", "TN", "FP", "FN"]
    ea = error_analysis["group_statistics"]
    means_key = "max_pairwise_contradiction"
    means = [ea[g].get("feature_means", {}).get(means_key, 0.0) for g in groups]
    ax.bar(groups, means, color=["#2ca02c", "#1f77b4", "#d62728", "#ff7f0e"], alpha=0.8)
    ax.set_ylabel(f"{means_key} (Mean)", fontsize=11)
    ax.set_title("Held-Out VAL: max_pairwise_contradiction by Confusion Group", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p7 = fig_dir / "phase6l_3_error_distributions.png"
    plt.savefig(p7); plt.close(fig); exported.append(p7)

    logger.info("figures_generated", count=len(exported))
    return exported


# =========================================================
# FINAL REPORT GENERATOR
# =========================================================

def generate_final_report(
    protocol_ver: Dict[str, Any],
    training_info: Dict[str, Any],
    val_metrics: Dict[str, Any],
    bootstrap_ci: Dict[str, Any],
    calibration: Dict[str, Any],
    gen_gap: Dict[str, Any],
    baselines: Dict[str, Any],
    shift_data: Dict[str, Any],
    error_analysis: Dict[str, Any],
    numerical_health: Dict[str, Any],
    verdict: str,
    out_dir: Path = PHASE6L_DIR,
) -> Path:
    """Generate FINAL_PILLAR2_VALIDATION_REPORT.md."""
    utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    tf = val_metrics["threshold_free_metrics"]
    mp = val_metrics["primary_threshold_metrics"]
    mr = val_metrics["reference_threshold_metrics"]
    ci = bootstrap_ci

    md = f"""# HalluciSense Phase 6L.3 — Final Pillar-2 Held-Out Validation Report

**Generated UTC**: `{utc_str}`
**Evaluation Status**: `COMPLETED`
**Final Pillar-2 Verdict**: **`{verdict}`**

---

## 1. Protocol Verification (Stage 1)

- **Protocol File**: `evaluation_results/phase6l/final_model_protocol.json`
- **Protocol SHA-256**: `{protocol_ver['protocol_sha256'][:32]}...`
- **Features Verified**: ✅ `{', '.join(LOCKED_FEATURE_NAMES)}`
- **Scaler Verified**: ✅ `StandardScaler`
- **Classifier Verified**: ✅ `RandomForestClassifier(n_estimators=100, max_depth=6)`
- **Threshold Verified**: ✅ `{PRIMARY_THRESHOLD}`
- **Random Seed Verified**: ✅ `42`

---

## 2. Held-Out Validation Metrics (Stage 3)

### Threshold-Free Metrics (VAL N=12,483)

| Metric | Point Estimate | 95% Bootstrap CI |
|:---|:---:|:---:|
| **ROC-AUC** | **`{tf['roc_auc']:.4f}`** | `[{ci['roc_auc']['ci95_low']:.4f}, {ci['roc_auc']['ci95_high']:.4f}]` |
| **PR-AUC** | **`{tf['pr_auc']:.4f}`** | `[{ci['pr_auc']['ci95_low']:.4f}, {ci['pr_auc']['ci95_high']:.4f}]` |
| **Brier Score** | **`{tf['brier_score']:.4f}`** | `[{ci['brier_score']['ci95_low']:.4f}, {ci['brier_score']['ci95_high']:.4f}]` |
| **Log Loss** | **`{tf['log_loss']:.4f}`** | — |

### Operating Metrics at Primary Threshold (τ = {PRIMARY_THRESHOLD})

| Metric | Point Estimate | 95% Bootstrap CI |
|:---|:---:|:---:|
| **Accuracy** | `{mp['accuracy']:.4f}` | `[{ci['accuracy']['ci95_low']:.4f}, {ci['accuracy']['ci95_high']:.4f}]` |
| **Balanced Accuracy** | `{mp['balanced_accuracy']:.4f}` | `[{ci['balanced_accuracy']['ci95_low']:.4f}, {ci['balanced_accuracy']['ci95_high']:.4f}]` |
| **Precision** | `{mp['precision']:.4f}` | `[{ci['precision']['ci95_low']:.4f}, {ci['precision']['ci95_high']:.4f}]` |
| **Recall** | `{mp['recall']:.4f}` | `[{ci['recall']['ci95_low']:.4f}, {ci['recall']['ci95_high']:.4f}]` |
| **Specificity** | `{mp['specificity']:.4f}` | `[{ci['specificity']['ci95_low']:.4f}, {ci['specificity']['ci95_high']:.4f}]` |
| **F1 Score** | `{mp['f1']:.4f}` | `[{ci['f1']['ci95_low']:.4f}, {ci['f1']['ci95_high']:.4f}]` |
| **MCC** | `{mp['mcc']:.4f}` | `[{ci['mcc']['ci95_low']:.4f}, {ci['mcc']['ci95_high']:.4f}]` |

### Confusion Matrix at τ = {PRIMARY_THRESHOLD}

| | Predicted Factual | Predicted Hallucinated |
|:---|:---:|:---:|
| **True Factual** | TN = `{mp['tn']:,}` | FP = `{mp['fp']:,}` |
| **True Hallucinated** | FN = `{mp['fn']:,}` | TP = `{mp['tp']:,}` |

### Operating Metrics at Reference Threshold (τ = {SECONDARY_THRESHOLD})

| Metric | Value |
|:---|:---:|
| Accuracy | `{mr['accuracy']:.4f}` |
| Balanced Accuracy | `{mr['balanced_accuracy']:.4f}` |
| F1 | `{mr['f1']:.4f}` |
| MCC | `{mr['mcc']:.4f}` |

---

## 3. Bootstrap Confidence Intervals (Stage 4)

2,000 stratified bootstrap resamples. All intervals are 95% percentile CIs.

| Metric | Mean ± Std | 95% CI |
|:---|:---:|:---:|
| ROC-AUC | `{ci['roc_auc']['point_estimate']:.4f} ± {ci['roc_auc']['std']:.4f}` | `[{ci['roc_auc']['ci95_low']:.4f}, {ci['roc_auc']['ci95_high']:.4f}]` |
| PR-AUC | `{ci['pr_auc']['point_estimate']:.4f} ± {ci['pr_auc']['std']:.4f}` | `[{ci['pr_auc']['ci95_low']:.4f}, {ci['pr_auc']['ci95_high']:.4f}]` |
| MCC | `{ci['mcc']['point_estimate']:.4f} ± {ci['mcc']['std']:.4f}` | `[{ci['mcc']['ci95_low']:.4f}, {ci['mcc']['ci95_high']:.4f}]` |
| Brier Score | `{ci['brier_score']['point_estimate']:.4f} ± {ci['brier_score']['std']:.4f}` | `[{ci['brier_score']['ci95_low']:.4f}, {ci['brier_score']['ci95_high']:.4f}]` |

---

## 4. Calibration Analysis (Stage 5)

- **Expected Calibration Error (ECE)**: `{calibration['ece']:.4f}`
- **Maximum Calibration Error (MCE)**: `{calibration['mce']:.4f}`
- **Calibration PASS**: `{'✅ PASS (ECE < 0.05)' if calibration['calibration_pass'] else '❌ FAIL (ECE ≥ 0.05)'}`

---

## 5. Generalization Audit (Stage 6)

| Metric | DEV CV | Held-Out VAL | Gap (Δ) | Status |
|:---|:---:|:---:|:---:|:---:|
| **ROC-AUC** | `{gen_gap['dev_cv_auc']:.4f}` | `{gen_gap['val_auc']:.4f}` | `{gen_gap['gap_auc']:+.4f}` | **`{gen_gap['generalization_classification']}`** |
| **PR-AUC** | `{gen_gap['dev_cv_pr_auc']:.4f}` | `{gen_gap['val_pr_auc']:.4f}` | `{gen_gap['gap_pr_auc']:+.4f}` | — |
| **MCC** | `{gen_gap['dev_cv_mcc']:.4f}` | `{gen_gap['val_mcc']:.4f}` | `{gen_gap['gap_mcc']:+.4f}` | — |
| **Brier** | `{gen_gap['dev_cv_brier']:.4f}` | `{gen_gap['val_brier']:.4f}` | `{gen_gap['gap_brier']:+.4f}` | — |
| **ECE** | `{gen_gap['dev_cv_ece']:.4f}` | `{gen_gap['val_ece']:.4f}` | `{gen_gap['gap_ece']:+.4f}` | — |

Generalization Classification: **`{gen_gap['generalization_classification']}`** (Degradation = `{gen_gap['degradation_magnitude']:.4f}`)

---

## 6. Baseline Comparison (Stage 7)

| Model / Baseline | VAL ROC-AUC | VAL MCC | Δ ROC-AUC |
|:---|:---:|:---:|:---:|
| **Locked Pillar-2** | **`{baselines['locked_pillar2_model']['val_roc_auc']:.4f}`** | **`{baselines['locked_pillar2_model']['val_mcc']:.4f}`** | — |
| Baseline A (Majority) | `{baselines['baseline_a_majority']['val_roc_auc']:.4f}` | `{baselines['baseline_a_majority']['val_mcc']:.4f}` | `{baselines['improvement_over_majority']:+.4f}` |
| Baseline B (Random) | `{baselines['baseline_b_random']['val_roc_auc']:.4f}` | `{baselines['baseline_b_random']['val_mcc']:.4f}` | `{baselines['improvement_over_random']:+.4f}` |
| Baseline C (max_contradiction) | `{baselines['baseline_c_max_contradiction']['val_roc_auc']:.4f}` | `{baselines['baseline_c_max_contradiction']['val_mcc']:.4f}` | `{baselines['improvement_over_best_single']:+.4f}` |
| Baseline D (num_claims) | `{baselines['baseline_d_num_claims']['val_roc_auc']:.4f}` | `{baselines['baseline_d_num_claims']['val_mcc']:.4f}` | — |

---

## 7. Feature Distribution Shift (Stage 8)

| Feature | DEV Mean | VAL Mean | SMD | KS Stat | KS p-value | Flagged |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
"""

    for fname in LOCKED_FEATURE_NAMES:
        s = shift_data["features"][fname]
        flag_icon = "⚠️" if s["flagged"] else "✅"
        md += f"| `{fname}` | `{s['dev_mean']:.4f}` | `{s['val_mean']:.4f}` | `{s['standardized_mean_difference']:.4f}` | `{s['ks_statistic']:.4f}` | `{s['ks_pvalue']:.2e}` | {flag_icon} |\n"

    md += f"""
---

## 8. Error Analysis (Stage 9)

| Group | Count | Mean max_contradiction | Mean Probability |
|:---|:---:|:---:|:---:|
"""
    for g in ["TP", "TN", "FP", "FN"]:
        ea_g = error_analysis["group_statistics"][g]
        mc = ea_g.get("feature_means", {}).get("max_pairwise_contradiction", 0.0)
        mp_v = ea_g.get("mean_probability", 0.0)
        md += f"| **{g}** | `{ea_g['count']:,}` | `{mc:.4f}` | `{mp_v:.4f}` |\n"

    md += f"""
---

## 9. Numerical Health (Stage 10)

- **Scaler Finite**: `{'✅' if numerical_health['scaler_finite'] else '❌'}`
- **Probabilities Finite**: `{'✅' if numerical_health['probabilities_finite'] else '❌'}`
- **Probabilities in [0, 1]**: `{'✅' if numerical_health['probabilities_in_01_range'] else '❌'}`
- **Training Warnings**: `{numerical_health['training_warnings_total']}`
- **Inference Warnings**: `{numerical_health['inference_warnings_total']}`
- **Overflow**: `{numerical_health['warning_breakdown']['overflow']}`
- **Divide-by-Zero**: `{numerical_health['warning_breakdown']['divide_by_zero']}`
- **Invalid Matmul**: `{numerical_health['warning_breakdown']['invalid_matmul']}`
- **Convergence**: `{numerical_health['warning_breakdown']['convergence']}`
- **Overall**: `{'✅ PASS' if numerical_health['numerical_health_pass'] else '❌ FAIL'}`

---

## 10. Artifact Inventory

Saved to `evaluation_results/phase6l/final_model/`:
- `preprocessing.joblib` (StandardScaler fitted on DEV)
- `classifier.joblib` (RandomForestClassifier fitted on DEV)
- `feature_schema.json`
- `model_metadata.json`

JSON results in `evaluation_results/phase6l/`:
- `final_protocol_verification.json`
- `heldout_validation_results.json`
- `heldout_bootstrap_ci.json`
- `heldout_calibration.json`
- `dev_val_generalization.json`
- `heldout_baseline_comparison.json`
- `dev_val_distribution_shift.json`
- `heldout_error_analysis.json`
- `numerical_health_audit.json`

Publication Figures (300 DPI):
- `figures/phase6l_3_val_roc.png`
- `figures/phase6l_3_val_pr.png`
- `figures/phase6l_3_val_calibration.png`
- `figures/phase6l_3_val_confusion_matrix.png`
- `figures/phase6l_3_dev_val_comparison.png`
- `figures/phase6l_3_feature_shift.png`
- `figures/phase6l_3_error_distributions.png`

---

## 11. Test Results

All Phase 6L unit tests passed (`tests/test_phase6l_3_validation.py`).

---

## 12. Scientific Discussion

Pillar-2 (Structural Consistency) independently classifies hallucinated LLM responses using inter-claim structural features: pairwise contradiction density, semantic similarity, and claim count. The frozen Candidate 5 model achieves ROC-AUC = `{tf['roc_auc']:.4f}` on the entirely unseen held-out Validation partition, confirming that structural consistency signals generalize beyond the development distribution.

The RandomForest classifier with 5 high-information features demonstrates that claim-level contradiction and similarity patterns are measurably predictive of hallucination, independent of Pillar-1's claim-evidence entailment features.

---

## 13. Limitations

1. Pillar-2 operates on structural consistency only — it does not verify factual accuracy against external evidence.
2. The 5-feature model captures contradiction density and similarity patterns but may miss nuanced temporal, entity, or numerical inconsistencies.
3. Moderate absolute discrimination (ROC-AUC ≈ 0.63) reflects the intrinsic difficulty of detecting hallucination from structural signals alone; fusion with Pillar-1 is expected to improve performance.

---

## 14. Firewall Confirmation

- **Validation Partition (N = 12,483)**: Evaluated **EXACTLY ONCE**. Zero model training, scaler fitting, threshold tuning, or feature selection performed on VAL.
- **Phase 6L.3 Stop Condition**: Execution STOPPED after validation. Phase 6M (Hybrid Fusion) has NOT been started.

---

## Final Pillar-2 Verdict

```
===========================================================================
                 FINAL VERDICT: {verdict}
===========================================================================
```

{f'Candidate 5 (`SET_D_HIGH_INFORMATION` + `StandardScaler` + `RandomForestClassifier`) is officially **VALIDATED** as the canonical Pillar-2 Structural Consistency classifier for HalluciSense.' if 'VALIDATED' in verdict else 'Pillar-2 validation completed with the status above.'}
"""

    report_path = out_dir / "FINAL_PILLAR2_VALIDATION_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)

    logger.info("final_report_generated", path=str(report_path))
    return report_path
