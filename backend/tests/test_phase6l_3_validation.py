"""Unit tests for Phase 6L.3 Held-Out Validation Pipeline (Pillar 2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.phase6l.config import PHASE6L_DIR, STRUCTURAL_FEATURE_COLUMNS
from evaluation.phase6l.heldout_validation import (
    LOCKED_FEATURE_NAMES,
    PRIMARY_THRESHOLD,
    compute_bootstrap_ci,
    compute_calibration,
    compute_generalization_gap,
    compute_distribution_shift,
    compute_error_analysis,
    compute_numerical_health,
)


def test_locked_feature_names():
    """Verify locked feature names match protocol."""
    expected = [
        "max_pairwise_contradiction",
        "mean_pairwise_contradiction",
        "max_pairwise_similarity",
        "fraction_contradictory_pairs",
        "num_claims",
    ]
    assert LOCKED_FEATURE_NAMES == expected
    assert PRIMARY_THRESHOLD == 0.57


def test_protocol_verification_file_exists():
    """Verify final_model_protocol.json exists and is locked."""
    protocol_path = PHASE6L_DIR / "final_model_protocol.json"
    assert protocol_path.exists()

    with open(protocol_path, "r") as f:
        protocol = json.load(f)

    assert protocol["protocol_locked"] is True
    assert protocol["feature_names"] == LOCKED_FEATURE_NAMES
    assert protocol["classifier"] == "RandomForestClassifier"
    assert protocol["scaler"] == "StandardScaler"
    assert protocol["decision_threshold"] == 0.57


def test_bootstrap_ci():
    """Verify bootstrap CI computation produces valid intervals."""
    np.random.seed(42)
    y = np.random.randint(0, 2, size=200)
    p = np.random.uniform(0, 1, size=200)

    ci = compute_bootstrap_ci(y, p, threshold=0.50, n_bootstrap=100, seed=42)

    assert "roc_auc" in ci
    assert "mcc" in ci
    assert "brier_score" in ci
    assert "precision" in ci
    assert "recall" in ci
    assert "specificity" in ci

    for metric_name, metric_ci in ci.items():
        assert metric_ci["ci95_low"] <= metric_ci["ci95_high"]
        assert 0.0 <= metric_ci["ci95_low"] or metric_name == "mcc"  # MCC can be negative


def test_calibration():
    """Verify ECE and MCE computation."""
    np.random.seed(42)
    y = np.random.randint(0, 2, size=500)
    p = np.random.uniform(0, 1, size=500)

    cal = compute_calibration(y, p, n_bins=10)

    assert "ece" in cal
    assert "mce" in cal
    assert 0.0 <= cal["ece"] <= 1.0
    assert 0.0 <= cal["mce"] <= 1.0
    assert cal["n_bins"] == 10
    assert "calibration_pass" in cal


def test_generalization_gap():
    """Verify generalization gap classification."""
    dev_summary = {"roc_auc_mean": 0.65, "pr_auc_mean": 0.60, "best_mcc": 0.20, "brier_score_mean": 0.23, "ece": 0.01}
    val_metrics = {
        "threshold_free_metrics": {"roc_auc": 0.64, "pr_auc": 0.59, "brier_score": 0.24, "log_loss": 0.66},
        "primary_threshold_metrics": {"mcc": 0.19},
    }
    cal = {"ece": 0.02}

    gap = compute_generalization_gap(dev_summary, val_metrics, cal)

    assert gap["generalization_classification"] == "STABLE"
    assert gap["gap_auc"] == pytest.approx(-0.01, abs=0.001)


def test_distribution_shift():
    """Verify feature distribution shift with SMD and KS."""
    np.random.seed(42)
    X_dev = np.random.randn(1000, 24)
    X_val = np.random.randn(200, 24)

    shift = compute_distribution_shift(X_dev, X_val)

    assert "features" in shift
    for fname in LOCKED_FEATURE_NAMES:
        assert fname in shift["features"]
        s = shift["features"][fname]
        assert "standardized_mean_difference" in s
        assert "ks_statistic" in s
        assert "ks_pvalue" in s


def test_error_analysis():
    """Verify error analysis group statistics."""
    np.random.seed(42)
    y_val = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    p_val = np.array([0.3, 0.8, 0.9, 0.2, 0.4, 0.7, 0.6, 0.5])
    X_val_selected = np.random.randn(8, 5)

    ea = compute_error_analysis(y_val, p_val, X_val_selected)

    assert "group_statistics" in ea
    for g in ["TP", "TN", "FP", "FN"]:
        assert g in ea["group_statistics"]
        assert "count" in ea["group_statistics"][g]

    total = sum(ea["group_statistics"][g]["count"] for g in ["TP", "TN", "FP", "FN"])
    assert total == len(y_val)


def test_numerical_health():
    """Verify numerical health audit."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier

    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, size=50)

    scaler = StandardScaler().fit(X)
    clf = RandomForestClassifier(n_estimators=10, random_state=42).fit(scaler.transform(X), y)
    p = clf.predict_proba(scaler.transform(X))[:, 1]

    health = compute_numerical_health(scaler, clf, p, {"training_warnings": []}, [])

    assert health["numerical_health_pass"] is True
    assert health["scaler_finite"] is True
    assert health["probabilities_finite"] is True
