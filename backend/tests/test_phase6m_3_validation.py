"""Exhaustive Unit Test Suite for Phase 6M.3 Final Held-Out Validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import RobustScaler

from evaluation.phase6m.config import HYBRID_FEATURE_SCHEMA, PHASE6M_DIR
from evaluation.phase6m.heldout_validation import (
    verify_protocol,
    train_locked_hybrid_model,
    run_heldout_inference,
    compute_bootstrap_ci,
    compute_generalization_gap,
    compute_distribution_shift_mitigation,
    compute_baseline_comparison,
    freeze_final_model_artifacts,
)


def test_verify_protocol():
    """Verify loading and checksumming locked final_hybrid_protocol.json."""
    res = verify_protocol(PHASE6M_DIR)
    assert res["protocol_sha256"] is not None
    assert res["protocol_contents"]["protocol_locked"] is True
    assert res["protocol_contents"]["decision_threshold"] == 0.54


def test_train_locked_hybrid_model_small():
    """Verify training locked model on small synthetic matrix."""
    np.random.seed(42)
    X = np.random.randn(50, 19)
    y = np.random.randint(0, 2, size=50)
    protocol = {"decision_threshold": 0.54}

    scaler, clf = train_locked_hybrid_model(X, y, protocol)
    assert scaler is not None
    assert clf is not None


def test_run_heldout_inference_small():
    """Verify held-out inference function on small synthetic matrix."""
    np.random.seed(42)
    X_dev = np.random.randn(50, 19)
    y_dev = np.random.randint(0, 2, size=50)
    X_val = np.random.randn(20, 19)
    y_val = np.random.randint(0, 2, size=20)
    protocol = {"decision_threshold": 0.54}

    scaler, clf = train_locked_hybrid_model(X_dev, y_dev, protocol)
    metrics = run_heldout_inference(X_val, y_val, scaler, clf, protocol)

    assert "threshold_free" in metrics
    assert "threshold_dependent" in metrics
    assert 0.0 <= metrics["threshold_free"]["roc_auc"] <= 1.0


def test_compute_bootstrap_ci():
    """Verify 2,000 stratified bootstrap CI computation."""
    np.random.seed(42)
    y = np.random.randint(0, 2, size=100)
    p = np.random.uniform(0, 1, size=100)

    ci = compute_bootstrap_ci(y, p, threshold=0.54, n_bootstrap=100, seed=42)

    assert "roc_auc" in ci
    assert "pr_auc" in ci
    assert "mcc" in ci
    assert ci["roc_auc"]["ci95_low"] <= ci["roc_auc"]["ci95_high"]


def test_compute_generalization_gap():
    """Verify DEV vs VAL generalization gap classification."""
    dev_summary = {"roc_auc": 0.7267, "mcc": 0.3370, "ece": 0.0066}
    val_metrics = {
        "threshold_free": {"roc_auc": 0.7212, "pr_auc": 0.7500, "brier_score": 0.2100, "log_loss": 0.6000},
        "threshold_dependent": {"mcc": 0.3300, "ece": 0.0080},
    }

    gap = compute_generalization_gap(dev_summary, val_metrics)

    assert gap["generalization_classification"] == "STABLE"
    assert gap["delta_roc_auc"] == pytest.approx(-0.0055, abs=1e-3)


def test_compute_distribution_shift_mitigation():
    """Verify distribution shift SMD and KS calculations."""
    np.random.seed(42)
    X_dev = np.random.randn(100, 19)
    X_val = np.random.randn(50, 19)

    res = compute_distribution_shift_mitigation(X_dev, X_val, HYBRID_FEATURE_SCHEMA)

    assert "feature_shifts" in res
    assert len(res["feature_shifts"]) == 19
    assert "p1_prob_smd" in res


def test_compute_baseline_comparison():
    """Verify baseline comparisons and DeLong test."""
    np.random.seed(42)
    y = np.random.randint(0, 2, size=100)
    p = np.random.uniform(0, 1, size=100)
    X = np.random.uniform(0, 1, size=(100, 19))

    res = compute_baseline_comparison(y, p, X, HYBRID_FEATURE_SCHEMA)

    assert "hybrid_val_auc" in res
    assert "delong_test_vs_pillar1" in res
    assert "statistically_superior_to_pillar1" in res


def test_freeze_final_model_artifacts(tmp_path: Path):
    """Verify freezing model artifacts in final_hybrid_model/."""
    scaler = RobustScaler()
    clf = HistGradientBoostingClassifier(max_iter=10)
    protocol = {"decision_threshold": 0.54}

    model_dir = freeze_final_model_artifacts(scaler, clf, protocol, out_dir=tmp_path)

    assert (model_dir / "preprocessing.joblib").exists()
    assert (model_dir / "hybrid_meta_classifier.joblib").exists()
    assert (model_dir / "feature_schema.json").exists()
    assert (model_dir / "model_metadata.json").exists()
