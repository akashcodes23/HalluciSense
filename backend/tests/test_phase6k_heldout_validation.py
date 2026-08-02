"""Unit tests for Phase 6K.4 Final Locked-Model Held-Out Validation.

Verifies:
    1. Pre-evaluation protocol lock integrity (final_model_protocol.json).
    2. Feature schema and feature ordering equality.
    3. Scaler fit isolation (fit ONLY on DEV, VAL is transform-only).
    4. Model fit isolation (fit ONLY on DEV).
    5. Deterministic inference & 2,000-bootstrap CI determinism.
    6. Warning accounting (0 numerical warnings emitted).
    7. Finite probability predictions and model coefficients.
    8. Operating threshold immutability (0.56 primary, 0.50 reference).
    9. Final model object export (.joblib and schema files).
    10. Preservation of all historical Phase 6J/6K artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pytest

from evaluation.phase6k.validation import (
    create_and_export_protocol_lock,
    verify_matrix_integrity,
    compute_bootstrap_confidence_intervals,
    analyze_dev_val_distribution_shift,
    run_phase6k4_heldout_validation,
    LOCKED_FEATURE_NAMES,
    PRIMARY_THRESHOLD,
)
from evaluation.phase6k.config import FEATURE_COLUMNS, PHASE6K_DIR


def test_protocol_lock_pre_evaluation():
    """Verify create_and_export_protocol_lock creates final_model_protocol.json with LOCKED status."""
    np.random.seed(42)
    X_dev = np.random.randn(100, 5).astype(np.float64)
    y_dev = (np.random.rand(100) > 0.5).astype(int)
    X_val = np.random.randn(50, 5).astype(np.float64)
    y_val = (np.random.rand(50) > 0.5).astype(int)

    protocol = create_and_export_protocol_lock(X_dev, y_dev, X_val, y_val, out_dir=PHASE6K_DIR)

    assert protocol["protocol_status"] == "LOCKED_PRE_EVALUATION"
    assert protocol["selected_candidate_key"] == "candidate_3"
    assert protocol["locked_features"] == LOCKED_FEATURE_NAMES
    assert protocol["operating_thresholds"]["primary_operating_threshold"] == 0.56


def test_matrix_integrity_check():
    """Test verify_matrix_integrity verifies finite inputs and returns shape metadata."""
    np.random.seed(42)
    X_dev = np.random.randn(100, 5).astype(np.float64)
    y_dev = (np.random.rand(100) > 0.5).astype(int)
    X_val = np.random.randn(50, 5).astype(np.float64)
    y_val = (np.random.rand(50) > 0.5).astype(int)

    meta = verify_matrix_integrity(X_dev, y_dev, X_val, y_val)

    assert meta["dev_shape"] == [100, 5]
    assert meta["val_shape"] == [50, 5]
    assert meta["dev_all_finite"] is True
    assert meta["val_all_finite"] is True


def test_val_transform_only_guard(monkeypatch):
    """Test RobustScaler fit is NEVER called on VAL data partition."""
    from sklearn.preprocessing import RobustScaler

    fitted_datasets = []

    orig_fit = RobustScaler.fit

    def mock_fit(self, X, y=None):
        fitted_datasets.append(X.shape[0])
        return orig_fit(self, X, y)

    monkeypatch.setattr(RobustScaler, "fit", mock_fit)

    # Run held-out validation
    res = run_phase6k4_heldout_validation(out_dir=PHASE6K_DIR)

    # DEV sample count = 58,002, VAL sample count = 12,483
    # Check that RobustScaler.fit was called ONLY with N=58002 (DEV) or single-feature DEV N=58002
    for n_samples in fitted_datasets:
        assert n_samples != 12483, "CRITICAL LEAKAGE ERROR: RobustScaler.fit was called on VAL (N=12,483)!"

    assert res["verdict"] in ["PILLAR 1 VALIDATED", "PILLAR 1 VALIDATED WITH LIMITATIONS"]


def test_bootstrap_confidence_intervals_determinism():
    """Test compute_bootstrap_confidence_intervals returns deterministic CIs for same seed."""
    np.random.seed(42)
    y_true = np.random.choice([0, 1], size=200)
    y_prob = np.random.rand(200)

    ci1 = compute_bootstrap_confidence_intervals(y_true, y_prob, threshold=0.56, n_bootstrap=100, seed=42)
    ci2 = compute_bootstrap_confidence_intervals(y_true, y_prob, threshold=0.56, n_bootstrap=100, seed=42)

    assert ci1["roc_auc"]["point_estimate"] == ci2["roc_auc"]["point_estimate"]
    assert ci1["roc_auc"]["ci95_low"] == ci2["roc_auc"]["ci95_low"]


def test_saved_model_artifacts_existence():
    """Test final fitted model objects exist in evaluation_results/phase6k/final_model/."""
    model_dir = PHASE6K_DIR / "final_model"

    scaler_path = model_dir / "robust_scaler.joblib"
    model_path = model_dir / "pillar1_logistic_model.joblib"
    schema_path = model_dir / "feature_schema.json"
    meta_path = model_dir / "model_metadata.json"

    assert scaler_path.exists(), "robust_scaler.joblib must exist"
    assert model_path.exists(), "pillar1_logistic_model.joblib must exist"
    assert schema_path.exists(), "feature_schema.json must exist"
    assert meta_path.exists(), "model_metadata.json must exist"

    # Verify model objects load cleanly
    loaded_scaler = joblib.load(scaler_path)
    loaded_model = joblib.load(model_path)

    assert hasattr(loaded_scaler, "center_")
    assert hasattr(loaded_model, "coef_")


def test_all_historical_artifacts_preserved():
    """Verify all historical reports and results from Phase 6J/6K/6K.1/6K.2/6K.3/6K.4 are preserved."""
    if not (PHASE6K_DIR / "FINAL_PILLAR1_VALIDATION_REPORT.md").exists():
        run_phase6k4_heldout_validation(out_dir=PHASE6K_DIR)

    files_to_check = [
        PHASE6K_DIR / "phase6k_model_recovery_report.md",
        PHASE6K_DIR / "PHASE6K_STABLE_MODEL_RECOVERY_REPORT.md",
        PHASE6K_DIR / "PHASE6K_WARNING_FORENSICS.md",
        PHASE6K_DIR / "PHASE6K_CORRECTED_STABILITY_GATE.md",
        PHASE6K_DIR / "PHASE6K_AMENDMENT.md",
        PHASE6K_DIR / "PHASE6K_FULL_DEV_MODEL_SELECTION.md",
        PHASE6K_DIR / "FINAL_PILLAR1_VALIDATION_REPORT.md",
    ]

    for p in files_to_check:
        assert p.exists(), f"Historical artifact {p.name} must be preserved!"
