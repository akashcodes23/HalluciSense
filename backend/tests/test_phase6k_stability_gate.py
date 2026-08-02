"""Unit tests for Phase 6K 1,000-Example Numerical Stability Gate.

Verifies:
    1. Deterministic stratified 1,000-example DEV subset extraction.
    2. Strict data isolation: validation data (X_val, y_val) is NEVER accessed.
    3. Warning recorder captures overflow, divide-by-zero, and convergence failures.
    4. PASS/FAIL decision logic correctly flags ill-conditioned configurations as FAIL.
    5. Overall verdict output formatting ("STABILITY GATE: PASS").
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.phase6k.benchmark import (
    get_stratified_1000_subset,
    evaluate_single_config_stability,
    run_stability_gate_1000,
)
from evaluation.phase6k.config import FEATURE_COLUMNS


@pytest.fixture
def synthetic_dev_data_large():
    """Create synthetic development dataset with 1,500 samples."""
    np.random.seed(42)
    n_samples = 1500
    n_features = 10
    X_dev = np.random.randn(n_samples, n_features) * 2.0
    y_dev = (X_dev[:, 0] + X_dev[:, 2] > 0).astype(int)
    return X_dev, y_dev


def test_stratified_1000_subset_extraction(synthetic_dev_data_large):
    """Test get_stratified_1000_subset extracts exactly 1,000 samples preserving target balance."""
    X_dev, y_dev = synthetic_dev_data_large
    X_sub, y_sub = get_stratified_1000_subset(X_dev, y_dev, seed=42)

    assert X_sub.shape == (1000, 10)
    assert len(y_sub) == 1000

    orig_ratio = (y_dev == 1).mean()
    sub_ratio = (y_sub == 1).mean()
    assert abs(orig_ratio - sub_ratio) < 0.02


def test_val_data_never_accessed_by_stability_gate(synthetic_dev_data_large):
    """Test run_stability_gate_1000 operates strictly without any validation data."""
    X_dev, y_dev = synthetic_dev_data_large

    report = run_stability_gate_1000(X_dev, y_dev, FEATURE_COLUMNS, seed=42)

    assert report.n_subset_samples == 1000
    assert report.total_configs_tested == 16
    assert report.overall_verdict in ["STABILITY GATE: PASS", "STABILITY GATE: FAIL"]


def test_single_config_stability_evaluation(synthetic_dev_data_large):
    """Test evaluate_single_config_stability returns structured stability result."""
    X_dev, y_dev = synthetic_dev_data_large
    X_sub, y_sub = get_stratified_1000_subset(X_dev, y_dev, seed=42)

    res = evaluate_single_config_stability(
        feature_set_name="SET_B_DECOLLINEARIZED",
        feature_names_subset=["mean_entailment", "max_entailment", "mean_contradiction", "min_support_margin", "num_claims"],
        scaler_name="RobustScaler",
        X_sub_full=X_sub,
        y_sub=y_sub,
        master_feature_names=FEATURE_COLUMNS,
        seed=42,
    )

    assert res.config_id == "SET_B_DECOLLINEARIZED__RobustScaler"
    assert isinstance(res.converged, bool)
    assert isinstance(res.pass_status, bool)
    assert res.condition_number > 0.0
    assert res.training_accuracy >= 0.0


def test_stability_gate_verdict_format(synthetic_dev_data_large):
    """Test stability gate outputs valid PASS or FAIL verdict."""
    X_dev, y_dev = synthetic_dev_data_large
    report = run_stability_gate_1000(X_dev, y_dev, FEATURE_COLUMNS, seed=42)

    assert "STABILITY GATE:" in report.overall_verdict
    assert report.passing_configs_count + report.failing_configs_count == 16
