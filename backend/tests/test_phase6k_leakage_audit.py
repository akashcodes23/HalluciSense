"""Unit tests for Phase 6K Leakage & Shortcut Audit.

Verifies:
    1. Target leakage detection correctly flags artificial target-correlated features.
    2. Label permutation sanity test verifies performance collapse to chance (~0.50).
    3. DEV-VAL exact feature vector duplicate overlap calculation.
    4. Single-feature ablation analysis measures delta ROC-AUC.
    5. Audit verdict assignment (PASS / PASS WITH WARNINGS / FAIL).
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.phase6k.feasibility import (
    run_leakage_shortcut_audit,
    LeakageShortcutReport,
)
from evaluation.phase6k.config import FEATURE_COLUMNS


@pytest.fixture
def synthetic_audit_data():
    """Create synthetic development and validation feature datasets."""
    np.random.seed(42)
    n_dev = 200
    n_val = 50
    n_feat = 10

    X_dev = np.random.randn(n_dev, n_feat)
    y_dev = (X_dev[:, 0] + X_dev[:, 2] > 0).astype(int)

    X_val = np.random.randn(n_val, n_feat)
    y_val = (X_val[:, 0] + X_val[:, 2] > 0).astype(int)

    return X_dev, y_dev, X_val, y_val


def test_label_permutation_collapses_to_chance(synthetic_audit_data):
    """Test label permutation test shows permuted ROC-AUC collapses near 0.50."""
    X_dev, y_dev, X_val, y_val = synthetic_audit_data
    report = run_leakage_shortcut_audit(
        X_dev=X_dev,
        y_dev=y_dev,
        X_val=X_val,
        y_val=y_val,
        feature_names=FEATURE_COLUMNS,
        seed=42,
    )

    assert report.permutation_test.collapsed_to_chance is True
    assert abs(report.permutation_test.mean_permuted_roc_auc - 0.50) < 0.10


def test_target_leakage_detection(synthetic_audit_data):
    """Test run_leakage_shortcut_audit flags artificially injected target leakage feature."""
    X_dev, y_dev, X_val, y_val = synthetic_audit_data

    # Inject exact target into feature index 0
    X_dev_leaked = X_dev.copy()
    X_dev_leaked[:, 0] = y_dev.astype(float) * 100.0

    report = run_leakage_shortcut_audit(
        X_dev=X_dev_leaked,
        y_dev=y_dev,
        X_val=X_val,
        y_val=y_val,
        feature_names=FEATURE_COLUMNS,
        seed=42,
    )

    assert report.target_leakage_detected is True
    assert report.leakage_feature_name == FEATURE_COLUMNS[0]
    assert report.overall_verdict == "FAIL"


def test_dev_val_overlap_calculation(synthetic_audit_data):
    """Test run_leakage_shortcut_audit detects exact feature vector overlap between DEV and VAL."""
    X_dev, y_dev, X_val, y_val = synthetic_audit_data

    # Make first 5 rows of X_val identical to first 5 rows of X_dev
    X_val_overlapped = X_val.copy()
    X_val_overlapped[:5] = X_dev[:5]

    report = run_leakage_shortcut_audit(
        X_dev=X_dev,
        y_dev=y_dev,
        X_val=X_val_overlapped,
        y_val=y_val,
        feature_names=FEATURE_COLUMNS,
        seed=42,
    )

    assert report.dev_val_overlap_count == 5
    assert report.dev_val_overlap_ratio == pytest.approx(5 / len(X_val))


def test_feature_ablation_analysis(synthetic_audit_data):
    """Test feature ablation outputs delta ROC-AUC for all 10 features."""
    X_dev, y_dev, X_val, y_val = synthetic_audit_data
    report = run_leakage_shortcut_audit(
        X_dev=X_dev,
        y_dev=y_dev,
        X_val=X_val,
        y_val=y_val,
        feature_names=FEATURE_COLUMNS,
        seed=42,
    )

    assert len(report.ablation_results) == 10
    for abl in report.ablation_results:
        assert abl.ablated_feature in FEATURE_COLUMNS
        assert np.isfinite(abl.delta_roc_auc)
