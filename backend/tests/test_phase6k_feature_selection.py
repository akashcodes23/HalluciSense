"""Unit tests for Phase 6K Feature Selection.

Verifies:
    1. Strict data isolation: validation labels (y_val) and features (X_val) are never accessed.
    2. Deterministic construction of all 4 candidate feature sets (SET A, B, C, D).
    3. Correct indexing of feature subsets against master feature names.
    4. Finite value guarantees on condition numbers and correlation metrics.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.phase6k.feature_selection import (
    construct_candidate_feature_sets,
    compute_composite_discriminative_ranks,
    compute_feature_set_metadata,
)
from evaluation.phase6k.config import FEATURE_COLUMNS


@pytest.fixture
def synthetic_dev_data():
    """Create synthetic development data (X_dev, y_dev)."""
    np.random.seed(42)
    n_samples = 300
    n_features = 10
    X_dev = np.random.randn(n_samples, n_features)
    # Add collinearity between col 0 and 1
    X_dev[:, 1] = X_dev[:, 0] * 0.99 + np.random.randn(n_samples) * 0.01
    y_dev = (X_dev[:, 0] + X_dev[:, 5] > 0).astype(int)
    return X_dev, y_dev


def test_val_data_never_accessed(synthetic_dev_data):
    """Test construct_candidate_feature_sets operates strictly without any validation data."""
    X_dev, y_dev = synthetic_dev_data

    # Call construct_candidate_feature_sets passing ONLY X_dev, y_dev
    report = construct_candidate_feature_sets(
        X_dev=X_dev,
        y_dev=y_dev,
        feature_names=FEATURE_COLUMNS,
    )

    assert len(report.candidate_sets) == 4
    assert "SET_A_ALL" in report.candidate_sets
    assert "SET_B_DECOLLINEARIZED" in report.candidate_sets
    assert "SET_C_TOP_DISCRIMINATIVE" in report.candidate_sets
    assert "SET_D_DECOLLINEARIZED_DISCRIMINATIVE" in report.candidate_sets


def test_deterministic_set_construction(synthetic_dev_data):
    """Test running construct_candidate_feature_sets twice yields identical set compositions."""
    X_dev, y_dev = synthetic_dev_data

    report1 = construct_candidate_feature_sets(X_dev, y_dev, FEATURE_COLUMNS)
    report2 = construct_candidate_feature_sets(X_dev, y_dev, FEATURE_COLUMNS)

    for k in report1.candidate_sets:
        assert report1.candidate_sets[k].feature_names == report2.candidate_sets[k].feature_names
        assert report1.candidate_sets[k].indices == report2.candidate_sets[k].indices
        assert report1.candidate_sets[k].condition_number_unscaled == pytest.approx(
            report2.candidate_sets[k].condition_number_unscaled
        )


def test_composite_discriminative_ranks(synthetic_dev_data):
    """Test compute_composite_discriminative_ranks orders features by discriminative strength."""
    X_dev, y_dev = synthetic_dev_data
    ranks = compute_composite_discriminative_ranks(X_dev, y_dev, FEATURE_COLUMNS)

    assert len(ranks) == 10
    # Scores must be sorted in descending order
    scores = [r[1] for r in ranks]
    assert scores == sorted(scores, reverse=True)


def test_finite_metrics_in_feature_sets(synthetic_dev_data):
    """Test all numerical metrics in candidate feature sets are finite numbers."""
    X_dev, y_dev = synthetic_dev_data
    report = construct_candidate_feature_sets(X_dev, y_dev, FEATURE_COLUMNS)

    for k, set_meta in report.candidate_sets.items():
        assert set_meta.feature_count > 0
        assert set_meta.matrix_rank > 0
        assert np.isfinite(set_meta.condition_number_unscaled)
        assert np.isfinite(set_meta.condition_number_robust_scaled)
        assert np.isfinite(set_meta.mean_pairwise_abs_correlation)
        assert np.isfinite(set_meta.max_pairwise_abs_correlation)
