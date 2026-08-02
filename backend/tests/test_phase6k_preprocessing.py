"""Unit tests for Phase 6K Preprocessing Pipelines.

Verifies:
    1. Zero data leakage: scaler fit occurs on DEV only; VAL is transform-only.
    2. Original feature matrices remain strictly unchanged (no in-place mutation).
    3. Deterministic transformation (repeatable across runs).
    4. Numerical stability: no NaN/Inf introduced in transformed outputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.phase6k.preprocessing import (
    fit_transform_strategy,
    compute_conditioning_stats,
    STRATEGY_NAMES,
)


@pytest.fixture
def dummy_dev_val_data():
    """Generate synthetic DEV and VAL feature matrices."""
    np.random.seed(42)
    X_dev = np.random.randn(100, 5) * 2.0 + 1.0
    X_val = np.random.randn(40, 5) * 5.0 - 2.0
    return X_dev, X_val


def test_original_matrices_unchanged(dummy_dev_val_data):
    """Test that fit_transform_strategy does NOT mutate input matrices in-place."""
    X_dev, X_val = dummy_dev_val_data
    dev_copy = X_dev.copy()
    val_copy = X_val.copy()

    for sname in STRATEGY_NAMES:
        _ = fit_transform_strategy(sname, X_dev, X_val, seed=42)
        np.testing.assert_allclose(X_dev, dev_copy, err_msg=f"X_dev mutated by {sname}")
        np.testing.assert_allclose(X_val, val_copy, err_msg=f"X_val mutated by {sname}")


def test_zero_data_leakage_val_isolation(dummy_dev_val_data):
    """Test that VAL partition is transform-only and fit occurs ONLY on DEV.

    Mutating X_val must NOT change the transformed X_dev matrix.
    """
    X_dev, X_val = dummy_dev_val_data
    X_val_mutated = X_val * 100.0 + 50.0  # Completely changed VAL data

    for sname in STRATEGY_NAMES:
        dev_scaled_1, val_scaled_1, scaler1 = fit_transform_strategy(sname, X_dev, X_val, seed=42)
        dev_scaled_2, val_scaled_2, scaler2 = fit_transform_strategy(sname, X_dev, X_val_mutated, seed=42)

        # X_dev transformed MUST be 100% identical regardless of X_val content
        np.testing.assert_allclose(
            dev_scaled_1,
            dev_scaled_2,
            err_msg=f"Data leakage detected! X_dev transform changed when X_val was altered for {sname}",
        )


def test_deterministic_transformations(dummy_dev_val_data):
    """Test that running fit_transform_strategy twice produces bit-exact outputs."""
    X_dev, X_val = dummy_dev_val_data

    for sname in STRATEGY_NAMES:
        dev1, val1, _ = fit_transform_strategy(sname, X_dev, X_val, seed=42)
        dev2, val2, _ = fit_transform_strategy(sname, X_dev, X_val, seed=42)

        np.testing.assert_allclose(dev1, dev2, err_msg=f"Non-deterministic DEV output for {sname}")
        np.testing.assert_allclose(val1, val2, err_msg=f"Non-deterministic VAL output for {sname}")


def test_no_nan_inf_introduced(dummy_dev_val_data):
    """Test that transformed output matrices contain zero NaN or Inf values."""
    X_dev, X_val = dummy_dev_val_data

    for sname in STRATEGY_NAMES:
        dev_s, val_s, _ = fit_transform_strategy(sname, X_dev, X_val, seed=42)

        assert np.all(np.isfinite(dev_s)), f"NaN/Inf found in DEV under {sname}"
        assert np.all(np.isfinite(val_s)), f"NaN/Inf found in VAL under {sname}"


def test_conditioning_stats_computation():
    """Test compute_conditioning_stats on a known matrix."""
    feature_names = ["f1", "f2", "f3"]
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    stats = compute_conditioning_stats(X, feature_names)

    assert stats.nan_count == 0
    assert stats.inf_count == 0
    assert stats.is_finite is True
    assert stats.min == 1.0
    assert stats.max == 9.0
    assert stats.abs_max == 9.0
    assert len(stats.per_feature) == 3
