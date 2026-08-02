"""Unit tests for Phase 6L.1B Feature Validation & Mathematical Invariant Audits."""

from __future__ import annotations

import numpy as np
import pytest
from evaluation.phase6l.config import PHASE6L_DIR, STRUCTURAL_FEATURE_COLUMNS
from evaluation.phase6l.feature_validation import (
    audit_feature_distributions,
    audit_feature_correlations,
    verify_structural_invariants,
)


def test_audit_feature_distributions_small_matrix():
    """Test audit_feature_distributions on finite dummy 24-feature matrix."""
    np.random.seed(42)
    X_dummy = np.random.randn(20, 24).astype(np.float64)

    res = audit_feature_distributions(X_dummy, STRUCTURAL_FEATURE_COLUMNS, out_dir=PHASE6L_DIR)

    assert res["n_samples"] == 20
    assert res["n_features"] == 24
    assert res["all_finite"] is True


def test_verify_structural_invariants_clean():
    """Test verify_structural_invariants returns passed True when bounds are satisfied."""
    dummy_resp = [
        {
            "example_id": "r1",
            "num_claims": 2,
            "pair_count": 1,
            "features": {col: 0.5 for col in STRUCTURAL_FEATURE_COLUMNS},
        }
    ]

    res = verify_structural_invariants(dummy_resp)
    assert res["invariants_passed"] is True
    assert res["violation_count"] == 0
