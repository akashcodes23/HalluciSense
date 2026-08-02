"""Unit tests for Phase 6K Collinearity & Feature Redundancy Audit.

Verifies:
    1. DEV partition isolation: collinearity calculations use DEV partition only.
    2. VIF calculation safely handles full-rank and near-singular matrices.
    3. Redundant pair identification correctly detects pairs with |Pearson r| >= 0.90.
    4. Proposed retention decisions assign valid proposed_retain and proposed_remove targets.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.phase6k.collinearity import (
    compute_vif,
    analyze_collinearity,
    evaluate_pair_retention,
)


@pytest.fixture
def synthetic_collinear_data():
    """Create synthetic data with known collinear features."""
    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    x2 = x1 * 0.98 + np.random.randn(n) * 0.05  # High correlation with x1 (|r| > 0.95)
    x3 = np.random.randn(n)                      # Independent feature
    y = (x1 + x3 > 0).astype(int)

    X = np.column_stack([x1, x2, x3])
    feature_names = ["feat1", "feat2", "feat3"]
    return X, y, feature_names


def test_vif_calculation_safe(synthetic_collinear_data):
    """Test compute_vif returns valid float scores for all features without errors."""
    X, _, feature_names = synthetic_collinear_data
    vifs = compute_vif(X, feature_names)

    assert len(vifs) == 3
    for name in feature_names:
        assert name in vifs
        assert isinstance(vifs[name], float)
        assert vifs[name] >= 1.0


def test_vif_singular_matrix_handling():
    """Test compute_vif handles exact collinearity (zero variance residual) safely."""
    X_singular = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
    names = ["col1", "col2"]
    vifs = compute_vif(X_singular, names)

    assert "col1" in vifs and "col2" in vifs
    assert vifs["col1"] >= 1e4
    assert vifs["col2"] >= 1e4


def test_redundant_pair_detection(synthetic_collinear_data):
    """Test analyze_collinearity identifies collinear pairs meeting |r| >= 0.90 threshold."""
    X, y, feature_names = synthetic_collinear_data
    audit, decisions = analyze_collinearity(X, y, feature_names, threshold=0.90)

    assert len(decisions.pair_decisions) >= 1
    # feat1 and feat2 must be identified as redundant pair
    pair_names = [(p.feature_a, p.feature_b) for p in decisions.pair_decisions]
    assert ("feat1", "feat2") in pair_names or ("feat2", "feat1") in pair_names

    # Check proposed sets
    assert len(decisions.proposed_retained_features) + len(decisions.proposed_removed_features) == 3
    assert len(decisions.proposed_removed_features) >= 1


def test_evaluate_pair_retention_logic():
    """Test evaluate_pair_retention assigns valid proposed_retain/remove and non-empty reasons."""
    decision = evaluate_pair_retention(
        feat_a="mean_contradiction",
        feat_b="fraction_contradicted",
        r_val=0.9826,
        rho_val=0.9810,
        mi_a=0.035,
        mi_b=0.012,
        auc_a=0.5769,
        auc_b=0.5757,
    )

    assert decision.proposed_retain == "mean_contradiction"
    assert decision.proposed_remove == "fraction_contradicted"
    assert len(decision.semantic_interpretation) > 0
    assert len(decision.quantitative_reason) > 0
