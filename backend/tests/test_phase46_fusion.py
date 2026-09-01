"""Phase 46 — Adaptive Fusion Availability & Weighting Tests."""

import pytest
from app.core.engine.fusion import FusionEngine

@pytest.fixture
def fusion_engine():
    return FusionEngine()

def test_fusion_three_pillars_available(fusion_engine):
    """Full three-pillar fusion weights FE, CG, and CF proportionally."""
    score, weights, mask = fusion_engine.compute_adaptive_h_score(
        fe=0.10,
        cg=0.20,
        cf=0.15,
    )
    assert mask == [1, 1, 1]
    assert 0.0 <= score <= 1.0
    assert "alpha" in weights
    assert "beta" in weights
    assert "gamma" in weights

def test_fusion_missing_p2_not_treated_as_zero(fusion_engine):
    """Missing P2 is renormalized rather than assumed to be 0.0 risk."""
    score_p1_only, weights_p1, mask = fusion_engine.compute_adaptive_h_score(
        fe=0.80,
        cg=None,
        cf=None,
    )
    assert mask == [1, 0, 0]
    # When only P1 is available, effective alpha is 1.0
    assert weights_p1["alpha"] == 1.0
    assert score_p1_only == 0.80

def test_fusion_p1_and_p3(fusion_engine):
    """P1 + P3 renormalizes weights between alpha and gamma."""
    score, weights, mask = fusion_engine.compute_adaptive_h_score(
        fe=0.20,
        cg=None,
        cf=0.40,
    )
    assert mask == [1, 0, 1]
    assert weights["alpha"] > 0
    assert weights["gamma"] > 0
    assert weights["beta"] == 0.0
