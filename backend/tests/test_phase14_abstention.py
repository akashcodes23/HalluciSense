"""Phase 14 Selective Prediction & Abstention Tests."""

from __future__ import annotations

import pytest
from app.core.engine.calibration import SelectiveAbstentionGate
from app.core.engine.types import RiskLevel


class TestPhase14Abstention:
    def test_rejection_under_insufficient_evidence(self):
        """Verifies that retrieval deficit triggers INSUFFICIENT_EVIDENCE."""
        gate = SelectiveAbstentionGate(min_evidence_similarity=0.40)
        res = gate.evaluate(
            h_score=0.48,
            evidence_available=False,
            max_evidence_similarity=0.15,
            confidence_available=False,
            epistemic_uncertainty=0.90,
        )
        assert res.abstained is True
        assert res.decision == RiskLevel.INSUFFICIENT_EVIDENCE

    def test_abstention_on_ambiguous_decision_boundary(self):
        """Verifies that boundary claims with high uncertainty trigger ABSTAIN."""
        gate = SelectiveAbstentionGate(ambiguity_margin=0.08)
        res = gate.evaluate(
            h_score=0.41,
            evidence_available=True,
            max_evidence_similarity=0.85,
            confidence_available=True,
            epistemic_uncertainty=0.88,
        )
        assert res.abstained is True
        assert res.decision == RiskLevel.ABSTAIN

    def test_confident_prediction_does_not_abstain(self):
        """Verifies that low risk claims with strong evidence pass without abstention."""
        gate = SelectiveAbstentionGate()
        res = gate.evaluate(
            h_score=0.08,
            evidence_available=True,
            max_evidence_similarity=0.95,
            confidence_available=True,
            epistemic_uncertainty=0.10,
        )
        assert res.abstained is False
        assert res.decision == RiskLevel.VERIFIED
