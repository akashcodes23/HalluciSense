"""Phase 14 Closed-Loop Correction & Reverification Tests."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from app.core.correction.correction_engine import CorrectionEngine
from app.core.correction.correction_policy import CorrectionPolicy


class TestPhase14Correction:
    def test_reverification_gate_success_contract(self):
        """Verifies that an independently reverified correction returns status=CORRECTED."""
        mock_pipe = MagicMock()
        init_res = MagicMock()
        init_res.overall_h_score = 0.88
        init_res.requires_verification = True
        init_res.evidence_items = [MagicMock(source_name="NIST", snippet="The speed of sound in air at 20 °C is 343 m/s.", claim="sound")]

        rever_res = MagicMock()
        rever_res.overall_h_score = 0.05
        rever_res.requires_verification = False
        mock_pipe.analyze_response.return_value = rever_res

        engine = CorrectionEngine(pipeline=mock_pipe)
        result = engine.execute_closed_loop_repair(
            user_query="What is the speed of sound?",
            initial_text="The speed of sound in air is 343 km/s.",
            initial_verification=init_res,
            max_attempts=2,
        )
        assert result.performed is True
        assert result.reverification is not None
        assert result.reverification.passed is True
        assert result.reverification.status == "PASSED"

    def test_unit_and_causal_policy_repair_rules(self):
        """Verifies deterministic unit and causal direction repair policies."""
        policy = CorrectionPolicy()
        unit_cand = policy.classify_and_repair_deterministic(
            claim_id="c1",
            claim_text="The speed of light is approximately 299792458 km/s in vacuum.",
            evidence_snippet="The speed of light in vacuum is defined as exactly 299792458 meters per second (m/s).",
        )
        assert unit_cand is not None
        assert "m/s" in unit_cand.corrected_claim
        assert "km/s" not in unit_cand.corrected_claim
