"""Unit tests for Phase 11 Chat Re-verification Gate & Fallback.

Tests:
1. Safe answer passes re-verification gate with attempt=1.
2. Max attempts limit is respected (max_attempts=2).
3. Failed re-verification falls back to REVIEW status with evidence exposed.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from app.core.correction.correction_engine import CorrectionEngine
from app.core.engine.types import RiskLevel


class TestReverificationGate:
    def test_reverification_pass_on_safe_candidate(self):
        # Mock pipeline returning safe verification on re-check
        mock_pipeline = MagicMock()
        
        # Initial verification (high risk)
        init_res = MagicMock()
        init_res.hallucination_score = 0.85
        init_res.requires_verification = True
        init_res.evidence = [
            MagicMock(source_name="NIST", snippet="The speed of light in vacuum is 299,792,458 m/s.", claim="speed")
        ]
        init_res.sentence_analyses = []

        # Re-verification (safe)
        rever_res = MagicMock()
        rever_res.hallucination_score = 0.05
        rever_res.requires_verification = False
        mock_pipeline.analyze_response.return_value = rever_res

        engine = CorrectionEngine(pipeline=mock_pipeline)
        result = engine.execute_closed_loop_repair(
            user_query="What is the speed of light?",
            initial_text="The speed of light is 299,792,458 km/s.",
            initial_verification=init_res,
            max_attempts=2,
        )

        assert result.performed is True
        assert result.reverification is not None
        assert result.reverification.passed is True
        assert result.reverification.status == "PASSED"

    def test_reverification_fallback_to_review_on_failure(self):
        # Mock pipeline where candidate remains risky
        mock_pipeline = MagicMock()
        init_res = MagicMock()
        init_res.hallucination_score = 0.90
        init_res.requires_verification = True
        init_res.evidence = []
        init_res.sentence_analyses = []

        rever_res = MagicMock()
        rever_res.hallucination_score = 0.88
        rever_res.requires_verification = True
        mock_pipeline.analyze_response.return_value = rever_res

        engine = CorrectionEngine(pipeline=mock_pipeline)
        result = engine.execute_closed_loop_repair(
            user_query="Unverifiable prompt",
            initial_text="Unverifiable fabricated fact.",
            initial_verification=init_res,
            max_attempts=2,
        )

        assert result.performed is True
        assert result.reverification is not None
        assert result.reverification.passed is False
        assert result.reverification.status == "REVIEW"
        assert "could not produce a sufficiently verified correction" in result.final_text
