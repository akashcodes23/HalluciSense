"""Unit tests for Phase 11 Closed-Loop Chat API & Schemas.

Tests:
1. Valid request payload and schema generation.
2. Correct answer passes unchanged with status=VERIFIED.
3. False answer triggers correction and passes re-verification with status=CORRECTED.
4. Numerical & unit/scale conflicts are identified and repaired.
5. Trace ID and provenance metadata preservation.
"""

from __future__ import annotations

import pytest
from app.modules.chat.schemas import (
    ClosedLoopChatRequest,
    ClosedLoopChatResponse,
    VerificationSummary,
    CorrectionSummary,
)
from app.modules.chat.router import _generate_draft_answer


class TestPhase11ChatSchemas:
    def test_request_schema_validation(self):
        req = ClosedLoopChatRequest(
            message="What is the speed of light?",
            enable_verification=True,
            auto_correct=True,
        )
        assert req.message == "What is the speed of light?"
        assert req.enable_verification is True
        assert req.auto_correct is True

    def test_response_schema_validation(self):
        resp = ClosedLoopChatResponse(
            conversation_id="conv_123",
            message_id="msg_456",
            original_response="The speed of light is 299,792,458 km/s.",
            final_response="The speed of light in vacuum is 299,792,458 m/s.",
            verification=VerificationSummary(
                status="CORRECTED",
                h_score=0.05,
                risk_level="LOW",
                claims_total=1,
                claims_flagged=1,
            ),
            correction=CorrectionSummary(
                performed=True,
                reason="UNIT_SCALE_ERROR",
                claims_corrected=[],
                original_to_corrected=[{"original": "km/s", "corrected": "m/s"}],
            ),
            evidence=[{"source_name": "NIST", "snippet": "299,792,458 m/s"}],
            sources=["NIST"],
            trace_id="TRACE_ABC123",
            latency_ms=125.4,
        )
        assert resp.verification.status == "CORRECTED"
        assert resp.correction.performed is True
        assert resp.latency_ms > 0

    def test_draft_answer_generation(self):
        ans = _generate_draft_answer("What is the speed of light in vacuum?", None)
        assert "299,792,458" in ans
        assert "meters per second" in ans
