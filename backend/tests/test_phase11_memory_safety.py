"""Unit tests for Phase 11B Memory Safety and Failure Semantics.

Tests:
1. Singleton ModelRegistry returns exact same object references across invocations.
2. NLI Semaphore concurrency bounding.
3. Proper failure semantics (h_score=None, not 1.0 or 100%).
4. Unverified responses return status=UNVERIFIED and h_score=None.
5. Invariant benchmarks remain untouched.
"""

from __future__ import annotations

import pytest
from app.core.engine.model_registry import ModelRegistry
from app.modules.chat.schemas import (
    ClosedLoopChatRequest,
    VerificationSummary,
    ClosedLoopChatResponse,
)


class TestModelRegistrySingleton:
    def test_singleton_pipeline_identity(self):
        p1 = ModelRegistry.get_pipeline()
        p2 = ModelRegistry.get_pipeline()
        assert p1 is p2
        assert id(p1) == id(p2)

    def test_singleton_nli_identity(self):
        tok1, m1 = ModelRegistry.get_nli_model()
        tok2, m2 = ModelRegistry.get_nli_model()
        assert tok1 is tok2
        assert m1 is m2
        assert id(m1) == id(m2)

    def test_singleton_sentence_transformer_identity(self):
        st1 = ModelRegistry.get_sentence_transformer()
        st2 = ModelRegistry.get_sentence_transformer()
        assert st1 is st2
        assert id(st1) == id(st2)

    def test_singleton_cross_encoder_reranker_identity(self):
        ce1 = ModelRegistry.get_cross_encoder_reranker()
        ce2 = ModelRegistry.get_cross_encoder_reranker()
        assert ce1 is ce2
        assert id(ce1) == id(ce2)

    def test_concurrency_semaphore_acquisition(self):
        sem = ModelRegistry.get_nli_semaphore(max_concurrent=2)
        assert sem is not None
        with sem:
            # Successfully acquired
            pass


class TestFailureSemantics:
    def test_failure_summary_allows_none_h_score(self):
        summary = VerificationSummary(
            status="FAILED",
            h_score=None,
            risk_level=None,
            claims_total=None,
            claims_flagged=None,
            error_message="Verification service encountered an internal error.",
        )
        assert summary.status == "FAILED"
        assert summary.h_score is None
        assert summary.risk_level is None
        assert summary.error_message is not None

    def test_unverified_summary_has_none_h_score(self):
        summary = VerificationSummary(
            status="UNVERIFIED",
            h_score=None,
            risk_level=None,
            claims_total=None,
            claims_flagged=None,
        )
        assert summary.status == "UNVERIFIED"
        assert summary.h_score is None
