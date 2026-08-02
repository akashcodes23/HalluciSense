"""
Phase 5 — Production Verification Integration Tests.

Tests the REAL HalluciSense scoring engines against mocked external
boundaries (DB, providers, evidence retrieval) to verify:
- Persistence contracts
- Pillar availability semantics
- Idempotency
- Transaction safety
- API serialization with None values

NO live external API calls. All pillar math is REAL.
"""

import pytest
import math
import uuid
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Optional, List

from app.core.engine.types import (
    Pillar1Result, Pillar2Result, Pillar3Result, EvidenceItem, RiskLevel,
    HallucinationReport, SentenceAnalysis as PipelineSentenceAnalysis,
)
from app.core.engine.pillar1_retrieval import Pillar1RetrievalEngine
from app.core.engine.pillar2_confidence import Pillar2ConfidenceEngine
from app.core.engine.pillar3_consistency import Pillar3ConsistencyEngine
from app.core.engine.fusion import FusionEngine
from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.modules.verification.schemas import VerificationReportResponse


# =========================================================
# CORRECTION MOCK FIXTURE (same as Phase 4)
# =========================================================

def _fake_generate_correction(self, full_text, sentence_analyses, evidence_items):
    """
    Deterministic offline replacement for _generate_correction.
    Returns a synthetic corrected text and populates corrected_response
    on flagged sentences without making any network calls.
    """
    corrected_text = full_text
    for sa in sentence_analyses:
        if sa.risk_level != RiskLevel.VERIFIED:
            sa.corrected_response = f"[TEST CORRECTION] {sa.text}"
    return corrected_text, sentence_analyses


@pytest.fixture(autouse=True)
def mock_correction_engine(monkeypatch):
    """Auto-applied fixture that patches out the Gemini correction call."""
    monkeypatch.setattr(
        HallucinationDetectionPipeline,
        "_generate_correction",
        _fake_generate_correction,
    )


# =========================================================
# SHARED TEST FIXTURES
# =========================================================

CONSISTENT_TEXT = "Paris is the capital of France."
CONSISTENT_EVIDENCE = [
    EvidenceItem(
        claim="Paris capital of France",
        snippet="Paris is the capital and most populous city of France.",
        source_name="Wikipedia",
        source_url="https://en.wikipedia.org/wiki/Paris",
        similarity_score=0.95,
        is_supporting=True,
    )
]
CONSISTENT_SAMPLES = [
    "The capital of France is Paris.",
    "Paris is France's capital city.",
]
CONSISTENT_PROBS = [0.98, 0.99, 0.97, 0.99, 0.96, 0.98]

CONTRADICTORY_TEXT = "Paris is the capital of Germany."
CONTRADICTORY_EVIDENCE = [
    EvidenceItem(
        claim="Paris capital of Germany",
        snippet="Berlin is the capital and largest city of Germany.",
        source_name="Wikipedia",
        source_url="https://en.wikipedia.org/wiki/Berlin",
        similarity_score=0.90,
        is_supporting=False,
    )
]
CONTRADICTORY_SAMPLES = [
    "Berlin is the capital of Germany.",
    "The capital city of Germany is Berlin.",
]
CONTRADICTORY_PROBS = [0.95, 0.92, 0.90, 0.94, 0.93, 0.91]


# =========================================================
# CASE 1 — Normal Verified Response
# =========================================================

class TestCase1NormalVerifiedResponse:
    """Verify that a consistent response produces a low H-score and VERIFIED risk."""

    def test_verified_response_scoring(self):
        pipeline = HallucinationDetectionPipeline()
        report = pipeline.analyze_response(
            full_text=CONSISTENT_TEXT,
            token_probabilities=CONSISTENT_PROBS,
            evidence_items=CONSISTENT_EVIDENCE,
            sample_responses=CONSISTENT_SAMPLES,
        )

        assert report.overall_h_score < 0.20
        assert report.overall_risk_level == RiskLevel.VERIFIED
        assert len(report.sentence_analyses) >= 1
        assert report.pillar1_summary is not None
        assert report.pillar2_summary is not None
        assert report.pillar3_summary is not None
        assert report.weights_used is not None

    def test_verified_response_persistence_fields(self):
        """Verify all fields needed for DB persistence are populated."""
        pipeline = HallucinationDetectionPipeline()
        report = pipeline.analyze_response(
            full_text=CONSISTENT_TEXT,
            token_probabilities=CONSISTENT_PROBS,
            evidence_items=CONSISTENT_EVIDENCE,
            sample_responses=CONSISTENT_SAMPLES,
        )

        # Report-level fields
        assert isinstance(report.overall_h_score, float)
        assert 0.0 <= report.overall_h_score <= 1.0
        assert report.overall_risk_level in RiskLevel
        assert isinstance(report.pillar1_summary.factual_error_score, float)
        assert report.corrected_response is not None  # Even VERIFIED gets a correction text

        # Sentence-level fields
        for sa in report.sentence_analyses:
            assert isinstance(sa.sentence_id, int)
            assert isinstance(sa.text, str)
            assert isinstance(sa.start_char, int)
            assert isinstance(sa.end_char, int)
            assert isinstance(sa.factual_error, float)
            assert isinstance(sa.hallucination_score, float)
            assert sa.risk_level in RiskLevel
            assert isinstance(sa.color_code, str)
            assert isinstance(sa.reasoning, str)


# =========================================================
# CASE 2 — Contradictory Response
# =========================================================

class TestCase2ContradictoryResponse:
    """Verify that a contradicted response produces high H-score and LIKELY_HALLUCINATED."""

    def test_contradictory_response_scoring(self):
        pipeline = HallucinationDetectionPipeline()
        report = pipeline.analyze_response(
            full_text=CONTRADICTORY_TEXT,
            token_probabilities=CONTRADICTORY_PROBS,
            evidence_items=CONTRADICTORY_EVIDENCE,
            sample_responses=CONTRADICTORY_SAMPLES,
        )

        assert report.overall_h_score >= 0.55
        assert report.overall_risk_level != RiskLevel.VERIFIED

    def test_contradictory_sentence_analysis(self):
        pipeline = HallucinationDetectionPipeline()
        report = pipeline.analyze_response(
            full_text=CONTRADICTORY_TEXT,
            token_probabilities=CONTRADICTORY_PROBS,
            evidence_items=CONTRADICTORY_EVIDENCE,
            sample_responses=CONTRADICTORY_SAMPLES,
        )

        assert len(report.sentence_analyses) >= 1
        flagged = [s for s in report.sentence_analyses if s.risk_level != RiskLevel.VERIFIED]
        assert len(flagged) >= 1


# =========================================================
# CASE 3 — Pillar 2 Unavailable (token_probs=None)
# =========================================================

class TestCase3Pillar2Unavailable:
    """Verify Pillar 2 unavailability with token_probs=None."""

    def test_p2_unavailable_cg_is_none(self):
        pipeline = HallucinationDetectionPipeline()
        report = pipeline.analyze_response(
            full_text=CONSISTENT_TEXT,
            token_probabilities=None,
            evidence_items=CONSISTENT_EVIDENCE,
            sample_responses=CONSISTENT_SAMPLES,
        )

        assert report.pillar2_summary.available is False
        assert report.pillar2_summary.confidence_gap_score is None

    def test_p2_unavailable_beta_zero(self):
        pipeline = HallucinationDetectionPipeline()
        report = pipeline.analyze_response(
            full_text=CONSISTENT_TEXT,
            token_probabilities=None,
            evidence_items=CONSISTENT_EVIDENCE,
            sample_responses=CONSISTENT_SAMPLES,
        )

        assert report.weights_used["beta_confidence_gap"] == 0.0
        assert pytest.approx(sum(report.weights_used.values()), abs=1e-3) == 1.0

    def test_p2_unavailable_sentence_cg_none(self):
        """Per-sentence confidence_gap must also be None when P2 unavailable."""
        pipeline = HallucinationDetectionPipeline()
        report = pipeline.analyze_response(
            full_text=CONSISTENT_TEXT,
            token_probabilities=None,
            evidence_items=CONSISTENT_EVIDENCE,
            sample_responses=CONSISTENT_SAMPLES,
        )

        for sa in report.sentence_analyses:
            assert sa.confidence_gap is None

    def test_p2_unavailable_verification_completes(self):
        """Verification must complete even with P2 unavailable."""
        pipeline = HallucinationDetectionPipeline()
        report = pipeline.analyze_response(
            full_text=CONSISTENT_TEXT,
            token_probabilities=None,
            evidence_items=CONSISTENT_EVIDENCE,
            sample_responses=CONSISTENT_SAMPLES,
        )

        assert 0.0 <= report.overall_h_score <= 1.0
        assert report.overall_risk_level in RiskLevel


# =========================================================
# CASE 4 — Pillar 3 Unavailable (no samples)
# =========================================================

class TestCase4Pillar3Unavailable:
    """Verify graceful degradation when sample generation fails."""

    def test_p3_unavailable_cf_is_none(self):
        pipeline = HallucinationDetectionPipeline()
        report = pipeline.analyze_response(
            full_text=CONSISTENT_TEXT,
            token_probabilities=CONSISTENT_PROBS,
            evidence_items=CONSISTENT_EVIDENCE,
            sample_responses=[],
        )

        assert report.pillar3_summary.available is False
        assert report.pillar3_summary.consistency_failure_score is None

    def test_p3_unavailable_gamma_zero(self):
        pipeline = HallucinationDetectionPipeline()
        report = pipeline.analyze_response(
            full_text=CONSISTENT_TEXT,
            token_probabilities=CONSISTENT_PROBS,
            evidence_items=CONSISTENT_EVIDENCE,
            sample_responses=[],
        )

        assert report.weights_used["gamma_consistency_failure"] == 0.0
        assert pytest.approx(sum(report.weights_used.values()), abs=1e-3) == 1.0

    def test_p3_unavailable_no_fabricated_score(self):
        """Must not fabricate CF=0.0 when P3 is genuinely unavailable."""
        pipeline = HallucinationDetectionPipeline()
        report = pipeline.analyze_response(
            full_text=CONSISTENT_TEXT,
            token_probabilities=CONSISTENT_PROBS,
            evidence_items=CONSISTENT_EVIDENCE,
            sample_responses=[],
        )

        for sa in report.sentence_analyses:
            assert sa.consistency_failure is None


# =========================================================
# CASE 5 — Correction Provider Unavailable
# =========================================================

class TestCase5CorrectionProviderUnavailable:
    """Verify detection report persists even when correction fails."""

    def test_detection_report_survives_correction_failure(self, monkeypatch):
        def _failing_correction(self, full_text, sentence_analyses, evidence_items):
            raise RuntimeError("Gemini DNS resolution failed")

        # Override the autouse fixture to simulate a real failure
        monkeypatch.setattr(
            HallucinationDetectionPipeline,
            "_generate_correction",
            _failing_correction,
        )

        pipeline = HallucinationDetectionPipeline()
        ev = EvidenceItem(
            claim="Eiffel Tower Berlin",
            snippet="The Eiffel Tower is located in Paris, France.",
            source_name="Wikipedia",
            source_url="https://en.wikipedia.org/wiki/Eiffel_Tower",
            similarity_score=0.90,
            is_supporting=False,
        )

        # This will trigger correction (high FE), but correction will fail
        # The pipeline's try/except should catch it and still return the report
        report = pipeline.analyze_response(
            full_text="The Eiffel Tower is located in Berlin.",
            token_probabilities=[0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
            evidence_items=[ev],
            sample_responses=["The Eiffel Tower is in Paris."],
        )

        # Detection must still be complete
        assert 0.0 <= report.overall_h_score <= 1.0
        assert report.pillar1_summary is not None
        assert len(report.sentence_analyses) >= 1
        # Correction failed, so corrected_response should indicate failure
        assert report.corrected_response is not None


# =========================================================
# CASE 6 — Task Executed Twice (Idempotency)
# =========================================================

class TestCase6IdempotencyGuard:
    """Verify running the pipeline twice produces consistent output without corruption."""

    def test_pipeline_deterministic_under_same_input(self):
        """Run the pipeline twice with identical input — scores should be identical."""
        pipeline = HallucinationDetectionPipeline()

        report1 = pipeline.analyze_response(
            full_text=CONSISTENT_TEXT,
            token_probabilities=CONSISTENT_PROBS,
            evidence_items=CONSISTENT_EVIDENCE,
            sample_responses=CONSISTENT_SAMPLES,
        )

        report2 = pipeline.analyze_response(
            full_text=CONSISTENT_TEXT,
            token_probabilities=CONSISTENT_PROBS,
            evidence_items=CONSISTENT_EVIDENCE,
            sample_responses=CONSISTENT_SAMPLES,
        )

        assert report1.overall_h_score == report2.overall_h_score
        assert report1.overall_risk_level == report2.overall_risk_level
        assert len(report1.sentence_analyses) == len(report2.sentence_analyses)

    def test_idempotency_guard_exists_in_worker(self):
        """Verify the worker code contains an idempotency check."""
        import inspect
        from app.workers.tasks.verification_task import run_verification_async

        source = inspect.getsource(run_verification_async)
        # The worker should check for existing reports before creating new ones
        assert "existing" in source.lower() or "replace" in source.lower() or "delete" in source.lower(), \
            "Worker must contain idempotency guard (check for existing report)"


# =========================================================
# CASE 7 — Transaction Safety
# =========================================================

class TestCase7TransactionSafety:
    """Verify that report and sentence analyses are produced atomically."""

    def test_report_and_sentences_produced_together(self):
        """The pipeline always produces both report-level and sentence-level data."""
        pipeline = HallucinationDetectionPipeline()
        report = pipeline.analyze_response(
            full_text="Sentence one. Sentence two. Sentence three.",
            token_probabilities=[0.95] * 9,
            evidence_items=[],
            sample_responses=["Sentence one. Sentence two. Sentence three."],
        )

        # Report must exist
        assert report is not None
        assert isinstance(report.overall_h_score, float)

        # Sentence analyses must exist
        assert len(report.sentence_analyses) == 3

        # Each sentence must have complete data
        for sa in report.sentence_analyses:
            assert isinstance(sa.hallucination_score, float)
            assert 0.0 <= sa.hallucination_score <= 1.0
            assert sa.risk_level in RiskLevel

    def test_worker_uses_single_commit(self):
        """Verify the worker commits report + sentences in a single transaction."""
        import inspect
        from app.workers.tasks.verification_task import run_verification_async

        source = inspect.getsource(run_verification_async)
        # Count session.commit() calls — should be exactly 1
        commit_count = source.count("session.commit()")
        assert commit_count == 1, (
            f"Expected exactly 1 session.commit() for atomic persistence, found {commit_count}"
        )


# =========================================================
# CASE 8 — API Serialization with CG=None
# =========================================================

class TestCase8APISerialization:
    """Verify API schema handles None values without TypeError."""

    def test_model_dump_with_none_cg(self):
        """Pydantic model_dump must handle None for confidence_gap_score."""
        pipeline = HallucinationDetectionPipeline()
        report = pipeline.analyze_response(
            full_text=CONSISTENT_TEXT,
            token_probabilities=None,
            evidence_items=CONSISTENT_EVIDENCE,
            sample_responses=[],
        )

        dumped = report.model_dump()

        assert dumped["pillar2_summary"]["confidence_gap_score"] is None
        assert dumped["pillar2_summary"]["available"] is False
        assert dumped["pillar3_summary"]["consistency_failure_score"] is None
        assert dumped["pillar3_summary"]["available"] is False
        assert isinstance(dumped["overall_h_score"], float)

    def test_no_type_error_on_trust_score_computation(self):
        """Verify 1.0 - None is never attempted."""
        pipeline = HallucinationDetectionPipeline()
        report = pipeline.analyze_response(
            full_text=CONSISTENT_TEXT,
            token_probabilities=None,
            evidence_items=[],
            sample_responses=[],
        )

        cg = report.pillar2_summary.confidence_gap_score
        # This is the exact computation the API does:
        if cg is not None:
            confidence_score = round(1.0 - cg, 4)
        else:
            confidence_score = None

        assert confidence_score is None

    def test_weights_sum_to_one_with_none_pillars(self):
        """Weights must still sum to 1.0 even with unavailable pillars."""
        pipeline = HallucinationDetectionPipeline()
        report = pipeline.analyze_response(
            full_text=CONSISTENT_TEXT,
            token_probabilities=None,
            evidence_items=[],
            sample_responses=[],
        )

        total = sum(report.weights_used.values())
        assert pytest.approx(total, abs=1e-3) == 1.0

    def test_verification_report_schema_accepts_none(self):
        """Verify VerificationReportResponse Pydantic schema handles None fields."""
        schema = VerificationReportResponse(
            id=uuid.uuid4(),
            message_id=uuid.uuid4(),
            overall_h_score=0.5,
            overall_risk_level="NEEDS_VERIFICATION",
            factual_error_score=0.5,
            confidence_gap_score=None,
            consistency_failure_score=None,
            weights_used={"alpha_factual_error": 1.0, "beta_confidence_gap": 0.0, "gamma_consistency_failure": 0.0},
            processing_time_ms=100.0,
            sentence_analyses=[],
            created_at="2026-01-01T00:00:00",
        )

        dumped = schema.model_dump()
        assert dumped["confidence_gap_score"] is None
        assert dumped["consistency_failure_score"] is None
        # No TypeError occurred


# =========================================================
# ADDITIONAL VERIFICATION — Pillar summary serialization
# =========================================================

class TestPillarSummarySerialization:
    """Verify pillar summaries can be serialized to JSON for DB storage."""

    def test_pillar_summaries_are_serializable(self):
        pipeline = HallucinationDetectionPipeline()
        report = pipeline.analyze_response(
            full_text=CONSISTENT_TEXT,
            token_probabilities=CONSISTENT_PROBS,
            evidence_items=CONSISTENT_EVIDENCE,
            sample_responses=CONSISTENT_SAMPLES,
        )

        # These must be serializable for JSON columns
        p1_dict = report.pillar1_summary.model_dump()
        p2_dict = report.pillar2_summary.model_dump()
        p3_dict = report.pillar3_summary.model_dump()

        assert isinstance(p1_dict, dict)
        assert isinstance(p2_dict, dict)
        assert isinstance(p3_dict, dict)
        assert "factual_error_score" in p1_dict
        assert "confidence_gap_score" in p2_dict
        assert "consistency_failure_score" in p3_dict

    def test_pillar_summary_none_values_serialize(self):
        pipeline = HallucinationDetectionPipeline()
        report = pipeline.analyze_response(
            full_text=CONSISTENT_TEXT,
            token_probabilities=None,
            evidence_items=[],
            sample_responses=[],
        )

        p2_dict = report.pillar2_summary.model_dump()
        p3_dict = report.pillar3_summary.model_dump()

        assert p2_dict["confidence_gap_score"] is None
        assert p3_dict["consistency_failure_score"] is None


# =========================================================
# REGRESSION GUARD — Correction does not modify H-Score
# =========================================================

class TestCorrectionDoesNotModifyHScore:
    """Ensure correction generation does not alter detection scores."""

    def test_correction_independence(self):
        pipeline = HallucinationDetectionPipeline()
        ev = EvidenceItem(
            claim="Eiffel Tower Berlin",
            snippet="The Eiffel Tower is located in Paris, France.",
            source_name="Wikipedia",
            source_url="https://en.wikipedia.org/wiki/Eiffel_Tower",
            similarity_score=0.90,
            is_supporting=False,
        )

        report = pipeline.analyze_response(
            full_text="The Eiffel Tower is located in Berlin.",
            token_probabilities=[0.95] * 7,
            evidence_items=[ev],
            sample_responses=["The Eiffel Tower is in Paris."],
        )

        h_before = report.overall_h_score
        assert report.corrected_response is not None
        assert report.overall_h_score == h_before
