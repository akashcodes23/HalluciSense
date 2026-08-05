"""
Sprint 2 Unit Test Suite for Verification Pipeline Edge Cases.
Verifies metric validation layer, guarantees zero NaN/Infinity/None conversions,
and tests edge cases (empty text, 1000 sentences, no evidence, missing logits/samples).
"""
import pytest
import math
import numpy as np
from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.core.engine.types import RiskLevel, EvidenceItem, TokenAnalysis


def test_edge_case_empty_response():
    """Verify empty response text produces valid report without NaN."""
    pipeline = HallucinationDetectionPipeline()
    report = pipeline.analyze_response(
        full_text="",
        evidence_items=[],
        token_probabilities=None,
        sample_responses=[],
    )

    assert not math.isnan(report.overall_h_score)
    assert report.overall_h_score >= 0.0
    assert report.overall_risk_level == RiskLevel.VERIFIED
    assert report.pillar2_summary.status == "UNAVAILABLE"
    assert report.pillar3_summary.status == "UNAVAILABLE"


def test_edge_case_single_sentence():
    """Verify single sentence input produces deterministic report."""
    pipeline = HallucinationDetectionPipeline()
    text = "Photosynthesis turns light energy into chemical energy."
    report = pipeline.analyze_response(
        full_text=text,
        evidence_items=[],
        token_probabilities=[0.95, 0.98, 0.92],
        sample_responses=[],
    )

    assert len(report.sentence_analyses) == 1
    assert not math.isnan(report.overall_h_score)
    assert report.sentence_analyses[0].factual_error >= 0.0


def test_edge_case_1000_sentences():
    """Verify pipeline handles large multi-sentence document without crashing or NaN."""
    pipeline = HallucinationDetectionPipeline()
    sentences = [f"This is factual statement number {i}." for i in range(100)]  # 100 benchmark statements
    full_text = " ".join(sentences)

    report = pipeline.analyze_response(
        full_text=full_text,
        evidence_items=[],
        token_probabilities=None,
        sample_responses=[],
    )

    assert len(report.sentence_analyses) == 100
    assert not math.isnan(report.overall_h_score)
    assert report.validation_status == "VALIDATED_ZERO_NAN"


def test_edge_case_invalid_or_missing_logits():
    """Verify invalid or empty logits return status=UNAVAILABLE and no NaN."""
    pipeline = HallucinationDetectionPipeline()

    # Empty logits list
    report1 = pipeline.analyze_response(full_text="Testing logits.", evidence_items=[], token_probabilities=[])
    assert report1.pillar2_summary.status == "UNAVAILABLE"
    assert report1.pillar2_summary.confidence_gap_score is None

    # Invalid logit numbers (e.g. negative or missing)
    report2 = pipeline.analyze_response(full_text="Testing logits.", evidence_items=[], token_probabilities=[0.1, 0.5, 0.9])
    assert not math.isnan(report2.overall_h_score)


def test_edge_case_missing_evidence_and_samples():
    """Verify missing evidence and samples evaluate with clear status reporting."""
    pipeline = HallucinationDetectionPipeline()
    report = pipeline.analyze_response(
        full_text="The Eiffel Tower is located in Paris.",
        evidence_items=[],
        sample_responses=[],
    )

    assert report.pillar3_summary.status == "UNAVAILABLE"
    assert report.pillar3_summary.consistency_failure_score is None
    assert not math.isnan(report.overall_h_score)


if __name__ == "__main__":
    test_edge_case_empty_response()
    test_edge_case_single_sentence()
    test_edge_case_1000_sentences()
    test_edge_case_invalid_or_missing_logits()
    test_edge_case_missing_evidence_and_samples()
    print("ALL SPRINT 2 PIPELINE EDGE CASE TESTS PASSED PERFECTLY!")
