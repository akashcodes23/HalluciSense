"""Phase 50 — Pillar 2 and Pillar 3 Operational Execution Proof Tests."""

import pytest
from app.core.engine.pipeline import HallucinationDetectionPipeline


def test_pillar2_static_verification_mode():
    """Verify Pillar 2 executes in STATIC_VERIFICATION_CONFIDENCE mode for static inputs."""
    pipeline = HallucinationDetectionPipeline()
    report = pipeline.analyze("The capital of France is Paris.")
    p2 = report.pillar2_summary

    assert p2 is not None
    assert p2.status == "EXECUTED"
    assert p2.mode == "STATIC_VERIFICATION_CONFIDENCE"
    assert p2.available is True
    assert p2.confidence_gap_score is not None
    assert 0.0 <= p2.confidence_gap_score <= 1.0


def test_pillar3_single_claim_mode():
    """Verify Pillar 3 executes in SINGLE_CLAIM_CONSISTENCY mode for atomic text."""
    pipeline = HallucinationDetectionPipeline()
    report = pipeline.analyze("The capital of France is Paris.")
    p3 = report.pillar3_summary

    assert p3 is not None
    assert p3.status == "EXECUTED"
    assert p3.mode == "SINGLE_CLAIM_CONSISTENCY"
    assert p3.available is True
    assert p3.consistency_failure_score == 0.0


def test_pillar3_contradiction_detection():
    """Verify Pillar 3 detects genuine internal contradiction across contradictory claim pairs."""
    pipeline = HallucinationDetectionPipeline()
    text = "Paris is the capital of France. Berlin is the capital of France."
    report = pipeline.analyze(text)
    p3 = report.pillar3_summary

    assert p3 is not None
    assert p3.status == "EXECUTED"
    assert p3.mode == "INTRA_RESPONSE_CONSISTENCY"
    assert p3.consistency_failure_score is not None
    assert p3.consistency_failure_score > 0.50, f"Expected high contradiction failure score, got {p3.consistency_failure_score}"
