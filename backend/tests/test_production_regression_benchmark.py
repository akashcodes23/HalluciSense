"""Regression Benchmark Suite for HalluciSense Accuracy Repair.

Covers the mandatory production benchmark cases:
TEST 1: "2+1=7" -> CONTRADICTED, Correction: "2+1 = 3"
TEST 2: "Hubli is the capital of Karnataka." -> CONTRADICTED, Correction: "Bengaluru is the capital of Karnataka."
TEST 3: "Bengaluru is the capital of Karnataka." -> SUPPORTED
TEST 4: "New Delhi is the capital of India." -> SUPPORTED
TEST 5: "Paris is the capital of Germany." -> CONTRADICTED, Correction: "Berlin is the capital of Germany."
TEST 6: "Paris is the capital of France." -> SUPPORTED
TEST 7: "Unicorns currently govern Mars." -> INSUFFICIENT_EVIDENCE
TEST 8: "Water is composed of hydrogen and oxygen." -> SUPPORTED
TEST 9: "Paris is the capital of France. Berlin is the capital of France." -> INTERNAL_CONTRADICTION
TEST 10: "All prime numbers are even." -> CONTRADICTED
TEST 11: "2+2=4." -> SUPPORTED
TEST 12: "An unsupported scientific claim about an undiscovered species in a distant exoplanet." -> INSUFFICIENT_EVIDENCE
"""

import pytest
from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.core.engine.pillar3_consistency import Pillar3ConsistencyEngine
from app.core.correction.correction_engine import HallucinationCorrectionEngine
from app.core.engine.types import RiskLevel


@pytest.fixture(scope="module")
def pipeline():
    return HallucinationDetectionPipeline()


@pytest.fixture(scope="module")
def p3_engine():
    return Pillar3ConsistencyEngine()


@pytest.fixture(scope="module")
def correction_engine():
    return HallucinationCorrectionEngine()


def test_case_1_arithmetic_contradiction(pipeline, correction_engine):
    """TEST 1: 2+1=7 -> CONTRADICTED, Correction: 2+1 = 3."""
    report = pipeline.analyze("2+1=7")
    assert report.pillar1_summary.factual_error_score >= 0.90
    assert report.overall_risk_level == RiskLevel.LIKELY_HALLUCINATED

    candidate = correction_engine.generate_candidate(
        query="", original_response="2+1=7", overall_h_score=report.overall_h_score
    )
    assert candidate["status"] == "candidate_generated"
    assert "3" in candidate["corrected_text"]


def test_case_2_hubli_capital_contradiction(pipeline, correction_engine):
    """TEST 2: Hubli is the capital of Karnataka. -> CONTRADICTED, Correction contains Bengaluru."""
    report = pipeline.analyze("Hubli is the capital of Karnataka.")
    assert report.pillar1_summary.factual_error_score >= 0.80
    assert report.overall_risk_level == RiskLevel.LIKELY_HALLUCINATED

    candidate = correction_engine.generate_candidate(
        query="",
        original_response="Hubli is the capital of Karnataka.",
        retrieved_evidence=[{
            "snippet": "Bengaluru, also known as Bangalore, is the capital and largest city of the southern Indian state of Karnataka.",
            "score": 0.95
        }],
        overall_h_score=report.overall_h_score
    )
    assert candidate["status"] == "candidate_generated"
    assert "Bengaluru" in candidate["corrected_text"]


def test_case_3_bengaluru_capital_supported(pipeline):
    """TEST 3: Bengaluru is the capital of Karnataka. -> SUPPORTED."""
    report = pipeline.analyze("Bengaluru is the capital of Karnataka.")
    assert report.pillar1_summary.factual_error_score <= 0.20
    assert report.overall_risk_level in (RiskLevel.VERIFIED, RiskLevel.LOW_RISK)


def test_case_4_new_delhi_capital_supported(pipeline):
    """TEST 4: New Delhi is the capital of India. -> SUPPORTED."""
    report = pipeline.analyze("New Delhi is the capital of India.")
    assert report.pillar1_summary.factual_error_score <= 0.20
    assert report.overall_risk_level in (RiskLevel.VERIFIED, RiskLevel.LOW_RISK)


def test_case_5_paris_germany_contradiction(pipeline, correction_engine):
    """TEST 5: Paris is the capital of Germany. -> CONTRADICTED, Correction: Berlin."""
    report = pipeline.analyze("Paris is the capital of Germany.")
    assert report.pillar1_summary.factual_error_score >= 0.80
    assert report.overall_risk_level == RiskLevel.LIKELY_HALLUCINATED

    candidate = correction_engine.generate_candidate(
        query="",
        original_response="Paris is the capital of Germany.",
        retrieved_evidence=[{
            "snippet": "Berlin is the capital and largest city of Germany.",
            "score": 0.95
        }],
        overall_h_score=report.overall_h_score
    )
    assert candidate["status"] == "candidate_generated"
    assert "Berlin" in candidate["corrected_text"]


def test_case_6_paris_france_supported(pipeline):
    """TEST 6: Paris is the capital of France. -> SUPPORTED."""
    report = pipeline.analyze("Paris is the capital of France.")
    assert report.pillar1_summary.factual_error_score <= 0.20
    assert report.overall_risk_level in (RiskLevel.VERIFIED, RiskLevel.LOW_RISK)


def test_case_7_unicorns_mars_insufficient(pipeline, correction_engine):
    """TEST 7: Unicorns currently govern Mars. -> INSUFFICIENT_EVIDENCE."""
    report = pipeline.analyze("Unicorns currently govern Mars.")
    # In the absence of conclusive supporting or refuting evidence
    assert report.overall_risk_level in (RiskLevel.INSUFFICIENT_EVIDENCE, RiskLevel.NEEDS_VERIFICATION, RiskLevel.NEEDS_REVIEW)

    candidate = correction_engine.generate_candidate(
        query="",
        original_response="Unicorns currently govern Mars.",
        retrieved_evidence=[],
        overall_h_score=0.5
    )
    assert candidate["status"] == "abstained"


def test_case_8_water_composition_supported(pipeline):
    """TEST 8: Water is composed of hydrogen and oxygen. -> SUPPORTED."""
    report = pipeline.analyze("Water is composed of hydrogen and oxygen.")
    assert report.pillar1_summary.factual_error_score <= 0.20
    assert report.overall_risk_level in (RiskLevel.VERIFIED, RiskLevel.LOW_RISK)


def test_case_9_intra_response_contradiction(p3_engine):
    """TEST 9: Paris is the capital of France. Berlin is the capital of France. -> INTERNAL_CONTRADICTION."""
    text = "Paris is the capital of France. Berlin is the capital of France."
    res = p3_engine.analyze(text)
    assert res.mode == "INTRA_RESPONSE_CONSISTENCY"
    assert res.contradiction_score >= 0.90
    assert res.consistency_failure_score >= 0.90


def test_case_10_all_prime_numbers_even(pipeline):
    """TEST 10: All prime numbers are even. -> CONTRADICTED."""
    report = pipeline.analyze("All prime numbers are even.")
    assert report.pillar1_summary.factual_error_score >= 0.80
    assert report.overall_risk_level == RiskLevel.LIKELY_HALLUCINATED


def test_case_11_arithmetic_supported(pipeline):
    """TEST 11: 2+2=4. -> SUPPORTED."""
    report = pipeline.analyze("2+2=4.")
    assert report.pillar1_summary.factual_error_score == 0.0
    assert report.overall_risk_level in (RiskLevel.VERIFIED, RiskLevel.LOW_RISK)


def test_case_12_exoplanet_unsupported_claim(pipeline, correction_engine):
    """TEST 12: Unsupported scientific claim about undiscovered species -> INSUFFICIENT_EVIDENCE."""
    text = "An undiscovered species with silicone-based DNA breathes methane on exoplanet Gliese 581g."
    report = pipeline.analyze(text)
    assert report.overall_risk_level in (RiskLevel.INSUFFICIENT_EVIDENCE, RiskLevel.NEEDS_VERIFICATION, RiskLevel.NEEDS_REVIEW)

