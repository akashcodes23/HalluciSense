"""
Comprehensive Production End-to-End Test Suite (Phase 13).

Covers:
- Pillar 1: Evidence & Entailment Retrieval (10 benchmark claims)
- Adaptive Fusion: Complete 8-mask signal availability truth table
- Closed-Loop Correction: 9 real-world repair, reverification & abstention scenarios
- Chat Pipeline: Verified vs. hallucinated closed-loop flows
- Provider Logprobs: Genuine token probabilities, absent logprobs, and pasted text
- Pillar 3 Consistency: Insufficient sample gating, semantic equivalence, and contradiction
"""

import math
import pytest
from unittest.mock import patch, MagicMock

from app.core.config import settings
from app.core.engine.types import (
    RiskLevel,
    EvidenceItem,
    Pillar1Result,
    Pillar2Result,
    Pillar3Result,
    HallucinationReport,
)
from app.core.engine.pillar1_retrieval import Pillar1RetrievalEngine
from app.core.engine.pillar2_confidence import Pillar2ConfidenceEngine
from app.core.engine.pillar3_consistency import Pillar3ConsistencyEngine
from app.core.engine.fusion import FusionEngine
from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.core.correction.correction_engine import HallucinationCorrectionEngine, ClosedLoopRepairResult
from app.main import app
from fastapi.testclient import TestClient


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def p1_engine():
    return Pillar1RetrievalEngine()


@pytest.fixture(scope="module")
def p2_engine():
    return Pillar2ConfidenceEngine()


@pytest.fixture(scope="module")
def p3_engine():
    return Pillar3ConsistencyEngine()


@pytest.fixture(scope="module")
def fusion_engine():
    return FusionEngine()


@pytest.fixture(scope="module")
def correction_engine():
    return HallucinationCorrectionEngine()


@pytest.fixture(scope="module")
def pipeline():
    return HallucinationDetectionPipeline()


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ============================================================================
# SECTION A: PILLAR 1 — EVIDENCE & RETRIEVAL (10 TEST CASES)
# ============================================================================

def test_p1_bengaluru_capital_maharashtra_contradicted(p1_engine):
    """Claim: Bengaluru is capital of Maharashtra -> must be CONTRADICTED."""
    claim = "Bengaluru is the capital of Maharashtra."
    evidence = [
        EvidenceItem(
            claim="capital of Maharashtra",
            snippet="Mumbai is the capital city of the Indian state of Maharashtra.",
            source_name="Wikipedia: Maharashtra",
            is_supporting=True,
            similarity_score=0.95,
        )
    ]
    res = p1_engine.analyze(claim, provided_evidence=evidence)
    assert res.factual_error_score >= 0.50, f"Expected FE >= 0.50 for false capital, got {res.factual_error_score}"


def test_p1_bengaluru_capital_karnataka_supported(p1_engine):
    """Claim: Bengaluru is capital of Karnataka -> must be SUPPORTED."""
    claim = "Bengaluru is the capital of Karnataka."
    evidence = [
        EvidenceItem(
            claim="capital of Karnataka",
            snippet="Bengaluru (also called Bangalore) is the capital and largest city of the Indian state of Karnataka.",
            source_name="Wikipedia: Bangalore",
            is_supporting=True,
            similarity_score=0.95,
        )
    ]
    res = p1_engine.analyze(claim, provided_evidence=evidence)
    assert res.factual_error_score <= 0.35, f"Expected FE <= 0.35 for true capital, got {res.factual_error_score}"


def test_p1_mumbai_capital_maharashtra_supported(p1_engine):
    """Claim: Mumbai is capital of Maharashtra -> must be SUPPORTED."""
    claim = "Mumbai is the capital of Maharashtra."
    evidence = [
        EvidenceItem(
            claim="capital of Maharashtra",
            snippet="Mumbai is the capital city of the Indian state of Maharashtra.",
            source_name="Wikipedia: Maharashtra",
            is_supporting=True,
            similarity_score=0.95,
        )
    ]
    res = p1_engine.analyze(claim, provided_evidence=evidence)
    assert res.factual_error_score <= 0.35, f"Expected FE <= 0.35 for true capital, got {res.factual_error_score}"


def test_p1_apollo11_moon_landing_1969_supported(p1_engine):
    """Claim: Apollo 11 landed on Moon in 1969 -> must be SUPPORTED."""
    claim = "Apollo 11 landed on the Moon in 1969."
    evidence = [
        EvidenceItem(
            claim="Apollo 11 Moon landing 1969",
            snippet="Apollo 11 was the American spaceflight that first landed humans on the Moon on July 20, 1969.",
            source_name="Wikipedia: Apollo 11",
            is_supporting=True,
            similarity_score=0.95,
        )
    ]
    res = p1_engine.analyze(claim, provided_evidence=evidence)
    assert res.factual_error_score <= 0.35, f"Expected FE <= 0.35, got {res.factual_error_score}"


def test_p1_apollo12_moon_landing_1969_not_confused_with_apollo11(p1_engine):
    """Claim: Apollo 12 landed on Moon in 1969 -> must be SUPPORTED and NOT confused with Apollo 11."""
    claim = "Apollo 12 landed on the Moon in 1969."
    evidence = [
        EvidenceItem(
            claim="Apollo 12 landed on the Moon",
            snippet="Apollo 12 was the second mission to successfully land on the Moon in 1969.",
            source_name="Apollo 12",
            is_supporting=True,
            similarity_score=0.95,
        ),
        EvidenceItem(
            claim="Apollo 11 landed on the Moon",
            snippet="Apollo 11 was the first spaceflight that landed humans on the Moon in 1969.",
            source_name="Apollo 11",
            is_supporting=True,
            similarity_score=0.90,
        ),
    ]
    res = p1_engine.analyze(claim, provided_evidence=evidence)
    assert res.factual_error_score <= 0.35, (
        f"Apollo 12 should be supported (FE <= 0.35) and not contradicted by Apollo 11 sibling entity, got {res.factual_error_score}"
    )


def test_p1_water_formula_h2o_supported(p1_engine):
    """Claim: Water formula H2O -> must be SUPPORTED."""
    claim = "The chemical formula of water is H2O."
    evidence = [
        EvidenceItem(
            claim="chemical formula of water",
            snippet="Water is an inorganic compound with the chemical formula H2O.",
            source_name="Wikipedia: Water",
            is_supporting=True,
            similarity_score=0.95,
        )
    ]
    res = p1_engine.analyze(claim, provided_evidence=evidence)
    assert res.factual_error_score <= 0.35, f"Expected FE <= 0.35 for H2O formula, got {res.factual_error_score}"


def test_p1_water_formula_co2_contradicted(p1_engine):
    """Claim: Water formula CO2 -> must be CONTRADICTED."""
    claim = "The chemical formula of water is CO2."
    evidence = [
        EvidenceItem(
            claim="chemical formula of water",
            snippet="Water is a chemical compound with the chemical formula H2O.",
            source_name="Wikipedia: Water",
            is_supporting=True,
            similarity_score=0.95,
        )
    ]
    res = p1_engine.analyze(claim, provided_evidence=evidence)
    assert res.factual_error_score >= 0.50, f"Expected FE >= 0.50 for CO2 formula, got {res.factual_error_score}"


def test_p1_boeing737_two_engines_supported(p1_engine):
    """Claim: Boeing 737 has two engines -> must be SUPPORTED."""
    claim = "The Boeing 737 is powered by two turbofan engines."
    evidence = [
        EvidenceItem(
            claim="Boeing 737 engines",
            snippet="The Boeing 737 is a twin-engine narrow-body airliner powered by two CFM turbofan engines.",
            source_name="Wikipedia: Boeing 737",
            is_supporting=True,
            similarity_score=0.95,
        )
    ]
    res = p1_engine.analyze(claim, provided_evidence=evidence)
    assert res.factual_error_score <= 0.35, f"Expected FE <= 0.35 for 2 engines, got {res.factual_error_score}"


def test_p1_boeing737_four_engines_contradicted(p1_engine):
    """Claim: Boeing 737 has four engines -> must be CONTRADICTED."""
    claim = "The Boeing 737 is powered by four turbofan engines."
    evidence = [
        EvidenceItem(
            claim="Boeing 737 engines",
            snippet="The Boeing 737 is a twinjet airliner with two engines, not four.",
            source_name="Wikipedia: Boeing 737",
            is_supporting=True,
            similarity_score=0.95,
        )
    ]
    res = p1_engine.analyze(claim, provided_evidence=evidence)
    assert res.factual_error_score >= 0.50, f"Expected FE >= 0.50 for 4 engines, got {res.factual_error_score}"


def test_p1_mars_two_moons_supported(p1_engine):
    """Claim: Mars has two natural moons -> must be SUPPORTED."""
    claim = "Mars has two small natural moons, Phobos and Deimos."
    evidence = [
        EvidenceItem(
            claim="Moons of Mars",
            snippet="Mars has two small natural moons, Phobos and Deimos, which are thought to be captured asteroids.",
            source_name="Wikipedia: Moons of Mars",
            is_supporting=True,
            similarity_score=0.95,
        )
    ]
    res = p1_engine.analyze(claim, provided_evidence=evidence)
    assert res.factual_error_score <= 0.35, f"Expected FE <= 0.35 for Mars moons, got {res.factual_error_score}"


# ============================================================================
# SECTION B: FUSION MASKS (ALL 8 COMBINATIONS)
# ============================================================================

def test_fusion_mask_100_p1_only(fusion_engine):
    """Mask [1,0,0]: P1 only available -> weight on alpha=1.0, risk derived from FE."""
    p1 = Pillar1Result(factual_error_score=0.10, available=True, reasoning="P1 verified")
    p2 = Pillar2Result(confidence_gap_score=None, available=False, reasoning="P2 unavailable")
    p3 = Pillar3Result(consistency_failure_score=None, available=False, reasoning="P3 unavailable")

    h_score, risk, color, weights = fusion_engine.fuse(p1, p2, p3)
    assert risk == RiskLevel.VERIFIED
    assert weights["alpha_factual_error"] == 1.0
    assert weights["beta_confidence_gap"] == 0.0
    assert weights["gamma_consistency_failure"] == 0.0
    assert h_score == 0.10


def test_fusion_mask_010_p2_only_insufficient_evidence(fusion_engine):
    """Mask [0,1,0]: P2 only -> must return INSUFFICIENT_EVIDENCE governance rule."""
    p1 = Pillar1Result(factual_error_score=None, available=False, reasoning="P1 unavailable")
    p2 = Pillar2Result(confidence_gap_score=0.15, available=True, reasoning="P2 verified")
    p3 = Pillar3Result(consistency_failure_score=None, available=False, reasoning="P3 unavailable")

    h_score, risk, color, weights = fusion_engine.fuse(p1, p2, p3)
    assert risk == RiskLevel.INSUFFICIENT_EVIDENCE, "P2 alone without factual evidence cannot determine factual truth."


def test_fusion_mask_001_p3_only_insufficient_evidence(fusion_engine):
    """Mask [0,0,1]: P3 only -> must return INSUFFICIENT_EVIDENCE governance rule."""
    p1 = Pillar1Result(factual_error_score=None, available=False, reasoning="P1 unavailable")
    p2 = Pillar2Result(confidence_gap_score=None, available=False, reasoning="P2 unavailable")
    p3 = Pillar3Result(consistency_failure_score=0.10, available=True, reasoning="P3 verified")

    h_score, risk, color, weights = fusion_engine.fuse(p1, p2, p3)
    assert risk == RiskLevel.INSUFFICIENT_EVIDENCE, "P3 alone without factual evidence cannot determine factual truth."


def test_fusion_mask_110_p1_p2(fusion_engine):
    """Mask [1,1,0]: P1 + P2 -> both weights active, gamma=0.0."""
    p1 = Pillar1Result(factual_error_score=0.08, available=True, reasoning="P1 verified")
    p2 = Pillar2Result(confidence_gap_score=0.12, available=True, reasoning="P2 verified")
    p3 = Pillar3Result(consistency_failure_score=None, available=False, reasoning="P3 unavailable")

    h_score, risk, color, weights = fusion_engine.fuse(p1, p2, p3)
    assert risk == RiskLevel.VERIFIED
    assert weights["alpha_factual_error"] > 0.0
    assert weights["beta_confidence_gap"] > 0.0
    assert weights["gamma_consistency_failure"] == 0.0
    assert math.isclose(weights["alpha_factual_error"] + weights["beta_confidence_gap"], 1.0, abs_tol=1e-3)


def test_fusion_mask_101_p1_p3(fusion_engine):
    """Mask [1,0,1]: P1 + P3 -> both weights active, beta=0.0."""
    p1 = Pillar1Result(factual_error_score=0.05, available=True, reasoning="P1 verified")
    p2 = Pillar2Result(confidence_gap_score=None, available=False, reasoning="P2 unavailable")
    p3 = Pillar3Result(consistency_failure_score=0.10, available=True, reasoning="P3 verified")

    h_score, risk, color, weights = fusion_engine.fuse(p1, p2, p3)
    assert risk == RiskLevel.VERIFIED
    assert weights["alpha_factual_error"] > 0.0
    assert weights["beta_confidence_gap"] == 0.0
    assert weights["gamma_consistency_failure"] > 0.0
    assert math.isclose(weights["alpha_factual_error"] + weights["gamma_consistency_failure"], 1.0, abs_tol=1e-3)


def test_fusion_mask_011_p2_p3_no_p1_insufficient_evidence(fusion_engine):
    """Mask [0,1,1]: P2 + P3 without P1 -> must return INSUFFICIENT_EVIDENCE."""
    p1 = Pillar1Result(factual_error_score=None, available=False, reasoning="P1 unavailable")
    p2 = Pillar2Result(confidence_gap_score=0.15, available=True, reasoning="P2 verified")
    p3 = Pillar3Result(consistency_failure_score=0.20, available=True, reasoning="P3 verified")

    h_score, risk, color, weights = fusion_engine.fuse(p1, p2, p3)
    assert risk == RiskLevel.INSUFFICIENT_EVIDENCE, "P2+P3 without P1 cannot establish factual grounding."


def test_fusion_mask_111_all_three_pillars(fusion_engine):
    """Mask [1,1,1]: All 3 pillars active -> full three-pillar fusion."""
    p1 = Pillar1Result(factual_error_score=0.10, available=True, reasoning="P1 verified")
    p2 = Pillar2Result(confidence_gap_score=0.15, available=True, reasoning="P2 verified")
    p3 = Pillar3Result(consistency_failure_score=0.12, available=True, reasoning="P3 verified")

    h_score, risk, color, weights = fusion_engine.fuse(p1, p2, p3)
    assert risk == RiskLevel.VERIFIED
    assert weights["alpha_factual_error"] > 0.0
    assert weights["beta_confidence_gap"] > 0.0
    assert weights["gamma_consistency_failure"] > 0.0
    assert math.isclose(
        weights["alpha_factual_error"] + weights["beta_confidence_gap"] + weights["gamma_consistency_failure"],
        1.0,
        abs_tol=1e-3,
    )


def test_fusion_mask_000_no_pillars_available(fusion_engine):
    """Mask [0,0,0]: All pillars unavailable -> defaults to prior 0.50 and INSUFFICIENT_EVIDENCE."""
    p1 = Pillar1Result(factual_error_score=None, available=False, reasoning="P1 unavailable")
    p2 = Pillar2Result(confidence_gap_score=None, available=False, reasoning="P2 unavailable")
    p3 = Pillar3Result(consistency_failure_score=None, available=False, reasoning="P3 unavailable")

    h_score, risk, color, weights = fusion_engine.fuse(p1, p2, p3)
    assert risk == RiskLevel.INSUFFICIENT_EVIDENCE
    assert h_score == 0.50
    assert weights["alpha_factual_error"] == 0.0
    assert weights["beta_confidence_gap"] == 0.0
    assert weights["gamma_consistency_failure"] == 0.0


# ============================================================================
# SECTION C: CORRECTION BEHAVIOR (9 SCENARIOS)
# ============================================================================

def test_correction_scenario_1_correct_speed_of_light_no_correction(correction_engine):
    """Scenario 1: Correct speed-of-light answer -> VERIFIED, NO correction triggered."""
    cand = correction_engine.generate_candidate(
        query="What is the speed of light in vacuum?",
        original_response="The speed of light in vacuum is approximately 299,792,458 m/s.",
        overall_h_score=0.05,
    )
    assert cand["status"] == "not_needed"
    assert cand["method"] == "none"
    assert cand["corrected_text"] == "The speed of light in vacuum is approximately 299,792,458 m/s."


def test_correction_scenario_2_correct_water_molar_mass_no_correction(correction_engine):
    """Scenario 2: Correct water molar mass -> VERIFIED, NO correction triggered."""
    cand = correction_engine.generate_candidate(
        query="What is the molar mass of water?",
        original_response="The molar mass of water is approximately 18.015 g/mol.",
        overall_h_score=0.08,
    )
    assert cand["status"] == "not_needed"
    assert cand["method"] == "none"
    assert cand["corrected_text"] == "The molar mass of water is approximately 18.015 g/mol."


def test_correction_scenario_3_incorrect_speed_of_light_unit_corrected(correction_engine):
    """Scenario 3: Incorrect speed of light (km/s instead of m/s) -> unit mismatch correction."""
    cand = correction_engine.generate_candidate(
        query="What is the speed of light in vacuum?",
        original_response="The speed of light in vacuum is 299,792,458 km/s.",
        retrieved_evidence=[
            {
                "snippet": "The speed of light in vacuum, commonly denoted c, is exactly 299,792,458 m/s.",
                "score": 0.95,
            }
        ],
        overall_h_score=0.75,
    )
    assert cand["status"] == "candidate_generated"
    assert cand["method"] == "unit_correction"
    assert "299,792,458 m/s" in cand["corrected_text"]


def test_correction_scenario_4_incorrect_water_molar_mass_correction_attempted(correction_engine):
    """Scenario 4: Incorrect water molar mass (180.15 g/mol) -> correction attempted."""
    cand = correction_engine.generate_candidate(
        query="What is the molar mass of water?",
        original_response="The molar mass of water is 180.15 g/mol.",
        retrieved_evidence=[
            {
                "snippet": "Water has a standard molar mass of 18.015 g/mol.",
                "score": 0.95,
            }
        ],
        overall_h_score=0.85,
    )
    assert cand["status"] == "candidate_generated"
    assert cand["corrected_text"] is not None
    assert "18.015" in cand["corrected_text"]


def test_correction_scenario_5_mixed_multi_claim_correction(correction_engine):
    """Scenario 5: Multi-claim with one correct and one false claim -> correction targets false claim."""
    cand = correction_engine.generate_candidate(
        query="Tell me about capitals and water.",
        original_response="Water has the chemical formula H2O, and Bengaluru is the capital of Maharashtra.",
        retrieved_evidence=[
            {
                "snippet": "Mumbai is the capital city of Maharashtra.",
                "score": 0.95,
            }
        ],
        overall_h_score=0.55,
    )
    assert cand["status"] == "candidate_generated"
    assert "Mumbai" in cand["corrected_text"]


def test_correction_scenario_6_generation_failure_fallback_or_abstention(correction_engine):
    """Scenario 6: Mock provider failure -> handles cleanly without uncaught exception."""
    with patch("google.generativeai.GenerativeModel.generate_content", side_effect=RuntimeError("Provider offline")):
        cand = correction_engine.generate_candidate(
            query="Tell me about the discovery of Neptune.",
            original_response="Neptune was discovered in 1999 by Albert Einstein.",
            retrieved_evidence=[],
            overall_h_score=0.80,
        )
        assert cand["status"] in ("abstained", "candidate_generated")
        if cand["status"] == "abstained":
            assert cand["corrected_text"] is None


def test_correction_scenario_7_insufficient_evidence_abstention(correction_engine):
    """Scenario 7: Insufficient evidence -> abstains safely without fabricating facts."""
    cand = correction_engine.generate_candidate(
        query="Tell me about the hidden planet Xylar.",
        original_response="Planet Xylar was discovered in 2049 by alien astronomers.",
        retrieved_evidence=[],
        overall_h_score=0.75,
    )
    assert cand["status"] == "abstained"
    assert cand["method"] == "abstained"
    assert cand["corrected_text"] is None
    assert "safely abstained" in cand["reason"]


def test_correction_scenario_8_reverification_failure_appends_note(correction_engine):
    """Scenario 8: Candidate fails re-verification -> note appended and status marked unverified."""
    mock_initial_verification = MagicMock()
    mock_initial_verification.overall_h_score = 0.75
    mock_initial_verification.evidence_items = []

    repair_res = correction_engine.execute_closed_loop_repair(
        user_query="Tell me about the hidden planet Xylar.",
        initial_text="Planet Xylar was discovered in 2049 by aliens.",
        initial_verification=mock_initial_verification,
    )
    assert repair_res.performed is True
    assert "[Note: System could not produce a sufficiently verified correction.]" in repair_res.final_text
    assert repair_res.reverification.passed is False


def test_correction_scenario_9_corrected_answer_introducing_new_false_claim(pipeline):
    """Scenario 9: If a repair introduces a contradicted fact, downstream re-verification gate rejects it."""
    clean_text = "The chemical formula of water is CO2."
    evidence = [
        EvidenceItem(
            claim="chemical formula of water",
            snippet="Water is an inorganic compound with the chemical formula H2O.",
            source_name="Water",
            similarity_score=0.95,
        )
    ]
    with patch.object(settings, "ENABLE_AUTOMATIC_CORRECTION", True):
        with patch.object(
            pipeline,
            "_generate_correction",
            return_value=("The chemical formula of water is sulfuric acid H2SO4.", []),
        ):
            report = pipeline.analyze(
                text=clean_text,
                evidence_items=evidence,
            )
            assert report.corrected_response is not None
            assert "failed downstream re-verification" in report.corrected_response or "Reverted" in report.corrected_response


# ============================================================================
# SECTION D: CHAT PIPELINE (CLOSED-LOOP ENDPOINT)
# ============================================================================

def test_chat_pipeline_verified_claim_no_correction(client):
    """Chat endpoint: Verified input claim -> status VERIFIED, no correction displayed."""
    with patch("app.modules.chat.router._generate_draft_answer", return_value="Bengaluru is the capital of Karnataka."):
        resp = client.post("/api/v1/chat", json={
            "message": "What is the capital of Karnataka?",
            "enable_verification": True,
            "auto_correct": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["verification"]["status"] == "VERIFIED"
        assert data["correction"]["performed"] is False
        assert data["final_response"] == "Bengaluru is the capital of Karnataka."


def test_chat_pipeline_hallucinated_claim_closed_loop_correction(client):
    """Chat endpoint: Hallucinated input claim -> status CORRECTED, corrected answer returned."""
    with patch("app.modules.chat.router._generate_draft_answer", return_value="12 * 5 = 55"):
        resp = client.post("/api/v1/chat", json={
            "message": "What is 12 multiplied by 5?",
            "enable_verification": True,
            "auto_correct": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["correction"]["performed"] is True
        assert data["verification"]["status"] == "CORRECTED"
        assert "12 * 5 = 60" in data["final_response"]


# ============================================================================
# SECTION E: PROVIDER LOGPROB HANDLING
# ============================================================================

def test_logprobs_passed_p2_available(pipeline):
    """Token probabilities passed -> Pillar 2 is AVAILABLE and has beta weight > 0."""
    report = pipeline.analyze(
        text="Water is H2O.",
        token_probabilities=[0.99, 0.98, 0.95],
        evidence_items=[
            EvidenceItem(
                claim="Water is H2O",
                snippet="Water is chemical compound H2O.",
                source_name="Water",
                similarity_score=0.95,
            )
        ],
    )
    assert report.pillar2_summary.available is True
    assert report.pillar2_summary.confidence_gap_score is not None
    assert report.weights_used["beta_confidence_gap"] > 0.0


def test_no_logprobs_passed_p2_unavailable(pipeline):
    """No token probabilities -> Pillar 2 is UNAVAILABLE, m2=0, beta weight = 0.0, no synthetic probs."""
    report = pipeline.analyze(
        text="Water is H2O.",
        token_probabilities=None,
        evidence_items=[
            EvidenceItem(
                claim="Water is H2O",
                snippet="Water is chemical compound H2O.",
                source_name="Water",
                similarity_score=0.95,
            )
        ],
    )
    assert report.pillar2_summary.available is False
    assert report.weights_used["beta_confidence_gap"] == 0.0


def test_pasted_text_p2_always_unavailable(pipeline):
    """Pasted external text (no provider stream) -> Pillar 2 strictly UNAVAILABLE."""
    pasted_text = "The capital of France is Paris."
    report = pipeline.analyze(
        text=pasted_text,
        token_probabilities=[],  # Empty list simulates pasted user input
        evidence_items=[
            EvidenceItem(
                claim="Paris capital France",
                snippet="Paris is the capital of France.",
                source_name="France",
                similarity_score=0.95,
            )
        ],
    )
    assert report.pillar2_summary.available is False
    assert report.pillar2_summary.status == "UNAVAILABLE"
    assert report.weights_used["beta_confidence_gap"] == 0.0


# ============================================================================
# SECTION F: PILLAR 3 WITH INSUFFICIENT SAMPLES
# ============================================================================

def test_p3_single_sample_insufficient(p3_engine):
    """Single sample identical to primary -> sanitized away, handled as intra-response without crashing."""
    primary = "Albert Einstein developed the theory of relativity."
    res = p3_engine.analyze(primary, [primary])
    assert res.mode in ("SINGLE_CLAIM_CONSISTENCY", "INTRA_RESPONSE_CONSISTENCY", "MULTI_SAMPLE_REQUIRED")


def test_p3_empty_samples_unavailable(pipeline):
    """Empty samples list -> Pillar 3 marked UNAVAILABLE in pipeline analysis."""
    report = pipeline.analyze(
        text="Albert Einstein developed relativity.",
        sample_responses=[],
        evidence_items=[
            EvidenceItem(
                claim="Einstein developed relativity",
                snippet="Albert Einstein developed relativity in 1905.",
                source_name="Physics",
                similarity_score=0.95,
            )
        ],
    )
    assert report.pillar3_summary.available is False
    assert report.pillar3_summary.status == "UNAVAILABLE"
    assert report.weights_used["gamma_consistency_failure"] == 0.0


def test_p3_three_semantically_equivalent_samples_low_cf(p3_engine):
    """3 semantically equivalent samples -> high consensus, low CF (< 0.30)."""
    primary = "The capital of France is Paris."
    samples = [
        "Paris is the capital and largest city of France.",
        "The capital city of France is Paris.",
        "Paris serves as the capital of France.",
    ]
    res = p3_engine.analyze(primary, samples)
    assert res.available is True
    assert res.consistency_failure_score < 0.30, f"Expected CF < 0.30 for equivalent samples, got {res.consistency_failure_score}"


def test_p3_three_contradictory_samples_high_cf(p3_engine):
    """3 contradictory samples -> conflicting assertions, high CF (> 0.60)."""
    primary = "The capital of France is Paris."
    samples = [
        "The capital of France is Berlin.",
        "The capital of France is Rome.",
        "The capital of France is Madrid.",
    ]
    res = p3_engine.analyze(primary, samples)
    assert res.available is True
    assert res.consistency_failure_score > 0.60, f"Expected CF > 0.60 for contradictory samples, got {res.consistency_failure_score}"
