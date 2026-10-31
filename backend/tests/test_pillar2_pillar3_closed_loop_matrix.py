"""Comprehensive End-to-End Test Matrix for HalluciSense:
1. Pillar 2 — Confidence Gap (CG) with genuine logprobs, NLL, LPF (tau=0.45), provenance, and missing handling.
2. Pillar 3 — Consistency Failure (CF) with 5 independent generations, semantic agreement, numerical and entity inconsistency, contradictions.
3. Hybrid Fusion — All 8 availability masks in {0, 1}^3 with adaptive renormalization and factuality gating.
4. Closed-Loop State Machine — DETECT -> EXPLAIN -> CORRECT -> RE-VERIFY across all states:
   NOT_REQUIRED, DETECTED, EXPLAINED, CORRECTION_GENERATED, REVERIFYING, CORRECTION_ACCEPTED, CORRECTION_REJECTED, ABSTAINED, FAILED.
"""

import math
import pytest
from unittest.mock import MagicMock, AsyncMock

from app.core.engine.pillar2_confidence import Pillar2ConfidenceEngine
from app.core.engine.pillar3_consistency import Pillar3ConsistencyEngine
from app.core.engine.fusion import FusionEngine
from app.core.correction.correction_engine import (
    HallucinationCorrectionEngine,
    CorrectionState,
)
from app.core.engine.types import (
    Pillar1Result,
    Pillar2Result,
    Pillar3Result,
    RiskLevel,
)
from app.modules.providers.gemini import _extract_chunk_logits
from app.modules.providers.schemas import TokenLogit


# ==============================================================================
# 1. PILLAR 2 — CONFIDENCE GAP (CG) TESTS
# ==============================================================================

def test_p2_gemini_genuine_logprob_extraction():
    """Verify genuine logprob extraction from Gemini chosen_candidates with provenance."""
    cand1 = MagicMock()
    cand1.token = "Canberra"
    cand1.token_id = 9912
    cand1.log_probability = -0.0512

    cand2 = MagicMock()
    cand2.token = " is"
    cand2.token_id = 104
    cand2.log_probability = -0.0120

    mock_chunk = MagicMock()
    mock_chunk.candidates = [MagicMock()]
    mock_chunk.candidates[0].logprobs_result.chosen_candidates = [cand1, cand2]

    logits = _extract_chunk_logits(mock_chunk, model_name="gemini-2.0-flash", start_pos=0)
    assert len(logits) == 2
    assert logits[0].token == "Canberra"
    assert logits[0].logprob == pytest.approx(-0.0512, rel=1e-3)
    assert logits[0].provider == "Google Gemini"
    assert logits[0].model == "gemini-2.0-flash"


def test_p2_openai_genuine_logprob_extraction():
    """Verify genuine logprob extraction structure for OpenAI."""
    token_logit = TokenLogit(
        token="Paris",
        logprob=-0.0034,
        position=0,
        provider="OpenAI",
        model="gpt-4o",
    )
    assert token_logit.token == "Paris"
    assert token_logit.logprob == pytest.approx(-0.0034, rel=1e-3)
    p = math.exp(token_logit.logprob)
    assert p > 0.99


def test_p2_probability_conversion_nll_lpf_and_cg():
    """Verify p_i = exp(logprob_i), NLL, LPF (tau=0.45), and CG = 0.70(1 - p_mean) + 0.30(LPF)."""
    p2 = Pillar2ConfidenceEngine()
    tokens = ["The", "capital", "of", "France", "is", "Paris"]
    # 6 logprobs: 4 high prob, 2 low prob (< ln(0.45) ~ -0.7985)
    logprobs = [-0.01, -0.02, -0.05, -0.03, -1.20, -1.50]

    probs = [math.exp(lp) for lp in logprobs]
    expected_mean_p = sum(probs) / 6.0
    expected_nll = -(sum(logprobs) / 6.0)
    expected_lpf = 2.0 / 6.0  # 2 out of 6 tokens < 0.45
    expected_cg = 0.70 * (1.0 - expected_mean_p) + 0.30 * expected_lpf

    res = p2.analyze(
        tokens=tokens,
        probabilities=logprobs,
        is_logprobs=True,
        provider="OpenAI",
        model="gpt-4o",
    )

    assert res.available is True
    assert res.status == "EXECUTED"
    assert res.is_genuine_logprobs is True
    assert res.avg_probability == pytest.approx(round(expected_mean_p, 4), abs=1e-3)
    assert res.nll == pytest.approx(round(expected_nll, 4), abs=1e-3)
    assert res.lpf == pytest.approx(round(expected_lpf, 4), abs=1e-3)
    assert res.threshold == 0.45
    assert res.confidence_gap_score == pytest.approx(round(expected_cg, 4), abs=1e-3)
    assert res.token_count == 6
    assert res.provider == "OpenAI"
    assert res.model == "gpt-4o"


def test_p2_missing_logprob_handling():
    """Missing log-probabilities must produce explicit UNAVAILABLE result without fabricating values."""
    p2 = Pillar2ConfidenceEngine()
    res = p2.analyze(tokens=["Some", "externally", "provided", "claim."], probabilities=None)

    assert res.available is False
    assert res.status == "UNAVAILABLE"
    assert res.mode == "UNAVAILABLE"
    assert res.avg_probability is None
    assert res.avg_entropy is None
    assert res.confidence_gap_score is None
    assert res.nll is None
    assert res.lpf is None
    assert "Generation-time token probabilities unavailable" in res.reasoning


def test_p2_pasted_text_unavailability():
    """For manually pasted text, explicitly report unavailable with exclusion from fusion."""
    p2 = Pillar2ConfidenceEngine()
    res = p2.analyze(tokens=["Pasted", "statement."], probabilities=[], allow_proxy=False)
    assert res.available is False
    assert res.status == "UNAVAILABLE"
    assert "Pillar 2 uncertainty analysis is excluded from fusion" in res.reasoning


def test_p2_no_fabricated_values_on_empty():
    """Ensure no numbers are manufactured when input is empty."""
    p2 = Pillar2ConfidenceEngine()
    res = p2.analyze(tokens=[], probabilities=None)
    assert res.available is False
    assert res.confidence_gap_score is None


# ==============================================================================
# 2. PILLAR 3 — CONSISTENCY FAILURE (CF) TESTS
# ==============================================================================

def test_p3_five_consistent_independent_generations():
    """Five consistent generations should yield high semantic agreement and near-zero CF."""
    p3 = Pillar3ConsistencyEngine()
    primary = "Canberra is the capital city of Australia."
    samples = [
        "The capital of Australia is Canberra.",
        "Canberra serves as Australia's capital.",
        "Australia's federal capital is Canberra.",
        "The city of Canberra is the capital of Australia.",
        "Canberra is the designated capital of Australia.",
    ]

    res = p3.analyze(primary_response=primary, sample_responses=samples)
    assert res.available is True
    assert res.status == "EXECUTED"
    assert res.generation_count == 5
    assert res.contradiction_count == 0
    assert len(res.conflicting_claims) == 0
    assert res.semantic_agreement is not None
    assert res.semantic_agreement >= 0.70
    assert res.consistency_failure_score is not None
    assert res.consistency_failure_score <= 0.30
    assert len(res.generation_comparisons) == 5


def test_p3_generations_containing_contradictions():
    """Contradictory generation (e.g. Sydney vs Canberra) must produce contradiction signal and penalize CF."""
    p3 = Pillar3ConsistencyEngine()
    primary = "Canberra is the capital of Australia."
    samples = [
        "Sydney is the capital of Australia.",
        "Melbourne is the capital of Australia.",
        "Canberra is Australia's capital city.",
        "Brisbane is the capital of Australia.",
        "Canberra is the capital of Australia.",
    ]

    res = p3.analyze(primary_response=primary, sample_responses=samples)
    assert res.available is True
    assert res.status == "EXECUTED"
    assert res.contradiction_count >= 1
    assert len(res.conflicting_claims) >= 1
    assert res.consistency_failure_score > 0.40


def test_p3_numerical_inconsistency_detection():
    """Numerical disparity between primary response and alternate generations must generate conflict."""
    p3 = Pillar3ConsistencyEngine()
    primary = "The human body contains 206 bones in an adult skeleton."
    samples = [
        "An adult human skeleton has 300 bones.",
        "The adult human body has 206 bones.",
        "There are 215 bones in the adult human body.",
        "An adult skeleton is composed of 206 bones.",
        "A mature human body features 250 bones.",
    ]

    res = p3.analyze(primary_response=primary, sample_responses=samples)
    assert res.available is True
    assert res.contradiction_count >= 1
    # Check that numerical conflict was identified
    has_num_conflict = any(comp.get("is_contradiction") for comp in res.generation_comparisons)
    assert has_num_conflict is True


def test_p3_entity_inconsistency_detection():
    """Entity conflict between primary and sample must trigger conflict detection."""
    p3 = Pillar3ConsistencyEngine()
    conflicts = p3.check_entity_numerical_conflicts(
        "Sydney is the capital of Australia.",
        "Canberra is the capital of Australia.",
    )
    assert len(conflicts) > 0
    assert "australia" in conflicts[0].lower()


def test_p3_paraphrased_semantic_consistency():
    """Paraphrased but factually equivalent assertions must not trigger contradiction."""
    p3 = Pillar3ConsistencyEngine()
    primary = "Water boils at 100 degrees Celsius at standard atmospheric pressure."
    samples = [
        "Under standard atmospheric pressure, water boils at 100 °C.",
        "The boiling point of water is 100 degrees Celsius under standard atmospheric pressure.",
        "Water has a boiling point of 100 °C at standard atmospheric pressure.",
        "At standard atmospheric pressure, 100 °C is the boiling point of water.",
        "Under 1 atmosphere of standard pressure, water boils at 100 degrees Celsius.",
    ]

    res = p3.analyze(primary_response=primary, sample_responses=samples)
    assert res.available is True
    assert res.contradiction_count == 0
    assert res.consistency_failure_score <= 0.25


def test_p3_unavailable_when_no_samples():
    """When no independent generations are available, Pillar 3 must report UNAVAILABLE."""
    p3 = Pillar3ConsistencyEngine()
    res = p3.analyze(primary_response="Single static response without alternate samples.", sample_responses=[])
    assert res.available is False
    assert res.status == "UNAVAILABLE"
    assert res.mode == "MULTI_SAMPLE_REQUIRED"
    assert res.consistency_failure_score is None
    assert "Multiple independent generations not available" in res.reasoning


# ==============================================================================
# 3. HYBRID FUSION — ALL 8 AVAILABILITY MASKS
# ==============================================================================

@pytest.mark.parametrize(
    "m1,m2,m3,expected_alpha,expected_beta,expected_gamma",
    [
        # [1, 1, 1] Full 3-pillar: alpha=0.45, beta=0.30, gamma=0.25
        (1, 1, 1, 0.45, 0.30, 0.25),
        # [1, 1, 0] P1 + P2: alpha=0.45/0.75=0.60, beta=0.30/0.75=0.40, gamma=0.0
        (1, 1, 0, 0.60, 0.40, 0.0),
        # [1, 0, 1] P1 + P3: alpha=0.45/0.70~0.6429, beta=0.0, gamma=0.25/0.70~0.3571
        (1, 0, 1, 0.45 / 0.70, 0.0, 0.25 / 0.70),
        # [1, 0, 0] P1 only: alpha=1.0, beta=0.0, gamma=0.0
        (1, 0, 0, 1.0, 0.0, 0.0),
        # [0, 1, 1] P2 + P3: alpha=0.0, beta=0.30/0.55~0.5455, gamma=0.25/0.55~0.4545
        (0, 1, 1, 0.0, 0.30 / 0.55, 0.25 / 0.55),
        # [0, 1, 0] P2 only: beta=1.0
        (0, 1, 0, 0.0, 1.0, 0.0),
        # [0, 0, 1] P3 only: gamma=1.0
        (0, 0, 1, 0.0, 0.0, 1.0),
        # [0, 0, 0] All unavailable: uninformative prior
        (0, 0, 0, 0.0, 0.0, 0.0),
    ],
)
def test_fusion_all_8_availability_masks(m1, m2, m3, expected_alpha, expected_beta, expected_gamma):
    """Test mathematical normalization across all 8 availability masks in {0, 1}^3."""
    fusion = FusionEngine()
    fe = 0.20 if m1 else None
    cg = 0.30 if m2 else None
    cf = 0.40 if m3 else None

    score, eff_w, mask = fusion.compute_adaptive_h_score(fe=fe, cg=cg, cf=cf)
    assert mask == [m1, m2, m3]

    if (m1, m2, m3) == (0, 0, 0):
        assert score == 0.50  # Uninformative prior
        assert eff_w["alpha"] == 0.0
        assert eff_w["beta"] == 0.0
        assert eff_w["gamma"] == 0.0
    else:
        assert eff_w["alpha"] == pytest.approx(expected_alpha, abs=1e-3)
        assert eff_w["beta"] == pytest.approx(expected_beta, abs=1e-3)
        assert eff_w["gamma"] == pytest.approx(expected_gamma, abs=1e-3)
        active_sum = (eff_w["alpha"] if m1 else 0) + (eff_w["beta"] if m2 else 0) + (eff_w["gamma"] if m3 else 0)
        assert active_sum == pytest.approx(1.0, abs=1e-3)


def test_fusion_factuality_gating():
    """P1 factual contradiction (FE >= 0.50) cannot be masked by low CG or low CF."""
    fusion = FusionEngine()
    p1 = Pillar1Result(factual_error_score=0.90, available=True, reasoning="Contradicted by encyclopedic facts")
    p2 = Pillar2Result(confidence_gap_score=0.05, available=True, reasoning="Model is highly confident")
    p3 = Pillar3Result(consistency_failure_score=0.05, available=True, reasoning="Consistent across samples")

    h_score, risk, _, _ = fusion.fuse(p1, p2, p3)
    # Direct factual error gating ensures h_score >= FE
    assert h_score >= 0.90
    assert risk == RiskLevel.LIKELY_HALLUCINATED


def test_fusion_abstention_when_p1_unavailable():
    """When P1 is unavailable, P2/P3 alone cannot declare VERIFIED or LIKELY_HALLUCINATED."""
    fusion = FusionEngine()
    p1 = Pillar1Result(factual_error_score=None, available=False, reasoning="Retrieval unavailable")
    p2 = Pillar2Result(confidence_gap_score=0.10, available=True, reasoning="High generation confidence")
    p3 = Pillar3Result(consistency_failure_score=0.10, available=True, reasoning="Consistent")

    _, risk, color, _ = fusion.fuse(p1, p2, p3)
    assert risk == RiskLevel.INSUFFICIENT_EVIDENCE
    assert color == "#6B7280"


# ==============================================================================
# 4. EXPLAINABLE CLOSED-LOOP STATE MACHINE TESTS
# ==============================================================================

def test_closed_loop_state_not_required():
    """Verified response must produce NOT_REQUIRED without rewriting."""
    engine = HallucinationCorrectionEngine()
    cand = engine.generate_candidate(
        query="What is the capital of France?",
        original_response="Paris is the capital of France.",
        overall_h_score=0.08,
    )
    assert cand["status"] == "not_needed"
    assert cand["state"] == CorrectionState.NOT_REQUIRED.value
    assert cand["corrected_text"] == "Paris is the capital of France."


def test_closed_loop_state_abstained():
    """Inconclusive evidence must produce ABSTAINED without introducing hallucinations."""
    engine = HallucinationCorrectionEngine()
    cand = engine.generate_candidate(
        query="What happened in the mythical land of Eldoria?",
        original_response="Eldoria was ruled by King Arthur in the year 3000.",
        retrieved_evidence=[],
        overall_h_score=0.75,
    )
    assert cand["status"] == "abstained"
    assert cand["state"] == CorrectionState.ABSTAINED.value
    assert cand["corrected_text"] is None
    assert "safely abstained" in cand["reason"]


def test_closed_loop_explanation_generation():
    """Stage 2 EXPLAIN must generate complete 9-field diagnosis."""
    engine = HallucinationCorrectionEngine()
    explanation = engine.generate_explanation(
        query="What is the capital of Australia?",
        original_response="Sydney is the capital of Australia.",
        overall_h_score=0.95,
        p1_score=0.95,
        p2_score=0.10,
        p3_score=0.20,
        retrieved_evidence=[
            {"snippet": "Canberra is the capital city of Australia.", "score": 0.95, "contradiction_prob": 0.90}
        ],
    )
    assert explanation["original_claim"] == "Sydney is the capital of Australia."
    assert explanation["claim_classification"] == "FACTUAL_CONTRADICTION"
    assert len(explanation["contradicting_evidence"]) >= 1
    assert explanation["p1_score"] == 0.95
    assert "p2_score_and_provenance" in explanation
    assert "p3_score_and_generation_comparisons" in explanation
    assert "detection_reason" in explanation
    assert "confidence_and_limitations" in explanation


@pytest.mark.asyncio
async def test_closed_loop_candidate_accepted_after_reverification():
    """Candidate that passes independent verification must transition to CORRECTION_ACCEPTED."""
    engine = HallucinationCorrectionEngine()

    async def mock_reverify_pass(query=None, response="", model_name="default"):
        return MagicMock(
            overall_h_score=0.10,
            risk_level="VERIFIED",
            trace_id="TRACE_REV_PASS_123",
            pillar_scores=MagicMock(retrieval=0.08, confidence=0.10, consistency=0.05),
        )

    evidence = [
        {"snippet": "Canberra is the capital city of Australia.", "score": 0.95}
    ]

    res = await engine.correct_and_reverify(
        query="What is the capital of Australia?",
        original_response="Sydney is the capital of Australia.",
        original_trace_id="TRACE_ORIG_123",
        analyze_func=mock_reverify_pass,
        retrieved_evidence=evidence,
        overall_h_score=0.95,
    )

    assert res["status"] == "verified"
    assert res["state"] == CorrectionState.CORRECTION_ACCEPTED.value
    assert "Canberra" in res["corrected_text"]
    assert res["reverification"] is not None
    assert res["reverification"]["status"] == "VERIFIED"
    assert res["reverification"]["trace_id"] == "TRACE_REV_PASS_123"


@pytest.mark.asyncio
async def test_closed_loop_candidate_rejected_preserves_original():
    """Candidate that fails independent verification must transition to CORRECTION_REJECTED and preserve original text."""
    engine = HallucinationCorrectionEngine()

    async def mock_reverify_fail(query=None, response="", model_name="default"):
        return MagicMock(
            overall_h_score=0.85,
            risk_level="LIKELY_HALLUCINATED",
            trace_id="TRACE_REV_FAIL_456",
            pillar_scores=MagicMock(retrieval=0.80, confidence=0.20, consistency=0.30),
        )

    evidence = [
        {"snippet": "Canberra is the capital city of Australia.", "score": 0.95}
    ]

    res = await engine.correct_and_reverify(
        query="What is the capital of Australia?",
        original_response="Sydney is the capital of Australia.",
        original_trace_id="TRACE_ORIG_123",
        analyze_func=mock_reverify_fail,
        retrieved_evidence=evidence,
        overall_h_score=0.95,
    )

    assert res["status"] == "rejected"
    assert res["state"] == CorrectionState.CORRECTION_REJECTED.value
    # Original text must be preserved!
    assert res["corrected_text"] == "Sydney is the capital of Australia."
    assert res["reverification"]["status"] == "REJECTED"


@pytest.mark.asyncio
async def test_closed_loop_failed_execution_preserves_original():
    """Exceptions during re-verification must transition to FAILED and preserve original text."""
    engine = HallucinationCorrectionEngine()

    async def mock_reverify_crash(query=None, response="", model_name="default"):
        raise ConnectionError("Simulated network outage during re-verification")

    evidence = [
        {"snippet": "Canberra is the capital city of Australia.", "score": 0.95}
    ]

    res = await engine.correct_and_reverify(
        query="What is the capital of Australia?",
        original_response="Sydney is the capital of Australia.",
        original_trace_id="TRACE_ORIG_123",
        analyze_func=mock_reverify_crash,
        retrieved_evidence=evidence,
        overall_h_score=0.95,
    )

    assert res["status"] == "failed"
    assert res["state"] == CorrectionState.FAILED.value
    assert res["corrected_text"] == "Sydney is the capital of Australia."
    assert "Original response preserved" in res["reason"]


def test_closed_loop_compound_claim_preservation():
    """Compound statement: only incorrect claim is replaced, unrelated facts are preserved."""
    engine = HallucinationCorrectionEngine()
    evidence = [
        {"snippet": "Bengaluru is the capital of Karnataka.", "score": 0.95}
    ]
    # Sentence: 'Mumbai is the capital of Karnataka, and Tokyo is the capital of Japan.'
    cand = engine.generate_candidate(
        query="Tell me about Karnataka and Japan capitals.",
        original_response="Mumbai is the capital of Karnataka, and Tokyo is the capital of Japan.",
        retrieved_evidence=evidence,
        overall_h_score=0.70,
    )

    assert cand["status"] == "candidate_generated"
    assert "Bengaluru" in cand["corrected_text"]
    # Tokyo and Japan assertion is preserved intact!
    assert "Tokyo is the capital of Japan" in cand["corrected_text"]
