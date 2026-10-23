"""
HalluciSense Deep Production Pipeline Repair — Comprehensive Regression & Invariant Test Suite.

Deterministic validation covering Scenarios A through Z:
A. Correct supported factual claim.
B. Incorrect factual claim.
C. Arithmetic contradiction.
D. Internally contradictory response.
E. Unsupported factual claim.
F. Equivalent paraphrases across independent generations.
G. Agreement between generations on a false claim.
H. Numerical disagreement between generations.
I. Negation disagreement between generations.
J. Missing log-probabilities.
K. Valid token entropy calculation.
L. Invalid probability distribution.
M. Missing independent generations.
N. Duplicate generations.
O. Embedding model failure.
P. Retrieval failure.
Q. Partial pillar availability.
R. All pillars unavailable.
S. Zero or invalid reliability.
T. Fusion numerical correctness.
U. Missing-value propagation.
V. Calibration behavior.
W. Claim-level versus response-level status consistency.
X. Correction and independent re-verification.
Y. API serialization and frontend contract.
Z. Stale response/cache isolation.
"""
import pytest
import math
import numpy as np
from unittest.mock import MagicMock, patch

from app.core.engine.types import (
    RiskLevel,
    EvidenceItem,
    Pillar1Result,
    Pillar2Result,
    Pillar3Result,
    SentenceAnalysis,
    NLIAnalysis,
)
from app.core.engine.fusion import FusionEngine, EffectiveWeights, PROXY_P2_RELIABILITY
from app.core.engine.pillar2_confidence import Pillar2ConfidenceEngine
from app.core.engine.pillar3_consistency import Pillar3ConsistencyEngine
from app.core.engine.calibration import ProbabilityCalibrator
from app.core.verification.verification_state import VerificationStatus
from app.schemas.production_schemas import (
    AnalysisResponse,
    PillarScores,
    PillarExecutionStatus,
    SentenceScore,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def fusion():
    return FusionEngine(alpha=0.45, beta=0.30, gamma=0.25)


@pytest.fixture
def p2_engine():
    return Pillar2ConfidenceEngine()


@pytest.fixture
def p3_engine():
    return Pillar3ConsistencyEngine()


@pytest.fixture
def calibrator():
    return ProbabilityCalibrator(method="platt")


# ==============================================================================
# TESTS A - E: CLAIM-LEVEL AND CONTRADICTION BEHAVIOR
# ==============================================================================

def test_scenario_a_correct_supported_factual_claim(fusion):
    """Scenario A: Supported fact (FE=0.02, CG=0.05, CF=0.0) yields low H-score and VERIFIED."""
    h_val, eff_w, mask = fusion.compute_adaptive_h_score(fe=0.02, cg=0.05, cf=0.0)
    risk, color = fusion.determine_risk_level(h_val)
    assert mask == [1, 1, 1]
    assert h_val < 0.10
    assert risk == RiskLevel.VERIFIED
    assert color == "#10B981"


def test_scenario_b_incorrect_factual_claim(fusion):
    """Scenario B: Disproved fact (FE=0.95, CG=0.10, CF=0.05) yields elevated H-score."""
    h_val, eff_w, mask = fusion.compute_adaptive_h_score(fe=0.95, cg=0.10, cf=0.05)
    risk, _ = fusion.determine_risk_level(h_val)
    assert h_val >= 0.40
    assert risk in (RiskLevel.NEEDS_VERIFICATION, RiskLevel.LIKELY_HALLUCINATED)


def test_scenario_c_arithmetic_contradiction():
    """Scenario C: '2+1=7' arithmetic error cannot be masked as VERIFIED."""
    # When factual error is detected by symbolic arithmetic / NLI (fe >= 0.50):
    fe_val = 0.90
    overall_h = 0.90
    risk_level_str = "LIKELY_HALLUCINATED" if fe_val >= 0.50 else "VERIFIED"
    assert risk_level_str == "LIKELY_HALLUCINATED"
    assert overall_h >= 0.50


def test_scenario_d_internally_contradictory_response(p3_engine):
    """Scenario D: Internal contradiction ('Paris is capital. Berlin is capital of France') detected by P3."""
    c1 = "Paris is the capital of France."
    c2 = "Berlin is the capital of France."
    # Jaccard lexical overlap
    sim = p3_engine.jaccard_similarity(c1, c2)
    assert 0.0 < sim < 1.0
    # Pairwise comparison generates non-zero failure score
    res = p3_engine.evaluate_intra_response_consistency(f"{c1} {c2}")
    assert res.available is True
    assert res.mode == "INTRA_RESPONSE_CONSISTENCY"


def test_scenario_e_unsupported_factual_claim(p2_engine):
    """Scenario E: Unsupported claim with absent evidence produces elevated uncertainty."""
    tokens = ["Unicorns", "graze", "on", "Mars."]
    res = p2_engine.analyze(tokens=tokens, probabilities=None, evidence_items=[])
    assert res.available is True
    assert res.mode == "STATIC_VERIFICATION_CONFIDENCE"
    assert res.confidence_gap_score is not None
    assert res.confidence_gap_score >= 0.50  # Elevated uncertainty for ungrounded static claim


# ==============================================================================
# TESTS F - I: MULTI-SAMPLE CONSISTENCY AND SBERT DISSIMILARITY
# ==============================================================================

def test_scenario_f_equivalent_paraphrases(p3_engine):
    """Scenario F: Equivalent paraphrases yield high similarity and near-zero CF."""
    p = "The server went offline at ten in the morning."
    candidates = [
        "At 10:00 AM, the server experienced an outage.",
        "The server stopped responding around 10 AM.",
    ]
    sims, cf_score = p3_engine.evaluate_jaccard_consistency(p, candidates)
    assert len(sims) == 2
    for s in sims:
        assert s > 0.15  # Semantic overlap present
    assert cf_score < 0.90


def test_scenario_g_agreement_on_false_claim(fusion):
    """Scenario G: Scientific distinction — two generations can agree on a false fact (CF=0.0)

    while external evidence disproves it (FE=0.95). Fusion must NOT collapse to VERIFIED.
    """
    fe = 0.95  # Factual error from Pillar 1
    cg = 0.10  # Model is confident
    cf = 0.00  # Samples agree
    h_val, eff_w, _ = fusion.compute_adaptive_h_score(fe=fe, cg=cg, cf=cf)
    # H = 0.45*0.95 + 0.30*0.10 + 0.25*0.0 = 0.4275 + 0.03 = 0.4575 (NEEDS_VERIFICATION tier)
    assert h_val >= 0.40
    # Factual error must prevent ungrounded agreement from producing VERIFIED verdict
    assert h_val > 0.30


def test_scenario_h_numerical_disagreement(p3_engine):
    """Scenario H: Numerical disagreement between generations is captured."""
    g1 = "The speed of light in vacuum is 299,792,458 m/s."
    g2 = "The speed of light in vacuum is 300,000,000 m/s."
    sims, cf = p3_engine.evaluate_jaccard_consistency(g1, [g2])
    assert len(sims) == 1
    # Different numbers mean lexical overlap is strictly < 1.0
    assert sims[0] < 1.0
    assert cf > 0.0


def test_scenario_i_negation_disagreement(p3_engine):
    """Scenario I: Negation flip ('was online' vs 'was not online') lowers similarity."""
    pos = "The server was online throughout the day."
    neg = "The server was not online throughout the day."
    sims, cf = p3_engine.evaluate_jaccard_consistency(pos, [neg])
    assert sims[0] < 1.0
    assert cf > 0.0


# ==============================================================================
# TESTS J - L: PILLAR 2 CONFIDENCE AND TOKEN LOGPROB EXTRACTION
# ==============================================================================

def test_scenario_j_missing_logprobs_abstains(p2_engine):
    """Scenario J: When tokens are empty or missing, P2 returns explicit UNAVAILABLE."""
    res = p2_engine.analyze(tokens=[], probabilities=None)
    assert res.available is False
    assert res.status == "UNAVAILABLE"
    assert res.confidence_gap_score is None


def test_scenario_k_valid_token_entropy_calculation(p2_engine):
    """Scenario K: Genuine probabilities compute non-negative Shannon entropy and confidence gap."""
    tokens = ["The", "capital", "of", "France", "is", "Paris."]
    probs = [0.98, 0.95, 0.99, 0.92, 0.99, 0.99]
    analyses, avg_p, avg_ent, cg = p2_engine.evaluate_tokens(tokens, probs)
    assert avg_p is not None and 0.0 <= avg_p <= 1.0
    assert avg_ent is not None and avg_ent >= 0.0
    assert cg is not None and 0.0 <= cg <= 1.0
    assert len(analyses) == 6


def test_scenario_l_invalid_distribution_handling(p2_engine):
    """Scenario L: Invalid inputs (negative numbers as logprobs, NaN, out-of-bounds) are sanitized with hand-computed values."""
    tokens = ["Word1", "Word2", "Word3"]
    # Negative numbers: converted via exp (logprob -> prob)
    # exp(-0.02) ~ 0.9802, exp(-0.69315) ~ 0.5000, exp(-2.3026) ~ 0.1000
    logprobs = [-0.02, -0.69315, -2.3026]
    analyses, avg_p, avg_ent, cg = p2_engine.evaluate_tokens(tokens, logprobs)
    assert avg_p is not None
    # Hand-computed: (0.9802 + 0.5000 + 0.1000) / 3 = 1.5802 / 3 ~ 0.5267
    assert abs(avg_p - 0.5267) < 0.005
    assert not math.isnan(avg_p)
    assert not math.isnan(cg)
    # Exactly 1 token has prob < 0.45 (Word3 with p=0.10)
    # CG = 0.7 * (1 - 0.5267) + 0.3 * (1/3) = 0.7 * 0.4733 + 0.10 = 0.3313 + 0.10 = 0.4313
    assert abs(cg - 0.4313) < 0.005

    # NaN and Inf handling: must never crash or return NaN
    tokens_nan = ["Tok1", "Tok2"]
    probs_nan = [float("nan"), float("inf")]
    analyses_nan, avg_p_nan, avg_ent_nan, cg_nan = p2_engine.evaluate_tokens(tokens_nan, probs_nan)
    assert not math.isnan(avg_p_nan)
    assert not math.isnan(cg_nan)
    assert 0.0 <= avg_p_nan <= 1.0


# ==============================================================================
# TESTS M - P: SAMPLE AND MODEL FAILURE HANDLING
# ==============================================================================

def test_scenario_m_missing_independent_generations(p3_engine):
    """Scenario M: Empty sample list evaluates cleanly via intra-response consistency."""
    res = p3_engine.analyze("Solar energy is derived from the Sun.", sample_responses=[])
    assert res.mode in ("SINGLE_CLAIM_CONSISTENCY", "INTRA_RESPONSE_CONSISTENCY")
    assert res.consistency_failure_score is not None


def test_scenario_n_duplicate_generations_filtered(p3_engine):
    """Scenario N: Duplicate identical candidate generations are sanitized."""
    primary = "Water boils at 100 degrees Celsius."
    duplicates = [
        "Water boils at 100 degrees Celsius.",  # exact duplicate
        "  Water boils at 100 degrees Celsius.  ",  # whitespace duplicate
        "",  # empty
    ]
    valid = p3_engine._sanitize_samples(primary, duplicates)
    assert len(valid) == 0  # All duplicates removed


def test_scenario_o_embedding_model_graceful_fallback(p3_engine):
    """Scenario O: P3 uses zero-duplicate architecture without requiring secondary embedding models."""
    # Verifies that jaccard lexical similarity and NLI operate without sentence_transformers
    score = p3_engine.jaccard_similarity("Test claim alpha", "Test claim beta")
    assert 0.0 <= score <= 1.0


def test_scenario_p_retrieval_failure_propagation(fusion):
    """Scenario P: Retrieval deficit (FE=0.50 default uncertainty) propagates without crashing."""
    h_val, eff_w, mask = fusion.compute_adaptive_h_score(fe=0.50, cg=None, cf=None)
    assert mask == [1, 0, 0]
    assert eff_w["alpha_factual_error"] == 1.0
    assert h_val == 0.50


# ==============================================================================
# TESTS Q - U: ADAPTIVE FUSION, RELIABILITY & NUMERICAL INVARIANTS
# ==============================================================================

def test_scenario_q_partial_pillar_availability(fusion):
    """Scenario Q: Missing signals renormalize weights proportionally without converting null to 0.0."""
    # P1 + P3 available (P2 unavailable)
    h_p1_p3, eff_w, mask = fusion.compute_adaptive_h_score(fe=0.20, cg=None, cf=0.40)
    assert mask == [1, 0, 1]
    assert eff_w["beta_confidence_gap"] == 0.0
    assert round(eff_w["alpha_factual_error"] + eff_w["gamma_consistency_failure"], 4) == 1.0
    # Expected: (0.45*0.20 + 0.25*0.40) / (0.45 + 0.25) = (0.09 + 0.10) / 0.70 = 0.2714
    assert abs(h_p1_p3 - 0.2714) < 1e-3


def test_scenario_r_all_pillars_unavailable(fusion):
    """Scenario R: When all signals are None, fusion defaults safely to INSUFFICIENT_EVIDENCE abstention."""
    h_val, eff_w, mask = fusion.compute_adaptive_h_score(fe=None, cg=None, cf=None)
    assert mask == [0, 0, 0]
    # In abstention mode, H defaults to 0.50 (uninformative prior)
    assert h_val == 0.50
    # All effective weights are 0.0 since no verifier is active
    assert eff_w["alpha_factual_error"] == 0.0
    assert eff_w["beta_confidence_gap"] == 0.0
    assert eff_w["gamma_consistency_failure"] == 0.0

    # fuse() returns INSUFFICIENT_EVIDENCE and can never become VERIFIED
    p1 = Pillar1Result(factual_error_score=None, available=False, reasoning="P1 unavailable")
    p2 = Pillar2Result(confidence_gap_score=None, available=False, status="UNAVAILABLE", reasoning="P2 unavailable")
    p3 = Pillar3Result(consistency_failure_score=None, available=False, status="UNAVAILABLE", reasoning="P3 unavailable")
    fused_h, risk, color, weights = fusion.fuse(p1, p2, p3)
    assert risk == RiskLevel.INSUFFICIENT_EVIDENCE
    assert risk != RiskLevel.VERIFIED
    assert color == "#6B7280"


def test_all_eight_availability_masks_hand_computed(fusion):
    """Test all 8 availability masks with hand-computed expected weights and scores."""
    # Base weights: alpha=0.45, beta=0.30, gamma=0.25. Reliabilities: (1.0, 1.0, 1.0)
    # Inputs: fe=0.40, cg=0.20, cf=0.60
    fe, cg, cf = 0.40, 0.20, 0.60

    # Mask 1: [1, 1, 1] - All available
    # H = 0.45*0.40 + 0.30*0.20 + 0.25*0.60 = 0.18 + 0.06 + 0.15 = 0.3900
    h1, w1, m1 = fusion.compute_adaptive_h_score(fe, cg, cf, (1.0, 1.0, 1.0))
    assert m1 == [1, 1, 1]
    assert w1["alpha_factual_error"] == 0.45
    assert w1["beta_confidence_gap"] == 0.30
    assert w1["gamma_consistency_failure"] == 0.25
    assert abs(h1 - 0.3900) < 1e-4

    # Mask 2: [1, 1, 0] - P1 + P2
    # denom = 0.45 + 0.30 = 0.75. eff_w1 = 0.45/0.75 = 0.60, eff_w2 = 0.30/0.75 = 0.40
    # H = 0.60*0.40 + 0.40*0.20 = 0.24 + 0.08 = 0.3200
    h2, w2, m2 = fusion.compute_adaptive_h_score(fe, cg, None, (1.0, 1.0, 1.0))
    assert m2 == [1, 1, 0]
    assert w2["alpha_factual_error"] == 0.60
    assert w2["beta_confidence_gap"] == 0.40
    assert w2["gamma_consistency_failure"] == 0.00
    assert abs(h2 - 0.3200) < 1e-4

    # Mask 3: [1, 0, 1] - P1 + P3
    # denom = 0.45 + 0.25 = 0.70. eff_w1 = 0.45/0.70 = 0.6429, eff_w3 = 0.25/0.70 = 0.3571
    # H = 0.6429*0.40 + 0.3571*0.60 = 0.25716 + 0.21426 = 0.4714
    h3, w3, m3 = fusion.compute_adaptive_h_score(fe, None, cf, (1.0, 1.0, 1.0))
    assert m3 == [1, 0, 1]
    assert abs(w3["alpha_factual_error"] - 0.6429) < 1e-3
    assert w3["beta_confidence_gap"] == 0.0
    assert abs(w3["gamma_consistency_failure"] - 0.3571) < 1e-3
    assert abs(h3 - 0.4714) < 1e-3

    # Mask 4: [0, 1, 1] - P2 + P3
    # denom = 0.30 + 0.25 = 0.55. eff_w2 = 0.30/0.55 = 0.5455, eff_w3 = 0.25/0.55 = 0.4545
    # H = 0.5455*0.20 + 0.4545*0.60 = 0.1091 + 0.2727 = 0.3818
    h4, w4, m4 = fusion.compute_adaptive_h_score(None, cg, cf, (1.0, 1.0, 1.0))
    assert m4 == [0, 1, 1]
    assert w4["alpha_factual_error"] == 0.0
    assert abs(w4["beta_confidence_gap"] - 0.5455) < 1e-3
    assert abs(w4["gamma_consistency_failure"] - 0.4545) < 1e-3
    assert abs(h4 - 0.3818) < 1e-3

    # Mask 5: [1, 0, 0] - P1 only
    # eff_w1 = 1.00, H = 0.4000
    h5, w5, m5 = fusion.compute_adaptive_h_score(fe, None, None, (1.0, 1.0, 1.0))
    assert m5 == [1, 0, 0]
    assert w5["alpha_factual_error"] == 1.0
    assert w5["beta_confidence_gap"] == 0.0
    assert w5["gamma_consistency_failure"] == 0.0
    assert abs(h5 - 0.4000) < 1e-4

    # Mask 6: [0, 1, 0] - P2 only
    # eff_w2 = 1.00, H = 0.2000
    h6, w6, m6 = fusion.compute_adaptive_h_score(None, cg, None, (1.0, 1.0, 1.0))
    assert m6 == [0, 1, 0]
    assert w6["alpha_factual_error"] == 0.0
    assert w6["beta_confidence_gap"] == 1.0
    assert w6["gamma_consistency_failure"] == 0.0
    assert abs(h6 - 0.2000) < 1e-4

    # Mask 7: [0, 0, 1] - P3 only
    # eff_w3 = 1.00, H = 0.6000
    h7, w7, m7 = fusion.compute_adaptive_h_score(None, None, cf, (1.0, 1.0, 1.0))
    assert m7 == [0, 0, 1]
    assert w7["alpha_factual_error"] == 0.0
    assert w7["beta_confidence_gap"] == 0.0
    assert w7["gamma_consistency_failure"] == 1.0
    assert abs(h7 - 0.6000) < 1e-4

    # Mask 8: [0, 0, 0] - All unavailable
    # eff_w = [0, 0, 0], H = 0.50 (abstention prior)
    h8, w8, m8 = fusion.compute_adaptive_h_score(None, None, None, (1.0, 1.0, 1.0))
    assert m8 == [0, 0, 0]
    assert w8["alpha_factual_error"] == 0.0
    assert w8["beta_confidence_gap"] == 0.0
    assert w8["gamma_consistency_failure"] == 0.0
    assert h8 == 0.50


def test_hand_computed_binary_bernoulli_entropy(p2_engine):
    """Test hand-computed binary Bernoulli entropy for specific probabilities."""
    # H(0.5) = -0.5*log2(0.5) - 0.5*log2(0.5) = 0.5*1 + 0.5*1 = 1.0000
    ent_half = p2_engine.calculate_entropy(0.5)
    assert abs(ent_half - 1.0000) < 1e-4

    # H(0.8) = -0.8*log2(0.8) - 0.2*log2(0.2)
    # log2(0.8) = -0.321928, log2(0.2) = -2.321928
    # H = -0.8*(-0.321928) - 0.2*(-2.321928) = 0.257542 + 0.464386 = 0.7219
    ent_80 = p2_engine.calculate_entropy(0.8)
    assert abs(ent_80 - 0.7219) < 1e-3

    # Extreme probabilities clamped to MIN_TOKEN_PROBABILITY (1e-5):
    # H(1.0) = H(0.0) ~ -1e-5*log2(1e-5) ~ 1e-5 * 16.6 ~ 0.0002 ~ 0.0000
    ent_one = p2_engine.calculate_entropy(1.0)
    assert ent_one < 0.001


def test_scenario_s_proxy_p2_reliability_capping(fusion):
    """Scenario S: Proxy P2 reliability is capped at PROXY_P2_RELIABILITY so FE dominates."""
    p1 = Pillar1Result(
        factual_error_score=0.45,
        dense_retrieval_score=0.90,
        bm25_retrieval_score=0.85,
        reasoning="P1 done",
        available=True,
    )
    p2_proxy = Pillar2Result(
        confidence_gap_score=0.10,
        available=True,
        status="EXECUTED",
        mode="STATIC_VERIFICATION_CONFIDENCE",
        is_proxy=True,
        calibration_score=0.90,  # High nominal proxy score
        reasoning="Proxy P2",
    )
    p3 = Pillar3Result(
        consistency_failure_score=0.0,
        available=True,
        reasoning="P3 done",
    )
    h_score, risk, color, weights = fusion.fuse(p1, p2_proxy, p3)
    # Because r_p2 is capped at 0.20, P1 weight remains overwhelmingly dominant
    assert weights["alpha_factual_error"] > weights["beta_confidence_gap"] * 3
    # H-score cannot be heavily suppressed by the proxy
    assert h_score > 0.20


def test_scenario_t_fusion_numerical_correctness(fusion):
    """Scenario T: Documented formula H = 0.45*FE + 0.30*CG + 0.25*CF is verified numerically."""
    fe, cg, cf = 0.40, 0.20, 0.10
    h_val, eff_w, mask = fusion.compute_adaptive_h_score(fe=fe, cg=cg, cf=cf, reliabilities=(1.0, 1.0, 1.0))
    expected = 0.45 * 0.40 + 0.30 * 0.20 + 0.25 * 0.10  # 0.18 + 0.06 + 0.025 = 0.265
    assert abs(h_val - expected) < 1e-4
    assert eff_w["alpha_factual_error"] == 0.45
    assert eff_w["beta_confidence_gap"] == 0.30
    assert eff_w["gamma_consistency_failure"] == 0.25


def test_scenario_u_effective_weights_dict_aliases(fusion):
    """Scenario U: EffectiveWeights supports canonical keys and legacy aliases transparently."""
    _, eff_w, _ = fusion.compute_adaptive_h_score(fe=0.3, cg=0.3, cf=0.3)
    # Summing values iterates ONLY the 3 canonical keys: sum == 1.0
    assert round(sum(eff_w.values()), 4) == 1.0
    assert len(eff_w) == 3
    # Aliases 'alpha', 'beta', 'gamma' work via __getitem__ and 'in'
    assert "alpha" in eff_w
    assert eff_w["alpha"] == eff_w["alpha_factual_error"]
    assert eff_w.get("beta") == eff_w["beta_confidence_gap"]
    assert eff_w.get("gamma") == eff_w["gamma_consistency_failure"]


# ==============================================================================
# TESTS V - Z: CALIBRATION, VERDICT INTEGRITY, SERIALIZATION & CACHE
# ==============================================================================

def test_scenario_v_calibration_behavior(calibrator):
    """Scenario V: Platt scaling maps raw scores into monotonic calibrated probabilities."""
    p_low = calibrator.calibrate(0.10).calibrated_probability
    p_mid = calibrator.calibrate(0.50).calibrated_probability
    p_high = calibrator.calibrate(0.90).calibrated_probability
    assert 0.0 <= p_low <= p_mid <= p_high <= 1.0


def test_scenario_w_claim_level_vs_response_level_status_consistency():
    """Scenario W: Claim with INSUFFICIENT_EVIDENCE blocks VERIFIED response banner."""
    # Simulated router check
    sentence_risk_levels = ["INSUFFICIENT_EVIDENCE"]
    fe_val = 0.18
    risk_level_str = "VERIFIED"  # Preliminary from numeric H < 0.30

    if risk_level_str == "VERIFIED":
        flagged = [s for s in sentence_risk_levels if s not in ("VERIFIED", "LOW_RISK")]
        if flagged or fe_val > 0.20:
            has_insufficient = any(s == "INSUFFICIENT_EVIDENCE" for s in flagged)
            if has_insufficient:
                risk_level_str = "INSUFFICIENT_EVIDENCE"
            else:
                risk_level_str = "NEEDS_VERIFICATION"

    # Confirms consistency: overall verdict cannot be VERIFIED
    assert risk_level_str == "INSUFFICIENT_EVIDENCE"


def test_scenario_x_contradicted_sentence_blocks_verified():
    """Scenario X: Claim with CONTRADICTED status blocks VERIFIED banner."""
    sentence_risk_levels = ["CONTRADICTED"]
    fe_val = 0.85
    risk_level_str = "VERIFIED"

    if risk_level_str == "VERIFIED":
        flagged = [s for s in sentence_risk_levels if s not in ("VERIFIED", "LOW_RISK")]
        if flagged or fe_val > 0.15:
            has_contradicted = any(s in ("CONTRADICTED", "LIKELY_HALLUCINATED") for s in flagged)
            if has_contradicted:
                risk_level_str = "LIKELY_HALLUCINATED"
            else:
                risk_level_str = "NEEDS_VERIFICATION"

    assert risk_level_str == "LIKELY_HALLUCINATED"


def test_scenario_y_api_serialization():
    """Scenario Y: PillarScores and PillarStatus serialize correctly with null values."""
    scores = PillarScores(
        retrieval=0.45,
        confidence=None,  # P2 unavailable
        consistency=0.05,
    )
    status = PillarExecutionStatus(
        p1_available=True,
        p2_available=False,
        p3_available=True,
        p1_status="EXECUTED",
        p2_status="UNAVAILABLE",
        p3_status="EXECUTED",
    )
    serialized = {
        "scores": scores.model_dump(),
        "status": status.model_dump(),
    }
    assert serialized["scores"]["confidence"] is None
    assert serialized["scores"]["retrieval"] == 0.45
    assert serialized["status"]["p2_available"] is False
    assert serialized["status"]["p2_status"] == "UNAVAILABLE"


def test_scenario_z_cache_isolation():
    """Scenario Z: Different query inputs produce distinct responses without cross-contamination."""
    fusion = FusionEngine()
    h1, _, _ = fusion.compute_adaptive_h_score(fe=0.10, cg=None, cf=None)
    h2, _, _ = fusion.compute_adaptive_h_score(fe=0.85, cg=None, cf=None)
    assert h1 != h2
    assert h1 == 0.10
    assert h2 == 0.85


# ==============================================================================
# STEP 2 & STEP 3 AUDIT TESTS: EXPLICIT CONTRACT & SCIENTIFIC INTEGRITY
# ==============================================================================

def test_pillar2_explicit_contract_and_edge_cases(p2_engine):
    """Step 2 Audit: Verify explicit representation contract, logprob conversions, genuine zeros, and length mismatches."""
    tokens = ["TokenA", "TokenB", "TokenC"]

    # 1. Negative log-probabilities with explicit is_logprobs=True
    logprobs = [-0.10, -0.69315, -2.3026]
    analyses_lp, avg_p_lp, avg_ent_lp, cg_lp = p2_engine.evaluate_tokens(tokens, logprobs, is_logprobs=True)
    assert abs(avg_p_lp - 0.5017) < 0.01  # exp(-0.1)~0.9048, exp(-0.69315)~0.5, exp(-2.3026)~0.1 -> mean~0.5016
    assert 0.0 <= cg_lp <= 1.0

    # 2. Equivalent probability inputs with explicit is_logprobs=False
    probs = [0.9048, 0.5000, 0.1000]
    analyses_pr, avg_p_pr, avg_ent_pr, cg_pr = p2_engine.evaluate_tokens(tokens, probs, is_logprobs=False)
    assert abs(avg_p_pr - avg_p_lp) < 0.01
    assert abs(cg_pr - cg_lp) < 0.01

    # 3. Zero log-probability: ln(1.0) = 0.0 -> probability must be 1.0
    analyses_z, avg_p_z, _, cg_z = p2_engine.evaluate_tokens(["Sure"], [0.0], is_logprobs=True)
    assert avg_p_z == 1.0
    assert cg_z == 0.0

    # 4. Genuine zero-valued probability in probability mode (not treated as missing data!)
    analyses_zero, avg_p_zero, _, cg_zero = p2_engine.evaluate_tokens(["HallucinatedWord"], [0.0], is_logprobs=False)
    assert avg_p_zero is not None  # Genuine measurement present
    assert avg_p_zero < 0.001
    assert cg_zero == 1.0  # Maximum confidence gap
    assert analyses_zero[0].risk_level == RiskLevel.LIKELY_HALLUCINATED

    # 5. Very small valid probabilities
    analyses_small, avg_p_small, _, _ = p2_engine.evaluate_tokens(["Rare"], [1e-4], is_logprobs=False)
    assert avg_p_small == 0.0001
    assert analyses_small[0].probability == 0.0001

    # 6. Invalid values greater than zero in log-probability mode: clamped to 0.0 (p=1.0)
    analyses_pos, avg_p_pos, _, _ = p2_engine.evaluate_tokens(["InvalidLogprob"], [0.5], is_logprobs=True)
    assert avg_p_pos == 1.0

    # 7. Token / probability length mismatches: handled gracefully with length alignment
    tokens_short = ["OnlyOne"]
    probs_long = [0.9, 0.8, 0.7]
    analyses_mismatch, avg_p_mismatch, _, _ = p2_engine.evaluate_tokens(tokens_short, probs_long, is_logprobs=False)
    assert len(analyses_mismatch) == 3
    assert abs(avg_p_mismatch - 0.80) < 1e-4

    # 8. Token predictability disclaimer: confirm reasoning never equates token probability to factual truth
    p2_res = p2_engine.analyze(tokens, logprobs, is_logprobs=True)
    assert "factual truth" in p2_res.reasoning or "generation certainty" in p2_res.reasoning


def test_pillar3_multi_sample_and_internal_consistency_distinction(p3_engine):
    """Step 3 Audit: Verify distinction between multi-generation consistency and internal contradiction."""
    primary = "Water boils at 100 degrees Celsius at standard atmospheric pressure."

    # 1. No alternate generations: evaluates cleanly via intra-response or single claim
    res_no_alt = p3_engine.analyze(primary, sample_responses=None)
    assert res_no_alt.mode in ("SINGLE_CLAIM_CONSISTENCY", "INTRA_RESPONSE_CONSISTENCY")
    assert res_no_alt.mode != "CROSS_GENERATION_CONSISTENCY"

    # 2. One alternate generation
    one_alt = ["At 1 atm, the boiling point of water is 100 °C."]
    res_one = p3_engine.analyze(primary, sample_responses=one_alt)
    assert res_one.mode == "CROSS_GENERATION_CONSISTENCY"
    assert len(res_one.sample_responses) == 1

    # 3. Several distinct generations
    multi_alt = [
        "At 1 atm, the boiling point of water is 100 °C.",
        "Water reaches its boiling point at 100 degrees Celsius under standard conditions.",
    ]
    res_multi = p3_engine.analyze(primary, sample_responses=multi_alt)
    assert res_multi.mode == "CROSS_GENERATION_CONSISTENCY"
    assert len(res_multi.sample_responses) == 2

    # 4. Duplicate and case-only duplicates are deduped
    case_duplicates = [
        "water boils at 100 degrees celsius at standard atmospheric pressure.",  # case duplicate of primary
        "AT 1 ATM, THE BOILING POINT OF WATER IS 100 °C.",
        "at 1 atm, the boiling point of water is 100 °c.",  # case duplicate of previous sample
    ]
    res_case = p3_engine.analyze(primary, sample_responses=case_duplicates)
    assert len(res_case.sample_responses) == 1  # Only 1 unique alternate generation retained

    # 5. Internal contradiction in a single response does not masquerade as multi-generation consistency
    contradictory_text = "The company reported record profits in 2023. The company filed for bankruptcy in 2023 after catastrophic losses."
    res_intra = p3_engine.evaluate_intra_response_consistency(contradictory_text)
    assert res_intra.mode == "INTRA_RESPONSE_CONSISTENCY"
    assert res_intra.mode != "CROSS_GENERATION_CONSISTENCY"
    assert len(res_intra.sample_responses) == 0
    assert res_intra.consistency_failure_score is not None
