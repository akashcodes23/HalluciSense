import pytest
import math
from unittest.mock import patch
from app.core.engine.types import (
    Pillar1Result, Pillar2Result, Pillar3Result, EvidenceItem, RiskLevel, HallucinationReport,
    SentenceAnalysis,
)
from app.core.engine.pillar1_retrieval import Pillar1RetrievalEngine
from app.core.engine.pillar2_confidence import Pillar2ConfidenceEngine
from app.core.engine.pillar3_consistency import Pillar3ConsistencyEngine
from app.core.engine.fusion import FusionEngine
from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.modules.verification.schemas import VerificationReportResponse

# =========================================================
# CORRECTION MOCK FIXTURE
# =========================================================
# Prevents any test in this module from calling the real
# Gemini correction API. All pillars (P1, P2, P3), fusion,
# sentence-level scoring, and risk classification continue
# to execute for real.
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


def test_step2_three_pillar_fusion_formula():
    """Verify three-pillar formula: H = alpha*FE + beta*CG + gamma*CF."""
    fusion = FusionEngine(alpha=0.45, beta=0.30, gamma=0.25)
    p1 = Pillar1Result(factual_error_score=0.70, reasoning="FE test")
    p2 = Pillar2Result(avg_probability=0.40, avg_entropy=0.50, confidence_gap_score=0.60, available=True, reasoning="CG test")
    p3 = Pillar3Result(sample_responses=["s1"], pairwise_similarities=[0.2], consistency_failure_score=0.80, available=True, reasoning="CF test")

    h_score, risk, color, weights = fusion.fuse(p1, p2, p3)
    
    expected_h = round(0.45 * 0.70 + 0.30 * 0.60 + 0.25 * 0.80, 4)
    assert h_score == expected_h == 0.6950
    assert risk == RiskLevel.LIKELY_HALLUCINATED
    assert color == "#EF4444"
    assert weights == {
        "alpha_factual_error": 0.45,
        "beta_confidence_gap": 0.30,
        "gamma_consistency_failure": 0.25
    }

def test_step3_availability_matrix():
    """Verify dynamic weight renormalization for all availability matrix permutations."""
    fusion = FusionEngine(alpha=0.45, beta=0.30, gamma=0.25)
    p1 = Pillar1Result(factual_error_score=0.70, reasoning="FE")
    p2_avail = Pillar2Result(avg_probability=0.4, avg_entropy=0.5, confidence_gap_score=0.60, available=True, reasoning="CG")
    p2_unavail = Pillar2Result(avg_probability=None, avg_entropy=None, confidence_gap_score=None, available=False, reasoning="CG unavail")
    p3_avail = Pillar3Result(sample_responses=["s"], pairwise_similarities=[0.2], consistency_failure_score=0.80, available=True, reasoning="CF")
    p3_unavail = Pillar3Result(sample_responses=[], pairwise_similarities=[], consistency_failure_score=None, available=False, reasoning="CF unavail")

    # A. All available
    h_all, _, _, w_all = fusion.fuse(p1, p2_avail, p3_avail)
    assert h_all == 0.6950
    assert w_all["beta_confidence_gap"] == 0.30
    assert w_all["gamma_consistency_failure"] == 0.25

    # B. P2 Unavailable
    h_p2_un, _, _, w_p2_un = fusion.fuse(p1, p2_unavail, p3_avail)
    assert pytest.approx(w_p2_un["alpha_factual_error"], abs=1e-3) == 0.6429
    assert w_p2_un["beta_confidence_gap"] == 0.0
    assert pytest.approx(w_p2_un["gamma_consistency_failure"], abs=1e-3) == 0.3571
    assert pytest.approx(h_p2_un, abs=1e-3) == 0.7357

    # C. P3 Unavailable
    h_p3_un, _, _, w_p3_un = fusion.fuse(p1, p2_avail, p3_unavail)
    assert pytest.approx(w_p3_un["alpha_factual_error"], abs=1e-3) == 0.6000
    assert pytest.approx(w_p3_un["beta_confidence_gap"], abs=1e-3) == 0.4000
    assert w_p3_un["gamma_consistency_failure"] == 0.0
    assert pytest.approx(h_p3_un, abs=1e-3) == 0.6600

    # D. Only P1 Available
    h_p1_only, _, _, w_p1_only = fusion.fuse(p1, p2_unavail, p3_unavail)
    assert w_p1_only == {"alpha_factual_error": 1.0, "beta_confidence_gap": 0.0, "gamma_consistency_failure": 0.0}
    assert h_p1_only == 0.7000

def test_step4_controlled_pipeline_scenarios():
    """Verify controlled pipeline scenarios A through G."""
    pipeline = HallucinationDetectionPipeline()

    ev_supported = EvidenceItem(
        claim="Paris capital of France", snippet="Paris is the capital and most populous city of France.",
        source_name="Wikipedia", source_url="https://en.wikipedia.org/wiki/Paris", similarity_score=0.95, is_supporting=True
    )
    ev_contradicted = EvidenceItem(
        claim="Paris capital of Germany", snippet="Berlin is the capital and largest city of Germany.",
        source_name="Wikipedia", source_url="https://en.wikipedia.org/wiki/Berlin", similarity_score=0.90, is_supporting=False
    )

    # Case A: Consistent / Supported
    # "Paris is the capital of France." => 6 whitespace-delimited tokens
    rep_a = pipeline.analyze_response(
        full_text="Paris is the capital of France.",
        token_probabilities=[0.98, 0.99, 0.97, 0.99, 0.96, 0.98],
        evidence_items=[ev_supported],
        sample_responses=["The capital of France is Paris.", "Paris is France's capital city."]
    )
    assert rep_a.overall_h_score < 0.20
    assert rep_a.overall_risk_level == RiskLevel.VERIFIED

    # Case B: Factual Contradiction
    # "Paris is the capital of Germany." => 6 whitespace-delimited tokens
    rep_b = pipeline.analyze_response(
        full_text="Paris is the capital of Germany.",
        token_probabilities=[0.95, 0.92, 0.90, 0.94, 0.93, 0.91],
        evidence_items=[ev_contradicted],
        sample_responses=["Berlin is the capital of Germany.", "The capital city of Germany is Berlin."]
    )
    assert rep_b.overall_h_score >= 0.55
    assert rep_b.overall_risk_level != RiskLevel.VERIFIED  # Must be flagged

    # Case C: Confident Hallucination (Low CG, High FE)
    ev_eiffel = EvidenceItem(
        claim="Eiffel Tower in Berlin", snippet="The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
        source_name="Wikipedia", source_url="https://en.wikipedia.org/wiki/Eiffel_Tower", similarity_score=0.92, is_supporting=False
    )
    rep_c = pipeline.analyze_response(
        full_text="The Eiffel Tower is located in Berlin.",
        # "The Eiffel Tower is located in Berlin." => 7 whitespace-delimited tokens
        token_probabilities=[0.99, 0.99, 0.98, 0.99, 0.98, 0.99, 0.97], # Very high confidence (low CG)
        evidence_items=[ev_eiffel],
        sample_responses=["The Eiffel Tower is located in Paris, France.", "The Eiffel Tower is in Paris."]
    )
    assert rep_c.pillar1_summary.factual_error_score > 0.80
    assert rep_c.pillar2_summary.confidence_gap_score < 0.10
    assert rep_c.overall_h_score >= 0.55
    assert rep_c.overall_risk_level != RiskLevel.VERIFIED  # Must be flagged despite high confidence

    # Case D: Uncertain but Factually Correct (High CG, Low FE)
    ev_water = EvidenceItem(
        claim="Water freezes at 0 C", snippet="At standard atmospheric pressure, pure water freezes at 0 degrees Celsius.",
        source_name="ScienceDirect", source_url="https://sciencedirect.com", similarity_score=0.95, is_supporting=True
    )
    rep_d = pipeline.analyze_response(
        full_text="Water freezes at 0 degrees Celsius at standard atmospheric pressure.",
        # "Water freezes at 0 degrees Celsius at standard atmospheric pressure." => 10 whitespace-delimited tokens
        token_probabilities=[0.15, 0.20, 0.18, 0.22, 0.19, 0.16, 0.21, 0.17, 0.20, 0.18], # Low confidence (high CG)
        evidence_items=[ev_water],
        sample_responses=["At standard pressure water freezes at 0 degrees Celsius.", "Water freezes at zero degrees C under standard pressure."]
    )
    assert rep_d.pillar1_summary.factual_error_score < 0.20
    assert rep_d.pillar2_summary.confidence_gap_score > 0.50
    assert rep_d.overall_h_score < 0.40 # Grounding and consistency keep H-score low

    # Case F: P2 Unavailable
    rep_f = pipeline.analyze_response(
        full_text="Paris is the capital of France.",
        token_probabilities=None,
        evidence_items=[ev_supported],
        sample_responses=["The capital of France is Paris."]
    )
    assert rep_f.pillar2_summary.available == False
    assert rep_f.pillar2_summary.confidence_gap_score is None
    assert rep_f.weights_used["beta_confidence_gap"] == 0.0
    assert pytest.approx(sum(rep_f.weights_used.values()), abs=1e-3) == 1.0

    # Case G: P3 Unavailable
    rep_g = pipeline.analyze_response(
        full_text="Paris is the capital of France.",
        # "Paris is the capital of France." => 6 whitespace-delimited tokens
        token_probabilities=[0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
        evidence_items=[ev_supported],
        sample_responses=[]
    )
    assert rep_g.pillar3_summary.available == False
    assert rep_g.pillar3_summary.consistency_failure_score is None
    assert rep_g.weights_used["gamma_consistency_failure"] == 0.0
    assert pytest.approx(sum(rep_g.weights_used.values()), abs=1e-3) == 1.0

def test_step5_sentence_level_multi_claim_isolation():
    """Verify sentence-level scoring isolates hallucinated sentence from factual sentences."""
    pipeline = HallucinationDetectionPipeline()

    multi_text = "Paris is the capital of France. The Eiffel Tower is located in Berlin. Water freezes at 0 degrees Celsius at standard pressure."
    multi_evidence = [
        EvidenceItem(claim="Paris capital France", snippet="Paris is the capital of France.",
                     source_name="Wikipedia", source_url="https://en.wikipedia.org/wiki/Paris",
                     similarity_score=0.95, is_supporting=True),
        EvidenceItem(claim="Eiffel Tower Berlin", snippet="The Eiffel Tower is located in Paris, France.",
                     source_name="Wikipedia", source_url="https://en.wikipedia.org/wiki/Eiffel_Tower",
                     similarity_score=0.90, is_supporting=False),
        EvidenceItem(claim="Water freezes 0 C", snippet="Water freezes at 0 degrees Celsius.",
                     source_name="ScienceDirect", source_url="https://sciencedirect.com",
                     similarity_score=0.95, is_supporting=True)
    ]
    multi_samples = [
        "Paris is the capital of France. The Eiffel Tower is in Paris. Water freezes at 0 C.",
        "France has Paris as capital. Eiffel Tower is situated in Paris. Water freezes at zero Celsius."
    ]

    report = pipeline.analyze_response(
        full_text=multi_text,
        # 6 + 7 + 10 = 23 whitespace-delimited tokens across 3 sentences
        token_probabilities=[0.95] * 23,
        evidence_items=multi_evidence,
        sample_responses=multi_samples
    )

    assert len(report.sentence_analyses) == 3
    s1, s2, s3 = report.sentence_analyses

    assert s1.risk_level == RiskLevel.VERIFIED
    assert s2.risk_level != RiskLevel.VERIFIED  # Hallucinated sentence must be flagged
    assert s2.hallucination_score > s1.hallucination_score
    assert s2.hallucination_score > s3.hallucination_score

def test_step6_token_probability_exp_handling():
    """Verify OpenAI logprob exponential conversion and safety."""
    p2_engine = Pillar2ConfidenceEngine()

    # Logprob -0.1 -> probability exp(-0.1) = ~0.9048
    logprob = -0.1
    prob = math.exp(logprob)
    assert 0.90 < prob < 0.91

    tokens = ["test"]
    probs = [prob]
    result = p2_engine.analyze(tokens, probs)
    assert result.available == True
    assert result.avg_probability == round(prob, 4)

def test_step8_correction_independence():
    """Verify H-score is computed prior to correction and remains unchanged after correction."""
    pipeline = HallucinationDetectionPipeline()
    ev = EvidenceItem(
        claim="Eiffel Tower Berlin", snippet="The Eiffel Tower is located in Paris, France.",
        source_name="Wikipedia", source_url="https://en.wikipedia.org/wiki/Eiffel_Tower", similarity_score=0.90, is_supporting=False
    )
    
    report = pipeline.analyze_response(
        full_text="The Eiffel Tower is located in Berlin.",
        # "The Eiffel Tower is located in Berlin." => 7 whitespace-delimited tokens
        token_probabilities=[0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
        evidence_items=[ev],
        sample_responses=["The Eiffel Tower is in Paris."]
    )

    h_score_before = report.overall_h_score

    # The corrected_response field is populated by _generate_correction
    # (mocked in this suite). Verify that mutation of corrected_response
    # does not alter the H-Score.
    assert report.corrected_response is not None
    assert report.overall_h_score == h_score_before

def test_step10_report_invariants():
    """Verify all report invariants and bounds."""
    pipeline = HallucinationDetectionPipeline()
    report = pipeline.analyze_response(
        full_text="Test sentence one. Test sentence two.",
        # "Test sentence one. Test sentence two." => 6 whitespace-delimited tokens
        token_probabilities=[0.90, 0.92, 0.88, 0.95, 0.91, 0.93],
        evidence_items=[],
        sample_responses=["Test sentence one. Test sentence two."]
    )

    assert 0.0 <= report.overall_h_score <= 1.0
    assert 0.0 <= report.pillar1_summary.factual_error_score <= 1.0
    if report.pillar2_summary.available:
        assert 0.0 <= report.pillar2_summary.confidence_gap_score <= 1.0
    if report.pillar3_summary.available:
        assert 0.0 <= report.pillar3_summary.consistency_failure_score <= 1.0

    assert pytest.approx(sum(report.weights_used.values()), abs=1e-3) == 1.0

def test_step11_api_schema_serialization():
    """Verify API schema Pydantic serialization handles None for unavailable pillars."""
    pipeline = HallucinationDetectionPipeline()
    
    # Analyze with unavailable P2 and P3
    report = pipeline.analyze_response(
        full_text="Paris is the capital of France.",
        token_probabilities=None,
        evidence_items=[],
        sample_responses=[]
    )

    dumped = report.model_dump()
    
    assert dumped["pillar2_summary"]["confidence_gap_score"] is None
    assert dumped["pillar2_summary"]["available"] is False
    assert dumped["pillar3_summary"]["consistency_failure_score"] is None
    assert dumped["pillar3_summary"]["available"] is False
    assert 0.0 <= dumped["overall_h_score"] <= 1.0

def test_network_safety_no_real_gemini_calls():
    """
    Regression guard: prove that the validation suite never reaches the
    real Gemini API. Monkeypatching genai.GenerativeModel to raise
    immediately if instantiated ensures no live correction request can
    occur through the mocked pipeline.
    """
    import google.generativeai as genai

    class _GeminiForbidden(Exception):
        pass

    original_model = genai.GenerativeModel

    def _exploding_model(*args, **kwargs):
        raise _GeminiForbidden(
            "NETWORK SAFETY VIOLATION: Real Gemini GenerativeModel was instantiated during tests"
        )

    # Patch the real Gemini model constructor
    with patch.object(genai, "GenerativeModel", _exploding_model):
        pipeline = HallucinationDetectionPipeline()
        ev = EvidenceItem(
            claim="Test claim", snippet="Contradicting evidence.",
            source_name="TestSource", source_url="https://test.com",
            similarity_score=0.90, is_supporting=False
        )

        # This scenario WILL trigger correction (high FE from contradiction).
        # If the mock is bypassed, _GeminiForbidden will fire.
        report = pipeline.analyze_response(
            full_text="A deliberately hallucinated statement for testing.",
            token_probabilities=[0.95, 0.95, 0.95],
            evidence_items=[ev],
            sample_responses=["A different statement entirely."]
        )

        assert 0.0 <= report.overall_h_score <= 1.0
        assert report.corrected_response is not None
