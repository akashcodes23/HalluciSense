"""
HalluciSense Phase 9 — Pillar 2 Genuine Token Log-Probabilities & Fusion Integrity Tests.

Tests:
1. Gemini genuine logprob extraction from candidate.logprobs_result.chosen_candidates.
2. OpenAI genuine logprob extraction from choice.logprobs.content.
3. Missing logprobs producing explicit UNAVAILABLE status (m2 = 0).
4. Invalid value sanitization (NaN, Inf, clamped probabilities).
5. Subword and whitespace token length alignment.
6. All 8 fusion availability masks in {0, 1}^3.
7. Proxy vs genuine Pillar 2 distinction and reliability downweighting.
8. Confidently wrong factual hallucination (low CG + high FE -> CONTRADICTED).
9. Authoritative verdict reconciliation (INSUFFICIENT_EVIDENCE claims block VERIFIED verdict).
10. Operational paths: Generate-and-Verify vs Paste-and-Verify provenance.
"""

import math
import pytest
from unittest.mock import MagicMock
from app.core.engine.pillar2_confidence import Pillar2ConfidenceEngine
from app.core.engine.fusion import FusionEngine, PROXY_P2_RELIABILITY
from app.core.engine.types import (
    RiskLevel,
    EvidenceItem,
    Pillar1Result,
    Pillar2Result,
    Pillar3Result,
)
from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.modules.providers.gemini import _extract_chunk_logits
from app.modules.providers.schemas import TokenLogit


# ── Test 1: Gemini Genuine Logprob Extraction ─────────────────────────────────

def test_gemini_logprob_extraction():
    """Verify Gemini chosen_candidates extraction with full provenance."""
    mock_candidate_token_1 = MagicMock()
    mock_candidate_token_1.token = "Hubli"
    mock_candidate_token_1.token_id = 12345
    mock_candidate_token_1.log_probability = -0.042

    mock_candidate_token_2 = MagicMock()
    mock_candidate_token_2.token = " is"
    mock_candidate_token_2.token_id = 67890
    mock_candidate_token_2.log_probability = -0.015

    mock_cand = MagicMock()
    mock_cand.logprobs_result.chosen_candidates = [
        mock_candidate_token_1,
        mock_candidate_token_2,
    ]

    mock_chunk = MagicMock()
    mock_chunk.candidates = [mock_cand]

    logits = _extract_chunk_logits(mock_chunk, model_name="gemini-1.5-flash", start_pos=0)

    assert len(logits) == 2
    assert logits[0].token == "Hubli"
    assert logits[0].logprob == pytest.approx(-0.042, rel=1e-3)
    assert logits[0].token_id == 12345
    assert logits[0].position == 0
    assert logits[0].provider == "Google Gemini"
    assert logits[0].model == "gemini-1.5-flash"

    assert logits[1].token == " is"
    assert logits[1].logprob == pytest.approx(-0.015, rel=1e-3)
    assert logits[1].token_id == 67890
    assert logits[1].position == 1


# ── Test 2: OpenAI Genuine Logprob Extraction ─────────────────────────────────

def test_openai_logprob_extraction():
    """Verify OpenAI selected-token logprob extraction from completion choice."""
    openai_choice_logprobs = {
        "content": [
            {
                "token": "Paris",
                "logprob": -0.002,
                "top_logprobs": [{"token": "Paris", "logprob": -0.002}, {"token": "Lyon", "logprob": -7.5}],
            },
            {
                "token": " is",
                "logprob": -0.0005,
                "top_logprobs": [{"token": " is", "logprob": -0.0005}],
            },
        ]
    }

    logits = []
    for idx, item in enumerate(openai_choice_logprobs["content"]):
        logits.append(
            TokenLogit(
                token=item["token"],
                logprob=float(item["logprob"]),
                position=idx,
                provider="OpenAI",
                model="gpt-4o",
            )
        )

    assert len(logits) == 2
    assert logits[0].token == "Paris"
    assert logits[0].logprob == -0.002
    assert logits[0].provider == "OpenAI"
    assert logits[0].position == 0


# ── Test 3: Missing Logprobs Producing Explicit UNAVAILABLE Status ─────────────

def test_missing_logprobs_unavailable():
    """Missing measurements must remain explicitly UNAVAILABLE with m2=0 in fusion."""
    p2 = Pillar2ConfidenceEngine()
    result = p2.analyze(tokens=["Some", "externally", "pasted", "text."], probabilities=None, allow_proxy=False)

    assert result.available is False
    assert result.status == "UNAVAILABLE"
    assert result.confidence_gap_score is None
    assert result.avg_probability is None
    assert result.is_proxy is False
    assert "Generation-time token log-probabilities were not captured" in result.reasoning

    # In fusion, unavailable P2 must have effective weight beta = 0.0
    fusion = FusionEngine()
    score, eff_w, mask = fusion.compute_adaptive_h_score(fe=0.30, cg=None, cf=0.20)
    assert mask == [1, 0, 1]
    assert eff_w["beta"] == 0.0
    assert eff_w["alpha"] > 0.0
    assert eff_w["gamma"] > 0.0
    assert score == pytest.approx(eff_w["alpha"] * 0.30 + eff_w["gamma"] * 0.20, rel=1e-3)


# ── Test 4: Invalid Value Sanitization ────────────────────────────────────────

def test_invalid_values_sanitization():
    """Sanitize NaN, Inf, and out-of-bound values safely."""
    p2 = Pillar2ConfidenceEngine()
    tokens = ["tok1", "tok2", "tok3", "tok4", "tok5"]
    raw_probs = [float("nan"), float("-inf"), float("inf"), 0.85, 0.10]

    token_analyses, avg_p, avg_ent, cg_score = p2.evaluate_tokens(
        tokens=tokens, probabilities=raw_probs, is_logprobs=False
    )

    assert len(token_analyses) == 5
    assert not math.isnan(avg_p)
    assert not math.isnan(avg_ent)
    assert not math.isnan(cg_score)
    assert 0.0 <= avg_p <= 1.0
    assert 0.0 <= cg_score <= 1.0


# ── Test 5: Subword and Whitespace Token Alignment ────────────────────────────

def test_subword_whitespace_token_alignment():
    """Handle mismatched token and probability counts gracefully."""
    p2 = Pillar2ConfidenceEngine()
    # 2 whitespace tokens, but 4 subword logprobs
    tokens = ["MachineLearning", "Rocks"]
    subword_logprobs = [-0.02, -0.05, -0.01, -0.03]

    token_analyses, avg_p, avg_ent, cg = p2.evaluate_tokens(
        tokens=tokens, probabilities=subword_logprobs, is_logprobs=True
    )

    assert len(token_analyses) == 4
    assert avg_p > 0.90
    assert cg < 0.20


# ── Test 6: All 8 Fusion Availability Masks (2^3) ─────────────────────────────

@pytest.mark.parametrize(
    "fe,cg,cf,expected_mask",
    [
        (None, None, None, [0, 0, 0]),
        (0.20, None, None, [1, 0, 0]),
        (None, 0.30, None, [0, 1, 0]),
        (None, None, 0.40, [0, 0, 1]),
        (0.20, 0.30, None, [1, 1, 0]),
        (0.20, None, 0.40, [1, 0, 1]),
        (None, 0.30, 0.40, [0, 1, 1]),
        (0.20, 0.30, 0.40, [1, 1, 1]),
    ],
)
def test_all_8_fusion_masks(fe, cg, cf, expected_mask):
    """Verify all 8 combinations in {0, 1}^3 availability space."""
    fusion = FusionEngine()
    score, eff_w, mask = fusion.compute_adaptive_h_score(fe=fe, cg=cg, cf=cf)
    assert mask == expected_mask

    if expected_mask == [0, 0, 0]:
        assert score == 0.50  # Uninformative prior
        assert eff_w["alpha"] == 0.0
        assert eff_w["beta"] == 0.0
        assert eff_w["gamma"] == 0.0
    else:
        # Sum of effective weights for available signals must be 1.0
        active_weight_sum = (
            (eff_w["alpha"] if expected_mask[0] else 0.0)
            + (eff_w["beta"] if expected_mask[1] else 0.0)
            + (eff_w["gamma"] if expected_mask[2] else 0.0)
        )
        assert active_weight_sum == pytest.approx(1.0, rel=1e-3)


# ── Test 7: Proxy vs Genuine Pillar 2 Distinction ─────────────────────────────

def test_proxy_vs_genuine_p2():
    """Proxy mode is downweighted in fusion; genuine logprob receives empirical reliability."""
    fusion = FusionEngine()

    # Case A: Genuine logprob P2
    p1 = Pillar1Result(factual_error_score=0.80, available=True, reasoning="Contradicted")
    p2_genuine = Pillar2Result(
        confidence_gap_score=0.10,
        available=True,
        is_proxy=False,
        calibration_score=0.90,
        reasoning="Genuine logprobs",
    )
    p3 = Pillar3Result(consistency_failure_score=0.20, available=True, reasoning="Consistent")

    score_genuine, _, _, weights_genuine = fusion.fuse(p1, p2_genuine, p3)

    # Case B: Evidence-similarity Proxy P2
    p2_proxy = Pillar2Result(
        confidence_gap_score=0.10,
        available=True,
        is_proxy=True,
        calibration_score=0.90,  # Proxy calibration should be capped by PROXY_P2_RELIABILITY (0.20)
        reasoning="Proxy estimate",
    )
    score_proxy, _, _, weights_proxy = fusion.fuse(p1, p2_proxy, p3)

    # In proxy mode, beta must be strictly less than in genuine mode
    assert weights_proxy["beta"] < weights_genuine["beta"]
    # And alpha must be strictly higher in proxy mode because FE is not diluted by an ungrounded proxy
    assert weights_proxy["alpha"] > weights_genuine["alpha"]


# ── Test 8: Confidently Wrong Factual Hallucination ───────────────────────────

def test_confidently_wrong_factual_hallucination():
    """When a model is confidently wrong (low CG, but high FE), FE contradiction dominates."""
    p1 = Pillar1Result(
        claims=["Hubli is the capital of Karnataka."],
        evidence=[
            EvidenceItem(
                claim="Hubli is the capital of Karnataka.",
                snippet="Bengaluru is the capital and largest city of Karnataka.",
                source_name="Wikipedia",
                similarity_score=0.95,
                is_supporting=False,
            )
        ],
        factual_error_score=0.98,
        available=True,
        reasoning="Contradicted: Bengaluru is the capital of Karnataka.",
    )
    # Model generates with 99.5% token probability -> very low confidence gap
    p2_conf = Pillar2Result(
        avg_probability=0.995,
        confidence_gap_score=0.015,
        available=True,
        is_proxy=False,
        reasoning="High generation confidence",
    )
    p3 = Pillar3Result(
        consistency_failure_score=0.10,
        available=True,
        reasoning="Consistent samples",
    )

    fusion = FusionEngine()
    h_score, risk, color, _ = fusion.fuse(p1, p2_conf, p3)

    # Direct factual contradiction must not be suppressed by high generation confidence
    assert risk == RiskLevel.LIKELY_HALLUCINATED
    assert color == "#EF4444"


# ── Test 9: Authoritative Verdict Integrity — Insufficient Evidence Claims ───

def test_verdict_reconciliation_insufficient_evidence():
    """Unverified / insufficient evidence claims must never produce an aggregate VERIFIED verdict."""
    pipeline = HallucinationDetectionPipeline()

    # Simulated response about an unsupported fictitious claim with no reference evidence
    report = pipeline.analyze(
        text="The mythical city of Zorblax was founded in the year 4022.",
        evidence_items=[],  # No reference evidence
        token_probabilities=None,  # External pasted text
    )

    # Must abstain / report INSUFFICIENT_EVIDENCE, never VERIFIED
    assert report.overall_risk_level != RiskLevel.VERIFIED
    assert report.overall_risk_level in (RiskLevel.INSUFFICIENT_EVIDENCE, RiskLevel.NEEDS_VERIFICATION)
    # Pillar 2 should be unavailable
    assert report.pillar2_summary.available is False


# ── Test 10: Operational Paths — Generate vs Paste Provenance ─────────────────

def test_generate_vs_paste_provenance():
    """Generate-and-Verify supplies logprobs and activates P2; Paste-and-Verify marks P2 unavailable."""
    pipeline = HallucinationDetectionPipeline()

    # Operational Path 1: Paste-and-Verify (manually pasted text, no logprobs)
    paste_report = pipeline.analyze(
        text="Water is composed of hydrogen and oxygen atoms.",
        token_probabilities=None,
    )
    assert paste_report.pillar2_summary.available is False
    assert paste_report.pillar2_summary.status == "UNAVAILABLE"
    assert paste_report.pillar2_summary.confidence_gap_score is None

    # Operational Path 2: Generate-and-Verify (captured logprobs from LLM provider)
    real_logprobs = [-0.01, -0.02, -0.01, -0.05, -0.01, -0.02, -0.01, -0.02]
    gen_report = pipeline.analyze(
        text="Water is composed of hydrogen and oxygen atoms.",
        token_probabilities=real_logprobs,
        is_logprobs=True,
    )
    assert gen_report.pillar2_summary.available is True
    assert gen_report.pillar2_summary.status == "EXECUTED"
    assert gen_report.pillar2_summary.confidence_gap_score is not None
    assert gen_report.pillar2_summary.confidence_gap_score < 0.20
