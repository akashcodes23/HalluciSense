"""Phase 46 — Pillar 2 Static Verification Confidence & Mode Integrity Tests."""

import pytest
from app.core.engine.pillar2_confidence import Pillar2ConfidenceEngine
from app.core.engine.types import EvidenceItem

@pytest.fixture
def p2_engine():
    return Pillar2ConfidenceEngine()

def test_p2_static_factual_claim_with_evidence(p2_engine):
    """P2 computes static confidence when token probabilities are absent and evidence is present."""
    tokens = ["Paris", "is", "the", "capital", "of", "France."]
    evidence = [
        EvidenceItem(
            claim="Paris is the capital of France.",
            snippet="Paris is the capital and most populous city of France.",
            source_name="Wikipedia",
            similarity_score=0.92,
            is_supporting=True,
        )
    ]
    res = p2_engine.analyze(tokens=tokens, probabilities=None, evidence_items=evidence)
    assert res.available is True
    assert res.status == "EXECUTED"
    assert res.mode == "STATIC_VERIFICATION_CONFIDENCE"
    assert res.confidence_gap_score is not None
    assert res.confidence_gap_score < 0.20  # High confidence -> small gap
    assert res.avg_probability is not None
    assert res.avg_probability > 0.80

def test_p2_static_unsupported_claim(p2_engine):
    """P2 indicates uncertainty when evidence is missing."""
    tokens = ["Unicorns", "graze", "on", "Mars."]
    res = p2_engine.analyze(tokens=tokens, probabilities=None, evidence_items=[])
    assert res.available is True
    assert res.status == "EXECUTED"
    assert res.mode == "STATIC_VERIFICATION_CONFIDENCE"
    assert res.confidence_gap_score == 0.60  # Elevated uncertainty
    assert res.avg_probability == 0.40

def test_p2_generation_mode_with_real_logprobs(p2_engine):
    """P2 operates in GENERATION_LOGPROB mode when probabilities are explicitly provided."""
    tokens = ["The", "sun", "rises", "in", "the", "east."]
    probs = [0.99, 0.98, 0.95, 0.99, 0.98, 0.99]
    res = p2_engine.analyze(tokens=tokens, probabilities=probs)
    assert res.available is True
    assert res.status == "EXECUTED"
    assert res.mode == "GENERATION_LOGPROB"
    assert res.token_logprobs is not None
    assert len(res.token_logprobs) == len(probs)
    assert res.confidence_gap_score < 0.15

def test_p2_empty_tokens(p2_engine):
    """P2 returns unavailable when tokens are empty."""
    res = p2_engine.analyze(tokens=[], probabilities=None)
    assert res.available is False
    assert res.status == "UNAVAILABLE"
