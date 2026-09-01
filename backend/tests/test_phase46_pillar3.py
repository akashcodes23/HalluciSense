"""Phase 46 — Pillar 3 Intra-Response Consistency & Scaled Safety Tests."""

import pytest
from app.core.engine.pillar3_consistency import Pillar3ConsistencyEngine

@pytest.fixture
def p3_engine():
    return Pillar3ConsistencyEngine()

def test_p3_single_claim_consistency(p3_engine):
    """Single claim evaluates as SINGLE_CLAIM_CONSISTENCY with CF=0.0."""
    text = "The capital of France is Paris."
    res = p3_engine.analyze(primary_response=text, sample_responses=[])
    assert res.available is True
    assert res.status == "EXECUTED"
    assert res.mode == "SINGLE_CLAIM_CONSISTENCY"
    assert res.consistency_failure_score == 0.0
    assert res.sentence_consistency_score == 1.0

def test_p3_two_consistent_claims(p3_engine):
    """Consistent multi-claims have low consistency failure score."""
    text = "Paris is the capital of France. Berlin is the capital of Germany."
    res = p3_engine.analyze(primary_response=text, sample_responses=[])
    assert res.available is True
    assert res.status == "EXECUTED"
    assert res.mode == "INTRA_RESPONSE_CONSISTENCY"
    assert res.consistency_failure_score is not None
    assert len(res.nli_analyses) == 1

def test_p3_two_contradictory_claims(p3_engine):
    """Contradictory claims in the same text produce elevated CF score."""
    text = "Paris is the capital of France. Berlin is the capital of France."
    res = p3_engine.analyze(primary_response=text, sample_responses=[])
    assert res.available is True
    assert res.status == "EXECUTED"
    assert res.mode == "INTRA_RESPONSE_CONSISTENCY"
    assert res.consistency_failure_score is not None
    assert res.consistency_failure_score > 0.10

def test_p3_claim_count_cap(p3_engine):
    """Pairwise comparisons are capped at 15 claims to prevent quadratic explosion."""
    # 20 claims
    sentences = [f"Claim number {i} provides factual statement." for i in range(20)]
    text = " ".join(sentences)
    res = p3_engine.analyze(primary_response=text, sample_responses=[])
    assert res.available is True
    # Maximum 15 claims -> 15*14/2 = 105 pairs
    assert len(res.nli_analyses) <= 105

def test_p3_cross_generation_mode(p3_engine):
    """When sample responses are supplied, P3 evaluates CROSS_GENERATION_CONSISTENCY."""
    text = "Photosynthesis converts light into chemical energy."
    samples = [
        "Photosynthesis uses sunlight to create energy in plants.",
        "Plants generate chemical energy through photosynthesis.",
    ]
    res = p3_engine.analyze(primary_response=text, sample_responses=samples)
    assert res.available is True
    assert res.status == "EXECUTED"
    assert res.consistency_failure_score is not None
    assert res.consistency_failure_score < 0.30
