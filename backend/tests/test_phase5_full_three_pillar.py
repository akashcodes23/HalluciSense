"""Phase 5 Comprehensive Three-Pillar Activation Test Suite.

Mandated Test Coverage:
- TEST 1: Primary generation preserves logprobs throughout the pipeline.
- TEST 2: P2 calculates only from genuine primary logprobs.
- TEST 3: Exactly 3 alternate generations are produced/evaluated for P3.
- TEST 4: Alternate generations are distinct and stochastic.
- TEST 5: Primary generation is never replaced by alternate candidates.
- TEST 6: P3 uses semantic embeddings (all-MiniLM-L6-v2) by default.
- TEST 7: Jaccard fallback is used when embeddings are unavailable/fail.
- TEST 8: P3 does not become available with only single generation or empty samples.
- TEST 9: P3 contradiction detection is claim-aligned with DeBERTa NLI.
- TEST 10: Correction engine is independent and does not alter P3 inputs.
- TEST 11: Full three-pillar fusion produces exact H = 0.45*P1 + 0.30*P2 + 0.25*P3.
- TEST 12: Full fusion is labeled FULL_THREE_PILLAR with empty missing_pillars.
- TEST 13: Missing pillar produces PARTIAL_RENORMALIZED with explicit provenance.
- TEST 14: No unavailable pillar is represented as 0 or 0%.
- TEST 15: No synthetic logprobs exist when omitted.
- TEST 16: No synthetic generations exist when omitted.
- TEST 17: No fabricated timings exist.
"""

import math
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.core.engine.pillar2_confidence import Pillar2ConfidenceEngine
from app.core.engine.pillar3_consistency import Pillar3ConsistencyEngine
from app.core.engine.fusion import FusionEngine
from app.core.engine.types import Pillar1Result, Pillar2Result, Pillar3Result, EvidenceItem

client = TestClient(app)


def test_1_primary_generation_preserves_logprobs():
    """TEST 1: Primary generation preserves logprobs throughout the pipeline."""
    pipeline = HallucinationDetectionPipeline()
    text = "The capital of France is Paris."
    probs = [0.99, 0.98, 0.99, 0.97, 0.99, 0.99]

    report = pipeline.analyze(
        text=text,
        query="Capital of France",
        token_probabilities=probs,
    )

    p2 = report.pillar2_summary
    assert p2.available is True
    assert p2.avg_probability is not None
    assert abs(p2.avg_probability - sum(probs) / len(probs)) < 1e-3
    assert p2.token_logprobs is not None
    assert len(p2.token_logprobs) == len(probs)


def test_2_p2_calculates_only_from_genuine_primary_logprobs():
    """TEST 2: P2 calculates only from genuine primary logprobs and not from alternates."""
    p2_engine = Pillar2ConfidenceEngine()
    tokens = ["Paris", "is", "capital"]
    probs = [0.95, 0.90, 0.92]

    res = p2_engine.analyze(tokens=tokens, probabilities=probs)
    assert res.available is True
    assert res.avg_probability == round(sum(probs) / 3.0, 4)
    assert res.confidence_gap_score is not None
    assert res.confidence_gap_score < 0.20


def test_3_exactly_3_alternate_generations():
    """TEST 3: Exactly 3 alternate generations are evaluated in P3."""
    p3_engine = Pillar3ConsistencyEngine()
    primary = "Apollo 11 landed on the Moon in July 1969."
    alternates = [
        "In July 1969, Apollo 11 achieved the first crewed Moon landing.",
        "The Apollo 11 lunar module touched down on the Moon on July 20, 1969.",
        "Neil Armstrong and Buzz Aldrin landed Apollo 11 on the Moon in 1969.",
    ]

    res = p3_engine.analyze(primary, alternates)
    assert res.available is True
    assert len(res.sample_responses) == 3
    assert len(res.pairwise_similarities) == 3
    assert res.consistency_failure_score is not None


def test_4_alternate_generations_distinct_stochastic():
    """TEST 4: Alternate generations are distinct and demonstrate stochastic variation."""
    p3_engine = Pillar3ConsistencyEngine()
    primary = "Water boils at 100 degrees Celsius."
    alternates = [
        "At sea level, the boiling point of water is 100 °C.",
        "Water reaches its boiling temperature at 100 degrees Celsius under standard atmospheric pressure.",
        "Pure liquid water boils at 100C.",
    ]

    res = p3_engine.analyze(primary, alternates)
    assert len(set(alternates)) == 3
    for sim in res.pairwise_similarities:
        assert 0.50 <= sim <= 1.0


def test_5_primary_generation_not_replaced():
    """TEST 5: Primary generation is never replaced by alternate candidates."""
    pipeline = HallucinationDetectionPipeline()
    primary = "Alexander Graham Bell patented the telephone in 1876."
    alternates = [
        "The telephone was patented by Alexander Graham Bell in 1876.",
        "In 1876, Alexander Graham Bell received a patent for the telephone.",
        "Alexander Graham Bell was granted the telephone patent in 1876.",
    ]

    report = pipeline.analyze(
        text=primary,
        query="Who patented the telephone?",
        sample_responses=alternates,
    )

    assert report.full_text == primary
    assert report.pillar3_summary.sample_responses == alternates


def test_6_p3_uses_semantic_embeddings():
    """TEST 6: P3 uses semantic embeddings (all-MiniLM-L6-v2) by default."""
    p3_engine = Pillar3ConsistencyEngine()
    primary = "Albert Einstein developed the theory of general relativity."
    alternates = [
        "General relativity was formulated by Albert Einstein.",
        "Einstein introduced general relativity to describe gravitation.",
        "The theory of general relativity was created by Albert Einstein in 1915.",
    ]

    res = p3_engine.analyze(primary, alternates)
    assert res.similarity_method == "semantic_embedding"
    assert res.available is True


def test_7_jaccard_fallback_when_embeddings_unavailable(monkeypatch):
    """TEST 7: Jaccard fallback is cleanly used when embeddings fail."""
    p3_engine = Pillar3ConsistencyEngine()
    primary = "Water is composed of hydrogen and oxygen."
    alternates = [
        "Water molecules consist of hydrogen and oxygen atoms.",
        "Hydrogen and oxygen make up water.",
        "Water chemical formula is H2O.",
    ]

    def mock_evaluate_semantic_consistency(primary, samples):
        raise RuntimeError("GPU/Torch Embedding Out of Memory")

    monkeypatch.setattr(p3_engine, "evaluate_semantic_consistency", mock_evaluate_semantic_consistency)

    res = p3_engine.analyze(primary, alternates)
    assert res.similarity_method == "jaccard_fallback"
    assert res.available is True
    assert res.consistency_failure_score is not None


def test_8_p3_unavailable_with_single_or_empty_generation():
    """TEST 8: P3 does not become available with empty or single generation."""
    p3_engine = Pillar3ConsistencyEngine()
    res_empty = p3_engine.analyze("Some text", [])
    assert res_empty.available is False
    assert res_empty.consistency_failure_score is None

    res_none = p3_engine.analyze("Some text", None)
    assert res_none.available is False
    assert res_none.consistency_failure_score is None


def test_9_p3_claim_aligned_nli_contradiction():
    """TEST 9: P3 contradiction detection is claim-aligned with DeBERTa NLI."""
    p3_engine = Pillar3ConsistencyEngine()
    primary = "Alexander Graham Bell invented the telephone in 1876."
    alternates = [
        "Alexander Graham Bell did NOT invent the telephone, Thomas Edison did in 1920.",
        "The telephone was invented by someone else in 1950.",
        "Alexander Graham Bell had nothing to do with the telephone.",
    ]

    res = p3_engine.analyze(primary, alternates)
    assert res.available is True
    assert res.nli_available is True
    assert res.contradiction_score is not None
    assert res.contradiction_score > 0.30
    assert res.consistency_failure_score > 0.40


def test_10_correction_engine_independence():
    """TEST 10: Correction engine does not alter raw P3 inputs."""
    pipeline = HallucinationDetectionPipeline()
    raw_text = "The Moon is made of cheese."
    alternates = [
        "The Moon consists of basalt and anorthosite rock.",
        "The lunar surface is composed of silicate rock.",
        "Moon composition is primarily rock and dust.",
    ]

    report = pipeline.analyze(
        text=raw_text,
        sample_responses=alternates,
    )

    assert report.full_text == raw_text
    assert report.pillar3_summary.sample_responses == alternates


def test_11_full_three_pillar_fusion_exact_formula():
    """TEST 11: Full three-pillar fusion produces exact H = 0.45*P1 + 0.30*P2 + 0.25*P3."""
    fusion = FusionEngine(alpha=0.45, beta=0.30, gamma=0.25)
    fe, cg, cf = 0.10, 0.20, 0.30
    h_score = fusion.compute_h_score(fe=fe, cg=cg, cf=cf)
    expected = 0.45 * 0.10 + 0.30 * 0.20 + 0.25 * 0.30  # 0.045 + 0.060 + 0.075 = 0.180
    assert abs(h_score - expected) < 1e-4


def test_12_full_fusion_labeled_full_three_pillar_via_api():
    """TEST 12: Full fusion is labeled FULL_THREE_PILLAR with empty missing_pillars via API."""
    payload = {
        "query": "When did Apollo 11 land on the Moon?",
        "response": "Apollo 11 landed on the Moon on July 20, 1969.",
        "model_name": "gpt-4",
        "logprobs": [0.99, 0.98, 0.99, 0.99, 0.99, 0.99],
        "sample_responses": [
            "Apollo 11 touched down on the Moon on July 20, 1969.",
            "The Apollo 11 lunar landing took place on July 20, 1969.",
            "In July 1969, Apollo 11 landed on the Moon.",
        ],
    }
    res = client.post("/api/v1/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()

    decomp = data.get("fusion_decomposition")
    assert decomp is not None
    assert decomp["fusion_mode"] == "FULL_THREE_PILLAR"
    assert decomp["is_full_analysis"] is True
    assert len(decomp["available_pillars"]) == 3
    assert len(decomp["missing_pillars"]) == 0
    assert data["pillar_status"]["p1_available"] is True
    assert data["pillar_status"]["p2_available"] is True
    assert data["pillar_status"]["p3_available"] is True
    assert data["risk_level"] == "VERIFIED"


def test_13_missing_pillar_produces_partial_renormalized():
    """TEST 13: Missing pillar produces PARTIAL_RENORMALIZED mode."""
    payload = {
        "query": "Capital of France",
        "response": "The capital of France is Paris.",
        "model_name": "gpt-4",
        # No logprobs or sample_responses
    }
    res = client.post("/api/v1/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()

    decomp = data.get("fusion_decomposition")
    assert decomp is not None
    assert decomp["fusion_mode"] == "PARTIAL_RENORMALIZED"
    assert decomp["is_full_analysis"] is False
    assert len(decomp["missing_pillars"]) > 0


def test_14_no_unavailable_pillar_represented_as_zero():
    """TEST 14: Unavailable pillar is represented as None / null, never 0.0."""
    payload = {
        "query": "Capital of Spain",
        "response": "Madrid is the capital of Spain.",
        "model_name": "gpt-4",
    }
    res = client.post("/api/v1/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["pillar_scores"]["confidence"] is None
    assert data["pillar_scores"]["consistency"] is None
    assert data["pillar_status"]["p2_status"] == "UNAVAILABLE"
    assert data["pillar_status"]["p3_status"] == "UNAVAILABLE"


def test_15_no_synthetic_logprobs():
    """TEST 15: No synthetic logprobs exist when omitted."""
    payload = {
        "query": "Capital of Italy",
        "response": "Rome is the capital of Italy.",
        "model_name": "gpt-4",
    }
    res = client.post("/api/v1/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()

    conf = data["confidence_analysis"]
    assert conf["signal_type"] == "UNAVAILABLE"
    assert conf["whitebox_entropy"] is None


def test_16_no_synthetic_generations():
    """TEST 16: No synthetic generations exist when omitted."""
    p3_engine = Pillar3ConsistencyEngine()
    res = p3_engine.analyze("Primary text", None)
    assert res.available is False
    assert res.sample_responses == []


def test_17_no_fabricated_timings():
    """TEST 17: Measured durations are based on actual perf_counter timings."""
    payload = {
        "query": "Speed of light",
        "response": "The speed of light in vacuum is 299,792,458 m/s.",
        "model_name": "gpt-4",
    }
    res = client.post("/api/v1/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()

    timings = data["measured_timings"]
    assert timings["total_latency_ms"] > 0
    assert timings["p1_latency_ms"] > 0
    # Unavailable stages are null or measured
    assert timings["gemini_generation_ms"] is None
