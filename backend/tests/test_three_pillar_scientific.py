"""Three-Pillar Scientific Validation Test Suite for HalluciSense v1.0.

Covers Phase 11 Mandates:
- TEST 1: Fully supported factual claim (Apollo 11)
- TEST 2: Unsupported factual claim (Moon made of solid gold)
- TEST 3: Contradictory factual claim (Sun revolves around Earth)
- TEST 4: P2 unavailable handling when logprobs omitted (not 0)
- TEST 5: P3 insufficient generations handling (single generation -> unavailable)
- TEST 6: Partial fusion mode transparency (PARTIAL_RENORMALIZED, no fake P2/P3)
- TEST 7: Full three-pillar fusion numerical precision (H = 0.45*P1 + 0.30*P2 + 0.25*P3)
- TEST 8: Real timing provenance & non-zero execution measurements
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.engine.fusion import FusionEngine
from app.core.engine.types import Pillar1Result, Pillar2Result, Pillar3Result

client = TestClient(app)


def test_1_fully_supported_factual_claim():
    """TEST 1: Fully supported factual claim (Apollo 11)."""
    payload = {
        "query": "When did Apollo 11 land on the Moon?",
        "response": "Apollo 11 landed on the Moon on July 20, 1969.",
        "model_name": "gpt-4",
    }
    res = client.post("/api/v1/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["risk_level"] == "VERIFIED"
    assert data["overall_h_score"] < 0.35
    assert data["pillar_status"]["p1_available"] is True
    assert data["pillar_scores"]["retrieval"] is not None
    assert data["trace_id"].startswith("TRACE_")


def test_2_unsupported_claim():
    """TEST 2: Unsupported claim should yield high factual error and non-verified verdict."""
    payload = {
        "query": "What is the composition of the Moon?",
        "response": "The Moon is made entirely of solid gold and diamonds.",
        "model_name": "gpt-4",
    }
    res = client.post("/api/v1/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["risk_level"] in ("MODERATE_RISK", "LIKELY_HALLUCINATED")
    assert data["overall_h_score"] >= 0.50
    assert data["pillar_scores"]["retrieval"] >= 0.50


def test_3_contradictory_claim():
    """TEST 3: Contradictory claim should be flagged with non-verified risk."""
    payload = {
        "query": "Does the Sun revolve around the Earth?",
        "response": "The Sun revolves around the Earth once every 24 hours.",
        "model_name": "gpt-4",
    }
    res = client.post("/api/v1/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["risk_level"] in ("NEEDS_VERIFICATION", "MODERATE_RISK", "LIKELY_HALLUCINATED")
    assert data["overall_h_score"] >= 0.35
    assert data["pillar_scores"]["retrieval"] >= 0.35


def test_4_p2_unavailable_without_logprobs():
    """TEST 4: When token logprobs are omitted, P2 must be UNAVAILABLE, never fake 0.0."""
    payload = {
        "query": "Capital of France",
        "response": "The capital of France is Paris.",
        "model_name": "gpt-4",
    }
    res = client.post("/api/v1/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["pillar_status"]["p2_available"] is False
    assert data["pillar_status"]["p2_status"] == "UNAVAILABLE"
    assert data["pillar_scores"]["confidence"] is None
    assert data["confidence_analysis"]["signal_type"] == "UNAVAILABLE"
    assert data["confidence_analysis"]["whitebox_entropy"] is None


def test_5_p3_insufficient_generations():
    """TEST 5: Single generation without sample_responses must mark P3 as UNAVAILABLE."""
    payload = {
        "query": "What is water?",
        "response": "Water is a chemical compound consisting of two hydrogen atoms and one oxygen atom (H2O).",
        "model_name": "gpt-4",
    }
    res = client.post("/api/v1/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["pillar_status"]["p3_available"] is False
    assert data["pillar_status"]["p3_status"] == "UNAVAILABLE"
    assert data["pillar_scores"]["consistency"] is None


def test_6_partial_fusion_mode_transparency():
    """TEST 6: When P2/P3 are unavailable, fusion_mode must be PARTIAL_RENORMALIZED."""
    payload = {
        "query": "Capital of Italy",
        "response": "Rome is the capital of Italy.",
        "model_name": "gpt-4",
    }
    res = client.post("/api/v1/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()

    decomp = data.get("fusion_decomposition")
    assert decomp is not None
    assert decomp["fusion_mode"] == "PARTIAL_RENORMALIZED"
    assert decomp["is_full_analysis"] is False
    assert "Pillar 1: Evidence Grounding" in decomp["available_pillars"]
    assert len(decomp["missing_pillars"]) > 0
    assert "explanation" in decomp
    assert "Partial renormalized fusion" in decomp["explanation"]


def test_7_full_three_pillar_fusion_numerical_precision():
    """TEST 7: Full three-pillar fusion calculates H = 0.45*P1 + 0.30*P2 + 0.25*P3 precisely."""
    engine = FusionEngine(alpha=0.45, beta=0.30, gamma=0.25)
    
    p1 = Pillar1Result(claims=["claim1"], evidence=[], factual_error_score=0.12, reasoning="grounded")
    p2 = Pillar2Result(confidence_gap_score=0.21, avg_probability=0.85, available=True, reasoning="confident")
    p3 = Pillar3Result(consistency_failure_score=0.09, available=True, reasoning="consistent")
    
    h_score, risk, color, weights = engine.fuse(p1, p2, p3)
    
    # Expected: 0.45 * 0.12 + 0.30 * 0.21 + 0.25 * 0.09 = 0.054 + 0.063 + 0.0225 = 0.1395
    assert abs(h_score - 0.1395) < 1e-4
    assert weights["alpha_factual_error"] == 0.45
    assert weights["beta_confidence_gap"] == 0.30
    assert weights["gamma_consistency_failure"] == 0.25
    assert risk.value == "VERIFIED"


def test_8_timing_integrity_and_measured_latencies():
    """TEST 8: Ensure measured durations are based on actual execution and not weight multiplications."""
    payload = {
        "query": "Speed of light in vacuum",
        "response": "The speed of light in vacuum is approximately 299,792,458 meters per second.",
        "model_name": "gpt-4",
    }
    res = client.post("/api/v1/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()

    timings = data.get("measured_timings")
    assert timings is not None
    assert timings["total_latency_ms"] > 0.0
    assert timings["p1_latency_ms"] > 0.0
    assert timings["fusion_latency_ms"] >= 0.0

    # Ensure P1 timing is NOT equal to alpha * total_latency
    assert abs(timings["p1_latency_ms"] - (0.45 * timings["total_latency_ms"])) > 0.001
