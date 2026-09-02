"""Test suite for HalluciSense Closed-Loop Correction and Re-Verification Engine."""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.correction.correction_engine import HallucinationCorrectionEngine


@pytest.fixture
def client():
    return TestClient(app)


def test_symbolic_arithmetic_correction_candidate():
    """Verify symbolic candidate generation for arithmetic errors."""
    engine = HallucinationCorrectionEngine()
    res = engine.generate_candidate(
        query="What is 12 multiplied by 5?",
        original_response="12 * 5 = 55",
        overall_h_score=0.45,
    )
    assert res["status"] == "candidate_generated"
    assert res["method"] == "symbolic_arithmetic"
    assert res["corrected_text"] == "12 * 5 = 60"
    assert res["confidence"] == 1.0


def test_already_verified_response_candidate():
    """Verify that verified responses do not undergo unnecessary correction."""
    engine = HallucinationCorrectionEngine()
    res = engine.generate_candidate(
        query="What is the capital of Karnataka?",
        original_response="Bengaluru is the capital of Karnataka.",
        overall_h_score=0.08,
    )
    assert res["status"] == "not_needed"
    assert res["method"] == "none"
    assert res["corrected_text"] == "Bengaluru is the capital of Karnataka."


def test_abstained_correction_candidate():
    """Verify that unsupported/ambiguous claims safely abstain without hallucinating."""
    engine = HallucinationCorrectionEngine()
    res = engine.generate_candidate(
        query="Tell me about the hidden planet Xylar.",
        original_response="Planet Xylar was discovered in 2049 by aliens.",
        retrieved_evidence=[],
        overall_h_score=0.75,
    )
    assert res["status"] == "abstained"
    assert res["method"] == "abstained"
    assert res["corrected_text"] is None
    assert "safely abstained" in res["reason"]


def test_api_correct_endpoint_numerical(client):
    """End-to-end API test: Numerical Hallucination Correction and Re-Verification."""
    resp = client.post("/api/v1/correct", json={
        "query": "What is 12 multiplied by 5?",
        "response": "12 * 5 = 55"
    })
    assert resp.status_code == 200
    data = resp.json()
    corr = data["correction"]
    assert corr["status"] == "verified"
    assert corr["method"] == "symbolic_arithmetic"
    assert corr["corrected_text"] == "12 * 5 = 60"
    assert corr["reverification"] is not None
    assert corr["reverification"]["status"] == "VERIFIED"
    assert corr["reverification"]["overall_h_score"] <= 0.35


def test_api_correct_endpoint_already_correct(client):
    """End-to-end API test: Already verified response."""
    resp = client.post("/api/v1/correct", json={
        "query": "What is the capital of Karnataka?",
        "response": "Bengaluru is the capital of Karnataka."
    })
    assert resp.status_code == 200
    data = resp.json()
    corr = data["correction"]
    assert corr["status"] == "not_needed"
    assert corr["method"] == "none"
