"""Pytest Production API Test Suite for HalluciSense v1.0 Sprint 1.

Covering:
- Unit & Schema Validation (empty, missing, invalid model, oversized payload)
- Canonical Analysis & Explainability Endpoints
- Debug Traces & Metrics Endpoints
- Health & Readiness Probes
- Regression Statements (Paris, Berlin, Telephone, Water, Photosynthesis, Speed of Light, Romeo & Juliet, Earth/Sun)
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify GET / returns running status and endpoints map."""
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["message"] == "HalluciSense API is running"
    assert "endpoints" in data


def test_health_and_readiness_endpoints():
    """Verify GET /health and GET /ready probes."""
    res_h = client.get("/health")
    assert res_h.status_code == 200
    assert res_h.json()["status"] == "healthy"

    res_r = client.get("/ready")
    assert res_r.status_code == 200
    data_r = res_r.json()
    assert data_r["status"] == "ready"
    assert data_r["components"]["pipeline"] is True


def test_canonical_analyze_endpoint():
    """Verify POST /api/v1/analyze canonical response payload."""
    payload = {
        "query": "Who invented the telephone?",
        "response": "Alexander Graham Bell invented the telephone in 1876.",
        "model_name": "gpt-4"
    }
    res = client.post("/api/v1/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert "trace_id" in data
    assert data["trace_id"].startswith("TRACE_")
    assert "overall_h_score" in data
    assert 0.0 <= data["overall_h_score"] <= 1.0
    assert "risk_level" in data
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert "pillar_scores" in data
    assert "retrieval" in data["pillar_scores"]
    assert "confidence" in data["pillar_scores"]
    assert "consistency" in data["pillar_scores"]
    assert "failure_taxonomy" in data
    assert "processing_time_ms" in data
    assert data["version"] == "1.0.0"


def test_explain_endpoint():
    """Verify POST /api/v1/explain returns detailed explainability info."""
    payload = {
        "query": "What is photosynthesis?",
        "response": "Photosynthesis is the process by which green plants convert sunlight into chemical energy using chlorophyll.",
        "model_name": "gpt-4"
    }
    res = client.post("/api/v1/explain", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert "trace_id" in data
    assert "retrieved_evidence" in data
    assert "supporting_passages" in data
    assert "reasoning_chain" in data
    assert "fusion_contribution" in data
    assert "adaptive_weights" in data
    assert "confidence_explanation" in data


def test_debug_and_metrics_endpoints():
    """Verify GET /api/v1/debug/latest and GET /api/v1/metrics."""
    # Execute one analysis to guarantee a trace exists
    client.post("/api/v1/analyze", json={"query": "Capital of France", "response": "Paris"})

    res_debug = client.get("/api/v1/debug/latest")
    assert res_debug.status_code == 200
    d_data = res_debug.json()
    assert "trace_id" in d_data

    res_metrics = client.get("/api/v1/metrics")
    assert res_metrics.status_code == 200
    m_data = res_metrics.json()
    assert m_data["requests"] > 0
    assert m_data["average_latency_ms"] >= 0.0
    assert m_data["memory_mb"] > 0.0


def test_empty_string_validation():
    """Verify empty string or whitespace-only inputs return 400 Bad Request."""
    res = client.post("/api/v1/analyze", json={"query": "   ", "response": "Paris"})
    assert res.status_code == 400
    data = res.json()
    assert data["status"] == "error"
    assert data["error_code"] == "BAD_REQUEST"


def test_missing_fields_validation():
    """Verify missing required fields return 422 Unprocessable Entity."""
    res = client.post("/api/v1/analyze", json={"model_name": "gpt-4"})
    assert res.status_code == 422
    data = res.json()
    assert data["status"] == "error"
    assert data["error_code"] == "VALIDATION_ERROR"


def test_oversized_payload_validation():
    """Verify payloads exceeding 100KB return 413 Payload Too Large."""
    huge_string = "A" * (105 * 1024)
    res = client.post("/api/v1/analyze", json={"query": "Test", "response": huge_string})
    assert res.status_code == 413
    data = res.json()
    assert data["status"] == "error"
    assert data["error_code"] == "PAYLOAD_TOO_LARGE"


def test_unsupported_model_validation():
    """Verify unsupported model names return 400 Bad Request."""
    res = client.post("/api/v1/analyze", json={
        "query": "Test",
        "response": "Test response",
        "model_name": "unknown_super_model_xyz"
    })
    assert res.status_code == 400
    data = res.json()
    assert data["status"] == "error"
    assert data["error_code"] == "BAD_REQUEST"


@pytest.mark.parametrize(
    "query,response,expected_risk",
    [
        ("Capital of France", "The capital of France is Paris.", "VERIFIED"),
        ("Capital of France", "The capital of France is Berlin.", "LIKELY_HALLUCINATED"),
        ("Who invented telephone?", "Alexander Graham Bell invented the telephone in 1876.", "VERIFIED"),
        ("Who invented telephone?", "Albert Einstein invented the telephone in 1920.", "LIKELY_HALLUCINATED"),
        ("What is water?", "Water is H2O.", "VERIFIED"),
        ("What is water?", "Water boils at 50 degrees Celsius at sea level.", "LIKELY_HALLUCINATED"),
        ("What is photosynthesis?", "Photosynthesis is the process by which green plants convert sunlight into chemical energy using chlorophyll.", "VERIFIED"),
        ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by Charles Dickens in 1920.", "LIKELY_HALLUCINATED"),
    ]
)
def test_regression_statements(query: str, response: str, expected_risk: str):
    """Verify canonical regression prompts pass through real pipeline."""
    res = client.post("/api/v1/analyze", json={"query": query, "response": response, "model_name": "gpt-4"})
    assert res.status_code == 200
    data = res.json()
    assert data["risk_level"] == expected_risk
