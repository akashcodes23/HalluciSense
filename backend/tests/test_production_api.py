"""HalluciSense v1.0 Production API Unit & Integration Test Suite."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "HalluciSense API is running"
    assert "version" in data


def test_health_endpoints():
    for endpoint in ["/health", "/healthz", "/ready", "/readyz"]:
        response = client.get(endpoint)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


def test_canonical_analyze_endpoint():
    payload = {
        "query": "What is the capital of France?",
        "response": "The capital of France is Paris.",
        "model_name": "GPT-4",
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "overall_h_score" in data
    assert "hallucination" in data
    assert "risk_level" in data
    assert "pillar_scores" in data
    assert "token_heatmap" in data
    assert "evidence" in data
    assert "processing_time_ms" in data
    assert data["version"] == "1.0.0"


def test_analyze_invalid_payload():
    payload = {"query": ""}
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422
