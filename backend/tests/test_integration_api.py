"""Integration Tests for REST Endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_predict_endpoint_factual():
    response = client.post(
        "/api/v1/hallucisense/predict",
        json={"response_text": "Paris is the capital of France."},
    )
    assert response.status_code == 200
    data = response.json()
    assert "hallucination_probability" in data
    assert "is_hallucinated" in data
    assert "explanation" in data
    assert data["is_hallucinated"] is False
    assert data["explanation"]["verdict"] == "FACTUAL"


def test_explain_endpoint():
    response = client.post(
        "/api/v1/hallucisense/explain",
        json={"response_text": "Paris is the capital of France."},
    )
    assert response.status_code == 200
    data = response.json()
    assert "explanation_breakdown" in data


def test_health_check_endpoint():
    response = client.get("/api/v1/hallucisense/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["active_model"] == "hybrid"
    assert data["loaded_successfully"] is True


def test_version_endpoint():
    response = client.get("/api/v1/hallucisense/version")
    assert response.status_code == 200
    data = response.json()
    assert data["framework"] == "HalluciSense"
    assert "git_sha" in data


def test_mlops_dashboard_endpoint():
    response = client.get("/api/v1/mlops/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert "uptime_seconds" in data
    assert "drift" in data
    assert "latency" in data
