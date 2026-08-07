"""Exhaustive Unit Test Suite for Phase 7 Final Packaging & Productionization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.pipeline import pipeline
from app.main import app
from app.models.registry import registry

client = TestClient(app)


def test_version_json_exists():
    """Verify version.json metadata file."""
    version_file = Path(__file__).resolve().parent.parent / "config" / "version.json"
    assert version_file.exists()

    with open(version_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["framework"] == "HalluciSense"
    assert "version" in data
    assert "dataset_fingerprints" in data


def test_model_registry_verification():
    """Verify model registry loading and checksum verification."""
    checksums = registry.verify_checksums()
    assert checksums["hybrid_classifier_exists"] is True
    assert checksums["hybrid_scaler_exists"] is True


def test_production_pipeline_inference():
    """Verify production inference pipeline end-to-end output."""
    res = pipeline.predict(
        response_text="Albert Einstein discovered general relativity in 1915.",
        claims=["Albert Einstein discovered general relativity in 1915."],
    )

    assert "is_hallucinated" in res
    assert "hallucination_probability" in res
    assert "explanation" in res
    assert 0.0 <= res["hallucination_probability"] <= 1.0


def test_fastapi_predict_endpoint():
    """Verify POST /api/v1/hallucisense/predict endpoint."""
    payload = {
        "response_text": "The Eiffel Tower is located in Paris, France.",
        "claims": ["The Eiffel Tower is located in Paris, France."],
    }
    response = client.post("/api/v1/hallucisense/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "is_hallucinated" in data
    assert "hallucination_probability" in data


def test_fastapi_explain_endpoint():
    """Verify POST /api/v1/hallucisense/explain endpoint."""
    payload = {
        "response_text": "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    }
    response = client.post("/api/v1/hallucisense/explain", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "explanation_breakdown" in data


def test_fastapi_health_endpoint():
    """Verify GET /api/v1/hallucisense/health endpoint."""
    response = client.get("/api/v1/hallucisense/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"].lower() in ["ok", "healthy", "degraded"]


def test_fastapi_version_endpoint():
    """Verify GET /api/v1/hallucisense/version endpoint."""
    response = client.get("/api/v1/hallucisense/version")
    assert response.status_code == 200

    data = response.json()
    assert "framework" in data
