"""
Integration tests for HalluciSense Pillar 2 — Modules 10.10 & 10.11: FastAPI Endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import create_application

app = create_application()
client = TestClient(app)


def test_pillar2_health():
    res = client.get("/api/v1/pillar2/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert "Wikipedia" in data["evidence_providers"]


def test_pillar2_version():
    res = client.get("/api/v1/pillar2/version")
    assert res.status_code == 200
    data = res.json()
    assert data["version"] == "10.0.0"


def test_pillar2_providers():
    res = client.get("/api/v1/pillar2/providers")
    assert res.status_code == 200
    data = res.json()
    assert "Wikipedia" in data["evidence_providers"]
    assert "Gemini" in data["llm_verifiers"]


def test_pillar2_extract_claims():
    payload = {"text": "Albert Einstein discovered relativity in 1905."}
    res = client.post("/api/v1/pillar2/claims", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total_claims"] == 1
    assert data["extracted_claims"][0]["claim_text"] == payload["text"]


def test_pillar2_evidence_retrieval():
    payload = {"query": "CRISPR gene editing", "max_results_per_provider": 1}
    res = client.post("/api/v1/pillar2/evidence", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total_retrieved"] > 0


def test_pillar2_hscore():
    payload = {
        "pillar1_probability": 0.25,
        "support_ratio": 0.90,
        "contradiction_ratio": 0.0,
        "authority_score": 0.95,
        "consensus_confidence": 0.90,
        "max_contradiction_severity": 0.0,
    }
    res = client.post("/api/v1/pillar2/hallucination-score", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert 0.0 <= data["hallucisense_score"] <= 50.0
    assert data["risk_category"] in ["VERY_LOW", "LOW"]


def test_pillar2_full_verify():
    payload = {
        "text": "Albert Einstein published relativity papers in 1905.",
        "pillar1_probability": 0.15,
    }
    res = client.post("/api/v1/pillar2/verify", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "verification_id" in data
    assert "hallucisense_score" in data
    assert "dashboard_ui" in data
    ui = data["dashboard_ui"]
    assert "risk_indicator" in ui
    assert "confidence_gauge" in ui
    assert len(ui["claim_cards"]) >= 1
    assert len(ui["timeline"]) >= 5
