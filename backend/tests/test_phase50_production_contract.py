"""Phase 50 — Production API Schema & Operational Contract Tests."""

import pytest
from fastapi.testclient import TestClient
from app.main import create_application


def test_production_api_canonical_response_contract():
    """Verify production API response schema adherence and pillar execution status."""
    app = create_application()
    client = TestClient(app)

    resp = client.post("/api/v1/analyze", json={"response": "The capital of France is Paris."})
    assert resp.status_code == 200
    data = resp.json()

    assert "overall_h_score" in data
    assert "risk_level" in data
    assert "pillar_status" in data
    assert "pillar_scores" in data

    p_status = data["pillar_status"]
    assert p_status.get("p1_status") == "EXECUTED"
    assert p_status.get("p2_status") == "EXECUTED"
    assert p_status.get("p3_status") == "EXECUTED"
    assert p_status.get("is_full_analysis") is True
