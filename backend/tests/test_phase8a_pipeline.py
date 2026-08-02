"""Exhaustive Unit Test Suite for Phase 8A Production Inference Integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.inference.claim_extractor import extract_claims
from app.core.inference.pillar1_engine import Pillar1Engine
from app.core.inference.pillar2_engine import Pillar2Engine
from app.core.pipeline import pipeline
from app.main import app

client = TestClient(app)


def test_claim_extractor_sentence_segmentation():
    """Verify claim extractor handles abbreviations without premature sentence splitting."""
    text = "Dr. Smith visited the U.S. capital in Washington. He met with e.g. several representatives at 5 p.m. Yesterday was Tuesday."
    claims = extract_claims(text)

    assert len(claims) >= 2
    assert claims[0]["claim_id"] == 0
    assert "Dr. Smith" in claims[0]["text"]
    assert "Washington" in claims[0]["text"]


def test_pillar1_engine_real_inference():
    """Verify Pillar 1 Engine extracts 5 evidence features and base P1 probability."""
    engine = Pillar1Engine()
    claims = [{"claim_id": 0, "text": "Paris is the capital of France."}]
    feats, p1_prob, attributions = engine.extract_features_and_predict(claims)

    assert len(feats) == 5
    assert 0.0 <= p1_prob <= 1.0
    assert len(attributions) == 1
    assert "top_entailment" in attributions[0]


def test_pillar2_engine_real_inference():
    """Verify Pillar 2 Engine extracts 5 structural features and base P2 probability."""
    engine = Pillar2Engine()
    claims = [
        {"claim_id": 0, "text": "The sun rises in the east."},
        {"claim_id": 1, "text": "The sun sets in the west."},
    ]
    feats, p2_prob, diagnostics = engine.extract_features_and_predict(claims)

    assert len(feats) == 5
    assert 0.0 <= p2_prob <= 1.0
    assert "graph_stats" in diagnostics


def test_production_pipeline_real_end_to_end():
    """Verify unified production pipeline executes real research models without synthetic defaults."""
    res = pipeline.predict("Water boils at 100 degrees Celsius at sea level.")

    assert "is_hallucinated" in res
    assert "hallucination_probability" in res
    assert 0.0 <= res["hallucination_probability"] <= 1.0
    assert res["operating_threshold"] == 0.54

    exp = res["explanation"]
    assert "pillar_contributions" in exp
    assert "pillar_1_probability" in exp["pillar_contributions"]
    assert "pillar_2_probability" in exp["pillar_contributions"]
    assert "structural_analysis" in exp
    assert "claim_analysis" in exp


def test_fastapi_predict_endpoint_real_pipeline():
    """Verify POST /api/v1/hallucisense/predict returns real prediction."""
    payload = {"response_text": "Albert Einstein won the Nobel Prize in Physics in 1921."}
    response = client.post("/api/v1/hallucisense/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "is_hallucinated" in data
    assert "explanation" in data
    assert "pillar_contributions" in data["explanation"]


def test_fastapi_explain_endpoint_real_pipeline():
    """Verify POST /api/v1/hallucisense/explain returns rich explanation."""
    payload = {"response_text": "The Earth is the third planet from the Sun."}
    response = client.post("/api/v1/hallucisense/explain", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "explanation_breakdown" in data
    assert "primary_driver" in data["explanation_breakdown"]
