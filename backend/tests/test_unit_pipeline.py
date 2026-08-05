"""Unit Tests for HalluciSense Production Pipeline Components."""

import pytest
import numpy as np
from app.core.inference.claim_extractor import extract_claims
from app.core.inference.pillar1_engine import _relevance_to_nli, Pillar1Engine
from app.models.registry import ModelRegistry, registry


def test_claim_extraction():
    text = "Paris is the capital of France. The moon is made of green cheese."
    claims = extract_claims(text)
    assert len(claims) >= 1
    assert any("Paris" in c.get("text", "") for c in claims)


def test_relevance_to_nli_conversion():
    # High relevance
    ent_hi, con_hi, neu_hi = _relevance_to_nli(0.999)
    assert 0.25 <= ent_hi <= 1.0
    assert con_hi < 0.05
    assert abs((ent_hi + con_hi + neu_hi) - 1.0) < 1e-5

    # Low relevance
    ent_lo, con_lo, neu_lo = _relevance_to_nli(0.01)
    assert ent_lo < 0.01
    assert con_lo > 0.5
    assert abs((ent_lo + con_lo + neu_lo) - 1.0) < 1e-5


def test_model_registry_resolution():
    checksums = registry.verify_checksums()
    assert checksums["hybrid_classifier_exists"] is True
    assert checksums["hybrid_scaler_exists"] is True
    assert checksums["hybrid_classifier_valid_size"] is True


def test_pillar1_engine_features():
    p1 = Pillar1Engine()
    claims_struct = [{"claim_id": 0, "text": "Paris is the capital of France."}]
    feats, prob, evidence = p1.extract_features_and_predict(claims_struct)

    assert len(feats) == 5
    assert 0.0 <= prob <= 1.0
    assert isinstance(evidence, list)
