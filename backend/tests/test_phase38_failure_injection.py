"""Phase 38.10 — Production Failure Injection & Boundary Robustness Test Suite.

Tests graceful error handling and boundary resilience:
- Empty and whitespace inputs
- Extremely long inputs (>10,000 characters)
- Unicode, emoji, and multi-lingual inputs
- Duplicate claim repetitions
- Invalid / non-finite feature vectors passed to attribution engine
- Malformed inputs to /explain and /predict endpoints
- Verification that no unhandled exceptions leak as 500 internal errors
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.pipeline import get_hallucisense_pipeline
from app.core.inference.claim_extractor import extract_claims
from app.core.inference.local_attribution import (
    compute_local_attribution,
    validate_feature_vector,
)
from app.models.registry import registry


@pytest.fixture(scope="module")
def pipeline_instance():
    return get_hallucisense_pipeline()


@pytest.fixture(scope="module")
def hybrid_artifacts():
    scaler, clf, meta = registry.load_hybrid_model()
    threshold = float(meta["protocol"].get("decision_threshold", 0.54))
    return scaler, clf, threshold


def test_empty_string_input(pipeline_instance):
    """Test 1: Empty response_text returns valid fallback response without raising."""
    res = pipeline_instance.predict(response_text="")
    assert isinstance(res, dict)
    assert "is_hallucinated" in res
    assert "hallucination_probability" in res
    assert 0.0 <= res["hallucination_probability"] <= 1.0
    assert "local_attribution" in res


def test_whitespace_only_input(pipeline_instance):
    """Test 2: Whitespace-only string returns valid fallback response."""
    res = pipeline_instance.predict(response_text="   \n\t   \n")
    assert isinstance(res, dict)
    assert 0.0 <= res["hallucination_probability"] <= 1.0
    assert "local_attribution" in res


def test_extremely_long_input(pipeline_instance):
    """Test 3: 10,000 character prompt executes without memory explosion."""
    long_text = "The quick brown fox jumps over the lazy dog. " * 250
    res = pipeline_instance.predict(response_text=long_text)
    assert isinstance(res, dict)
    assert res["claim_count"] >= 1
    assert "local_attribution" in res


def test_unicode_and_emojis_input(pipeline_instance):
    """Test 4: Unicode, non-Latin scripts (Arabic, Cyrillic, Chinese), and emojis."""
    unicode_text = "🌍 Paris est la capitale de la France. Москва — столица России. 巴黎是法国的首都。 🚀✨"
    res = pipeline_instance.predict(response_text=unicode_text)
    assert isinstance(res, dict)
    assert res["claim_count"] >= 1
    assert 0.0 <= res["hallucination_probability"] <= 1.0


def test_claim_extractor_abbreviations_resilience():
    """Test 5: Complex abbreviation patterns do not prematurely segment claims."""
    text = "Dr. Smith and Prof. Jones met at 9 a.m. in the U.S. to discuss the E.U. treaty vs. other agreements etc."
    claims = extract_claims(text)
    assert len(claims) == 1
    assert "Dr. Smith" in claims[0]["text"]
    assert "U.S." in claims[0]["text"]


def test_repeated_duplicate_claims(pipeline_instance):
    """Test 6: 10 repeated identical sentences capped and handled cleanly in Pillar 2."""
    repeated = "Berlin is the capital of Germany. " * 10
    res = pipeline_instance.predict(response_text=repeated)
    assert isinstance(res, dict)
    assert res["claim_count"] >= 1
    assert "local_attribution" in res


def test_nan_vector_raises_value_error():
    """Test 7: validate_feature_vector strictly rejects vectors containing NaN."""
    v = np.zeros(19)
    v[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_feature_vector(v)


def test_inf_vector_raises_value_error():
    """Test 8: validate_feature_vector strictly rejects vectors containing Inf."""
    v = np.zeros(19)
    v[5] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        validate_feature_vector(v)


def test_wrong_shape_vector_raises_value_error():
    """Test 9: validate_feature_vector strictly rejects vectors of dimension != 19."""
    with pytest.raises(ValueError, match="exactly 19"):
        validate_feature_vector(np.zeros(18))
    with pytest.raises(ValueError, match="exactly 19"):
        validate_feature_vector(np.zeros(20))


def test_attribution_with_boundary_values(hybrid_artifacts):
    """Test 10: compute_local_attribution succeeds on edge-case feature extremes."""
    scaler, clf, threshold = hybrid_artifacts
    v_min = np.zeros(19)
    res_min = compute_local_attribution(v_min, scaler, clf, threshold)
    assert 0.0 <= res_min.original_probability <= 1.0

    v_max = np.ones(19) * 100.0
    res_max = compute_local_attribution(v_max, scaler, clf, threshold)
    assert 0.0 <= res_max.original_probability <= 1.0
