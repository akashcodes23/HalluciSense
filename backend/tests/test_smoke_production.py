"""Production Smoke Test Suite for HalluciSense Backend.

Tests live or local HTTP endpoints to verify:
1. System health & memory telemetry
2. System readiness & component initialization
3. Active hybrid model status & zero fallback
4. True claim factual verification
5. False claim hallucination detection
6. Fast cache retrieval
7. Dedicated hybrid 19-feature classifier execution
"""

import os
import time
import pytest
import requests

TARGET_URL = os.getenv("TARGET_URL", "https://hallucisense-production.up.railway.app").rstrip("/")


def test_production_health():
    """Verify /health returns 200 and reports active hybrid model."""
    url = f"{TARGET_URL}/health"
    resp = requests.get(url, timeout=10)
    assert resp.status_code == 200, f"/health returned {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data.get("status") == "healthy"
    assert data.get("active_model") == "hybrid"
    assert data.get("hybrid_available") is True
    assert data.get("fallback_active") is False
    assert "memory_mb" in data


def test_production_readiness():
    """Verify /ready returns 200 and confirms pipeline readiness."""
    url = f"{TARGET_URL}/ready"
    resp = requests.get(url, timeout=10)
    assert resp.status_code == 200, f"/ready returned {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data.get("status") == "ready"
    assert data.get("ready") is True
    assert data.get("active_model") == "hybrid"
    assert data.get("hybrid_available") is True
    assert data.get("fallback_active") is False


def test_production_true_claim_analysis():
    """Verify factual claim is classified as VERIFIED."""
    url = f"{TARGET_URL}/api/v1/analyze"
    payload = {
        "query": "What is the capital of Karnataka?",
        "response": "The capital of Karnataka is Bengaluru.",
    }
    resp = requests.post(url, json=payload, timeout=20)
    assert resp.status_code == 200, f"/analyze returned {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data.get("risk_level") == "VERIFIED"
    assert data.get("overall_h_score") < 0.50
    assert "trace_id" in data
    assert len(data.get("evidence", [])) > 0


def test_production_false_claim_analysis():
    """Verify false claim is classified as LIKELY_HALLUCINATED."""
    url = f"{TARGET_URL}/api/v1/analyze"
    payload = {
        "query": "What is the capital of Karnataka?",
        "response": "The capital of Karnataka is Mumbai.",
    }
    resp = requests.post(url, json=payload, timeout=20)
    assert resp.status_code == 200, f"/analyze returned {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data.get("risk_level") == "LIKELY_HALLUCINATED"
    assert data.get("overall_h_score") > 0.50
    assert "trace_id" in data


def test_production_cached_analysis():
    """Verify repeated request hits cache and returns quickly."""
    url = f"{TARGET_URL}/api/v1/analyze"
    payload = {
        "query": "What is the capital of Karnataka?",
        "response": "The capital of Karnataka is Bengaluru.",
    }
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=10)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("risk_level") == "VERIFIED"
    # Cached processing time reported by server is typically under 50ms
    assert data.get("processing_time_ms", elapsed_ms) < 200


def test_production_hybrid_direct_predict():
    """Verify dedicated hybrid classifier endpoint executes."""
    url = f"{TARGET_URL}/api/v1/hallucisense/predict"
    payload = {
        "response_text": "The capital of Karnataka is Bengaluru.",
    }
    resp = requests.post(url, json=payload, timeout=15)
    assert resp.status_code == 200, f"/predict returned {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data.get("is_hallucinated") is False
    assert data.get("operating_threshold") == 0.54
    assert data.get("hallucination_probability") < 0.54
    assert "explanation" in data


if __name__ == "__main__":
    print(f"Running production smoke tests against {TARGET_URL}...")
    test_production_health()
    print("✓ /health passed")
    test_production_readiness()
    print("✓ /ready passed")
    test_production_true_claim_analysis()
    print("✓ True claim analysis passed (VERIFIED)")
    test_production_false_claim_analysis()
    print("✓ False claim analysis passed (LIKELY_HALLUCINATED)")
    test_production_cached_analysis()
    print("✓ Cached repeat passed")
    test_production_hybrid_direct_predict()
    print("✓ Hybrid direct prediction passed (threshold=0.54)")
    print("\nALL SMOKE TESTS PASSED SUCCESSFULLY!")
