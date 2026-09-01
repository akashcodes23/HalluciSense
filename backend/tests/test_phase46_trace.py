"""Phase 46 — Trace Schema & Multi-Pillar Response Tests."""

import pytest
from app.core.pipeline import HalluciSensePipeline

@pytest.fixture
def pipeline():
    return HalluciSensePipeline()

def test_pipeline_predict_trace_structure(pipeline):
    res = pipeline.predict(response_text="2 + 2 = 4.")
    assert "request_id" in res
    assert "trace_id" in res
    assert "is_hallucinated" in res
    assert "hallucination_probability" in res
    assert "verification_summary" in res
    
    summary = res["verification_summary"]
    assert "total_claims" in summary
    assert "verified_claims" in summary
    assert summary["verified_claims"] >= 1
    assert summary["primary_status"] == "ALL_VERIFIED"
