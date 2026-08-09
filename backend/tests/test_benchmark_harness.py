"""Unit Tests for Phase 1 Benchmark Harness (scripts/benchmark_phase1.py).

Verifies:
1. Benchmark harness passes BOTH query and response to pipeline.analyze().
2. Every output record contains all 21 required telemetry & pillar metric fields.
3. Machine-readable JSON report file is saved and schema-compliant.
"""

import json
from pathlib import Path
import pytest
from app.core.engine.pipeline import HallucinationDetectionPipeline
from scripts.benchmark_phase1 import run_case, CASES, JSON_OUTPUT_PATH


def test_benchmark_run_case_query_and_response_integration():
    """Verify run_case passes query and response and extracts complete metrics dictionary."""
    pipeline = HallucinationDetectionPipeline()
    name, query, response = CASES[0]  # Case A: Correct

    record = run_case(pipeline, name, query, response)

    required_keys = [
        "case_name",
        "query",
        "response",
        "total_latency",
        "retrieval_total_latency",
        "wikipedia_latency",
        "bm25_latency",
        "reranker_latency",
        "cache_hits",
        "cache_misses",
        "cache_hit_rate",
        "search_requests",
        "extraction_requests",
        "nli_pair_count",
        "nli_batch_count",
        "nli_inference_latency",
        "p1_score",
        "p2_score",
        "p3_score",
        "overall_h_score",
        "overall_risk_level",
    ]

    for key in required_keys:
        assert key in record, f"Missing required metric key: '{key}'"

    assert record["case_name"] == name
    assert record["query"] == query
    assert record["response"] == response
    assert isinstance(record["total_latency"], float)
    assert record["p1_score"] is not None
    assert 0.0 <= record["p1_score"] <= 1.0
    assert 0.0 <= record["overall_h_score"] <= 1.0


def test_benchmark_json_output_persistence():
    """Verify JSON_OUTPUT_PATH is created and contains records for all 5 test cases."""
    assert JSON_OUTPUT_PATH.exists(), f"Expected JSON report at {JSON_OUTPUT_PATH}"

    with open(JSON_OUTPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) == 5

    case_names = [item["case_name"] for item in data]
    expected_names = ["A_correct", "B_obvious_hallucination", "C_partially_incorrect", "D_highly_confident_hallucination", "E_ambiguous"]
    assert case_names == expected_names
