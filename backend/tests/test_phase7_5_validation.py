"""Exhaustive Unit Test Suite for Phase 7.5 Production System Validation & Engineering Audit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.phase7_5.validation_engine import (
    run_deployment_verification,
    run_api_verification,
    run_frontend_verification,
    run_curated_prompt_evaluation,
    run_performance_benchmarking,
    run_stress_testing,
    run_explainability_audit,
    run_engineering_bug_audit,
    run_deployment_readiness_audit,
    run_final_deployment_signoff,
)

BASE_DIR = Path(__file__).resolve().parent.parent
GOLD_SET_PATH = BASE_DIR / "evaluation_data" / "gold_regression_set.jsonl"


def test_gold_regression_set_exists():
    """Verify 200+ Gold Labeled Regression Set file exists and contains valid JSON lines."""
    assert GOLD_SET_PATH.exists()

    count = 0
    categories = set()
    with open(GOLD_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
                rec = json.loads(line)
                assert "prompt_id" in rec
                assert "category_code" in rec
                categories.add(rec["category_code"])

    assert count >= 200
    assert len(categories) == 20


def test_deployment_verification():
    """Verify local deployment audit."""
    res = run_deployment_verification()
    assert res["status"] in ["HEALTHY", "DEGRADED"]
    assert res["pipeline_loaded"] is True


def test_api_verification():
    """Verify API endpoint automated tests."""
    res = run_api_verification()
    assert res["api_resiliency_status"] == "PASSED"
    assert "/predict" in res["endpoints"]
    assert res["endpoints"]["/predict"]["status_code"] == 200


def test_frontend_verification():
    """Verify frontend UI audit report."""
    res = run_frontend_verification()
    assert res["nextjs_connected"] is True
    assert res["confidence_visualizer_verified"] is True


def test_performance_benchmarking():
    """Verify performance latency benchmark outputs."""
    res = run_performance_benchmarking()
    assert "p50_latency_ms" in res
    assert "p95_latency_ms" in res
    assert res["p50_latency_ms"] >= 0.0


def test_explainability_audit():
    """Verify explainability field presence."""
    res = run_explainability_audit()
    assert res["is_complete"] is True


def test_final_deployment_signoff():
    """Verify final deployment decision generation."""
    res = run_final_deployment_signoff()
    assert res["decision"] in ["GO", "NO GO"]
    assert len(res["checklist"]) == 8
