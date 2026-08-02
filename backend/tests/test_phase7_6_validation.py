"""Exhaustive Unit Test Suite for Phase 7.6 Real-World Deployment Validation & Acceptance Testing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.phase7_6.realworld_validation_engine import (
    run_deployment_startup_report,
    run_api_validation_report,
    run_frontend_validation,
    run_real_world_test_suite,
    run_edge_case_validation,
    run_explainability_validation,
    run_performance_benchmark,
    run_stress_test_results,
    run_failure_recovery_report,
    run_final_deployment_audit,
)

BASE_DIR = Path(__file__).resolve().parent.parent
REAL_WORLD_DATA_PATH = BASE_DIR / "evaluation_data" / "real_world_multi_llm_set.jsonl"
OUT_DIR = BASE_DIR / "evaluation_results" / "phase7_6"


def test_real_world_multi_llm_set_exists():
    """Verify 350 real-world multi-LLM response dataset file exists."""
    assert REAL_WORLD_DATA_PATH.exists()

    count = 0
    models = set()
    domains = set()

    with open(REAL_WORLD_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
                rec = json.loads(line)
                assert "response_id" in rec
                assert "llm_model" in rec
                assert "domain" in rec
                models.add(rec["llm_model"])
                domains.add(rec["domain"])

    assert count >= 300
    assert len(models) >= 7
    assert len(domains) == 10


def test_deployment_startup_report():
    """Verify deployment startup audit."""
    res = run_deployment_startup_report(OUT_DIR)
    assert res["backend_started"] is True
    assert res["model_registry_loaded"] is True


def test_api_validation_report():
    """Verify REST API endpoint validation report."""
    res = run_api_validation_report(OUT_DIR)
    assert res["api_validation_status"] == "PASSED"
    assert "/predict" in res["endpoints"]


def test_frontend_validation():
    """Verify frontend UI component audit."""
    res = run_frontend_validation(OUT_DIR)
    assert res["upload_workflow"] == "VERIFIED"
    assert res["dark_mode"] == "VERIFIED"


def test_edge_case_validation():
    """Verify 19 edge case tests."""
    res = run_edge_case_validation(OUT_DIR)
    assert res["edge_case_status"] == "PASSED"
    assert len(res["cases"]) >= 18


def test_explainability_validation():
    """Verify explainability fields completeness."""
    res = run_explainability_validation(OUT_DIR)
    assert res["valid"] is True


def test_performance_benchmark():
    """Verify performance benchmarking outputs."""
    res = run_performance_benchmark(OUT_DIR)
    assert "p50_latency_ms" in res
    assert "requests_per_second" in res


def test_failure_recovery_report():
    """Verify simulated failure recovery report."""
    res = run_failure_recovery_report(OUT_DIR)
    assert res["verdict"] == "PASSED"


def test_final_deployment_audit():
    """Verify master deployment decision audit."""
    res = run_final_deployment_audit(OUT_DIR)
    assert res["decision"] in ["GO", "NO GO"]
    assert res["checklist"]["blocking_issues"] == 0
