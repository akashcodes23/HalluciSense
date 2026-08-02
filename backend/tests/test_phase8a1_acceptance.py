"""Exhaustive Unit Test Suite for Phase 8A.1 Scientific Acceptance Validation & Pre-Publication Audit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.phase8a1.acceptance_engine import (
    run_api_validation,
    run_retrieval_validation,
    run_pillar1_validation,
    run_pillar2_validation,
    run_hybrid_validation,
    run_explainability_validation,
    run_robustness_validation,
    run_performance_validation,
    run_reproducibility_audit,
    run_publication_readiness_review,
    run_final_acceptance_report,
)

BASE_DIR = Path(__file__).resolve().parent.parent
SUITE_PATH = BASE_DIR / "evaluation_data" / "phase8a1_acceptance_suite.jsonl"
RESULTS_DIR = BASE_DIR / "evaluation_results" / "phase8a1"
DOCS_DIR = BASE_DIR.parent / "docs"


def test_acceptance_suite_file_exists():
    """Verify 500+ Scientific Acceptance Suite file exists and contains 20 categories."""
    assert SUITE_PATH.exists()

    count = 0
    categories = set()
    with open(SUITE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
                rec = json.loads(line)
                assert "prompt_id" in rec
                assert "category_code" in rec
                categories.add(rec["category_code"])

    assert count >= 500
    assert len(categories) == 20


def test_api_validation():
    """Verify REST API validation audit."""
    res = run_api_validation(RESULTS_DIR)
    assert res["api_validation_status"] == "PASSED"
    assert "/predict" in res["endpoints"]


def test_pillar1_validation():
    """Verify Pillar 1 audit report."""
    res = run_pillar1_validation(RESULTS_DIR)
    assert res["status"] == "PASSED"
    assert res["mean_entailment_verified"] is True


def test_pillar2_validation():
    """Verify Pillar 2 audit report."""
    res = run_pillar2_validation(RESULTS_DIR)
    assert res["status"] == "PASSED"
    assert res["pairwise_nli_verified"] is True


def test_hybrid_validation():
    """Verify hybrid fusion engine audit."""
    res = run_hybrid_validation(RESULTS_DIR)
    assert res["status"] == "PASSED"
    assert res["hybrid_19_feature_vector_assembled"] is True


def test_explainability_validation():
    """Verify explainability audit report."""
    res = run_explainability_validation(RESULTS_DIR)
    assert res["valid"] is True


def test_robustness_validation():
    """Verify system robustness edge case audit."""
    res = run_robustness_validation(RESULTS_DIR)
    assert res["robustness_status"] == "PASSED"


def test_performance_validation():
    """Verify performance benchmarking outputs."""
    res = run_performance_validation(RESULTS_DIR)
    assert res["performance_status"] == "ACCEPTABLE"
    assert "p50_latency_ms" in res


def test_reproducibility_audit():
    """Verify reproducibility audit report."""
    res = run_reproducibility_audit(DOCS_DIR)
    assert res["status"] == "PASSED"


def test_publication_readiness_review():
    """Verify publication readiness review."""
    res = run_publication_readiness_review(DOCS_DIR)
    assert res["status"] == "APPROVED"


def test_final_acceptance_report():
    """Verify master final acceptance decision report."""
    res = run_final_acceptance_report(RESULTS_DIR)
    assert res["decision"] in ["GO", "GO WITH MINOR OBSERVATIONS", "NO GO"]
