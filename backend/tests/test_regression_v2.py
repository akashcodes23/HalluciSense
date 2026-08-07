"""Pytest Test Suite for Phase 25 Regression Suite v2 & Quality Gates."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.core.engine.tracer import PipelineTracer, get_latest_trace
from evaluation.phase25.retrieval_diagnostics import run_retrieval_diagnostics
from evaluation.phase25.nli_diagnostics import run_nli_diagnostics

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DATA_DIR = BASE_DIR / "evaluation_data"


def test_regression_v2_dataset_exists():
    """Verify regression_suite_v2.jsonl contains 1000 records."""
    path = EVAL_DATA_DIR / "regression_suite_v2.jsonl"
    assert path.exists()

    count = 0
    categories = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
                rec = json.loads(line)
                assert "test_id" in rec
                assert "category" in rec
                categories.add(rec["category"])

    assert count >= 1000
    assert len(categories) == 17


def test_gold_longform_dataset_exists():
    """Verify gold_longform_dataset.jsonl contains 500 records."""
    path = EVAL_DATA_DIR / "gold_longform_dataset.jsonl"
    assert path.exists()

    count = 0
    domains = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
                rec = json.loads(line)
                assert "prompt_id" in rec
                assert "domain" in rec
                domains.add(rec["domain"])

    assert count >= 500
    assert len(domains) == 10


def test_photosynthesis_resolution():
    """Verify Photosynthesis complex definition is correctly resolved as VERIFIED (H < 0.35)."""
    pipeline = HallucinationDetectionPipeline()
    text = "Photosynthesis is the process by which green plants convert sunlight into chemical energy using chlorophyll."
    report = pipeline.analyze(text=text)

    assert float(report.overall_h_score) < 0.35
    assert str(report.overall_risk_level.value) == "VERIFIED"


def test_pipeline_tracer_generation():
    """Verify PipelineTracer records stage timings and persists JSON."""
    tracer = PipelineTracer()
    tracer.record_stage("unit_test_stage", 12.5, {"test_key": "test_val"}, confidence=0.95)
    payload = tracer.finalize(final_h_score=0.10, risk_level="VERIFIED", root_cause="VERIFIED")

    assert payload["trace_id"] == tracer.trace_id
    assert "unit_test_stage" in payload["stages"]
    assert payload["summary"]["final_h_score"] == 0.10

    latest = get_latest_trace()
    assert latest is not None
    assert "summary" in latest


def test_ir_diagnostics_execution():
    """Verify retrieval_diagnostics calculates IR metrics."""
    res = run_retrieval_diagnostics(["The capital of France is Paris."])
    assert "recall_at_5" in res
    assert "mrr" in res
    assert res["recall_at_5"] >= 0.0


def test_nli_diagnostics_execution():
    """Verify nli_diagnostics calculates NLI metric distributions."""
    pairs = [
        {"claim": "The capital of France is Paris.", "evidence": "Paris is the capital of France.", "expected_label": "entailment"}
    ]
    res = run_nli_diagnostics(pairs)
    assert "mean_entailment_prob" in res
    assert res["mean_entailment_prob"] >= 0.0
