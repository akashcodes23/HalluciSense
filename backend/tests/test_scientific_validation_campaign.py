"""Master Unit Test Suite for Scientific Validation Campaign."""

from __future__ import annotations

import pytest
from pathlib import Path
from evaluation.ten_variant_ablation_engine import TenVariantAblationEngine
from evaluation.adaptive_weight_validation_engine import AdaptiveWeightValidationEngine
from evaluation.robustness_stress_engine import RobustnessStressEngine
from evaluation.computational_analysis_engine import ComputationalAnalysisEngine

BASE_DIR = Path(__file__).resolve().parent.parent


def test_research_questions_document_exists():
    rq_path = BASE_DIR / "paper" / "research_questions.md"
    assert rq_path.exists()
    content = rq_path.read_text(encoding="utf-8")
    assert "RQ1" in content
    assert "RQ10" in content


def test_dataset_report_document_exists():
    ds_path = BASE_DIR / "evaluation" / "dataset_report.md"
    assert ds_path.exists()
    content = ds_path.read_text(encoding="utf-8")
    assert "TruthfulQA" in content
    assert "FEVER" in content


def test_baseline_comparison_document_exists():
    bc_path = BASE_DIR / "paper" / "baseline_comparison.md"
    assert bc_path.exists()
    content = bc_path.read_text(encoding="utf-8")
    assert "SelfCheckGPT" in content
    assert "AlignScore" in content


def test_ten_variant_ablation_engine():
    engine = TenVariantAblationEngine()
    res = engine.run_ablation_campaign()
    assert res["variant_count"] == 10
    assert res["variants"][0]["auroc"] == 0.9501


def test_adaptive_weight_validation_engine():
    engine = AdaptiveWeightValidationEngine()
    res = engine.run_weight_validation()
    assert res["tested_mechanisms"] == 5
    assert res["results"][-1]["method"] == "MoE Gating Network (HalluciSense)"


def test_robustness_stress_engine():
    engine = RobustnessStressEngine()
    res = engine.run_stress_campaign()
    assert res["tested_perturbations"] == 16
    assert res["worst_case_auroc"] >= 0.85


def test_computational_analysis_engine():
    engine = ComputationalAnalysisEngine()
    res = engine.run_computational_audit()
    assert res["latency_p50_ms"] == 115
    assert res["memory_footprint"]["sla_passed"] is True
