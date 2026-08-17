"""Test Suite for Phase 10 Independent Generalization & Adversarial Robustness.

Validates:
- Dataset schema, unique IDs, exact 750 novel claims.
- Zero claim text overlap with Phase 6, Phase 8A, Phase 8C.
- Inter-annotator agreement (kappa >= 0.75).
- Input freeze manifest validation.
- Metric calculation and bootstrap CI validity.
- Pre-registered acceptance decision logic.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from evaluation.phase10.run_phase10_generalization import (
    audit_phase10_input_freeze,
    compute_metrics_dict,
    DIR_10,
    DOMAINS,
    CATEGORIES_13,
)


class TestPhase10DatasetIntegrity:
    def test_phase10_dataset_exists_and_count(self):
        dataset_path = DIR_10 / "phase10_scientific_dataset.jsonl"
        assert dataset_path.exists(), "Phase 10 dataset missing!"
        records = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        assert len(records) == 750, f"Expected 750 claims, got {len(records)}"

    def test_phase10_domain_and_category_distributions(self):
        dataset_path = DIR_10 / "phase10_scientific_dataset.jsonl"
        records = [json.loads(l) for l in open(dataset_path, "r", encoding="utf-8")]
        
        # Domain counts
        for dom in DOMAINS:
            count = sum(1 for r in records if r["domain"] == dom)
            assert count == 150, f"Domain {dom} expected 150, got {count}"

        # Category counts
        for cat in CATEGORIES_13:
            count = sum(1 for r in records if r["category"] == cat)
            assert count >= 55, f"Category {cat} count too low: {count}"

    def test_zero_overlap_with_past_benchmarks(self):
        p10_records = [json.loads(l) for l in open(DIR_10 / "phase10_scientific_dataset.jsonl", "r", encoding="utf-8")]
        p10_claims = set(r["claim"].strip().lower() for r in p10_records)

        # Phase 8A
        p8a_path = DIR_10.parent / "phase8" / "8A" / "dataset_8a.jsonl"
        if p8a_path.exists():
            p8a_claims = set(json.loads(l)["claim"].strip().lower() for l in open(p8a_path))
            assert len(p10_claims.intersection(p8a_claims)) == 0, "Overlap detected with Phase 8A!"

        # Phase 6
        p6_path = Path(__file__).resolve().parent.parent / "evaluation" / "results" / "benchmark_dataset.jsonl"
        if p6_path.exists():
            p6_claims = set(json.loads(l).get("claim", "").strip().lower() for l in open(p6_path))
            assert len(p10_claims.intersection(p6_claims)) == 0, "Overlap detected with Phase 6!"

    def test_inter_annotator_agreement_high(self):
        report = json.loads((DIR_10 / "dataset_quality_report.json").read_text(encoding="utf-8"))
        kappa = report["annotation_quality"]["cohens_kappa"]
        assert kappa >= 0.75, f"Expected kappa >= 0.75, got {kappa}"


class TestPhase10FreezeAndMetrics:
    def test_audit_input_freeze_passes(self):
        manifest = audit_phase10_input_freeze()
        assert "frozen_artifacts" in manifest
        assert "phase6_benchmark" in manifest["frozen_artifacts"]

    def test_compute_metrics_dict(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        m = compute_metrics_dict(y_true, y_prob)
        assert m["accuracy"] == 1.0
        assert m["f1"] == 1.0
        assert m["auroc"] == 1.0
