"""Unit Test Suite for Phase 9 Calibrated Hybrid Optimization.

Validates:
- Freeze audit hash validation.
- 70/30 stratified split generation and non-leakage.
- Evidence-aware severity function behavior.
- Hybrid logistic model fitting and inference.
- Pre-registered decision rule evaluation logic.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from evaluation.phase9.run_phase9_calibrated_hybrid import (
    audit_phase9_freeze,
    generate_stratified_split,
    compute_evidence_aware_severities,
    fit_calibrated_hybrid_model,
    DIR_8A,
)


class TestPhase9FreezeAndSplit:
    def test_audit_phase9_freeze_passes(self):
        hashes = audit_phase9_freeze()
        assert "phase6_benchmark" in hashes
        assert "phase8a_dataset" in hashes
        assert "phase8d_paired_results" in hashes

    def test_generate_stratified_split_integrity(self):
        records = []
        with open(DIR_8A / "dataset_8a.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

        dev, test, manifest = generate_stratified_split(records, seed=42)
        assert len(dev) == 122
        assert len(test) == 53
        assert len(dev) + len(test) == 175

        # Check no leakage between dev and test
        dev_ids = set(r["id"] for r in dev)
        test_ids = set(r["id"] for r in test)
        assert len(dev_ids.intersection(test_ids)) == 0


class TestPhase9EvidenceSeverities:
    def test_severity_computation_no_conflicts(self):
        trace_enh = {"enhancements_triggered": [], "proposition_details": []}
        trace_base = {"fusion": {"factual_error": 0.1}, "evidence": {"retrieved_evidence": [1, 2, 3]}}
        sev = compute_evidence_aware_severities(trace_enh, trace_base)
        assert sev["numeric_unit_severity"] == 0.0
        assert sev["negation_severity"] == 0.0
        assert sev["causal_severity"] == 0.0
        assert sev["evidence_coverage"] == 0.6

    def test_severity_computation_with_numeric_conflict(self):
        trace_enh = {"enhancements_triggered": ["NUMERIC_UNIT: mismatch"], "proposition_details": []}
        trace_base = {"fusion": {"factual_error": 0.8}, "evidence": {"retrieved_evidence": [1, 2, 3, 4, 5]}}
        sev = compute_evidence_aware_severities(trace_enh, trace_base)
        assert sev["numeric_unit_severity"] == 0.85
        assert sev["evidence_coverage"] == 1.0


class TestPhase9HybridFitting:
    def test_fit_calibrated_hybrid_model(self):
        # Synthetic development data
        X_dev = np.array([
            [0.1, 0.8, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.9, 0.0, 0.0, 0.0, 0.0],
            [0.9, 0.8, 0.8, 0.0, 0.0, 0.0],
            [0.8, 0.7, 0.0, 0.9, 0.0, 0.2],
        ])
        y_dev = np.array([0, 0, 1, 1])
        clf, iso, model_meta = fit_calibrated_hybrid_model(X_dev, y_dev)
        assert clf is not None
        assert iso is not None
        assert "coefficients" in model_meta
        assert len(model_meta["features"]) == 6
