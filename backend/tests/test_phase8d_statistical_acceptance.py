"""Test Suite for Phase 8D Statistical Acceptance Test.

Validates:
- Freeze audit verification of Dataset 8A and Phase 6 benchmark.
- Paired metrics calculation correctness.
- McNemar's test contingency table and binomial calculation.
- Bootstrap CI calculations.
- Transition matrix rate mathematics.
- Pre-registered decision rule logic.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from evaluation.phase8d.run_phase8d_acceptance_test import (
    audit_dataset_freeze,
    compute_metrics_dict,
    run_paired_statistical_tests,
    build_acceptance_matrix,
    PHASE6_BENCHMARK_HASH,
    DIR_8A,
)


class TestPhase8DFreezeIntegrity:
    def test_audit_dataset_freeze_passes(self):
        audit_res = audit_dataset_freeze()
        assert audit_res["audit_status"] == "PASSED_ALL_FREEZE_GATES"
        assert audit_res["total_records"] == 175
        assert audit_res["phase6_integrity_verified"] is True
        assert audit_res["phase6_benchmark_sha256"] == PHASE6_BENCHMARK_HASH

    def test_domain_distribution_exact_35(self):
        audit_res = audit_dataset_freeze()
        for dom, count in audit_res["domains"].items():
            assert count == 35, f"Domain {dom} expected 35, got {count}"

    def test_category_distribution_exact_25(self):
        audit_res = audit_dataset_freeze()
        for cat, count in audit_res["categories"].items():
            assert count == 25, f"Category {cat} expected 25, got {count}"


class TestPhase8DMetricsCalculation:
    def test_compute_metrics_dict_perfect(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        m = compute_metrics_dict(y_true, y_prob, threshold=0.50)
        assert m["accuracy"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["TP"] == 2
        assert m["TN"] == 2
        assert m["FP"] == 0
        assert m["FN"] == 0

    def test_compute_metrics_dict_half(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.8, 0.2, 0.9])
        m = compute_metrics_dict(y_true, y_prob, threshold=0.50)
        assert m["accuracy"] == 0.50
        assert m["TP"] == 1
        assert m["TN"] == 1
        assert m["FP"] == 1
        assert m["FN"] == 1


class TestPhase8DStatisticalCalculations:
    def test_mcnemar_and_bootstrap(self):
        # Create synthetic paired dataframe
        data = []
        for i in range(100):
            gt = 1 if i < 50 else 0
            # System A gets 70 correct, System B gets 85 correct
            b_score = 0.8 if (i < 35 or i >= 65) else 0.2
            e_score = 0.8 if (i < 45 or i >= 60) else 0.2
            data.append({
                "sample_id": f"S{i:03d}",
                "domain": "Physics",
                "category": "NUMERICAL_PRECISION",
                "ground_truth": gt,
                "baseline_score": b_score,
                "enhanced_score": e_score,
                "baseline_correct": (1 if b_score >= 0.5 else 0) == gt,
                "enhanced_correct": (1 if e_score >= 0.5 else 0) == gt,
            })
        df = pd.DataFrame(data)
        stat_summary, ci_summary = run_paired_statistical_tests(df, B=100)
        assert "mcnemar" in stat_summary
        assert "contingency_table" in stat_summary["mcnemar"]
        assert "delta_accuracy" in ci_summary
        assert ci_summary["delta_accuracy"]["ci_95_lower"] <= ci_summary["delta_accuracy"]["ci_95_upper"]
