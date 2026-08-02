"""Unit tests for Phase 6K.1 Warning Forensics.

Verifies:
    1. Mutually exclusive warning categorization: one warning increments exactly ONE category.
    2. Input matrix stats and percentiles calculation.
    3. Direct matrix multiplication stress testing.
    4. Solver isolation testing (lbfgs, liblinear, newton-cg, saga).
    5. Manual logit and probability numerical equivalence.
"""

from __future__ import annotations

import warnings
import numpy as np
import pytest

from evaluation.phase6k.forensics import (
    categorize_warning,
    summarize_warning_records,
    run_warning_forensics,
)
from evaluation.phase6k.config import FEATURE_COLUMNS


def test_mutually_exclusive_warning_categorization():
    """Test categorize_warning assigns exactly ONE category per warning object without double counting."""
    # Create mock warning objects
    w1 = warnings.WarningMessage(
        message=RuntimeWarning("overflow encountered in matmul"),
        category=RuntimeWarning,
        filename="extmath.py",
        lineno=203,
    )
    w2 = warnings.WarningMessage(
        message=RuntimeWarning("divide by zero encountered in matmul"),
        category=RuntimeWarning,
        filename="extmath.py",
        lineno=203,
    )
    w3 = warnings.WarningMessage(
        message=UserWarning("Stochastic Optimizer: Maximum iterations (1000) reached and failed to converge"),
        category=UserWarning,
        filename="stochastic.py",
        lineno=100,
    )

    rec1 = categorize_warning(w1)
    rec2 = categorize_warning(w2)
    rec3 = categorize_warning(w3)

    assert rec1.mutually_exclusive_category == "overflow_matmul"
    assert rec2.mutually_exclusive_category == "divide_by_zero_matmul"
    assert rec3.mutually_exclusive_category == "convergence_warning"

    summary = summarize_warning_records([rec1, rec2, rec3])
    assert summary["overflow_matmul"] == 1
    assert summary["divide_by_zero_matmul"] == 1
    assert summary["convergence_warning"] == 1
    assert summary["invalid_matmul"] == 0
    assert summary["other_runtime_warning"] == 0
    assert sum(summary.values()) == 3  # Total count equals record count


def test_warning_forensics_execution():
    """Test run_warning_forensics executes all 8 steps cleanly and returns structured dictionary."""
    forensic_data = run_warning_forensics()

    assert "step1_warning_counting" in forensic_data
    assert "step2_matrix_inspection" in forensic_data
    assert "step3_direct_matmul_test" in forensic_data
    assert "step4_solver_isolation" in forensic_data
    assert "step5_regularization_forensics" in forensic_data
    assert "step6_manual_logit_check" in forensic_data
    assert "step7_environment" in forensic_data
    assert "step8_standalone_reproduction" in forensic_data

    # Check Step 4 solver isolation results
    solvers = [s["solver"] for s in forensic_data["step4_solver_isolation"]]
    assert "lbfgs" in solvers
    assert "liblinear" in solvers
    assert "newton-cg" in solvers
    assert "saga" in solvers

    # Check Step 6 manual logit check
    s6 = forensic_data["step6_manual_logit_check"]
    assert s6["z_all_finite"] is True
    assert s6["prob_manual_all_finite"] is True
    assert s6["probabilities_match_tolerance"] is True
