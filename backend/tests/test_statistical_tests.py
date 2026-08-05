"""Tests for Phase 15 Statistical Validation Suite."""

import pytest
import numpy as np
from evaluation.phase15.statistical_analysis import (
    compute_bootstrap_ci,
    compute_mcnemar_test,
    compute_cohens_d,
)


def test_bootstrap_ci():
    rng = np.random.default_rng(42)
    y_true = np.array([0]*50 + [1]*50)
    y_prob = np.concatenate([rng.uniform(0.0, 0.4, 50), rng.uniform(0.6, 1.0, 50)])

    ci = compute_bootstrap_ci(y_true, y_prob, threshold=0.50, n_bootstraps=500, seed=42)
    assert "accuracy" in ci
    assert "f1_score" in ci
    assert "auroc" in ci
    assert ci["accuracy"]["ci_lower_95"] <= ci["accuracy"]["mean"] <= ci["accuracy"]["ci_upper_95"]


def test_mcnemar_test():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred1 = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred2 = np.array([1, 1, 0, 0, 1, 0, 1, 0])

    mcnemar = compute_mcnemar_test(y_true, y_pred1, y_pred2)
    assert "b_model1_win" in mcnemar
    assert "p_value" in mcnemar
    assert mcnemar["b_model1_win"] == 8


def test_cohens_d():
    x1 = np.array([0.8, 0.9, 0.85, 0.88, 0.92])
    x2 = np.array([0.4, 0.5, 0.45, 0.48, 0.52])

    d = compute_cohens_d(x1, x2)
    assert d > 1.5  # Large effect size
