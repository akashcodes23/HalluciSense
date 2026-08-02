"""Exhaustive Unit Test Suite for Phase 6M.2 Development Model Selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.phase6m.config import HYBRID_FEATURE_SCHEMA, PHASE6M_DIR
from evaluation.phase6m.fusion_models import get_candidate_configs, get_preprocessor
from evaluation.phase6m.model_selection import (
    compute_ece,
    compute_delong_pvalue,
    compute_mcnemar_test,
    evaluate_candidate_cv,
)


def test_get_candidate_configs():
    """Verify nominated candidate configurations."""
    cands = get_candidate_configs()
    assert len(cands) == 6
    assert "Candidate 1" in cands
    assert "Candidate 4" in cands


def test_get_preprocessor():
    """Verify scaler instantiator."""
    assert get_preprocessor("StandardScaler") is not None
    assert get_preprocessor("RobustScaler") is not None
    assert get_preprocessor(None) is None


def test_compute_ece():
    """Verify Expected Calibration Error calculation."""
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6])

    res = compute_ece(y, p, n_bins=5)
    assert "ece" in res
    assert 0.0 <= res["ece"] <= 1.0


def test_compute_delong_pvalue():
    """Verify DeLong test calculation."""
    np.random.seed(42)
    y = np.random.randint(0, 2, size=100)
    p1 = np.random.uniform(0, 1, size=100)
    p2 = np.random.uniform(0, 1, size=100)

    res = compute_delong_pvalue(y, p1, p2)
    assert "z_stat" in res
    assert "p_value" in res
    assert 0.0 <= res["p_value"] <= 1.0


def test_compute_mcnemar_test():
    """Verify McNemar's test calculation."""
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    pred1 = np.array([0, 0, 1, 1, 0, 0, 0, 1])
    pred2 = np.array([0, 1, 1, 0, 0, 1, 0, 0])

    res = compute_mcnemar_test(y, pred1, pred2)
    assert "mcnemar_statistic" in res
    assert "p_value" in res


def test_evaluate_candidate_cv_small():
    """Verify cross-validation evaluation on small array."""
    np.random.seed(42)
    X = np.random.randn(50, 19)
    y = np.random.randint(0, 2, size=50)

    cand_configs = get_candidate_configs()
    c_cfg = cand_configs["Candidate 1"]

    res = evaluate_candidate_cv(X, y, HYBRID_FEATURE_SCHEMA, "Candidate 1", c_cfg, n_splits=2, n_repeats=1)

    assert "summary_metrics" in res
    assert "roc_auc_mean" in res["summary_metrics"]
    assert "best_mcc" in res["summary_metrics"]
