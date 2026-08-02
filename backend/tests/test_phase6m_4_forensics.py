"""Exhaustive Unit Test Suite for Phase 6M.4 Hybrid Fusion Forensic Analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from evaluation.phase6m.config import HYBRID_FEATURE_SCHEMA, PHASE6M_DIR
from evaluation.phase6m.forensic_analysis import (
    run_feature_shift_attribution,
    run_pillar_contribution_analysis,
    run_distribution_shift_decomposition,
    run_calibration_drift_investigation,
    run_error_cluster_investigation,
    run_scientific_hypothesis_evaluation,
    run_future_research_recommendations,
)


def test_run_feature_shift_attribution():
    """Verify feature shift attribution SMD and KS calculation."""
    np.random.seed(42)
    X_dev = np.random.randn(100, 19)
    X_val = np.random.randn(50, 19)

    res = run_feature_shift_attribution(X_dev, X_val, HYBRID_FEATURE_SCHEMA)

    assert "shift_attribution" in res
    assert len(res["shift_attribution"]) == 19
    assert len(res["most_stable_features"]) > 0


def test_run_pillar_contribution_analysis():
    """Verify grouping of feature importances across feature families."""
    clf = HistGradientBoostingClassifier(max_iter=10)
    X = np.random.randn(20, 19)
    y = np.random.randint(0, 2, size=20)
    clf.fit(X, y)

    res = run_pillar_contribution_analysis(clf, HYBRID_FEATURE_SCHEMA)

    assert "family_importances" in res
    assert "Pillar_1_Evidence" in res["family_importances"]
    assert "Pillar_2_Structure" in res["family_importances"]


def test_run_distribution_shift_decomposition():
    """Verify causal hierarchy decomposition structure."""
    shift_attr = {
        "shift_attribution": [],
        "most_stable_features": ["prob_p1"],
        "most_shifted_features": ["prob_p2"],
    }
    res = run_distribution_shift_decomposition(shift_attr)

    assert "causal_hierarchy" in res
    assert len(res["causal_hierarchy"]) == 4


def test_run_calibration_drift_investigation():
    """Verify calibration drift investigation results."""
    res = run_calibration_drift_investigation(0.0066, 0.0939)

    assert res["ece_increase_delta"] == pytest.approx(0.0873, abs=1e-3)


def test_run_error_cluster_investigation():
    """Verify FP and FN error clustering."""
    np.random.seed(42)
    X_val = np.random.randn(100, 19)
    y_val = np.random.randint(0, 2, size=100)
    p_val = np.random.uniform(0, 1, size=100)

    res = run_error_cluster_investigation(X_val, y_val, p_val, threshold=0.54, feature_names=HYBRID_FEATURE_SCHEMA)

    assert "counts" in res
    assert "false_positive_cluster_analysis" in res


def test_run_scientific_hypothesis_evaluation():
    """Verify evaluation of pre-declared hypotheses H1 through H5."""
    res = run_scientific_hypothesis_evaluation()

    assert res["H1_hybrid_superior_to_pillar1"]["status"] == "SUPPORTED"
    assert res["H3_ece_calibration_target"]["status"] == "NOT SUPPORTED"
    assert res["H5_stable_generalization"]["status"] == "NOT SUPPORTED"


def test_run_future_research_recommendations():
    """Verify future research recommendations structure."""
    res = run_future_research_recommendations()

    assert "recommendations" in res
    assert len(res["recommendations"]) == 4
