"""Unit tests for Phase 6L.2 Development Model Selection Pipeline (Pillar 2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.phase6l.config import PHASE6L_DIR, STRUCTURAL_FEATURE_COLUMNS
from evaluation.phase6l.dataset import load_and_validate_full_dev_matrix
from evaluation.phase6l.collinearity import evaluate_preprocessing_scalers, run_collinearity_analysis
from evaluation.phase6l.discrimination import run_feature_discrimination_audit
from evaluation.phase6l.stability_gate import run_numerical_stability_gate
from evaluation.phase6l.leakage import run_data_leakage_audit
from evaluation.phase6l.protocol import export_final_model_protocol


def test_load_and_validate_full_dev_matrix():
    """Verify loading and schema validation of full DEV matrix."""
    val_data = load_and_validate_full_dev_matrix()
    X, y = val_data["X"], val_data["y"]

    assert X.shape[0] == 58002
    assert X.shape[1] == 24
    assert y.shape[0] == 58002
    assert np.isnan(X).sum() == 0
    assert np.isinf(X).sum() == 0
    assert val_data["validation_payload"]["status"] == "PASS"


def test_evaluate_preprocessing_scalers():
    """Verify preprocessing study computes condition numbers and finite statistics."""
    X = np.random.randn(100, 24)
    res = evaluate_preprocessing_scalers(X)

    assert "None" in res
    assert "StandardScaler" in res
    assert "RobustScaler" in res
    assert res["StandardScaler"]["finite_all"] is True
    assert res["RobustScaler"]["finite_all"] is True


def test_run_collinearity_analysis(tmp_path: Path):
    """Verify collinearity, VIF, and candidate feature set generation."""
    X = np.random.randn(200, 24)
    res = run_collinearity_analysis(X, out_dir=tmp_path)

    assert "pearson" in res["correlations"]
    assert "spearman" in res["correlations"]
    assert "vif_records" in res["vif"]
    assert len(res["candidate_sets"]) == 6
    assert "SET_A_FULL_SCHEMA" in res["candidate_sets"]
    assert "SET_D_HIGH_INFORMATION" in res["candidate_sets"]


def test_run_feature_discrimination_audit(tmp_path: Path):
    """Verify univariate feature discrimination metrics calculation."""
    X = np.random.randn(100, 24)
    y = np.random.randint(0, 2, size=100)
    res = run_feature_discrimination_audit(X, y, out_dir=tmp_path)

    assert len(res["feature_rankings"]) == 24
    assert "mutual_information" in res["feature_rankings"][0]
    assert "roc_auc" in res["feature_rankings"][0]
    assert "cohens_d" in res["feature_rankings"][0]


def test_run_numerical_stability_gate(tmp_path: Path):
    """Verify mutually-exclusive warning accounting in stability gate."""
    X = np.random.randn(100, 24)
    y = np.random.randint(0, 2, size=100)

    cand_sets = {
        "SET_A_FULL_SCHEMA": {
            "name": "SET_A_FULL_SCHEMA",
            "features": STRUCTURAL_FEATURE_COLUMNS,
        }
    }
    res = run_numerical_stability_gate(X, y, candidate_sets=cand_sets, out_dir=tmp_path)

    assert res["gate_payload"]["n_evaluated"] > 0
    assert "warning_forensics" in res


def test_run_data_leakage_audit(tmp_path: Path):
    """Verify 5-point data leakage and firewall audit."""
    X = np.random.randn(100, 24)
    y = np.random.randint(0, 2, size=100)
    res = run_data_leakage_audit(X, y, out_dir=tmp_path)

    assert res["status"] == "PASS"
    assert res["held_out_val_sample_count"] == 12483
    assert res["held_out_val_status"] == "STRICTLY_SEALED_AND_UNTOUCHED"


def test_protocol_lock_export(tmp_path: Path):
    """Verify immutable protocol locking export."""
    mock_cand = {
        "classifier_name": "Candidate 4 (SET_D + RobustScaler + LogisticRegression)",
        "feature_set": "SET_D_HIGH_INFORMATION",
        "scaler_type": "RobustScaler",
        "features": STRUCTURAL_FEATURE_COLUMNS[:5],
        "summary_metrics": {
            "roc_auc_mean": 0.7200,
            "pr_auc_mean": 0.7100,
            "best_mcc_threshold": 0.52,
            "best_mcc": 0.4500,
            "accuracy_at_best_thresh": 0.7250,
            "f1_at_best_thresh": 0.7150,
        },
    }
    res = export_final_model_protocol(mock_cand, out_dir=tmp_path)

    assert res["selected_candidate"] == mock_cand["classifier_name"]
    assert res["feature_set_name"] == "SET_D_HIGH_INFORMATION"
    assert res["protocol_locked"] is True
