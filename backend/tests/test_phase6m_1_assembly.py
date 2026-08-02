"""Exhaustive Unit Test Suite for Phase 6M.1 Hybrid Feature Assembly & Preflight Validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.phase6m.config import PHASE6M_DIR, HYBRID_FEATURE_SCHEMA, CANDIDATE_SUBSETS
from evaluation.phase6m.dataset import compute_logit, load_and_assemble_hybrid_matrix
from evaluation.phase6m.preflight_assembly import (
    audit_dataset_integrity,
    validate_hybrid_matrix,
    compute_feature_distribution_statistics,
    compute_correlation_audit,
    compute_probability_diagnostics,
    audit_data_leakage,
    audit_numerical_health,
    export_candidate_subsets,
    evaluate_decision_gate,
)


def test_compute_logit():
    """Verify logit (log-odds) computation with bounds clipping."""
    assert np.isfinite(compute_logit(0.5))
    assert compute_logit(0.5) == pytest.approx(0.0, abs=1e-5)
    assert np.isfinite(compute_logit(0.0))
    assert np.isfinite(compute_logit(1.0))


def test_load_and_assemble_hybrid_matrix_dev():
    """Verify loading and assembling DEV hybrid matrix (N=58,002)."""
    dev_data = load_and_assemble_hybrid_matrix("development")

    X = dev_data["X"]
    y = dev_data["y"]
    example_ids = dev_data["example_ids"]
    feature_names = dev_data["feature_names"]

    assert X.shape == (58002, 19)
    assert y.shape == (58002,)
    assert len(example_ids) == 58002
    assert feature_names == HYBRID_FEATURE_SCHEMA
    assert np.isfinite(X).all()
    assert int((y == 1).sum()) + int((y == 0).sum()) == 58002


def test_load_and_assemble_hybrid_matrix_val():
    """Verify loading and assembling VAL hybrid matrix (N=12,483)."""
    val_data = load_and_assemble_hybrid_matrix("validation")

    X = val_data["X"]
    y = val_data["y"]
    example_ids = val_data["example_ids"]

    assert X.shape == (12483, 19)
    assert y.shape == (12483,)
    assert len(example_ids) == 12483
    assert np.isfinite(X).all()


def test_audit_dataset_integrity(tmp_path: Path):
    """Verify dataset integrity audit functionality."""
    dev_data = {"example_ids": [f"dev_{i}" for i in range(58002)]}
    val_data = {"example_ids": [f"val_{i}" for i in range(12483)]}

    res = audit_dataset_integrity(dev_data, val_data, out_dir=tmp_path)

    assert res["integrity_status"] == "PASS"
    assert res["dev_duplicate_ids"] == 0
    assert res["val_duplicate_ids"] == 0
    assert res["dev_val_id_overlap"] == 0
    assert (tmp_path / "hybrid_integrity_report.json").exists()


def test_validate_hybrid_matrix():
    """Verify matrix validation checks (0 NaN, 0 Inf)."""
    X_dev = np.random.randn(100, 19)
    X_val = np.random.randn(50, 19)

    res = validate_hybrid_matrix(X_dev, X_val, HYBRID_FEATURE_SCHEMA)

    assert res["matrix_validation_status"] == "PASS"
    assert res["dev_nan_count"] == 0
    assert res["dev_inf_count"] == 0
    assert res["duplicate_columns_count"] == 0


def test_compute_correlation_audit(tmp_path: Path):
    """Verify Pearson, Spearman correlation audit."""
    X_dev = np.random.randn(200, 19)

    res = compute_correlation_audit(X_dev, HYBRID_FEATURE_SCHEMA, out_dir=tmp_path)

    assert "pearson_correlation_matrix" in res
    assert "spearman_correlation_matrix" in res
    assert len(res["pearson_correlation_matrix"]) == 19
    assert (tmp_path / "hybrid_correlations.json").exists()


def test_audit_data_leakage(tmp_path: Path):
    """Verify 5-point data leakage audit."""
    X_dev = np.random.randn(100, 19)
    y_dev = np.random.randint(0, 2, 100)
    X_val = np.random.randn(50, 19)
    y_val = np.random.randint(0, 2, 50)

    res = audit_data_leakage(X_dev, y_dev, X_val, y_val, out_dir=tmp_path)

    assert res["leakage_audit_status"] == "PASS"
    assert res["labels_embedded_in_features"] is False
    assert (tmp_path / "hybrid_leakage_report.json").exists()


def test_export_candidate_subsets(tmp_path: Path):
    """Verify pre-defined candidate feature subsets serialization."""
    payload = export_candidate_subsets(out_dir=tmp_path)

    assert payload["total_feature_count"] == 19
    assert "SET_A_FULL_HYBRID" in payload["candidate_subsets"]
    assert len(payload["candidate_subsets"]) == 6
    assert (tmp_path / "hybrid_schema.json").exists()


def test_evaluate_decision_gate():
    """Verify 9-question decision gate evaluation."""
    integrity = {"dev_record_count": 58002, "val_record_count": 12483, "dev_duplicate_ids": 0, "val_duplicate_ids": 0}
    matrix_val = {"feature_count": 19, "duplicate_columns_count": 0, "dev_nan_count": 0, "val_nan_count": 0, "dev_inf_count": 0, "val_inf_count": 0}
    leakage = {"leakage_audit_status": "PASS"}
    num_health = {"numerical_health_status": "PASS"}

    checklist = evaluate_decision_gate(integrity, matrix_val, leakage, num_health)

    assert checklist["9_phase6m2_scientifically_cleared"] == "GO"
    assert checklist["1_assembled_correctly"] == "YES"
    assert checklist["2_rows_perfectly_aligned"] == "YES"
