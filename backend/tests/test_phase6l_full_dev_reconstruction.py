"""Unit tests for Phase 6L.1C Full DEV Structural Feature Reconstruction Engine."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from evaluation.phase6l.config import DEV_FEATURES_JSONL, PHASE6L_DIR, STRUCTURAL_FEATURE_COLUMNS
from evaluation.phase6l.full_dev_reconstruction import (
    verify_dev_dataset_integrity,
    execute_full_dev_sharded_reconstruction,
)
from evaluation.phase6l.preflight_activation import run_rare_feature_activation_preflight


def test_verify_dev_dataset_integrity():
    """Verify DEV dataset integrity returns exact expected record and pair counts."""
    res = verify_dev_dataset_integrity(DEV_FEATURES_JSONL)

    assert res["total_responses"] == 58002
    assert res["unique_response_ids"] == 58002
    assert res["total_unordered_pairs"] == 964637
    assert res["total_directional_inferences"] == 1929274
    assert len(res["dataset_sha256"]) == 64


def test_run_rare_feature_activation_preflight():
    """Test label-free rare feature activation preflight on first 500 DEV records."""
    res = run_rare_feature_activation_preflight(dev_path=DEV_FEATURES_JSONL, max_scan=500, out_dir=PHASE6L_DIR)

    assert res["total_responses_scanned"] == 500
    assert res["preflight_status"] == "PASS"
    assert "activation_counts" in res
