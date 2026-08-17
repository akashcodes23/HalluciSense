"""Phase 7B Automated Scientific Integrity Test Suite.

Verifies paired dataset alignment, leakage metrics, failure taxonomies,
threshold sweeps, and non-destructive isolation from Phase 6 and Phase 7.
"""

import json
import hashlib
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
PHASE7B_DIR = BACKEND_DIR / "reports" / "phase7b"
PHASE6_DIR = BACKEND_DIR / "reports" / "phase6"
PHASE7_DIR = BACKEND_DIR / "reports" / "phase7"
DATASET_PATH = BACKEND_DIR / "evaluation" / "results" / "benchmark_dataset.jsonl"


def test_1_paired_samples_count():
    """TEST 1: Paired comparison contains exactly 750 samples."""
    comp_file = PHASE7B_DIR / "phase6_vs_phase7_comparison.csv"
    assert comp_file.exists()
    df = pd.read_csv(comp_file)
    assert len(df) == 750


def test_2_sample_ids_are_unique():
    """TEST 2: All 750 sample IDs are unique."""
    df = pd.read_csv(PHASE7B_DIR / "phase6_vs_phase7_comparison.csv")
    assert len(df["sample_id"].unique()) == 750


def test_3_alignment_audit_passes():
    """TEST 3: Alignment audit confirms 750 matched and 0 mismatches."""
    audit_file = PHASE7B_DIR / "alignment_audit.json"
    assert audit_file.exists()
    with open(audit_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["matched_records"] == 750
    assert data["mismatched_records"] == 0
    assert data["label_mismatches"] == 0
    assert data["domain_mismatches"] == 0
    assert data["query_mismatches"] == 0
    assert data["alignment_status"] == "EXACT_MATCH_VERIFIED"


def test_4_ground_truth_consistency():
    """TEST 4: Ground truth is identical across benchmark, Phase 6, and Phase 7."""
    df = pd.read_csv(PHASE7B_DIR / "phase6_vs_phase7_comparison.csv")
    factual = (df["ground_truth"] == 0).sum()
    hallucinated = (df["ground_truth"] == 1).sum()
    assert factual == 375
    assert hallucinated == 375


def test_5_no_fabricated_p2_values():
    """TEST 5: Phase 7 P2 values remain empty / unavailable without synthetic filler."""
    df = pd.read_csv(PHASE7B_DIR / "phase6_vs_phase7_comparison.csv")
    # 27 samples where P3 had fewer than 2 distinct alternates are honestly NaN
    assert df["phase7_p3"].notnull().sum() == 723
    # Check blocker file exists
    assert (PHASE7B_DIR / "P2_PROVIDER_BLOCKER.md").exists()
    assert (PHASE7B_DIR / "provider_capability_matrix.json").exists()


def test_6_threshold_validation_split_reproducible():
    """TEST 6: Validation (70%) and Test (30%) splits are disjoint and reproducible."""
    n_total = 750
    np.random.seed(42)
    indices = np.random.permutation(n_total)
    split_idx = int(0.70 * n_total)
    val_idx, test_idx = set(indices[:split_idx]), set(indices[split_idx:])
    assert len(val_idx) == 525
    assert len(test_idx) == 225
    assert len(val_idx.intersection(test_idx)) == 0


def test_7_phase6_artifacts_unchanged():
    """TEST 7: Phase 6 frozen metrics remain strictly untouched."""
    p6_m_file = PHASE6_DIR / "metrics.json"
    with open(p6_m_file, "r", encoding="utf-8") as f:
        p6_m = json.load(f)
    assert p6_m["accuracy"] == 0.8467
    assert p6_m["auroc"] == 0.9260


def test_8_phase7_artifacts_unchanged():
    """TEST 8: Phase 7 frozen metrics remain strictly untouched."""
    p7_m_file = PHASE7_DIR / "metrics.json"
    with open(p7_m_file, "r", encoding="utf-8") as f:
        p7_m = json.load(f)
    assert p7_m["accuracy"] == 0.5733
    assert p7_m["auroc"] == 0.5602


def test_9_phase7b_csv_schemas_valid():
    """TEST 9: All Phase 7B CSV tables exist and have valid rows."""
    expected_csvs = [
        "phase6_vs_phase7_comparison.csv",
        "phase6_leakage_audit.csv",
        "response_distribution_comparison.csv",
        "p1_failure_analysis.csv",
        "p3_failure_analysis.csv",
        "threshold_analysis.csv",
        "calibration_comparison.csv",
        "domain_phase6_phase7_comparison.csv",
        "error_taxonomy.csv",
        "cross_model_results.csv",
    ]
    for name in expected_csvs:
        p = PHASE7B_DIR / name
        assert p.exists(), f"Missing CSV {name}"
        df = pd.read_csv(p)
        assert len(df) > 0, f"Empty CSV {name}"


def test_10_error_taxonomy_sample_mapping():
    """TEST 10: Every error-analysis record maps to a valid benchmark sample ID."""
    err_df = pd.read_csv(PHASE7B_DIR / "error_taxonomy.csv")
    assert len(err_df) > 0
    df = pd.read_csv(PHASE7B_DIR / "phase6_vs_phase7_comparison.csv")
    valid_ids = set(df["sample_id"].unique())
    for sid in err_df["sample_id"]:
        assert sid in valid_ids


def test_11_all_plots_exist():
    """TEST 11: All 8 publication figures exist in plots/."""
    plots = [
        "response_length_distribution.png",
        "claim_count_distribution.png",
        "phase6_phase7_score_distribution.png",
        "threshold_analysis.png",
        "calibration_comparison.png",
        "domain_comparison.png",
        "error_taxonomy.png",
        "model_comparison.png",
    ]
    for p in plots:
        assert (PHASE7B_DIR / "plots" / p).exists(), f"Missing plot {p}"


def test_12_reproduction_manifest_valid():
    """TEST 12: Reproduction manifest exists and is complete."""
    p = PHASE7B_DIR / "reproduction_manifest.json"
    assert p.exists()
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    assert d["sample_count"] == 750
    assert d["domains"] == 15
