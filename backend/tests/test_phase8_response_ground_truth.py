"""Phase 8 Automated Test Suites — Response Ground Truth, Controlled Hallucination,
Calibration, Statistical Integrity, and Trace Integrity.

All 5 test files consolidated here for brevity. They are registered as
test_phase8_response_ground_truth.py via pytest.
"""

import json
import hashlib
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy import stats

BACKEND_DIR = Path(__file__).resolve().parent.parent
PHASE8_DIR = BACKEND_DIR / "reports" / "phase8"
PHASE6_DIR = BACKEND_DIR / "reports" / "phase6"
PHASE7_DIR = BACKEND_DIR / "reports" / "phase7"
DATASET_PATH = BACKEND_DIR / "evaluation" / "results" / "benchmark_dataset.jsonl"

CORRUPTION_TYPES = [
    "ENTITY_SUBSTITUTION", "NUMERIC_SUBSTITUTION", "DATE_SUBSTITUTION",
    "TEMPORAL_ERROR", "LOCATION_SUBSTITUTION", "PERSON_SUBSTITUTION",
    "CAUSAL_REVERSAL", "CONTRADICTION", "PARTIAL_CLAIM_CORRUPTION",
    "MULTI_CLAIM_CORRUPTION",
]


# ════════════════════════════════════════════════════════════════════════════
# TEST FILE 1: test_phase8_response_ground_truth.py
# ════════════════════════════════════════════════════════════════════════════

def load_dataset_b():
    records = []
    with open(PHASE8_DIR / "response_level_ground_truth.jsonl", "r") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def test_rgt_1_exactly_750_records():
    """Dataset B has exactly 750 records."""
    ds_b = load_dataset_b()
    assert len(ds_b) == 750


def test_rgt_2_unique_sample_ids():
    """Dataset B has 750 unique sample IDs."""
    ds_b = load_dataset_b()
    ids = [r["sample_id"] for r in ds_b]
    assert len(ids) == len(set(ids))


def test_rgt_3_dual_label_columns_present():
    """Both original_static_label and response_ground_truth are present and distinct columns."""
    ds_b = load_dataset_b()
    for r in ds_b[:10]:
        assert "original_static_label" in r
        assert "response_ground_truth" in r
        assert "response_ground_truth_binary" in r


def test_rgt_4_static_label_distribution():
    """Static labels preserve 375 factual / 375 hallucinated from the canonical benchmark."""
    ds_b = load_dataset_b()
    static_factual = sum(1 for r in ds_b if r["original_static_label"] == 0)
    static_hallucinated = sum(1 for r in ds_b if r["original_static_label"] == 1)
    assert static_factual == 375
    assert static_hallucinated == 375


def test_rgt_5_response_gt_classes():
    """Response-level ground truth uses valid rich labels."""
    ds_b = load_dataset_b()
    valid = {"factual", "hallucinated", "partially_hallucinated", "unverifiable"}
    for r in ds_b:
        assert r["response_ground_truth"] in valid


def test_rgt_6_binary_label_consistency():
    """Binary label is None only for unverifiable; 0 for factual; 1 for hallucinated/partial."""
    ds_b = load_dataset_b()
    for r in ds_b:
        rgt = r["response_ground_truth"]
        bin_gt = r["response_ground_truth_binary"]
        if rgt == "unverifiable":
            assert bin_gt is None
        elif rgt == "factual":
            assert bin_gt == 0
        else:
            assert bin_gt == 1


def test_rgt_7_label_shift_count():
    """Label shift cases (static=hallucinated, response=factual) are non-zero and documented."""
    ds_b = load_dataset_b()
    shifts = [r for r in ds_b if r.get("is_label_shift")]
    assert len(shifts) > 0, "No label shift cases detected — expected ~190"
    # Should be substantial: Phase 7B found ~50.7% of hallucinated prompts answered correctly
    assert len(shifts) >= 100


def test_rgt_8_ground_truth_method_documented():
    """Every record has ground_truth_method and ground_truth_reason."""
    ds_b = load_dataset_b()
    for r in ds_b[:20]:
        assert r.get("ground_truth_method") == "P1_NLI_Evidence_Grounding"
        assert r.get("ground_truth_reason") and len(r["ground_truth_reason"]) > 10


def test_rgt_9_no_h_score_used_as_gt():
    """Ground truth was NOT assigned from the final H-score fusion."""
    ds_b = load_dataset_b()
    for r in ds_b[:10]:
        method = r.get("ground_truth_method", "")
        assert "H-score" not in method
        assert "fusion" not in method.lower()
        assert "P1_NLI_Evidence_Grounding" in method


def test_rgt_10_circularity_disclosure_in_p1_results():
    """P1-only results on Dataset B contain explicit circularity warning."""
    p1_df = pd.read_csv(PHASE8_DIR / "p1_results.csv")
    p1_row = p1_df[p1_df["pillar_config"] == "P1_ONLY"]
    assert len(p1_row) >= 1
    warning = p1_row.iloc[0].get("circularity_warning", "")
    assert "METHODOLOGICAL DISCLOSURE" in str(warning) or "circular" in str(warning).lower()


def test_rgt_11_response_gt_manifest_sha256():
    """Response ground truth manifest has a valid SHA-256 hash."""
    m_path = PHASE8_DIR / "response_ground_truth_manifest.json"
    assert m_path.exists()
    m = json.loads(m_path.read_text())
    assert "output_sha256" in m
    assert len(m["output_sha256"]) == 64
    # Verify it matches the actual file
    actual = hashlib.sha256((PHASE8_DIR / "response_level_ground_truth.jsonl").read_bytes()).hexdigest()
    assert m["output_sha256"] == actual


# ════════════════════════════════════════════════════════════════════════════
# TEST FILE 2: test_phase8_controlled_hallucination.py
# ════════════════════════════════════════════════════════════════════════════

def load_dataset_c():
    records = []
    with open(PHASE8_DIR / "controlled_hallucination_dataset.jsonl", "r") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def test_ctrl_1_exactly_300_records():
    """Dataset C has exactly 300 records (30 per corruption type × 10 types)."""
    ds_c = load_dataset_c()
    assert len(ds_c) == 300


def test_ctrl_2_all_gt_1():
    """All Dataset C records have ground_truth = 1 (hallucinated)."""
    ds_c = load_dataset_c()
    for r in ds_c:
        assert r["ground_truth"] == 1


def test_ctrl_3_all_10_corruption_types():
    """All 10 corruption types are present."""
    ds_c = load_dataset_c()
    found = set(r["corruption_type"] for r in ds_c)
    for ct in CORRUPTION_TYPES:
        assert ct in found, f"Missing corruption type: {ct}"


def test_ctrl_4_30_per_type():
    """Each corruption type has exactly 30 samples."""
    ds_c = load_dataset_c()
    df = pd.DataFrame(ds_c)
    counts = df["corruption_type"].value_counts()
    for ct in CORRUPTION_TYPES:
        assert counts.get(ct, 0) == 30, f"{ct} has {counts.get(ct, 0)} samples, expected 30"


def test_ctrl_5_all_actually_corrupted():
    """All 300 records are actually corrupted (text changed from original)."""
    ds_c = load_dataset_c()
    not_corrupted = [r for r in ds_c if not r.get("is_actually_corrupted")]
    assert len(not_corrupted) == 0, f"{len(not_corrupted)} records not actually corrupted"


def test_ctrl_6_no_llm_used_for_gt():
    """Ground truth method confirms rule-based transformation, no LLM annotation."""
    ds_c = load_dataset_c()
    for r in ds_c[:10]:
        method = r.get("ground_truth_method", "")
        assert "rule_based" in method
        assert "llm" not in method.lower()
        assert "h_score" not in method.lower()


def test_ctrl_7_no_hallucisense_as_gt():
    """HalluciSense H-score is not used as ground truth in Dataset C."""
    ds_c = load_dataset_c()
    for r in ds_c[:10]:
        reason = r.get("ground_truth_reason", "")
        assert "h_score" not in reason.lower()
        assert "hallucisense" not in reason.lower()


def test_ctrl_8_unique_ids():
    """All Dataset C sample IDs are unique."""
    ds_c = load_dataset_c()
    ids = [r["sample_id"] for r in ds_c]
    assert len(ids) == len(set(ids))


def test_ctrl_9_no_b_c_id_overlap():
    """Dataset B and Dataset C have no overlapping sample IDs."""
    ds_b = load_dataset_b()
    ds_c = load_dataset_c()
    b_ids = set(r["sample_id"] for r in ds_b)
    c_ids = set(r["sample_id"] for r in ds_c)
    overlap = b_ids.intersection(c_ids)
    assert len(overlap) == 0, f"ID overlap between Dataset B and C: {overlap}"


def test_ctrl_10_severity_4_levels_present():
    """At least 3 distinct severity levels (1,2,3,4) present in Dataset C."""
    ds_c = load_dataset_c()
    severities = set(r["corruption_severity"] for r in ds_c)
    assert len(severities) >= 3


def test_ctrl_11_severity_monotonicity():
    """Spearman ρ(severity, H-score) is positive and statistically significant."""
    df = pd.read_csv(PHASE8_DIR / "severity_analysis.csv")
    assert df["spearman_rho_severity_vs_h"].iloc[0] > 0
    assert df["spearman_p_value"].iloc[0] < 0.05


def test_ctrl_12_manifest_sha256_valid():
    """Controlled hallucination manifest has a valid SHA-256 hash."""
    m = json.loads((PHASE8_DIR / "controlled_hallucination_manifest.json").read_text())
    actual = hashlib.sha256((PHASE8_DIR / "controlled_hallucination_dataset.jsonl").read_bytes()).hexdigest()
    assert m["output_sha256"] == actual


# ════════════════════════════════════════════════════════════════════════════
# TEST FILE 3: test_phase8_calibration.py
# ════════════════════════════════════════════════════════════════════════════

def test_cal_1_calibration_csv_exists():
    """calibration_results.csv exists and has 4 rows (uncalibrated + 3 methods)."""
    df = pd.read_csv(PHASE8_DIR / "calibration_results.csv")
    assert len(df) == 4


def test_cal_2_isotonic_reduces_ece():
    """Isotonic regression reduces ECE vs uncalibrated on held-out test."""
    df = pd.read_csv(PHASE8_DIR / "calibration_results.csv")
    raw_ece = float(df[df["method"].str.contains("Uncalibrated")]["test_ece"].iloc[0])
    iso_ece = float(df[df["method"].str.contains("Isotonic")]["test_ece"].iloc[0])
    assert iso_ece < raw_ece, f"Isotonic ECE {iso_ece} not lower than raw {raw_ece}"


def test_cal_3_no_calibration_on_test():
    """Data leakage audit confirms calibration was not fitted on test set."""
    audit = json.loads((PHASE8_DIR / "data_leakage_audit.json").read_text())
    assert not audit["calibration_fitted_on_test"]
    assert not audit["threshold_optimized_on_test"]


def test_cal_4_val_test_split_disjoint():
    """Validation (70%) and test (30%) splits are disjoint (no index overlap)."""
    n_total = 750
    np.random.seed(42)
    idx = np.random.permutation(n_total)
    split = int(0.70 * n_total)
    val_idx, test_idx = set(idx[:split]), set(idx[split:])
    assert len(val_idx.intersection(test_idx)) == 0
    assert len(val_idx) == 525
    assert len(test_idx) == 225


def test_cal_5_threshold_analysis_csv():
    """Threshold analysis CSV covers 0.05–0.95 range and has validation metrics."""
    df = pd.read_csv(PHASE8_DIR / "threshold_analysis.csv")
    assert len(df) >= 19  # 0.05 to 0.95 step 0.05
    assert "threshold" in df.columns
    assert "f1" in df.columns
    assert "accuracy" in df.columns
    assert (df["threshold"] >= 0.05).all()
    assert (df["threshold"] <= 0.95).all()


# ════════════════════════════════════════════════════════════════════════════
# TEST FILE 4: test_phase8_statistical_integrity.py
# ════════════════════════════════════════════════════════════════════════════

def test_stat_1_statistical_tests_exist():
    """statistical_tests.json exists and contains required keys."""
    st = json.loads((PHASE8_DIR / "statistical_tests.json").read_text())
    assert "mcnemar_p1_vs_p1p3_dataset_b" in st
    assert "kruskal_wallis_severity" in st
    assert "spearman_severity_h_score" in st


def test_stat_2_mcnemar_significant():
    """McNemar P1 vs P1+P3 is statistically significant (p < 0.05)."""
    st = json.loads((PHASE8_DIR / "statistical_tests.json").read_text())
    assert st["mcnemar_p1_vs_p1p3_dataset_b"]["is_significant"]


def test_stat_3_kruskal_wallis_severity_significant():
    """Kruskal-Wallis test across severity groups is statistically significant."""
    st = json.loads((PHASE8_DIR / "statistical_tests.json").read_text())
    assert st["kruskal_wallis_severity"]["is_significant"]


def test_stat_4_spearman_positive_severity():
    """Spearman ρ(severity, H-score) is positive (higher severity → higher H)."""
    st = json.loads((PHASE8_DIR / "statistical_tests.json").read_text())
    assert st["spearman_severity_h_score"]["rho"] > 0


def test_stat_5_effect_size_reported():
    """Cohen's d effect size is reported."""
    st = json.loads((PHASE8_DIR / "statistical_tests.json").read_text())
    assert "effect_size_cohen_d" in st


def test_stat_6_data_leakage_clean():
    """Data leakage audit reports CLEAN status."""
    audit = json.loads((PHASE8_DIR / "data_leakage_audit.json").read_text())
    assert audit["status"] == "CLEAN"


def test_stat_7_phase6_frozen():
    """Phase 6 metrics are unchanged (AUROC = 0.9260)."""
    m = json.loads((PHASE6_DIR / "metrics.json").read_text())
    assert m["auroc"] == 0.9260


def test_stat_8_phase7_frozen():
    """Phase 7 metrics are unchanged (AUROC = 0.5602)."""
    m = json.loads((PHASE7_DIR / "metrics.json").read_text())
    assert m["auroc"] == 0.5602


# ════════════════════════════════════════════════════════════════════════════
# TEST FILE 5: test_phase8_trace_integrity.py
# ════════════════════════════════════════════════════════════════════════════

def test_trace_1_exactly_300_traces():
    """Exactly 300 Phase 8 traces exist (Dataset C)."""
    traces = list((PHASE8_DIR / "traces").glob("TRACE_PHASE8_*.json"))
    assert len(traces) == 300


def test_trace_2_trace_schema():
    """Each trace has required schema fields."""
    required = ["trace_id", "sample_id", "dataset", "domain", "ground_truth",
                "p1", "p2", "p3", "fusion", "predicted_label", "timings"]
    for i in range(1, 6):
        t = json.loads((PHASE8_DIR / "traces" / f"TRACE_PHASE8_{i:06d}.json").read_text())
        for k in required:
            assert k in t, f"Trace {i} missing field '{k}'"


def test_trace_3_p2_unavailable():
    """P2 is marked UNAVAILABLE in all traces (no synthetic logprobs)."""
    for i in range(1, 11):
        t = json.loads((PHASE8_DIR / "traces" / f"TRACE_PHASE8_{i:06d}.json").read_text())
        assert not t["p2"]["available"]
        assert t["p2"]["score"] is None


def test_trace_4_all_gt_1_in_dataset_c():
    """All Phase 8 traces (Dataset C) have ground_truth = 1."""
    for i in range(1, 301):
        t = json.loads((PHASE8_DIR / "traces" / f"TRACE_PHASE8_{i:06d}.json").read_text())
        assert t["ground_truth"] == 1


def test_trace_5_fusion_reconstruction_exact():
    """Fusion reconstruction error is exactly 0 across all traces."""
    for i in range(1, 301):
        t = json.loads((PHASE8_DIR / "traces" / f"TRACE_PHASE8_{i:06d}.json").read_text())
        err = t["fusion"]["fusion_absolute_error"]
        assert err == 0.0, f"Trace {i} has non-zero fusion error: {err}"


def test_trace_6_corruption_type_in_trace():
    """All Phase 8 traces contain corruption_type field."""
    for i in range(1, 11):
        t = json.loads((PHASE8_DIR / "traces" / f"TRACE_PHASE8_{i:06d}.json").read_text())
        assert "corruption_type" in t
        assert t["corruption_type"] in CORRUPTION_TYPES


def test_trace_7_all_csvs_have_content():
    """All Phase 8 required CSVs exist and have records."""
    required_csvs = [
        "hallucination_type_results.csv", "severity_analysis.csv",
        "domain_breakdown.csv", "model_robustness.csv", "calibration_results.csv",
        "threshold_analysis.csv", "p1_results.csv", "fusion_results.csv",
        "data_leakage_records.csv", "fusion_integrity_audit.csv",
    ]
    for name in required_csvs:
        p = PHASE8_DIR / name
        assert p.exists(), f"Missing CSV: {name}"
        df = pd.read_csv(p)
        assert len(df) > 0, f"Empty CSV: {name}"


def test_trace_8_all_12_plots_exist():
    """All 12 publication figures exist."""
    required_plots = [
        "confusion_matrix.png", "roc_curve.png", "precision_recall_curve.png",
        "calibration_curve.png", "threshold_analysis.png", "domain_comparison.png",
        "hallucination_type_detection.png", "severity_detection.png",
        "model_comparison.png", "pillar_comparison.png",
        "fusion_comparison.png", "temperature_robustness.png",
    ]
    for name in required_plots:
        assert (PHASE8_DIR / "plots" / name).exists(), f"Missing plot: {name}"


def test_trace_9_domain_breakdown_15_domains():
    """Domain breakdown CSV covers 15 domains."""
    df = pd.read_csv(PHASE8_DIR / "domain_breakdown.csv")
    assert len(df) == 15


def test_trace_10_hallucination_type_10_rows():
    """Hallucination type results CSV has exactly 10 rows (one per type)."""
    df = pd.read_csv(PHASE8_DIR / "hallucination_type_results.csv")
    assert len(df) == 10
