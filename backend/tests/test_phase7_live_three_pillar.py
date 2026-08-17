"""Phase 7 Live Three-Pillar Automated Test Suite.

Verifies end-to-end live generation, P1/P2/P3 execution, adaptive fusion,
trace integrity, statistical outputs, and isolation from Phase 6.
"""

import json
import hashlib
from pathlib import Path
import pytest
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

BACKEND_DIR = Path(__file__).resolve().parent.parent
PHASE7_DIR = BACKEND_DIR / "reports" / "phase7"
TRACES_DIR = PHASE7_DIR / "traces"
PLOTS_DIR = PHASE7_DIR / "plots"
PHASE6_DIR = BACKEND_DIR / "reports" / "phase6"

PHASE6_FROZEN_ACCURACY = 0.8467
PHASE6_FROZEN_AUROC = 0.9260


def test_1_traces_count_exactly_750():
    """TEST 1: Exactly 750 traces exist."""
    traces = list(TRACES_DIR.glob("TRACE_PHASE7_*.json"))
    assert len(traces) == 750


def test_2_trace_schema_and_fields():
    """TEST 2: Traces conform to Phase 7 schema."""
    trace_file = TRACES_DIR / "TRACE_PHASE7_000001.json"
    assert trace_file.exists()
    with open(trace_file, "r", encoding="utf-8") as f:
        t = json.load(f)
    assert "trace_id" in t
    assert "sample_id" in t
    assert "domain" in t
    assert "ground_truth" in t
    assert "query" in t
    assert "generated_response" in t
    assert "p1" in t
    assert "p2" in t
    assert "p3" in t
    assert "fusion" in t
    assert "timings" in t


def test_3_p1_available_across_all_samples():
    """TEST 3: Pillar 1 is available for all 750 samples."""
    avail_file = PHASE7_DIR / "pillar_availability_audit.csv"
    assert avail_file.exists()
    lines = avail_file.read_text(encoding="utf-8").splitlines()[1:]
    assert len(lines) == 750
    for line in lines:
        parts = line.split(",")
        assert parts[3].strip() == "True"  # p1_available


def test_4_p2_unavailable_without_synthetic_filler():
    """TEST 4: Pillar 2 is honestly marked False when logprobs are omitted."""
    avail_file = PHASE7_DIR / "pillar_availability_audit.csv"
    lines = avail_file.read_text(encoding="utf-8").splitlines()[1:]
    for line in lines:
        parts = line.split(",")
        assert parts[4].strip() == "False"  # p2_available


def test_5_p3_live_alternates_present():
    """TEST 5: Pillar 3 consistency is available with multi-generation samples."""
    trace_file = TRACES_DIR / "TRACE_PHASE7_000005.json"
    with open(trace_file, "r", encoding="utf-8") as f:
        t = json.load(f)
    assert t["p3"]["available"] is True
    assert t["p3"]["sample_count"] == 3


def test_6_adaptive_fusion_effective_weights_sum_to_one():
    """TEST 6: Effective fusion weights strictly sum to 1.0."""
    trace_file = TRACES_DIR / "TRACE_PHASE7_000010.json"
    with open(trace_file, "r", encoding="utf-8") as f:
        t = json.load(f)
    w = t["fusion"]["effective_weights"]
    w_sum = w["alpha_factual_error"] + w["beta_confidence_gap"] + w["gamma_consistency_failure"]
    assert abs(w_sum - 1.0) < 1e-9


def test_7_fusion_reconstruction_error_below_threshold():
    """TEST 7: Fusion reconstruction error < 1e-9 across all 750 records."""
    fusion_file = PHASE7_DIR / "fusion_integrity_audit.csv"
    assert fusion_file.exists()
    lines = fusion_file.read_text(encoding="utf-8").splitlines()[1:]
    assert len(lines) == 750
    for line in lines:
        err = float(line.split(",")[-1].strip())
        assert err < 1e-9


def test_8_recomputed_metrics_match_stored_metrics():
    """TEST 8: Recomputed metrics from raw_predictions.jsonl match stored metrics."""
    raw_pred_file = PHASE7_DIR / "raw_predictions.jsonl"
    assert raw_pred_file.exists()
    records = [json.loads(line) for line in raw_pred_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 750

    y_true = np.array([r["ground_truth"] for r in records])
    y_prob = np.array([r["predicted_h_score"] for r in records])
    y_pred = (y_prob >= 0.50).astype(int)

    acc = round(accuracy_score(y_true, y_pred), 4)
    prec = round(precision_score(y_true, y_pred, zero_division=0), 4)
    rec = round(recall_score(y_true, y_pred, zero_division=0), 4)
    f1 = round(f1_score(y_true, y_pred, zero_division=0), 4)
    auroc = round(roc_auc_score(y_true, y_prob), 4)

    metrics_file = PHASE7_DIR / "metrics.json"
    with open(metrics_file, "r", encoding="utf-8") as f:
        m = json.load(f)

    assert m["accuracy"] == acc == 0.5733
    assert m["precision"] == prec == 0.7434
    assert m["recall"] == rec == 0.2240
    assert m["f1"] == f1 == 0.3443
    assert m["auroc"] == auroc == 0.5602


def test_9_confusion_matrix_dimensions():
    """TEST 9: Confusion matrix entries sum to exactly 750."""
    metrics_file = PHASE7_DIR / "metrics.json"
    with open(metrics_file, "r", encoding="utf-8") as f:
        m = json.load(f)
    cm = m["confusion_matrix"]
    assert cm["TP"] == 84
    assert cm["TN"] == 346
    assert cm["FP"] == 29
    assert cm["FN"] == 291
    assert cm["TP"] + cm["TN"] + cm["FP"] + cm["FN"] == 750


def test_10_phase6_artifacts_untouched():
    """TEST 10: Phase 6 frozen metrics remain strictly unchanged."""
    p6_metrics_file = PHASE6_DIR / "metrics.json"
    assert p6_metrics_file.exists()
    with open(p6_metrics_file, "r", encoding="utf-8") as f:
        p6_m = json.load(f)
    assert p6_m["accuracy"] == PHASE6_FROZEN_ACCURACY
    assert p6_m["auroc"] == PHASE6_FROZEN_AUROC


def test_11_all_phase7_reports_and_plots_exist():
    """TEST 11: All Phase 7 documentation and publication plots exist."""
    assert (PHASE7_DIR / "PHASE7_SCIENTIFIC_VALIDATION.md").exists()
    assert (PHASE7_DIR / "PHASE7_SCIENTIFIC_INTEGRITY_REPORT.md").exists()
    assert (PHASE7_DIR / "PHASE7_REPRODUCIBILITY.md").exists()
    assert (PHASE7_DIR / "PHASE7_CLAIMS_AUDIT.md").exists()
    assert (PLOTS_DIR / "roc_curve.png").exists()
    assert (PLOTS_DIR / "precision_recall_curve.png").exists()
    assert (PLOTS_DIR / "calibration_curve.png").exists()
    assert (PLOTS_DIR / "confusion_matrix.png").exists()
    assert (PLOTS_DIR / "domain_f1.png").exists()
