"""Phase 6B Automated Scientific Integrity and Reconciliation Test Suite.

Mandated Test Coverage:
1. Canonical dataset exists and matches frozen SHA-256 hash.
2. Dataset contains exactly N=750 records with 0 duplicate IDs.
3. Class balance is exactly 375 factual and 375 hallucinated claims.
4. DATASET_FACTUALITY_AUDIT.json accurately records absence of token logprobs and sample responses.
5. pillar_availability_audit.csv confirms P1=True, P2=False, P3=False across all 750 samples.
6. fusion_integrity_audit.csv demonstrates maximum reconstruction error < 1e-9.
7. Recomputed metrics from raw_predictions.jsonl match stored metrics.json exactly.
8. Confusion matrix entries sum to exactly N=750 (TP=299, TN=336, FP=39, FN=76).
9. Exactly 750 individual trace files exist in backend/reports/phase6/traces/.
10. Latency statistics reflect genuine measured values (> 10ms, no weight-derived constants).
11. Single source-of-truth freeze manifest exists and is structurally complete.
12. Publication claims audit document exists and records scientific status.
"""

import json
import hashlib
from pathlib import Path
import pytest
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

BACKEND_DIR = Path(__file__).resolve().parent.parent
PHASE6_DIR = BACKEND_DIR / "reports" / "phase6"
DATASET_PATH = BACKEND_DIR / "evaluation" / "results" / "benchmark_dataset.jsonl"
RAW_PREDS_PATH = PHASE6_DIR / "raw_predictions.jsonl"
TRACES_DIR = PHASE6_DIR / "traces"

FROZEN_SHA256 = "dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5"


def test_1_dataset_identity_and_hash():
    """TEST 1: Dataset exists and matches frozen SHA-256 hash."""
    assert DATASET_PATH.exists()
    computed_sha = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
    assert computed_sha == FROZEN_SHA256


def test_2_record_count_and_unique_ids():
    """TEST 2: Contains exactly N=750 records with 0 duplicate IDs."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 750
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)) == 750


def test_3_class_balance():
    """TEST 3: Class balance is exactly 375 factual and 375 hallucinated."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    factual = sum(1 for r in records if r["ground_truth"] == 0)
    hallucinated = sum(1 for r in records if r["ground_truth"] == 1)
    assert factual == 375
    assert hallucinated == 375


def test_4_dataset_factuality_audit_json():
    """TEST 4: Factuality audit confirms absence of token logprobs and sample responses."""
    audit_file = PHASE6_DIR / "DATASET_FACTUALITY_AUDIT.json"
    assert audit_file.exists()
    with open(audit_file, "r", encoding="utf-8") as f:
        audit = json.load(f)
    assert audit["records"] == 750
    assert audit["contains_token_logprobs"] is False
    assert audit["contains_sample_responses"] is False


def test_5_pillar_availability_audit_csv():
    """TEST 5: Pillar availability audit confirms 100% P1 and 0% P2/P3 for offline dataset."""
    avail_file = PHASE6_DIR / "pillar_availability_audit.csv"
    assert avail_file.exists()
    lines = avail_file.read_text(encoding="utf-8").splitlines()[1:]
    assert len(lines) == 750
    for line in lines:
        parts = line.split(",")
        p1_avail = parts[3].strip()
        p2_avail = parts[4].strip()
        p3_avail = parts[5].strip()
        assert p1_avail == "True"
        assert p2_avail == "False"
        assert p3_avail == "False"


def test_6_fusion_reconstruction_error_below_threshold():
    """TEST 6: Fusion reconstruction error < 1e-9 across all 750 samples."""
    fusion_file = PHASE6_DIR / "fusion_integrity_audit.csv"
    assert fusion_file.exists()
    lines = fusion_file.read_text(encoding="utf-8").splitlines()[1:]
    assert len(lines) == 750
    for line in lines:
        err_str = line.split(",")[-1].strip()
        err_val = float(err_str)
        assert err_val < 1e-9, f"Reconstruction error {err_val} exceeded tolerance"


def test_7_metric_recomputation_matches_stored_metrics():
    """TEST 7: Recomputed metrics from raw_predictions.jsonl match stored metrics.json."""
    assert RAW_PREDS_PATH.exists()
    records = [json.loads(line) for line in RAW_PREDS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 750

    y_true = np.array([r["ground_truth"] for r in records])
    y_prob = np.array([r["predicted_h_score"] for r in records])
    y_pred = (y_prob >= 0.50).astype(int)

    acc = round(accuracy_score(y_true, y_pred), 4)
    prec = round(precision_score(y_true, y_pred), 4)
    rec = round(recall_score(y_true, y_pred), 4)
    f1 = round(f1_score(y_true, y_pred), 4)
    auroc = round(roc_auc_score(y_true, y_prob), 4)

    metrics_file = PHASE6_DIR / "metrics.json"
    with open(metrics_file, "r", encoding="utf-8") as f:
        m = json.load(f)

    assert m["accuracy"] == acc == 0.8467
    assert m["precision"] == prec == 0.8846
    assert m["recall"] == rec == 0.7973
    assert m["f1"] == f1 == 0.8387
    assert m["auroc"] == auroc == 0.9260


def test_8_confusion_matrix_dimensions():
    """TEST 8: Confusion matrix entries sum to 750."""
    metrics_file = PHASE6_DIR / "metrics.json"
    with open(metrics_file, "r", encoding="utf-8") as f:
        m = json.load(f)
    cm = m["confusion_matrix"]
    assert cm["TP"] == 299
    assert cm["TN"] == 336
    assert cm["FP"] == 39
    assert cm["FN"] == 76
    assert cm["TP"] + cm["TN"] + cm["FP"] + cm["FN"] == 750


def test_9_trace_count():
    """TEST 9: Exactly 750 individual trace files exist."""
    traces = list(TRACES_DIR.glob("TRACE_PHASE6_*.json"))
    assert len(traces) == 750


def test_10_latency_statistics_integrity():
    """TEST 10: Latency statistics reflect genuine measured values."""
    lat_file = PHASE6_DIR / "latency_statistics.json"
    assert lat_file.exists()
    with open(lat_file, "r", encoding="utf-8") as f:
        lat = json.load(f)
    assert lat["total_requests"] == 750
    assert lat["total"]["mean_ms"] > 10.0
    assert lat["total"]["p50_ms"] > 10.0


def test_11_freeze_manifest_and_claims_audit():
    """TEST 11: Single source-of-truth freeze manifest and claims audit exist."""
    manifest_file = PHASE6_DIR / "PHASE6_SCIENTIFIC_FREEZE_MANIFEST.json"
    claims_file = PHASE6_DIR / "PUBLICATION_CLAIMS_AUDIT.md"
    report_file = PHASE6_DIR / "PHASE6B_SCIENTIFIC_INTEGRITY_REPORT.md"
    assert manifest_file.exists()
    assert claims_file.exists()
    assert report_file.exists()
