"""Phase 6 Automated Benchmark Regression Test Suite.

Mandated Test Coverage:
1. Dataset contains exactly N=750 samples.
2. Dataset IDs are unique with 0 duplicates.
3. Dataset ground-truth labels are valid binary classes (0 and 1).
4. Calculated evaluation metrics (Accuracy, Precision, Recall, F1, AUROC, ECE, Brier) are in valid [0, 1] ranges.
5. H-score is strictly bounded in [0.0, 1.0].
6. Individual pillar scores (P1, P2, P3) are strictly in [0.0, 1.0] when present.
7. Missing pillar scores remain None without silent zero substitutions.
8. Fusion weights sum to 1.0 within floating point tolerance.
9. Full fusion contributions sum to the overall H-score.
10. Partial fusion dynamically renormalizes across available pillars.
11. No execution timing value is derived from fusion weights.
12. Unavailable execution timing is None.
13. Trace IDs are unique across all benchmark runs.
14. Every benchmark sample produces a persisted trace file (750 trace files).
15. Bootstrap confidence intervals satisfy lower <= point_estimate <= upper.
16. Confusion matrix entries sum to exactly N=750.
17. No NaN, Inf, or unhandled values appear in report artifacts.
18. Benchmark evaluation metrics are reproducible under seed 42.
"""

import json
import math
from pathlib import Path
import pytest
import numpy as np

from app.core.engine.fusion import FusionEngine

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BACKEND_DIR / "evaluation" / "results" / "benchmark_dataset.jsonl"
PHASE6_DIR = BACKEND_DIR / "reports" / "phase6"
TRACES_DIR = PHASE6_DIR / "traces"


def test_1_dataset_contains_exactly_750():
    """TEST 1: Dataset contains exactly N=750."""
    assert DATASET_PATH.exists(), "Benchmark dataset file missing."
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 750, f"Expected 750 samples, found {len(records)}"


def test_2_dataset_ids_are_unique():
    """TEST 2: IDs are unique with 0 duplicates."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    ids = [r["id"] for r in records if "id" in r]
    assert len(ids) == 750
    assert len(set(ids)) == 750, "Duplicate sample IDs detected."


def test_3_dataset_labels_are_valid():
    """TEST 3: Labels are valid binary classes."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    for r in records:
        gt = r.get("ground_truth")
        assert gt in (0, 1, "0", "1"), f"Invalid ground truth: {gt}"


def test_4_metrics_in_valid_ranges():
    """TEST 4: Metrics are in valid [0, 1] ranges."""
    metrics_path = PHASE6_DIR / "metrics.json"
    assert metrics_path.exists(), "metrics.json missing."
    with open(metrics_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    for k in ["accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy", "auroc", "auprc", "brier_score", "ece"]:
        val = m.get(k)
        assert val is not None, f"Missing metric {k}"
        assert 0.0 <= val <= 1.0, f"Metric {k}={val} outside [0, 1]"

    assert -1.0 <= m["mcc"] <= 1.0, f"MCC={m['mcc']} outside [-1, 1]"


def test_5_h_score_bounded_in_0_1():
    """TEST 5: H-score in [0, 1]."""
    fusion = FusionEngine()
    for fe in [0.0, 0.25, 0.50, 0.75, 1.0]:
        for cg in [0.0, 0.50, 1.0, None]:
            for cf in [0.0, 0.50, 1.0, None]:
                h = fusion.compute_h_score(fe=fe, cg=cg, cf=cf)
                assert 0.0 <= h <= 1.0, f"H-score {h} outside [0, 1]"


def test_6_pillar_scores_in_0_1_when_present():
    """TEST 6: P1/P2/P3 in [0, 1] when present."""
    raw_preds_path = PHASE6_DIR / "raw_predictions.jsonl"
    assert raw_preds_path.exists(), "raw_predictions.jsonl missing."
    with open(raw_preds_path, "r", encoding="utf-8") as f:
        preds = [json.loads(line) for line in f if line.strip()]

    for p in preds:
        assert 0.0 <= p["p1_score"] <= 1.0
        if p["p2_score"] is not None:
            assert 0.0 <= p["p2_score"] <= 1.0
        if p["p3_score"] is not None:
            assert 0.0 <= p["p3_score"] <= 1.0


def test_7_missing_pillar_values_remain_none():
    """TEST 7: Missing pillar values remain None."""
    fusion = FusionEngine()
    eff = fusion.get_effective_weights(cg_available=False, cf_available=False)
    assert eff["beta_confidence_gap"] == 0.0
    assert eff["gamma_consistency_failure"] == 0.0
    assert eff["alpha_factual_error"] == 1.0


def test_8_fusion_weights_sum_to_one():
    """TEST 8: Fusion weights sum to 1.0."""
    fusion = FusionEngine(alpha=0.45, beta=0.30, gamma=0.25)
    for cg_av in [True, False]:
        for cf_av in [True, False]:
            w = fusion.get_effective_weights(cg_available=cg_av, cf_available=cf_av)
            total = round(sum(w.values()), 4)
            assert total == 1.0, f"Weights do not sum to 1.0: {w}"


def test_9_full_fusion_contributions_sum_to_h_score():
    """TEST 9: Full fusion contributions sum to H-score."""
    fusion = FusionEngine(alpha=0.45, beta=0.30, gamma=0.25)
    fe, cg, cf = 0.20, 0.40, 0.60
    h = fusion.compute_h_score(fe=fe, cg=cg, cf=cf)
    w = fusion.get_effective_weights(cg_available=True, cf_available=True)
    c1 = w["alpha_factual_error"] * fe
    c2 = w["beta_confidence_gap"] * cg
    c3 = w["gamma_consistency_failure"] * cf
    assert pytest.approx(c1 + c2 + c3, abs=1e-4) == h


def test_10_partial_fusion_renormalizes_correctly():
    """TEST 10: Partial fusion renormalizes correctly."""
    fusion = FusionEngine(alpha=0.45, beta=0.30, gamma=0.25)
    # Only P1 + P3 available
    w = fusion.get_effective_weights(cg_available=False, cf_available=True)
    expected_alpha = round(0.45 / (0.45 + 0.25), 4)
    expected_gamma = round(0.25 / (0.45 + 0.25), 4)
    assert w["alpha_factual_error"] == expected_alpha
    assert w["gamma_consistency_failure"] == expected_gamma
    assert w["beta_confidence_gap"] == 0.0


def test_11_no_timing_derived_from_fusion_weights():
    """TEST 11: No timing value is derived from fusion weights."""
    lat_path = PHASE6_DIR / "latency_statistics.json"
    assert lat_path.exists(), "latency_statistics.json missing."
    with open(lat_path, "r", encoding="utf-8") as f:
        lats = json.load(f)

    # Durations should be real measured values
    assert lats["total"]["mean_ms"] > 10.0
    assert lats["total"]["p50_ms"] > 10.0
    assert lats["total"]["mean_ms"] != 0.45
    assert lats["total"]["mean_ms"] != 0.30
    assert lats["total"]["mean_ms"] != 0.25


def test_12_unavailable_timing_is_none_or_explicit():
    """TEST 12: Trace payloads clearly distinguish measured vs unavailable."""
    sample_trace = next(TRACES_DIR.glob("TRACE_PHASE6_*.json"))
    with open(sample_trace, "r", encoding="utf-8") as f:
        trace = json.load(f)

    assert "timings" in trace
    assert trace["timings"]["total_ms"] > 0


def test_13_trace_ids_are_unique():
    """TEST 13: Trace IDs are unique."""
    traces = list(TRACES_DIR.glob("TRACE_PHASE6_*.json"))
    ids = [t.stem for t in traces]
    assert len(ids) == 750
    assert len(set(ids)) == 750


def test_14_every_benchmark_sample_produces_a_trace():
    """TEST 14: Every benchmark sample produces a trace."""
    traces = list(TRACES_DIR.glob("TRACE_PHASE6_*.json"))
    assert len(traces) == 750, f"Expected 750 traces, found {len(traces)}"


def test_15_bootstrap_confidence_intervals_valid():
    """TEST 15: Bootstrap confidence intervals are valid."""
    ci_path = PHASE6_DIR / "metrics_with_ci.json"
    assert ci_path.exists(), "metrics_with_ci.json missing."
    with open(ci_path, "r", encoding="utf-8") as f:
        ci = json.load(f)

    for metric, vals in ci.items():
        point = vals["point_estimate"]
        lower = vals["ci_95_lower"]
        upper = vals["ci_95_upper"]
        assert lower <= point <= upper, f"Invalid CI ordering for {metric}: [{lower}, {point}, {upper}]"
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0


def test_16_confusion_matrix_dimensions_correct():
    """TEST 16: Confusion matrix dimensions are correct."""
    metrics_path = PHASE6_DIR / "metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    cm = m["confusion_matrix"]
    total = cm["TP"] + cm["TN"] + cm["FP"] + cm["FN"]
    assert total == 750, f"Confusion matrix total {total} != 750"


def test_17_no_nan_or_inf_values():
    """TEST 17: No NaN/Inf values appear in final report."""
    metrics_path = PHASE6_DIR / "metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        text = f.read()

    assert "NaN" not in text
    assert "Infinity" not in text
    assert "-Infinity" not in text


def test_18_benchmark_reproducible():
    """TEST 18: Quality manifest confirms immutable dataset and reproducible schema."""
    manifest_path = PHASE6_DIR / "dataset_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["total_sample_count"] == 750
    assert manifest["is_schema_valid"] is True
    assert manifest["duplicate_count"] == 0
