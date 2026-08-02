"""Comprehensive Unit Tests for HalluciSense Phase 6A Evaluation Infrastructure.

Covers metrics calculations, zero-division handling, threshold sweeps, optimal threshold selection,
ablation configurations, calibration binning, Brier Score, dataset validation, and network isolation.
"""

from pathlib import Path
import pytest
from evaluation.ablation import (
    AblationConfig,
    compute_ablation_score,
    run_ablation_study,
)
from evaluation.calibration import (
    analyze_calibration,
    analyze_score_distributions,
    calculate_distribution_stats,
)
from evaluation.dataset import BenchmarkSample, DatasetLoader
from evaluation.metrics import (
    compute_accuracy,
    compute_all_metrics,
    compute_brier_score,
    compute_confusion_matrix,
    compute_ece,
    compute_f1,
    compute_precision,
    compute_pr_auc,
    compute_recall,
    compute_roc_auc,
    compute_specificity,
)
from evaluation.runner import EvaluationRunner


# =====================================================================
# 1. Confusion Matrix & Basic Classification Metrics Tests
# =====================================================================

def test_confusion_matrix_correctness():
    y_true = [0, 0, 1, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 1, 0, 0, 1, 0, 0]

    # TN: (0,0)->3, FP: (0,1)->1, FN: (1,0)->2, TP: (1,1)->2
    tp, tn, fp, fn = compute_confusion_matrix(y_true, y_pred)
    assert tp == 2
    assert tn == 3
    assert fp == 1
    assert fn == 2


def test_accuracy_precision_recall_specificity_f1():
    # TP=2, TN=3, FP=1, FN=2 (Total=8)
    tp, tn, fp, fn = 2, 3, 1, 2

    acc = compute_accuracy(tp, tn, fp, fn)
    assert acc == pytest.approx(5 / 8, abs=1e-4)

    prec = compute_precision(tp, fp)
    assert prec == pytest.approx(2 / 3, abs=1e-4)

    rec = compute_recall(tp, fn)
    assert rec == pytest.approx(2 / 4, abs=1e-4)

    spec = compute_specificity(tn, fp)
    assert spec == pytest.approx(3 / 4, abs=1e-4)

    f1 = compute_f1(prec, rec)
    # 2*(2/3 * 1/2) / (2/3 + 1/2) = (2/3) / (7/6) = 4/7
    assert f1 == pytest.approx(4 / 7, abs=1e-4)


# =====================================================================
# 2. Zero-Denominator & Edge Case Handling Tests
# =====================================================================

def test_zero_denominator_handling():
    # All Negative, TP=0, FP=0 -> Precision undefined
    assert compute_precision(tp=0, fp=0) is None

    # All Positive, TN=0, FP=0 -> Specificity undefined
    assert compute_specificity(tn=0, fp=0) is None

    # All Factual predicted, TP=0, FN=0 -> Recall undefined
    assert compute_recall(tp=0, fn=0) is None

    # Undefined Precision -> F1 returns None
    assert compute_f1(precision=None, recall=0.8) is None
    assert compute_f1(precision=0.0, recall=0.0) is None

    # Total 0 samples
    assert compute_accuracy(0, 0, 0, 0) is None


def test_single_class_auc_handling():
    y_true = [0, 0, 0, 0]  # Only factual class
    scores = [0.1, 0.2, 0.3, 0.4]

    assert compute_roc_auc(y_true, scores) is None
    assert compute_pr_auc(y_true, scores) is None


# =====================================================================
# 3. Threshold Sweep & Optimal Selection Tests
# =====================================================================

def test_threshold_sweep_and_best_selection():
    runner = EvaluationRunner()

    # Ground truth: 3 factual, 3 hallucinated
    y_true = [0, 0, 0, 1, 1, 1]
    scores = [0.05, 0.12, 0.25, 0.45, 0.75, 0.90]

    sweep_res = runner.run_threshold_sweep(y_true, scores, step=0.05)

    assert "sweep_points" in sweep_res
    assert len(sweep_res["sweep_points"]) > 10

    # At threshold 0.35: preds = [0, 0, 0, 1, 1, 1] -> perfect classification (F1 = 1.0)
    assert sweep_res["optimal_f1_threshold"] in (0.30, 0.35, 0.40)
    assert sweep_res["optimal_f1_score"] == 1.0
    assert sweep_res["optimal_youden_j_score"] == 1.0


# =====================================================================
# 4. Pillar Ablation Study Tests
# =====================================================================

def test_ablation_configurations_and_weight_handling():
    # P1=0.8, P2=0.2, P3=0.1
    score_full = compute_ablation_score(
        fe=0.8, cg=0.2, cf=0.1, use_p1=True, use_p2=True, use_p3=True
    )
    # alpha=0.45, beta=0.30, gamma=0.25 -> 0.45*0.8 + 0.30*0.2 + 0.25*0.1 = 0.445
    assert score_full == 0.445

    # P1_ONLY: weight alpha renormalized to 1.0 -> score = 0.8
    score_p1_only = compute_ablation_score(
        fe=0.8, cg=0.2, cf=0.1, use_p1=True, use_p2=False, use_p3=False
    )
    assert score_p1_only == pytest.approx(0.8, abs=1e-3)

    # P2_P3: alpha masked -> beta: 0.25/0.55 = 0.4545, gamma: 0.30/0.55 = 0.5455
    score_p2_p3 = compute_ablation_score(
        fe=0.8, cg=0.2, cf=0.1, use_p1=False, use_p2=True, use_p3=True
    )
    assert score_p2_p3 is not None
    assert 0.10 <= score_p2_p3 <= 0.20


def test_full_ablation_study_execution():
    y_true = [0, 0, 1, 1]
    p1 = [0.1, 0.2, 0.8, 0.9]
    p2 = [0.1, None, 0.7, None]
    p3 = [0.05, 0.1, 0.85, 0.95]

    ablation_res = run_ablation_study(y_true, p1, p2, p3, threshold=0.35)

    assert len(ablation_res) == 7
    for cfg in AblationConfig.CONFIGS.keys():
        assert cfg in ablation_res
        assert ablation_res[cfg]["sample_count"] > 0
        assert "metrics" in ablation_res[cfg]


# =====================================================================
# 5. Score Distribution & Calibration Tests
# =====================================================================

def test_brier_score_and_ece():
    y_true = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]

    # Brier: ((0.1-0)^2 + (0.2-0)^2 + (0.8-1)^2 + (0.9-1)^2)/4 = (0.01 + 0.04 + 0.04 + 0.01)/4 = 0.10 / 4 = 0.025
    brier = compute_brier_score(y_true, scores)
    assert brier == pytest.approx(0.025, abs=1e-4)

    ece = compute_ece(y_true, scores, num_bins=10)
    assert ece is not None
    assert 0.0 <= ece <= 1.0


def test_calibration_binning_and_distributions():
    y_true = [0, 0, 1, 1]
    scores = [0.05, 0.15, 0.75, 0.85]

    cal_res = analyze_calibration(y_true, scores, num_bins=10)
    assert cal_res["brier_score"] is not None
    assert len(cal_res["bins"]) == 10

    dist_res = analyze_score_distributions(y_true, scores)
    assert dist_res["factual"]["count"] == 2
    assert dist_res["factual"]["mean"] == pytest.approx(0.10, abs=1e-2)
    assert dist_res["hallucinated"]["count"] == 2
    assert dist_res["hallucinated"]["mean"] == pytest.approx(0.80, abs=1e-2)


# =====================================================================
# 6. Dataset Loader & Schema Validation Tests
# =====================================================================

def test_dataset_loader_validates_jsonl():
    fixture_path = (
        Path(__file__).parent.parent
        / "evaluation"
        / "datasets"
        / "development_fixture.jsonl"
    )

    samples = DatasetLoader.load_from_file(fixture_path)
    assert len(samples) == 10
    assert samples[0].id == "dev_001"
    assert samples[0].ground_truth_label == 0
    assert samples[5].ground_truth_label == 1


def test_dataset_schema_rejects_invalid_labels():
    with pytest.raises(ValueError, match="ground_truth_label must be 0"):
        BenchmarkSample(
            id="bad_1",
            prompt="Test prompt",
            response="Test response",
            ground_truth_label=2,  # Invalid label
        )


def test_dataset_schema_rejects_empty_response():
    with pytest.raises(ValueError, match="Prompt and response must be non-empty"):
        BenchmarkSample(
            id="bad_2",
            prompt="Test prompt",
            response="   ",  # Blank response
            ground_truth_label=0,
        )


# =====================================================================
# 7. Network Isolation Safeguard Test
# =====================================================================

def test_network_isolation_safeguard(monkeypatch):
    """Ensures evaluation functions do not attempt network connections."""
    def block_connect(*args, **kwargs):
        raise RuntimeError("Network call blocked during evaluation test!")

    import socket
    monkeypatch.setattr(socket, "create_connection", block_connect)

    fixture_path = (
        Path(__file__).parent.parent
        / "evaluation"
        / "datasets"
        / "development_fixture.jsonl"
    )
    samples = DatasetLoader.load_from_file(fixture_path)
    assert len(samples) == 10
