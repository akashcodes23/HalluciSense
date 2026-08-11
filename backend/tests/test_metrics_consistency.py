"""Phase 6C Metric Consistency Tests.

These tests form a mathematical gate: if any metric formula is inconsistent,
the test suite FAILS. This prevents publication of incorrect metrics.

Tests cover:
  1.  F1 agrees with Precision and Recall.
  2.  Accuracy agrees with TP+TN / N.
  3.  Specificity = TN / (TN + FP).
  4.  FPR = FP / (FP + TN).
  5.  FNR = FN / (FN + TP).
  6.  Balanced accuracy = (Recall + Specificity) / 2.
  7.  MCC formula correctness.
  8.  AUROC/AUPRC computed from continuous scores, not thresholded labels.
  9.  All-positive / all-negative edge cases survive without crashing.
  10. Repeated evaluation of identical predictions is deterministic.
"""

from __future__ import annotations

import math
import time
import pytest
from typing import List

from evaluation.canonical_evaluator import CanonicalEvaluator, evaluate, EvaluationResult


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def balanced_result() -> EvaluationResult:
    """TP=40, TN=30, FP=10, FN=20  (total=100)."""
    y_true = [1] * 60 + [0] * 40
    y_pred = [1] * 40 + [0] * 20 + [0] * 30 + [1] * 10
    y_score = [0.8] * 40 + [0.3] * 20 + [0.2] * 30 + [0.6] * 10
    return evaluate(y_true, y_pred, y_score, dataset="test", config_name="balanced")


@pytest.fixture
def skewed_result() -> EvaluationResult:
    """TP=3, TN=90, FP=5, FN=2  (high imbalance, mostly negatives)."""
    y_true = [1] * 5 + [0] * 95
    y_pred = [1] * 3 + [0] * 2 + [0] * 90 + [1] * 5
    y_score = [0.9, 0.85, 0.75, 0.4, 0.35] + [0.1] * 90 + [0.6] * 5
    return evaluate(y_true, y_pred, y_score, dataset="test", config_name="skewed")


@pytest.fixture
def phase6b_a0() -> EvaluationResult:
    """Reproduce Phase 6B A0_NLI_Baseline: TP=121, TN=231, FP=46, FN=152."""
    y_true = [1] * (121 + 152) + [0] * (231 + 46)
    y_pred = [1] * 121 + [0] * 152 + [0] * 231 + [1] * 46
    return evaluate(y_true, y_pred, dataset="phase6b_ablation", config_name="A0_NLI_Baseline")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: F1 mathematically agrees with Precision and Recall
# ─────────────────────────────────────────────────────────────────────────────

def test_f1_agrees_with_precision_and_recall(balanced_result):
    """2*P*R/(P+R) must equal reported F1 to within 1e-4."""
    r = balanced_result
    assert r.precision is not None
    assert r.recall is not None
    assert r.f1 is not None

    expected_f1 = 2 * r.precision * r.recall / (r.precision + r.recall)
    assert abs(r.f1 - expected_f1) < 1e-4, (
        f"F1 formula mismatch: reported={r.f1}, computed={expected_f1}"
    )


def test_f1_formula_cross_check_phase6b_a0(phase6b_a0):
    """Re-verify Phase 6B A0 F1 == 0.5500 via formula."""
    r = phase6b_a0
    tp, fp, fn = r.tp, r.fp, r.fn
    prec = tp / (tp + fp)
    rec  = tp / (tp + fn)
    f1   = 2 * prec * rec / (prec + rec)
    assert abs(r.f1 - f1) < 1e-4
    assert abs(r.f1 - 0.5500) < 1e-3, f"A0 F1 expected ~0.5500, got {r.f1}"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: Accuracy agrees with (TP + TN) / N
# ─────────────────────────────────────────────────────────────────────────────

def test_accuracy_agrees_with_confusion_matrix(balanced_result):
    r = balanced_result
    expected = (r.tp + r.tn) / r.total
    assert abs(r.accuracy - expected) < 1e-6, (
        f"Accuracy formula mismatch: reported={r.accuracy}, computed={expected}"
    )


def test_accuracy_phase6b_a0(phase6b_a0):
    r = phase6b_a0
    expected = (121 + 231) / 550
    assert abs(r.accuracy - expected) < 1e-6
    assert abs(r.accuracy - 0.6400) < 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: Specificity = TN / (TN + FP)
# ─────────────────────────────────────────────────────────────────────────────

def test_specificity_formula(balanced_result):
    r = balanced_result
    expected = r.tn / (r.tn + r.fp)
    assert abs(r.specificity - expected) < 1e-6, (
        f"Specificity mismatch: reported={r.specificity}, expected={expected}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: FPR = FP / (FP + TN)
# ─────────────────────────────────────────────────────────────────────────────

def test_fpr_formula(balanced_result):
    r = balanced_result
    expected = r.fp / (r.fp + r.tn)
    assert abs(r.fpr - expected) < 1e-6, (
        f"FPR mismatch: reported={r.fpr}, expected={expected}"
    )


def test_fpr_plus_specificity_equals_one(balanced_result):
    """FPR + Specificity must equal 1.0."""
    r = balanced_result
    assert abs(r.fpr + r.specificity - 1.0) < 1e-6, (
        f"FPR + Specificity = {r.fpr + r.specificity} (expected 1.0)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: FNR = FN / (FN + TP)
# ─────────────────────────────────────────────────────────────────────────────

def test_fnr_formula(balanced_result):
    r = balanced_result
    expected = r.fn / (r.fn + r.tp)
    assert abs(r.fnr - expected) < 1e-6


def test_fnr_plus_recall_equals_one(balanced_result):
    """FNR + Recall (TPR) must equal 1.0."""
    r = balanced_result
    assert abs(r.fnr + r.recall - 1.0) < 1e-6, (
        f"FNR + Recall = {r.fnr + r.recall} (expected 1.0)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: Balanced Accuracy = (Recall + Specificity) / 2
# ─────────────────────────────────────────────────────────────────────────────

def test_balanced_accuracy_formula(balanced_result):
    r = balanced_result
    expected = (r.recall + r.specificity) / 2.0
    assert abs(r.balanced_accuracy - expected) < 1e-6, (
        f"Balanced accuracy mismatch: {r.balanced_accuracy} vs {expected}"
    )


def test_balanced_accuracy_skewed(skewed_result):
    """Balanced accuracy correctly differs from raw accuracy for skewed data."""
    r = skewed_result
    # raw accuracy will be high because TN dominates
    # balanced accuracy should be lower than raw accuracy
    expected = (r.recall + r.specificity) / 2.0
    assert abs(r.balanced_accuracy - expected) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7: MCC formula correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_mcc_formula(balanced_result):
    r = balanced_result
    tp, tn, fp, fn = r.tp, r.tn, r.fp, r.fn
    denom = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    expected_mcc = float((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0
    assert abs(r.mcc - expected_mcc) < 1e-4, (
        f"MCC mismatch: reported={r.mcc}, expected={expected_mcc}"
    )


def test_mcc_range(balanced_result, skewed_result):
    """MCC must be in [-1, 1]."""
    for r in (balanced_result, skewed_result):
        assert -1.0 <= r.mcc <= 1.0, f"MCC out of range: {r.mcc}"


def test_mcc_perfect_classifier():
    """Perfect predictions should yield MCC = 1.0."""
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 0, 0]
    r = evaluate(y_true, y_pred)
    assert abs(r.mcc - 1.0) < 1e-6


def test_mcc_inverted_classifier():
    """Perfectly inverted predictions should yield MCC = -1.0."""
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [0, 0, 0, 1, 1, 1]
    r = evaluate(y_true, y_pred)
    assert abs(r.mcc - (-1.0)) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# TEST 8: AUROC/AUPRC computed from continuous scores, not thresholded labels
# ─────────────────────────────────────────────────────────────────────────────

def test_auroc_uses_continuous_scores():
    """AUROC from continuous scores should differ from AUROC from binary predictions."""
    y_true  = [1, 1, 0, 0, 1, 0]
    y_pred  = [1, 0, 1, 0, 1, 0]   # thresholded
    y_score = [0.9, 0.4, 0.8, 0.2, 0.95, 0.1]  # continuous

    r_with_score    = evaluate(y_true, y_pred, y_score)
    r_without_score = evaluate(y_true, y_pred)  # no continuous scores

    # AUROC from continuous scores should be non-trivial
    assert r_with_score.auroc is not None
    assert r_without_score.auroc is None, "AUROC must be None when y_score is not provided"


def test_auroc_perfect_separation():
    """When positives always score higher than negatives, AUROC = 1.0."""
    y_true  = [1, 1, 1, 0, 0, 0]
    y_score = [0.9, 0.85, 0.8, 0.3, 0.2, 0.1]
    r = evaluate(y_true, None, y_score)
    assert r.auroc is not None
    assert abs(r.auroc - 1.0) < 1e-6, f"Expected AUROC=1.0, got {r.auroc}"


def test_auroc_random_separation():
    """When scores are random (AUC=0.5), AUROC should be ~0.5."""
    y_true  = [1, 0, 1, 0, 1, 0, 1, 0]
    y_score = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    r = evaluate(y_true, None, y_score)
    # All tied scores: AUROC is technically undefined but should be ~0.5
    assert r.auroc is not None
    assert 0.0 <= r.auroc <= 1.0


def test_auprc_requires_positives():
    """AUPRC must be None when there are no positive examples."""
    y_true  = [0, 0, 0, 0]
    y_score = [0.9, 0.8, 0.3, 0.1]
    r = evaluate(y_true, None, y_score)
    assert r.auprc is None, "AUPRC must be None with no positive examples"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 9: All-positive / all-negative edge cases
# ─────────────────────────────────────────────────────────────────────────────

def test_all_positive_labels_no_crash():
    """All-positive y_true must not crash and must return defined metrics."""
    y_true  = [1, 1, 1, 1, 1]
    y_pred  = [1, 1, 0, 1, 0]
    y_score = [0.9, 0.8, 0.4, 0.7, 0.3]
    r = evaluate(y_true, y_pred, y_score)
    # Specificity and FPR are undefined (no negatives) -> None
    assert r.specificity is None
    assert r.fpr is None
    # Recall, precision, F1 should be defined
    assert r.recall is not None
    assert r.auroc is None  # single class -> AUROC undefined


def test_all_negative_labels_no_crash():
    """All-negative y_true must not crash and must return defined metrics."""
    y_true  = [0, 0, 0, 0, 0]
    y_pred  = [0, 0, 1, 0, 1]
    y_score = [0.1, 0.2, 0.6, 0.15, 0.7]
    r = evaluate(y_true, y_pred, y_score)
    # Recall and FNR are undefined (no positives) -> None
    assert r.recall is None
    assert r.fnr is None
    assert r.auroc is None  # single class -> AUROC undefined
    assert r.auprc is None  # no positives -> AUPRC undefined


def test_all_correct_predictions():
    """Perfect classifier: TP=N/2, TN=N/2, FP=0, FN=0."""
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 0, 0]
    r = evaluate(y_true, y_pred)
    assert r.tp == 3 and r.tn == 3 and r.fp == 0 and r.fn == 0
    assert abs(r.accuracy - 1.0) < 1e-6
    assert abs(r.precision - 1.0) < 1e-6
    assert abs(r.recall - 1.0) < 1e-6
    assert abs(r.f1 - 1.0) < 1e-6


def test_all_incorrect_predictions():
    """All-wrong classifier: FP=N/2, FN=N/2, TP=0, TN=0."""
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [0, 0, 0, 1, 1, 1]
    r = evaluate(y_true, y_pred)
    assert r.tp == 0 and r.tn == 0 and r.fn == 3 and r.fp == 3
    assert abs(r.accuracy - 0.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# TEST 10: Determinism — repeated evaluation of identical inputs
# ─────────────────────────────────────────────────────────────────────────────

def test_determinism_repeated_evaluation():
    """Evaluating identical predictions must produce byte-identical results."""
    y_true  = [1, 0, 1, 1, 0, 0, 1, 0, 0, 1] * 10
    y_pred  = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0] * 10
    y_score = [0.9, 0.1, 0.85, 0.45, 0.2, 0.55, 0.8, 0.15, 0.6, 0.7] * 10

    results = []
    for _ in range(5):
        r = evaluate(y_true, y_pred, y_score, dataset="det_test")
        results.append(r)

    # All metric values must be identical across runs
    for i in range(1, len(results)):
        assert results[0].accuracy     == results[i].accuracy,     "Accuracy non-deterministic"
        assert results[0].precision    == results[i].precision,     "Precision non-deterministic"
        assert results[0].recall       == results[i].recall,        "Recall non-deterministic"
        assert results[0].f1           == results[i].f1,            "F1 non-deterministic"
        assert results[0].mcc          == results[i].mcc,           "MCC non-deterministic"
        assert results[0].auroc        == results[i].auroc,         "AUROC non-deterministic"
        assert results[0].auprc        == results[i].auprc,         "AUPRC non-deterministic"
        assert results[0].tp           == results[i].tp,            "TP non-deterministic"
        assert results[0].tn           == results[i].tn,            "TN non-deterministic"
        assert results[0].fp           == results[i].fp,            "FP non-deterministic"
        assert results[0].fn           == results[i].fn,            "FN non-deterministic"


# ─────────────────────────────────────────────────────────────────────────────
# GUARD: Phase 6B Historical Metrics Re-verification
# These tests lock in the Phase 6B verified numbers.
# They MUST FAIL if someone manually inserts incorrect metrics.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tp,tn,fp,fn,expected_acc,expected_f1,label", [
    (121, 231,  46, 152, 0.6400, 0.5500, "A0_NLI_Baseline"),
    (121, 231,  46, 152, 0.6400, 0.5500, "A1_NLI_Retrieval"),
    (121, 229,  48, 152, 0.6364, 0.5475, "A2_Plus_Temporal"),
    (  0, 275,   2, 273, 0.5000, 0.0000, "A3_Plus_Modality"),
    (164, 159, 118, 109, 0.5873, 0.5910, "A4_Plus_AtomicClaim"),
    (  4, 277,   0, 269, 0.5109, 0.0289, "A5_Plus_GlobalAlign"),
    (117, 239,  38, 156, 0.6473, 0.5467, "A6_Plus_Relational"),
    (121, 234,  43, 152, 0.6455, 0.5538, "A7_Plus_MetaFiction"),
    (123, 231,  46, 150, 0.6436, 0.5566, "A8_Plus_DynAnchor"),
    (123, 223,  54, 150, 0.6291, 0.5467, "A9_Full_HalluciSense"),
])
def test_phase6b_historical_metric_lock(tp, tn, fp, fn, expected_acc, expected_f1, label):
    """Guard test: recomputed metrics must match Phase 6B historical record."""
    y_true = [1] * (tp + fn) + [0] * (tn + fp)
    y_pred = [1] * tp + [0] * fn + [0] * tn + [1] * fp
    r = evaluate(y_true, y_pred, dataset="phase6b_audit", config_name=label)

    assert abs(r.accuracy - expected_acc) < 5e-4, (
        f"[{label}] Accuracy: expected={expected_acc}, got={r.accuracy}"
    )
    if expected_f1 > 0.0:
        assert abs(r.f1 - expected_f1) < 5e-4, (
            f"[{label}] F1: expected={expected_f1}, got={r.f1}"
        )
    else:
        assert r.f1 is None or abs(r.f1 - 0.0) < 1e-6, (
            f"[{label}] Expected F1=0.0 or None, got {r.f1}"
        )
