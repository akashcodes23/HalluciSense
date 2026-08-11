"""Canonical Publication-Grade Evaluation Module for HalluciSense Phase 6C.

This module provides the single authoritative evaluation interface used in all
Phase 6C publication experiments. Every experiment that reports metrics must go
through this module.

Design Principles:
  - All metrics computed from raw predictions, never from rounded intermediates.
  - Structured return type (EvaluationResult dataclass) for machine-readability.
  - Metadata recording: seed, git SHA, timestamp, dataset name, N.
  - Safe edge-case handling: all-positive, all-negative, zero-division.
  - Deterministic: given identical inputs produces identical outputs.

Usage:
    from evaluation.canonical_evaluator import CanonicalEvaluator, evaluate

    result = evaluate(y_true, y_pred, y_score, dataset="halubench", seed=42)
    print(result.to_dict())
"""

from __future__ import annotations

import math
import subprocess
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """Structured, typed evaluation result for a single experiment configuration.

    All float metrics are rounded to 6 significant figures to avoid
    floating-point display noise while preserving precision.
    """

    # Confusion matrix
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    total: int = 0

    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    specificity: Optional[float] = None
    fpr: Optional[float] = None          # False positive rate
    fnr: Optional[float] = None          # False negative rate
    balanced_accuracy: Optional[float] = None
    mcc: Optional[float] = None          # Matthews correlation coefficient

    # Score-based metrics (require y_score)
    auroc: Optional[float] = None
    auprc: Optional[float] = None

    # Class distribution
    positive_count: int = 0
    negative_count: int = 0
    positive_ratio: Optional[float] = None
    negative_ratio: Optional[float] = None

    # Experiment metadata
    dataset: str = ""
    config_name: str = ""
    seed: Optional[int] = None
    git_sha: str = ""
    timestamp: str = ""
    n_samples: int = 0
    notes: str = ""

    def to_dict(self) -> Dict:
        """Return all fields as a JSON-serializable dictionary."""
        return asdict(self)

    def summary(self) -> str:
        """Return a compact one-line metric summary."""
        return (
            f"Acc={self.accuracy:.4f} P={self.precision:.4f} R={self.recall:.4f} "
            f"F1={self.f1:.4f} MCC={self.mcc:.4f} AUROC={self.auroc:.4f} "
            f"BAcc={self.balanced_accuracy:.4f} FPR={self.fpr:.4f} FNR={self.fnr:.4f}"
        )


# ---------------------------------------------------------------------------
# AUROC / AUPRC  (no external dependency)
# ---------------------------------------------------------------------------

def _compute_auroc(y_true: List[int], y_score: List[float]) -> Optional[float]:
    """Mann–Whitney U statistic AUROC. Returns None when single class."""
    pos_count = sum(y_true)
    neg_count = len(y_true) - pos_count
    if pos_count == 0 or neg_count == 0:
        return None
    paired = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    rank_sum = 0.0
    for rank, (_, gold) in enumerate(paired, 1):
        if gold:
            rank_sum += rank
    auroc = 1.0 - (rank_sum - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count)
    return max(0.0, min(1.0, auroc))


def _compute_auprc(y_true: List[int], y_score: List[float]) -> Optional[float]:
    """Trapezoidal Precision-Recall AUC. Returns None when no positives."""
    pos_count = sum(y_true)
    if pos_count == 0:
        return None
    paired = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    tp = fp = 0
    prev_rec = 0.0
    auprc = 0.0
    for score, gold in paired:
        if gold:
            tp += 1
        else:
            fp += 1
        prec = tp / (tp + fp)
        rec = tp / pos_count
        auprc += prec * (rec - prev_rec)
        prev_rec = rec
    return auprc


# ---------------------------------------------------------------------------
# MCC
# ---------------------------------------------------------------------------

def _compute_mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    """Matthews Correlation Coefficient. Returns 0.0 when denominator is 0."""
    denom = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if denom == 0.0:
        return 0.0
    return float((tp * tn) - (fp * fn)) / denom


# ---------------------------------------------------------------------------
# Git SHA helper
# ---------------------------------------------------------------------------

def _get_git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

class CanonicalEvaluator:
    """Publication-grade binary classification evaluator for HalluciSense."""

    def __init__(self, threshold: float = 0.50):
        """
        Args:
            threshold: Score threshold for positive class prediction.
                       Applied when y_pred is not supplied explicitly.
        """
        self.threshold = threshold

    def evaluate(
        self,
        y_true: List[int],
        y_pred: Optional[List[int]] = None,
        y_score: Optional[List[float]] = None,
        *,
        dataset: str = "",
        config_name: str = "",
        seed: Optional[int] = None,
        notes: str = "",
    ) -> EvaluationResult:
        """Compute all publication metrics and return a structured result.

        Args:
            y_true:  Ground-truth binary labels (1=hallucinated, 0=factual).
            y_pred:  Predicted binary labels. If None, thresholded from y_score.
            y_score: Continuous hallucination scores in [0, 1].
            dataset: Dataset identifier for metadata.
            config_name: Experiment configuration label.
            seed:    Random seed used (for metadata only).
            notes:   Free-text notes.

        Returns:
            EvaluationResult with all metrics populated.
        """
        if not y_true:
            raise ValueError("y_true must not be empty.")
        if y_score is not None and len(y_score) != len(y_true):
            raise ValueError("y_score and y_true must have the same length.")
        if y_pred is not None and len(y_pred) != len(y_true):
            raise ValueError("y_pred and y_true must have the same length.")

        # Derive y_pred from scores if not supplied
        if y_pred is None:
            if y_score is None:
                raise ValueError("At least one of y_pred or y_score must be supplied.")
            y_pred = [1 if s >= self.threshold else 0 for s in y_score]

        n = len(y_true)
        tp = tn = fp = fn = 0
        for t, p in zip(y_true, y_pred):
            if t == 1 and p == 1:
                tp += 1
            elif t == 0 and p == 0:
                tn += 1
            elif t == 0 and p == 1:
                fp += 1
            elif t == 1 and p == 0:
                fn += 1

        # Class distribution
        pos_count = sum(y_true)
        neg_count = n - pos_count

        # Scalar metrics (safe division)
        accuracy = (tp + tn) / n if n > 0 else None
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        specificity = tn / (tn + fp) if (tn + fp) > 0 else None
        f1 = (2 * precision * recall / (precision + recall)
              if (precision is not None and recall is not None and (precision + recall) > 0)
              else None)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else None
        fnr = fn / (fn + tp) if (fn + tp) > 0 else None
        balanced_accuracy = (
            (recall + specificity) / 2.0
            if (recall is not None and specificity is not None)
            else None
        )
        mcc = _compute_mcc(tp, tn, fp, fn)

        # Score-based metrics
        auroc = _compute_auroc(y_true, y_score) if y_score is not None else None
        auprc = _compute_auprc(y_true, y_score) if y_score is not None else None

        def _r(v: Optional[float], d: int = 6) -> Optional[float]:
            return round(v, d) if v is not None else None

        return EvaluationResult(
            tp=tp, tn=tn, fp=fp, fn=fn, total=n,
            accuracy=_r(accuracy),
            precision=_r(precision),
            recall=_r(recall),
            f1=_r(f1),
            specificity=_r(specificity),
            fpr=_r(fpr),
            fnr=_r(fnr),
            balanced_accuracy=_r(balanced_accuracy),
            mcc=_r(mcc),
            auroc=_r(auroc),
            auprc=_r(auprc),
            positive_count=pos_count,
            negative_count=neg_count,
            positive_ratio=_r(pos_count / n) if n > 0 else None,
            negative_ratio=_r(neg_count / n) if n > 0 else None,
            dataset=dataset,
            config_name=config_name,
            seed=seed,
            git_sha=_get_git_sha(),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            n_samples=n,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

_DEFAULT_EVALUATOR = CanonicalEvaluator()


def evaluate(
    y_true: List[int],
    y_pred: Optional[List[int]] = None,
    y_score: Optional[List[float]] = None,
    *,
    dataset: str = "",
    config_name: str = "",
    seed: Optional[int] = None,
    threshold: float = 0.50,
    notes: str = "",
) -> EvaluationResult:
    """Convenience wrapper around CanonicalEvaluator."""
    ev = CanonicalEvaluator(threshold=threshold)
    return ev.evaluate(
        y_true, y_pred, y_score,
        dataset=dataset, config_name=config_name, seed=seed, notes=notes,
    )
