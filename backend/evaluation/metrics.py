"""Metrics Engine for HalluciSense Phase 6A Evaluation.

Provides zero-division safe functions for Confusion Matrix, Accuracy, Precision, Recall, Specificity,
F1 Score, ROC-AUC, PR-AUC, Brier Score, and Expected Calibration Error (ECE).
"""

from typing import Dict, List, Optional, Tuple, Union
import math


def compute_confusion_matrix(
    y_true: List[int], y_pred: List[int]
) -> Tuple[int, int, int, int]:
    """Computes (TP, TN, FP, FN) for binary classification where 1=hallucinated, 0=factual."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have identical length.")

    tp = tn = fp = fn = 0
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 0:
            tn += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1

    return tp, tn, fp, fn


def compute_accuracy(tp: int, tn: int, fp: int, fn: int) -> Optional[float]:
    total = tp + tn + fp + fn
    if total == 0:
        return None
    return (tp + tn) / total


def compute_precision(tp: int, fp: int) -> Optional[float]:
    if tp + fp == 0:
        return None
    return tp / (tp + fp)


def compute_recall(tp: int, fn: int) -> Optional[float]:
    if tp + fn == 0:
        return None
    return tp / (tp + fn)


def compute_specificity(tn: int, fp: int) -> Optional[float]:
    if tn + fp == 0:
        return None
    return tn / (tn + fp)


def compute_f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return None
    return 2.0 * (precision * recall) / (precision + recall)


def compute_brier_score(y_true: List[int], scores: List[float]) -> Optional[float]:
    if not y_true or len(y_true) != len(scores):
        return None
    squared_errors = [(score - true) ** 2 for score, true in zip(scores, y_true)]
    return sum(squared_errors) / len(y_true)


def compute_ece(
    y_true: List[int], scores: List[float], num_bins: int = 10
) -> Optional[float]:
    """Calculates Expected Calibration Error (ECE) across num_bins equal bins in [0, 1]."""
    if not y_true or len(y_true) != len(scores):
        return None

    bin_boundaries = [i / num_bins for i in range(num_bins + 1)]
    total_samples = len(y_true)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Extract samples falling in [bin_lower, bin_upper) or [bin_lower, bin_upper] for last bin
        bin_indices = []
        for idx, s in enumerate(scores):
            if i == num_bins - 1:
                if bin_lower <= s <= bin_upper:
                    bin_indices.append(idx)
            else:
                if bin_lower <= s < bin_upper:
                    bin_indices.append(idx)

        bin_size = len(bin_indices)
        if bin_size > 0:
            avg_confidence = sum(scores[idx] for idx in bin_indices) / bin_size
            avg_accuracy = sum(y_true[idx] for idx in bin_indices) / bin_size
            ece += (bin_size / total_samples) * abs(avg_accuracy - avg_confidence)

    return ece


def compute_roc_auc(y_true: List[int], scores: List[float]) -> Optional[float]:
    """Computes Area Under Receiver Operating Characteristic Curve via trapezoidal rule."""
    if not y_true or len(y_true) != len(scores):
        return None

    # Check for single-class dataset
    num_pos = sum(1 for y in y_true if y == 1)
    num_neg = sum(1 for y in y_true if y == 0)
    if num_pos == 0 or num_neg == 0:
        return None

    # Sort scores descending
    desc_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    sorted_y = [y_true[i] for i in desc_indices]
    sorted_scores = [scores[i] for i in desc_indices]

    fpr_list = [0.0]
    tpr_list = [0.0]

    current_tp = 0
    current_fp = 0

    for i in range(len(sorted_scores)):
        if sorted_y[i] == 1:
            current_tp += 1
        else:
            current_fp += 1

        # Push point when score changes or at end
        if i == len(sorted_scores) - 1 or sorted_scores[i] != sorted_scores[i + 1]:
            fpr_list.append(current_fp / num_neg)
            tpr_list.append(current_tp / num_pos)

    # Calculate trapezoidal AUC
    auc = 0.0
    for i in range(1, len(fpr_list)):
        dx = fpr_list[i] - fpr_list[i - 1]
        avg_height = (tpr_list[i] + tpr_list[i - 1]) / 2.0
        auc += dx * avg_height

    return auc


def compute_pr_auc(y_true: List[int], scores: List[float]) -> Optional[float]:
    """Computes Area Under Precision-Recall Curve via trapezoidal rule."""
    if not y_true or len(y_true) != len(scores):
        return None

    num_pos = sum(1 for y in y_true if y == 1)
    if num_pos == 0:
        return None

    # Sort descending by score
    desc_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    sorted_y = [y_true[i] for i in desc_indices]
    sorted_scores = [scores[i] for i in desc_indices]

    recalls = [0.0]
    precisions = [1.0]

    current_tp = 0
    current_fp = 0

    for i in range(len(sorted_scores)):
        if sorted_y[i] == 1:
            current_tp += 1
        else:
            current_fp += 1

        if i == len(sorted_scores) - 1 or sorted_scores[i] != sorted_scores[i + 1]:
            r = current_tp / num_pos
            p = current_tp / (current_tp + current_fp)
            recalls.append(r)
            precisions.append(p)

    # Trapezoidal area under PR curve
    auc = 0.0
    for i in range(1, len(recalls)):
        dx = recalls[i] - recalls[i - 1]
        avg_height = (precisions[i] + precisions[i - 1]) / 2.0
        auc += dx * avg_height

    return auc


def compute_balanced_accuracy(
    recall: Optional[float], specificity: Optional[float]
) -> Optional[float]:
    if recall is None or specificity is None:
        return None
    return (recall + specificity) / 2.0


def compute_false_positive_rate(fp: int, tn: int) -> Optional[float]:
    if fp + tn == 0:
        return None
    return fp / (fp + tn)


def compute_false_negative_rate(fn: int, tp: int) -> Optional[float]:
    if fn + tp == 0:
        return None
    return fn / (fn + tp)


def compute_all_metrics(
    y_true: List[int], y_pred: List[int], scores: Optional[List[float]] = None
) -> Dict[str, Union[int, float, None]]:
    """Calculates all binary classification and score-based metrics safely."""
    tp, tn, fp, fn = compute_confusion_matrix(y_true, y_pred)
    acc = compute_accuracy(tp, tn, fp, fn)
    prec = compute_precision(tp, fp)
    rec = compute_recall(tp, fn)
    spec = compute_specificity(tn, fp)
    f1 = compute_f1(prec, rec)
    bal_acc = compute_balanced_accuracy(rec, spec)
    fpr = compute_false_positive_rate(fp, tn)
    fnr = compute_false_negative_rate(fn, tp)

    roc_auc = compute_roc_auc(y_true, scores) if scores is not None else None
    pr_auc = compute_pr_auc(y_true, scores) if scores is not None else None
    brier = compute_brier_score(y_true, scores) if scores is not None else None
    ece = compute_ece(y_true, scores) if scores is not None else None

    denom = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = float((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0
    youden_j = (rec + spec - 1.0) if (rec is not None and spec is not None) else None

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round(acc, 4) if acc is not None else None,
        "balanced_accuracy": round(bal_acc, 4) if bal_acc is not None else None,
        "precision": round(prec, 4) if prec is not None else None,
        "recall": round(rec, 4) if rec is not None else None,
        "specificity": round(spec, 4) if spec is not None else None,
        "false_positive_rate": round(fpr, 4) if fpr is not None else None,
        "false_negative_rate": round(fnr, 4) if fnr is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "mcc": round(mcc, 4),
        "youden_j": round(youden_j, 4) if youden_j is not None else None,
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "brier_score": round(brier, 4) if brier is not None else None,
        "ece": round(ece, 4) if ece is not None else None,
    }
