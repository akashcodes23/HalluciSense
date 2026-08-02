"""Calibration and score distribution analysis for HalluciSense Phase 6A evaluation.

Evaluates how well predicted H-Scores calibrate with actual ground-truth hallucination risk.
"""

import math
from typing import Any, Dict, List, Optional
from evaluation.metrics import compute_brier_score, compute_ece


def calculate_distribution_stats(values: List[float]) -> Dict[str, Any]:
    """Calculates summary statistics for a list of score values."""
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "p25": None,
            "p75": None,
        }

    n = len(values)
    sorted_v = sorted(values)
    mean_val = sum(sorted_v) / n

    # Variance & Std
    variance = sum((x - mean_val) ** 2 for x in sorted_v) / n
    std_val = math.sqrt(variance)

    def percentile(p: float) -> float:
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_v[int(k)]
        return sorted_v[int(f)] * (c - k) + sorted_v[int(c)] * (k - f)

    return {
        "count": n,
        "mean": round(mean_val, 4),
        "median": round(percentile(0.50), 4),
        "std": round(std_val, 4),
        "min": round(sorted_v[0], 4),
        "max": round(sorted_v[-1], 4),
        "p25": round(percentile(0.25), 4),
        "p75": round(percentile(0.75), 4),
    }


def analyze_score_distributions(
    y_true: List[int], scores: List[float]
) -> Dict[str, Dict[str, Any]]:
    """Calculates H-Score distributions separately for factual (0) vs hallucinated (1) responses."""
    factual_scores = [score for true, score in zip(y_true, scores) if true == 0]
    hallucinated_scores = [score for true, score in zip(y_true, scores) if true == 1]

    return {
        "factual": calculate_distribution_stats(factual_scores),
        "hallucinated": calculate_distribution_stats(hallucinated_scores),
    }


def analyze_calibration(
    y_true: List[int], scores: List[float], num_bins: int = 10
) -> Dict[str, Any]:
    """Bins H-Scores and computes calibration curve data, Brier Score, and ECE."""
    if not y_true or len(y_true) != len(scores):
        return {
            "brier_score": None,
            "ece": None,
            "bins": [],
        }

    bins_data = []
    bin_size_step = 1.0 / num_bins

    for i in range(num_bins):
        bin_lower = i * bin_size_step
        bin_upper = (i + 1) * bin_size_step

        bin_samples = []
        for true, score in zip(y_true, scores):
            if i == num_bins - 1:
                if bin_lower <= score <= bin_upper:
                    bin_samples.append((true, score))
            else:
                if bin_lower <= score < bin_upper:
                    bin_samples.append((true, score))

        count = len(bin_samples)
        if count > 0:
            mean_predicted = sum(s for _, s in bin_samples) / count
            observed_frequency = sum(t for t, _ in bin_samples) / count
        else:
            mean_predicted = None
            observed_frequency = None

        bins_data.append(
            {
                "bin_range": f"{bin_lower:.1f}-{bin_upper:.1f}",
                "count": count,
                "mean_predicted_h_score": (
                    round(mean_predicted, 4) if mean_predicted is not None else None
                ),
                "observed_hallucination_freq": (
                    round(observed_frequency, 4)
                    if observed_frequency is not None
                    else None
                ),
            }
        )

    brier = compute_brier_score(y_true, scores)
    ece = compute_ece(y_true, scores, num_bins=num_bins)

    return {
        "brier_score": round(brier, 4) if brier is not None else None,
        "ece": round(ece, 4) if ece is not None else None,
        "bins": bins_data,
    }
