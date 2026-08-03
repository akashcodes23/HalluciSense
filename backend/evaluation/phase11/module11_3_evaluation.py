"""
HalluciSense Phase 11 — Module 11.3: Head-to-Head Evaluation Engine
===================================================================
Evaluates HalluciSense against all baseline detectors across 10 core performance metrics:
  ROC-AUC, PR-AUC, F1, MCC, Accuracy, ECE, Brier, Latency, Memory, Throughput (QPS).
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import structlog
from evaluation.phase11.module11_1_datasets import BenchmarkSample
from evaluation.phase11.module11_2_baselines import BaseHallucinationDetector
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = structlog.get_logger(__name__)


@dataclass
class HeadToHeadMetrics:
    system_name: str
    roc_auc: float
    pr_auc: float
    f1_score: float
    mcc: float
    accuracy: float
    ece: float
    brier_score: float
    mean_latency_ms: float
    memory_kb: float
    throughput_qps: float


class HeadToHeadEvaluator:
    """
    Computes rigorous head-to-head performance metrics.
    """

    def compute_ece(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error (ECE)."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n_samples = len(y_true)

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper) if i > 0 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
            bin_size = np.sum(in_bin)

            if bin_size > 0:
                acc_in_bin = np.mean(y_true[in_bin])
                conf_in_bin = np.mean(y_prob[in_bin])
                ece += (bin_size / n_samples) * np.abs(acc_in_bin - conf_in_bin)

        return round(float(ece), 4)

    def evaluate_system(
        self, detector: BaseHallucinationDetector, samples: List[BenchmarkSample]
    ) -> Tuple[HeadToHeadMetrics, np.ndarray, np.ndarray]:
        """
        Evaluate a single detector system across samples.

        Returns
        -------
        Tuple[HeadToHeadMetrics, y_true, y_prob]
        """
        t0 = time.perf_counter()
        tracemalloc.start()

        y_true = np.array([s.ground_truth_label for s in samples], dtype=int)
        probs = []
        preds = []
        latencies = []

        for s in samples:
            t_s = time.perf_counter()
            prob, pred = detector.predict_sample(s)
            lat_ms = (time.perf_counter() - t_s) * 1000.0
            probs.append(prob)
            preds.append(pred)
            latencies.append(lat_ms)

        mem_bytes = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        total_sec = time.perf_counter() - t0
        y_prob = np.array(probs, dtype=float)
        y_pred = np.array(preds, dtype=int)

        # Handle edge cases (e.g. constant predictions)
        try:
            r_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            r_auc = 0.50

        try:
            p_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            p_auc = float(np.mean(y_true))

        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        mcc = float(matthews_corrcoef(y_true, y_pred))
        acc = float(accuracy_score(y_true, y_pred))
        brier = float(brier_score_loss(y_true, y_prob))
        ece = self.compute_ece(y_true, y_prob)

        mean_lat = float(np.mean(latencies))
        throughput = float(len(samples) / total_sec) if total_sec > 0 else 0.0

        metrics = HeadToHeadMetrics(
            system_name=detector.name,
            roc_auc=round(r_auc, 4),
            pr_auc=round(p_auc, 4),
            f1_score=round(f1, 4),
            mcc=round(mcc, 4),
            accuracy=round(acc, 4),
            ece=ece,
            brier_score=round(brier, 4),
            mean_latency_ms=round(mean_lat, 2),
            memory_kb=round(mem_bytes / 1024.0, 2),
            throughput_qps=round(throughput, 1),
        )

        logger.info(
            "head_to_head_evaluated",
            system=detector.name,
            roc_auc=metrics.roc_auc,
            f1=metrics.f1_score,
            ece=metrics.ece,
        )

        return metrics, y_true, y_prob
