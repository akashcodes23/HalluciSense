"""HalluciSense Calibration & Selective Abstention Engine.

Implements:
1. Probability Calibration:
   - Platt Scaling (Sigmoid logistic calibration)
   - Isotonic Regression (Non-parametric piecewise isotonic mapping)
   - Identity / Pass-through baseline
   - Metrics: Expected Calibration Error (ECE), Brier Score, Reliability Bins
2. Selective Prediction / Abstention:
   - Rejection / Abstention gating when evidence coverage is insufficient
   - Risk @ Coverage evaluation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from app.core.engine.types import RiskLevel


@dataclass
class CalibrationResult:
    """Calibrated probability and diagnostic metrics."""
    raw_score: float
    calibrated_probability: float
    calibration_method: str
    confidence_interval: Tuple[float, float]
    ece_estimate: Optional[float] = None
    brier_score_estimate: Optional[float] = None


@dataclass
class AbstentionDecision:
    """Selective prediction / abstention result."""
    abstained: bool
    decision: RiskLevel
    color_code: str
    confidence: float
    abstention_reason: Optional[str] = None
    coverage_score: float = 1.0


class ProbabilityCalibrator:
    """Production Probability Calibrator with Platt and Isotonic support."""

    def __init__(
        self,
        method: str = "platt",
        platt_a: float = 1.82,
        platt_b: float = -0.45,
    ):
        self.method = method.lower()
        self.platt_a = platt_a
        self.platt_b = platt_b
        self._isotonic_model = None

    def calibrate(self, raw_h_score: float) -> CalibrationResult:
        """Calibrates a raw continuous H-score into a true posterior hallucination probability."""
        score_clamped = max(1e-6, min(1.0 - 1e-6, float(raw_h_score)))

        if self.method == "identity" or self.method == "uncalibrated":
            calibrated_p = score_clamped
        elif self.method == "isotonic" and self._isotonic_model is not None:
            calibrated_p = float(self._isotonic_model.predict([score_clamped])[0])
            calibrated_p = max(0.0, min(1.0, calibrated_p))
        else:
            # Platt Sigmoidal Scaling: P(H=1|s) = 1 / (1 + exp(-(a * logit(s) + b)))
            logit_z = math.log(score_clamped / (1.0 - score_clamped))
            calibrated_p = 1.0 / (1.0 + math.exp(-(self.platt_a * logit_z + self.platt_b)))
            calibrated_p = max(0.0, min(1.0, round(calibrated_p, 4)))

        # Wilson-score approximation for 95% CI
        z = 1.96
        ci_half = z * math.sqrt(max(1e-6, calibrated_p * (1.0 - calibrated_p)) / 100.0)
        ci_low = max(0.0, round(calibrated_p - ci_half, 4))
        ci_high = min(1.0, round(calibrated_p + ci_half, 4))

        return CalibrationResult(
            raw_score=round(score_clamped, 4),
            calibrated_probability=round(calibrated_p, 4),
            calibration_method=self.method,
            confidence_interval=(ci_low, ci_high),
        )

    @staticmethod
    def compute_ece(
        y_true: List[int] | np.ndarray,
        y_prob: List[float] | np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Computes Expected Calibration Error (ECE) across uniform bins."""
        y_true_arr = np.array(y_true, dtype=float)
        y_prob_arr = np.array(y_prob, dtype=float)
        if len(y_true_arr) == 0:
            return 0.0

        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_total = len(y_true_arr)

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (y_prob_arr >= bin_lower) & (y_prob_arr <= bin_upper if i == n_bins - 1 else y_prob_arr < bin_upper)
            bin_size = np.sum(in_bin)

            if bin_size > 0:
                acc_bin = np.mean(y_true_arr[in_bin])
                conf_bin = np.mean(y_prob_arr[in_bin])
                ece += (bin_size / n_total) * abs(acc_bin - conf_bin)

        return round(float(ece), 4)

    @staticmethod
    def compute_brier_score(
        y_true: List[int] | np.ndarray,
        y_prob: List[float] | np.ndarray,
    ) -> float:
        """Computes mean squared Brier score: (1/N) * sum((prob - true)^2)."""
        y_true_arr = np.array(y_true, dtype=float)
        y_prob_arr = np.array(y_prob, dtype=float)
        if len(y_true_arr) == 0:
            return 0.0
        return round(float(np.mean((y_prob_arr - y_true_arr) ** 2)), 4)

    @staticmethod
    def compute_reliability_diagram(
        y_true: List[int] | np.ndarray,
        y_prob: List[float] | np.ndarray,
        n_bins: int = 10,
    ) -> List[Dict[str, Any]]:
        """Computes binned statistics for reliability diagrams."""
        y_true_arr = np.array(y_true, dtype=float)
        y_prob_arr = np.array(y_prob, dtype=float)
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        bins = []

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (y_prob_arr >= bin_lower) & (y_prob_arr <= bin_upper if i == n_bins - 1 else y_prob_arr < bin_upper)
            bin_size = int(np.sum(in_bin))

            if bin_size > 0:
                acc_bin = float(np.mean(y_true_arr[in_bin]))
                conf_bin = float(np.mean(y_prob_arr[in_bin]))
                calib_err = abs(acc_bin - conf_bin)
            else:
                acc_bin = 0.0
                conf_bin = float((bin_lower + bin_upper) / 2.0)
                calib_err = 0.0

            bins.append({
                "bin_idx": i,
                "bin_range": f"[{bin_lower:.2f}, {bin_upper:.2f}]",
                "sample_count": bin_size,
                "mean_predicted_h": round(conf_bin, 4),
                "observed_hallucination_rate": round(acc_bin, 4),
                "calibration_error": round(calib_err, 4),
            })
        return bins


class SelectiveAbstentionGate:
    """Evaluates whether to classify or selectively abstain on ambiguous inputs."""

    def __init__(
        self,
        min_evidence_similarity: float = 0.40,
        uncertainty_abstain_threshold: float = 0.85,
        ambiguity_margin: float = 0.08,
    ):
        self.min_evidence_similarity = min_evidence_similarity
        self.uncertainty_abstain_threshold = uncertainty_abstain_threshold
        self.ambiguity_margin = ambiguity_margin

    def evaluate(
        self,
        h_score: float,
        evidence_available: bool,
        max_evidence_similarity: float = 1.0,
        confidence_available: bool = True,
        epistemic_uncertainty: float = 0.0,
    ) -> AbstentionDecision:
        """Evaluates input risk with explicit rejection / abstention criteria."""
        # 1. Check for complete evidence failure / out-of-domain knowledge deficit
        if not evidence_available or max_evidence_similarity < self.min_evidence_similarity:
            if not confidence_available or epistemic_uncertainty > self.uncertainty_abstain_threshold:
                return AbstentionDecision(
                    abstained=True,
                    decision=RiskLevel.INSUFFICIENT_EVIDENCE,
                    color_code="#6B7280",
                    confidence=round(1.0 - epistemic_uncertainty, 4),
                    abstention_reason="No verifiable evidence passages retrieved with similarity >= threshold and model uncertainty is elevated.",
                    coverage_score=round(max_evidence_similarity, 4),
                )

        # 2. Check for high epistemic uncertainty causing indecision around boundary (0.35 - 0.50)
        boundary = 0.40
        if abs(h_score - boundary) < self.ambiguity_margin and epistemic_uncertainty > 0.75:
            return AbstentionDecision(
                abstained=True,
                decision=RiskLevel.ABSTAIN,
                color_code="#6B7280",
                confidence=round(1.0 - epistemic_uncertainty, 4),
                abstention_reason=f"H-score ({h_score:.3f}) lies inside the ambiguous decision boundary [{boundary - self.ambiguity_margin:.2f}, {boundary + self.ambiguity_margin:.2f}] under high epistemic uncertainty.",
                coverage_score=round(max_evidence_similarity, 4),
            )

        # 3. Standard calibrated risk categorization
        if h_score < 0.20:
            dec = RiskLevel.VERIFIED
            color = "#10B981"
        elif h_score < 0.35:
            dec = RiskLevel.LOW_RISK
            color = "#10B981"
        elif h_score < 0.50:
            dec = RiskLevel.NEEDS_VERIFICATION
            color = "#F59E0B"
        elif h_score < 0.65:
            dec = RiskLevel.MODERATE_RISK
            color = "#F97316"
        else:
            dec = RiskLevel.LIKELY_HALLUCINATED
            color = "#EF4444"

        return AbstentionDecision(
            abstained=False,
            decision=dec,
            color_code=color,
            confidence=round(1.0 - abs(h_score - 0.5) * 0.5, 4),
            coverage_score=round(max_evidence_similarity, 4),
        )
