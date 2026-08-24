"""Phase 14 Probability Calibration & Reliability Tests."""

from __future__ import annotations

import pytest
import numpy as np
from app.core.engine.calibration import ProbabilityCalibrator


class TestPhase14Calibration:
    def test_ece_and_brier_score_bounds(self):
        """Validates ECE and Brier score mathematical invariants."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.15, 0.85, 0.95, 0.05])

        ece = ProbabilityCalibrator.compute_ece(y_true, y_prob, n_bins=5)
        brier = ProbabilityCalibrator.compute_brier_score(y_true, y_prob)

        assert 0.0 <= ece <= 1.0
        assert 0.0 <= brier <= 1.0
        assert brier < 0.10  # Well-calibrated predictions have low Brier score

    def test_platt_scaling_monotonicity(self):
        """Verifies that Platt scaling preserves score ranking monotonicity."""
        calibrator = ProbabilityCalibrator(method="platt", platt_a=1.82, platt_b=-0.45)
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        calibrated = [calibrator.calibrate(s).calibrated_probability for s in scores]

        for i in range(len(calibrated) - 1):
            assert calibrated[i] <= calibrated[i + 1]
