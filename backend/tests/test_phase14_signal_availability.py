"""Phase 14 Signal Availability Robustness & Zero-Logit Safety Tests."""

from __future__ import annotations

import pytest
from app.core.engine.fusion import FusionEngine
from app.core.engine.model_registry import ModelRegistry


class TestPhase14SignalAvailability:
    def test_all_seven_signal_mask_combinations(self):
        """Validates that all 7 binary signal masks renormalize weights correctly to sum=1.0."""
        fusion = FusionEngine(alpha=0.40, beta=0.30, gamma=0.30)
        masks = [
            (0.8, 0.6, 0.4, [1, 1, 1]),
            (0.8, None, 0.4, [1, 0, 1]),
            (0.8, 0.6, None, [1, 1, 0]),
            (None, 0.6, 0.4, [0, 1, 1]),
            (0.8, None, None, [1, 0, 0]),
            (None, 0.6, None, [0, 1, 0]),
            (None, None, 0.4, [0, 0, 1]),
        ]

        for fe, cg, cf, expected_mask in masks:
            h_val, eff_w, observed_mask = fusion.compute_adaptive_h_score(
                fe=fe,
                cg=cg,
                cf=cf,
            )
            assert observed_mask == expected_mask
            assert round(sum(eff_w.values()), 4) == 1.0
            assert 0.0 <= h_val <= 1.0

    def test_zero_logit_manufacturing_invariant(self):
        """Ensures that null/absent logprobs are strictly recorded as available=False."""
        pipeline = ModelRegistry.get_pipeline()
        report = pipeline.analyze("The speed of light in vacuum is 299,792,458 m/s.", token_probabilities=None)
        assert report.pillar2_summary.available is False
        assert report.pillar2_summary.confidence_gap_score is None
        assert report.weights_used.get("beta_confidence_gap", 0.0) == 0.0
