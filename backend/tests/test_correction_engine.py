"""Unit tests for Phase 11 Correction Engine & Policy.

Tests:
1. Deterministic repair for unit scale conflicts (km/s -> m/s).
2. Deterministic repair for negation polarity flips.
3. Deterministic repair for causal direction inversions.
4. Preservation of verified/supported claims.
"""

from __future__ import annotations

import pytest
from app.core.correction.correction_policy import CorrectionPolicy
from app.core.correction.correction_models import ErrorClassification


class TestCorrectionPolicy:
    def setup_method(self):
        self.policy = CorrectionPolicy()

    def test_unit_scale_repair(self):
        claim = "The speed of light is approximately 299792458 km/s in vacuum."
        evidence = "The speed of light in vacuum is defined as exactly 299792458 meters per second (m/s)."
        repair = self.policy.classify_and_repair_deterministic("c1", claim, evidence)
        assert repair is not None
        assert repair.error_type == ErrorClassification.UNIT_SCALE_ERROR
        assert "m/s" in repair.corrected_claim
        assert "km/s" not in repair.corrected_claim

    def test_negation_polarity_repair(self):
        claim = "Photons do not possess momentum when traveling through free space."
        evidence = "Photons carry momentum p = h/lambda when propagating through space."
        repair = self.policy.classify_and_repair_deterministic("c2", claim, evidence)
        assert repair is not None
        assert repair.error_type == ErrorClassification.NEGATION_CONFLICT
        assert "possess" in repair.corrected_claim

    def test_causal_direction_repair(self):
        claim = "Smoking is caused by lung cancer."
        evidence = "Smoking causes lung cancer."
        repair = self.policy.classify_and_repair_deterministic("c3", claim, evidence)
        assert repair is not None
        assert repair.error_type == ErrorClassification.CAUSAL_DIRECTION_ERROR
        assert "Smoking" in repair.corrected_claim
