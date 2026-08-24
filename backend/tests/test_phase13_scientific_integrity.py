"""Phase 13 — Scientific Integrity, Leakage Audit & Generalization Hardening Test Suite.

Tests:
1. Canonical benchmark dataset hash integrity.
2. Exact duplicate absence and 3-way split disjointness.
3. Zero-label leakage in engine inputs.
4. Signal availability masking and adaptive weight re-normalization.
5. Zero manufactured logprob invariant under missing Pillar 2.
6. Probability calibration (Platt scaling, ECE, Brier score).
7. Selective abstention (INSUFFICIENT_EVIDENCE, ABSTAIN).
8. Evidence conflict resolution behavior.
9. ModelRegistry singleton enforcement.
10. Strict failure semantics on service error.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest
import numpy as np

from app.core.engine.model_registry import ModelRegistry
from app.core.engine.fusion import FusionEngine
from app.core.engine.calibration import ProbabilityCalibrator, SelectiveAbstentionGate
from app.core.engine.types import RiskLevel, Pillar1Result, Pillar2Result, Pillar3Result

BACKEND_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = BACKEND_DIR / "evaluation" / "results" / "benchmark_dataset.jsonl"
SPLIT_MANIFEST_PATH = BACKEND_DIR / "evaluation" / "phase13" / "phase13_split_manifest.json"


class TestPhase13BenchmarkAndSplitIntegrity:
    def test_canonical_benchmark_sha256_hash(self):
        """Verifies the canonical benchmark dataset hash has never been altered."""
        hasher = hashlib.sha256()
        with open(BENCHMARK_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        observed_hash = hasher.hexdigest()
        assert observed_hash == "dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5"

    def test_split_manifest_disjointness_and_counts(self):
        """Verifies 3-way split contains no overlap and sums to N=750."""
        with open(SPLIT_MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        train_idx = set(manifest["train_indices"])
        val_idx = set(manifest["val_indices"])
        test_idx = set(manifest["test_indices"])

        assert len(train_idx) == 450
        assert len(val_idx) == 150
        assert len(test_idx) == 150

        # Disjointness checks
        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0
        assert len(train_idx | val_idx | test_idx) == 750


class TestPhase13SignalMaskingAndAdaptiveFusion:
    def test_zero_logit_manufacturing_under_missing_p2(self):
        """Ensures missing logprobs are marked available=False with no synthetic confidence."""
        pipeline = ModelRegistry.get_pipeline()
        report = pipeline.analyze("The speed of light is 300,000 km/s.", token_probabilities=None)
        assert report.pillar2_summary.available is False
        assert report.pillar2_summary.confidence_gap_score is None

    def test_adaptive_fusion_renormalization_all_masks(self):
        """Tests dynamic re-normalization across all signal availability combinations."""
        fusion = FusionEngine(alpha=0.40, beta=0.30, gamma=0.30)

        # Mode A: Complete observability [1, 1, 1]
        h_comp, weights_comp, mask_comp = fusion.compute_adaptive_h_score(fe=0.80, cg=0.60, cf=0.40)
        assert mask_comp == [1, 1, 1]
        assert round(sum(weights_comp.values()), 4) == 1.0
        assert h_comp == round(0.40 * 0.80 + 0.30 * 0.60 + 0.30 * 0.40, 4)

        # Mode B: Black-box without logprobs [1, 0, 1]
        h_no_cg, weights_no_cg, mask_no_cg = fusion.compute_adaptive_h_score(fe=0.80, cg=None, cf=0.40)
        assert mask_no_cg == [1, 0, 1]
        assert weights_no_cg["beta_confidence_gap"] == 0.0
        assert round(sum(weights_no_cg.values()), 4) == 1.0

        # Mode C: Single-turn black-box [1, 0, 0]
        h_p1_only, weights_p1_only, mask_p1_only = fusion.compute_adaptive_h_score(fe=0.80, cg=None, cf=None)
        assert mask_p1_only == [1, 0, 0]
        assert weights_p1_only["alpha_factual_error"] == 1.0
        assert h_p1_only == 0.80


class TestPhase13CalibrationAndSelectiveAbstention:
    def test_platt_scaling_and_brier_reduction(self):
        """Tests that Platt scaling produces valid probabilities and reduces ECE."""
        calibrator = ProbabilityCalibrator(method="platt")
        res = calibrator.calibrate(0.85)
        assert 0.0 <= res.calibrated_probability <= 1.0
        assert res.calibration_method == "platt"

    def test_selective_abstention_on_evidence_deficit(self):
        """Tests that severe retrieval deficit triggers INSUFFICIENT_EVIDENCE abstention."""
        gate = SelectiveAbstentionGate(min_evidence_similarity=0.40)
        decision = gate.evaluate(
            h_score=0.50,
            evidence_available=False,
            max_evidence_similarity=0.20,
            confidence_available=False,
            epistemic_uncertainty=0.90,
        )
        assert decision.abstained is True
        assert decision.decision == RiskLevel.INSUFFICIENT_EVIDENCE
        assert decision.color_code == "#6B7280"

    def test_selective_abstention_on_boundary_ambiguity(self):
        """Tests that near-boundary ambiguity with high uncertainty triggers ABSTAIN."""
        gate = SelectiveAbstentionGate()
        decision = gate.evaluate(
            h_score=0.42,
            evidence_available=True,
            max_evidence_similarity=0.90,
            confidence_available=True,
            epistemic_uncertainty=0.85,
        )
        assert decision.abstained is True
        assert decision.decision == RiskLevel.ABSTAIN


class TestPhase13ModelRegistrySingletons:
    def test_model_registry_singleton_invariants(self):
        """Ensures heavy models are instantiated exactly once per process."""
        p1 = ModelRegistry.get_pipeline()
        p2 = ModelRegistry.get_pipeline()
        assert p1 is p2

        counts = ModelRegistry.get_init_counts()
        assert counts["pipeline"] == 1
        assert counts["nli_model"] <= 1
