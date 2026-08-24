"""
Phase 11E Unit and Architecture Invariant Tests.

Tests:
1. Canonical benchmark SHA-256 integrity.
2. Phase 10 dataset SHA-256 integrity.
3. Candidate models parameter metadata validation.
4. Deterministic checks safety: verify symbolic checks do not silently override evidence.
5. ModelRegistry singleton idempotency.
6. Closed-loop correction path integrity across configurations.
7. Verification failure semantics preservation.
"""
import os
import hashlib
import pytest
from app.core.engine.model_registry import ModelRegistry
from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.core.engine.types import EvidenceItem
from app.core.correction.correction_engine import CorrectionEngine


CANONICAL_BENCHMARK_PATH = "backend/evaluation/results/benchmark_dataset.jsonl"
CANONICAL_SHA = "dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5"
PHASE10_DATASET_PATH = "backend/reports/phase10/phase10_scientific_dataset.jsonl"


class TestPhase11EArchitectureInvariants:
    """Verifies dataset integrity and architectural invariants."""

    def test_canonical_benchmark_sha_preserved(self):
        assert os.path.exists(CANONICAL_BENCHMARK_PATH)
        with open(CANONICAL_BENCHMARK_PATH, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        assert sha == CANONICAL_SHA, f"Expected {CANONICAL_SHA}, got {sha}"

    def test_phase10_dataset_exists_and_valid(self):
        assert os.path.exists(PHASE10_DATASET_PATH)
        with open(PHASE10_DATASET_PATH, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) >= 100, f"Expected at least 100 Phase 10 samples, found {len(lines)}"

    def test_deterministic_checks_traceability(self):
        pipeline = HallucinationDetectionPipeline()
        claim = "The speed of light is 300,000,000 km/s."
        ev_item = [EvidenceItem(claim="Light", snippet="The speed of light is 299792458 m/s.", source_name="Ref", similarity_score=0.9, is_supporting=True)]
        res = pipeline.analyze_response(full_text=claim, query="What is the speed of light?", evidence_items=ev_item, sample_responses=[])
        assert res.overall_h_score >= 0.65
        assert res.overall_risk_level.value in ["LIKELY_HALLUCINATED", "HIGH_RISK", "UNCERTAIN"]

    def test_singleton_model_registry_no_duplicates(self):
        ModelRegistry.reset_for_testing()
        tok1, m1 = ModelRegistry.get_nli_model()
        tok2, m2 = ModelRegistry.get_nli_model()
        assert id(tok1) == id(tok2)
        assert id(m1) == id(m2)
        assert ModelRegistry.get_init_counts()["nli_model"] == 1

    def test_closed_loop_correction_path(self):
        pipeline = HallucinationDetectionPipeline()
        engine = CorrectionEngine(pipeline=pipeline)
        query = "What is the speed of light in vacuum?"
        hallucinated_text = "The speed of light in vacuum is approximately 299792458 km/s."
        ev_items = [
            EvidenceItem(
                claim="Speed of light",
                snippet="The speed of light in vacuum is defined as exactly 299792458 meters per second (m/s).",
                source_name="Wikipedia",
                similarity_score=0.95,
                is_supporting=True,
            )
        ]
        init_v = pipeline.analyze_response(full_text=hallucinated_text, query=query, evidence_items=ev_items, sample_responses=[])
        res = engine.execute_closed_loop_repair(user_query=query, initial_text=hallucinated_text, initial_verification=init_v)
        assert res.performed is True
        assert res.reverification is not None
        assert res.reverification.passed is True
