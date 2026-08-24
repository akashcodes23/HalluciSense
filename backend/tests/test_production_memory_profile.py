"""
Production Memory Profile & Verification Test Suite.

Validates that HalluciSense in production-lite mode:
1. Loads exactly 1 NLI model.
2. Loads 0 CrossEncoder reranker models.
3. Loads 0 unused SentenceTransformer models.
4. Correctly verifies factual claims (True Speed of Light, True Water).
5. Correctly detects hallucinated claims (False Speed of Light, False Water, Negation).
6. Executes closed-loop correction and re-verification safely.
"""
import pytest
from app.core.config import settings
from app.core.engine.model_registry import ModelRegistry
from app.core.correction.correction_engine import CorrectionEngine
from app.core.engine.types import EvidenceItem


class TestProductionMemoryProfile:
    """Validates memory-safe single-transformer footprint and verification accuracy."""

    def test_production_configuration_defaults(self):
        """Verify production memory settings."""
        assert hasattr(settings, "HALLUCISENSE_ENABLE_RERANKER")
        assert settings.HALLUCISENSE_ENABLE_RERANKER is False

    def test_production_model_registry_initialization(self):
        """Verify pipeline loads single NLI model with zero CrossEncoder/SentenceTransformer."""
        ModelRegistry.reset_for_testing()
        pipeline = ModelRegistry.get_pipeline()
        assert pipeline is not None

        # Verify factual evaluation does not load extra models
        res = pipeline.analyze_response(
            full_text="The chemical formula of water is H2O.",
            query="What is the chemical formula of water?",
            sample_responses=[],
        )
        assert res is not None

        counts = ModelRegistry.get_init_counts()
        assert counts["nli_model"] == 1, "Must have exactly 1 NLI model"
        assert counts["cross_encoder_reranker"] == 0, "Must have 0 CrossEncoder reranker instances"
        assert counts["sentence_transformer"] == 0, "Must have 0 SentenceTransformer instances"
        assert counts["pipeline"] == 1, "Must have exactly 1 Pipeline orchestrator"

    def test_factual_claims_verification(self):
        """Verify true claims receive low hallucination score (VERIFIED)."""
        pipeline = ModelRegistry.get_pipeline()

        # Case 1: True Speed of Light
        ev_sol = [
            EvidenceItem(
                claim="Speed of light",
                snippet="The speed of light in vacuum is defined as exactly 299792458 meters per second (m/s).",
                source_name="Wikipedia: Speed of light",
                similarity_score=0.95,
                is_supporting=True,
            )
        ]
        res1 = pipeline.analyze_response(
            full_text="The speed of light in vacuum is approximately 299,792,458 m/s.",
            query="What is the speed of light in vacuum?",
            evidence_items=ev_sol,
            sample_responses=[],
        )
        assert res1.overall_h_score <= 0.35, f"Expected VERIFIED, got {res1.overall_h_score}"

        # Case 2: True Water formula
        ev_water = [
            EvidenceItem(
                claim="Water formula",
                snippet="Water is an inorganic compound with the chemical formula H2O.",
                source_name="Wikipedia: Water",
                similarity_score=0.95,
                is_supporting=True,
            )
        ]
        res2 = pipeline.analyze_response(
            full_text="Water has the chemical formula H2O.",
            query="What is the chemical formula of water?",
            evidence_items=ev_water,
            sample_responses=[],
        )
        assert res2.overall_h_score <= 0.35, f"Expected VERIFIED, got {res2.overall_h_score}"

    def test_hallucinated_claims_detection(self):
        """Verify false and contradictory claims receive high hallucination score."""
        pipeline = ModelRegistry.get_pipeline()

        # Case 1: False Speed of Light (km/s instead of m/s)
        ev_sol = [
            EvidenceItem(
                claim="Speed of light",
                snippet="The speed of light in vacuum is defined as exactly 299792458 meters per second (m/s).",
                source_name="Wikipedia: Speed of light",
                similarity_score=0.95,
                is_supporting=True,
            )
        ]
        res_false_sol = pipeline.analyze_response(
            full_text="The speed of light in vacuum is approximately 299,792,458 km/s.",
            query="What is the speed of light in vacuum?",
            evidence_items=ev_sol,
            sample_responses=[],
        )
        assert res_false_sol.overall_h_score >= 0.65, f"Expected HIGH RISK, got {res_false_sol.overall_h_score}"

        # Case 2: False Water formula (CO2)
        ev_water = [
            EvidenceItem(
                claim="Water formula",
                snippet="Water is an inorganic compound with the chemical formula H2O.",
                source_name="Wikipedia: Water",
                similarity_score=0.95,
                is_supporting=True,
            )
        ]
        res_false_water = pipeline.analyze_response(
            full_text="Water has the chemical formula CO2.",
            query="What is the chemical formula of water?",
            evidence_items=ev_water,
            sample_responses=[],
        )
        assert res_false_water.overall_h_score >= 0.65, f"Expected HIGH RISK, got {res_false_water.overall_h_score}"

        # Case 3: Negation error (Mitochondria do not produce ATP)
        ev_mito = [
            EvidenceItem(
                claim="Mitochondria ATP",
                snippet="Mitochondria are the cellular organelles that produce ATP in eukaryotic cells.",
                source_name="Wikipedia: Mitochondrion",
                similarity_score=0.95,
                is_supporting=True,
            )
        ]
        res_neg = pipeline.analyze_response(
            full_text="Mitochondria do not produce ATP in eukaryotic cells.",
            query="What role do mitochondria play in ATP production?",
            evidence_items=ev_mito,
            sample_responses=[],
        )
        assert res_neg.overall_h_score >= 0.65, f"Expected HIGH RISK for negation conflict, got {res_neg.overall_h_score}"

    def test_closed_loop_correction_and_reverification(self):
        """Verify closed-loop correction successfully repairs numerical unit errors."""
        pipeline = ModelRegistry.get_pipeline()
        engine = CorrectionEngine(pipeline=pipeline)

        ev_sol = [
            EvidenceItem(
                claim="Speed of light",
                snippet="The speed of light in vacuum is defined as exactly 299792458 meters per second (m/s).",
                source_name="Wikipedia: Speed of light",
                similarity_score=0.95,
                is_supporting=True,
            )
        ]
        text = "The speed of light in vacuum is approximately 299792458 km/s."
        query = "What is the speed of light in vacuum?"

        init_verif = pipeline.analyze_response(full_text=text, query=query, evidence_items=ev_sol, sample_responses=[])
        res = engine.execute_closed_loop_repair(user_query=query, initial_text=text, initial_verification=init_verif)

        assert res.performed is True
        assert "m/s" in res.final_text or "meters" in res.final_text
        assert res.reverification is not None
        assert res.reverification.passed is True
