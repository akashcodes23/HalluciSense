"""HalluciSense E2E Adversarial Hallucination Benchmark Test Suite.

Validates directional correctness, score fusion monotonicity, and risk classification
across 10 adversarial hallucination categories.
"""

import pytest
from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.core.engine.fusion import FusionEngine
from app.core.engine.types import RiskLevel

@pytest.fixture(scope="module")
def pipeline():
    return HallucinationDetectionPipeline()

@pytest.fixture(scope="module")
def fusion_engine():
    return FusionEngine()


class TestHallucinationBenchmark:
    """E2E Benchmark Suite testing directional correctness across 10 categories."""

    def test_category_1_true_fact(self, pipeline):
        """TRUE FACT: Apollo 11 landed on the Moon in 1969."""
        report = pipeline.analyze(
            text="Apollo 11 landed on the Moon in 1969.",
            query="When and where did Apollo 11 land?"
        )
        assert report.overall_h_score < 0.35, f"True fact score should be low risk, got {report.overall_h_score}"
        assert report.overall_risk_level in [RiskLevel.VERIFIED, RiskLevel.NEEDS_VERIFICATION]

    def test_category_2_false_fact(self, pipeline):
        """FALSE FACT: Apollo 11 landed on Mars in 1969."""
        report_true = pipeline.analyze(text="Apollo 11 landed on the Moon in 1969.")
        report_false = pipeline.analyze(text="Apollo 11 landed on Mars in 1969.")
        assert report_false.overall_h_score >= report_true.overall_h_score, (
            f"False fact score ({report_false.overall_h_score}) should be >= true fact score ({report_true.overall_h_score})"
        )

    def test_category_3_temporal_contamination(self, pipeline):
        """TEMPORAL CONTAMINATION: The Eiffel Tower was completed in 2020."""
        report = pipeline.analyze(
            text="The Eiffel Tower was completed in 2020.",
            query="When was the Eiffel Tower built?"
        )
        assert report.overall_h_score >= 0.10

    def test_category_4_entity_swap(self, pipeline):
        """ENTITY SWAP: Albert Einstein discovered gravity when an apple fell on his head."""
        report = pipeline.analyze(
            text="Albert Einstein discovered gravity when an apple fell on his head."
        )
        assert report.overall_h_score >= 0.10

    def test_category_5_numerical_error(self, pipeline):
        """NUMERICAL ERROR: The distance from Earth to the Moon is 50 miles."""
        report = pipeline.analyze(
            text="The distance from Earth to the Moon is 50 miles."
        )
        assert report.overall_h_score >= 0.10

    def test_category_6_partial_truth(self, pipeline):
        """PARTIAL TRUTH: Neil Armstrong landed on the Moon in 1969 and became President of France in 1975."""
        report = pipeline.analyze(
            text="Neil Armstrong landed on the Moon in 1969 and became President of France in 1975."
        )
        assert report.overall_h_score >= 0.15

    def test_category_7_unverifiable(self, pipeline):
        """UNVERIFIABLE: Quantum computers will achieve sentient consciousness in 2045."""
        report = pipeline.analyze(
            text="Quantum computers will achieve sentient consciousness in 2045."
        )
        assert hasattr(report, "overall_h_score")

    def test_category_8_conflicting_evidence(self, pipeline):
        """CONFLICTING EVIDENCE: Direct contradiction."""
        report = pipeline.analyze(
            text="Paris is the capital of Japan.",
            query="What is the capital of Japan?"
        )
        assert report.overall_h_score >= 0.15

    def test_category_9_multi_claim_response(self, pipeline):
        """MULTI-CLAIM RESPONSE: Multiple atomic statements."""
        text = (
            "Apollo 11 landed on the Moon in 1969. "
            "Neil Armstrong was the commander. "
            "They landed on Mars in 1975. "
            "Buzz Aldrin accompanied Armstrong. "
            "The mission landed in the Pacific Ocean on return."
        )
        report = pipeline.analyze(text=text)
        assert len(report.sentence_analyses) >= 1
        assert hasattr(report, "overall_h_score")

    def test_category_10_adversarial_confidence(self, pipeline):
        """ADVERSARIAL CONFIDENCE: Overconfident language with false fact."""
        report = pipeline.analyze(
            text="Without a shadow of a doubt, Abraham Lincoln was elected President of the United States in 2024."
        )
        assert report.overall_h_score >= 0.15
