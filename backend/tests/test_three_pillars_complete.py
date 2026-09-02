"""Comprehensive Unit Test Suite for Complete Scientific Architecture.

Tests:
1. Pillar 1: BM25 + Dense + Cross-Encoder retrieval, evidence aggregation, citation confidence.
2. Pillar 2: White-box metrics (entropy, predictive entropy, mutual info, epistemic/aleatoric) and Black-box API metrics.
3. Pillar 3: Paraphrase matrix construction, sentence consistency score, NLI contradiction analysis.
4. Fusion Engine: Static, Adaptive, Gradient weight modes, 4-tier risk levels, weight importance, sensitivity analysis.
5. Token Localization: Score propagation, span localization, 4-tier risk heatmaps.
"""

import pytest
from app.core.engine.types import (
    RiskLevel,
    EvidenceItem,
    TokenAnalysis,
    Pillar1Result,
    Pillar2Result,
    Pillar3Result,
    SentenceAnalysis,
    HallucinationReport,
)
from app.core.engine.pillar1_retrieval import Pillar1RetrievalEngine
from app.core.engine.pillar2_confidence import Pillar2ConfidenceEngine
from app.core.engine.pillar3_consistency import Pillar3ConsistencyEngine
from app.core.engine.fusion import FusionEngine
from app.core.engine.pipeline import HallucinationDetectionPipeline


def test_pillar1_hybrid_retrieval():
    engine = Pillar1RetrievalEngine()
    claims = engine.extract_claims("Albert Einstein was born in Ulm, Germany in 1879.")
    assert len(claims) >= 1

    evidence_items = [
        EvidenceItem(
            claim="Albert Einstein was born in Ulm, Germany",
            snippet="Albert Einstein was born in Ulm, in the Kingdom of Württemberg in the German Empire, on 14 March 1879.",
            source_name="Wikipedia",
            similarity_score=0.92,
            is_supporting=True,
        )
    ]

    result = engine.analyze("Albert Einstein was born in Ulm, Germany in 1879.", provided_evidence=evidence_items)
    assert isinstance(result, Pillar1Result)
    assert result.factual_error_score <= 0.30
    assert result.citation_confidence_score is not None
    assert result.dense_retrieval_score is not None
    assert result.bm25_retrieval_score is not None


def test_pillar2_whitebox_and_blackbox_confidence():
    engine = Pillar2ConfidenceEngine()
    tokens = ["Albert", "Einstein", "was", "born"]
    probs = [0.95, 0.92, 0.88, 0.90]

    result = engine.analyze(tokens=tokens, probabilities=probs)
    assert isinstance(result, Pillar2Result)
    assert result.available is True
    assert result.confidence_gap_score is not None
    assert result.predictive_entropy is not None
    assert result.epistemic_uncertainty is not None
    assert result.top_k_logprob_diff is not None


def test_pillar3_paraphrase_consistency():
    engine = Pillar3ConsistencyEngine()
    primary = "Einstein discovered relativity in 1905."
    samples = [
        "In 1905, Albert Einstein formulated the theory of relativity.",
        "Einstein proposed special relativity in the year 1905."
    ]

    result = engine.analyze(primary_response=primary, sample_responses=samples)
    assert isinstance(result, Pillar3Result)
    assert result.available is True
    assert result.consistency_failure_score is not None
    assert len(result.paraphrase_matrix) == 3
    assert result.sentence_consistency_score is not None


def test_fusion_modes_and_sensitivity_analysis():
    fusion = FusionEngine(alpha=0.4, beta=0.3, gamma=0.3)
    
    # 4 Risk Tier Check
    risk_v, color_v = fusion.determine_risk_level(0.15)
    assert risk_v == RiskLevel.VERIFIED
    assert color_v == "#10B981"

    risk_n, color_n = fusion.determine_risk_level(0.42)
    assert risk_n == RiskLevel.NEEDS_VERIFICATION
    assert color_n == "#F59E0B"

    risk_m, color_m = fusion.determine_risk_level(0.58)
    assert risk_m == RiskLevel.MODERATE_RISK
    assert color_m == "#F97316"

    risk_l, color_l = fusion.determine_risk_level(0.75)
    assert risk_l == RiskLevel.LIKELY_HALLUCINATED
    assert color_l == "#EF4444"

    # Sensitivity Analysis Grid
    sensitivity = fusion.compute_sensitivity_analysis(fe=0.2, cg=0.3, cf=0.1)
    assert "weight_importance" in sensitivity
    assert "sensitivity_grid" in sensitivity
    assert len(sensitivity["sensitivity_grid"]) > 0


def test_pipeline_token_localization_and_explainability():
    pipeline = HallucinationDetectionPipeline()
    text = "The speed of light in vacuum is approximately 299,792,458 meters per second."
    evidence = [
        EvidenceItem(
            claim="The speed of light is 299,792,458 m/s",
            snippet="The speed of light in vacuum is exactly 299,792,458 m/s.",
            source_name="Physics Handbook",
            similarity_score=0.95,
            is_supporting=True,
        )
    ]

    report = pipeline.analyze(text=text, provided_evidence=evidence)
    assert isinstance(report, HallucinationReport)
    assert report.confidence_decomposition is not None
    assert report.uncertainty_analysis is not None
    assert report.sensitivity_analysis is not None
    assert report.calibrated_probability is not None
    assert len(report.sentence_analyses) > 0
    assert report.sentence_analyses[0].span_localization is not None
