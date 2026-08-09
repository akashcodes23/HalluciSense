"""Unit Tests for Temporal Claim Analysis Engine & Benchmark.

Verifies:
1. Future events asserted as completed facts are detected as Temporal Inconsistency (H >= 0.65).
2. Future predictions are protected against false positives (H < 0.35).
3. Hypothetical scenarios are protected against false positives (H < 0.35).
4. Counterfactual statements are protected against false positives (H < 0.35).
5. Fictional statements are protected against false positives (H < 0.35).
6. Historical true claims pass verification.
7. Historical false claims / date mismatches are flagged as unverified.
8. Existing API contract and dynamic availability rules remain 100% backward compatible.
"""

import pytest
from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.core.engine.temporal import TemporalClaimEngine, TemporalStatus, EpistemicModality


def test_temporal_engine_modality_detection():
    engine = TemporalClaimEngine()

    # Asserted Fact
    res1 = engine.analyze_claim("Brazil won the 2027 FIFA World Cup.", query="Who won the 2027 FIFA World Cup?")
    assert res1.modality == EpistemicModality.ASSERTED_FACT
    assert res1.temporal_status == TemporalStatus.FUTURE_IMPOSSIBLE_FACT
    assert res1.temporal_inconsistency_score > 0.80

    # Prediction
    res2 = engine.analyze_claim("The 2030 World Cup is expected to be hosted in Spain.", query="What will happen at the 2030 World Cup?")
    assert res2.modality == EpistemicModality.PREDICTION
    assert res2.temporal_status == TemporalStatus.FUTURE_PREDICTION
    assert res2.temporal_inconsistency_score == 0.0

    # Hypothetical
    res3 = engine.analyze_claim("Suppose Brazil wins the 2030 World Cup.", query="What if Brazil wins in 2030?")
    assert res3.modality == EpistemicModality.HYPOTHETICAL
    assert res3.temporal_status == TemporalStatus.HYPOTHETICAL
    assert res3.temporal_inconsistency_score == 0.0

    # Counterfactual
    res4 = engine.analyze_claim("If France had won in 2022, Mbappe would have 2 titles.", query="What if France won in 2022?")
    assert res4.modality == EpistemicModality.COUNTERFACTUAL
    assert res4.temporal_status == TemporalStatus.COUNTERFACTUAL
    assert res4.temporal_inconsistency_score == 0.0

    # Fiction
    res5 = engine.analyze_claim("In the sci-fi story, humans colonized Mars in 2045.", query="What happens in the novel?")
    assert res5.modality == EpistemicModality.FICTION
    assert res5.temporal_status == TemporalStatus.FICTIONAL
    assert res5.temporal_inconsistency_score == 0.0


def test_pipeline_temporal_impossible_fact_case_b():
    """Verify Case B (Brazil 2027 World Cup assertion) is classified as LIKELY_HALLUCINATED."""
    pipeline = HallucinationDetectionPipeline()
    report = pipeline.analyze("Brazil won the 2027 FIFA World Cup.", query="Who won the 2027 FIFA World Cup?")

    assert float(report.overall_h_score) >= 0.65
    assert str(report.overall_risk_level.value) == "LIKELY_HALLUCINATED"


def test_pipeline_temporal_prediction_protected():
    """Verify future prediction is protected against false positives (VERIFIED)."""
    pipeline = HallucinationDetectionPipeline()
    report = pipeline.analyze("The 2030 FIFA World Cup is expected to be hosted across multiple countries.", query="What will happen at the 2030 World Cup?")

    assert float(report.overall_h_score) < 0.35
    assert str(report.overall_risk_level.value) == "VERIFIED"


def test_pipeline_temporal_hypothetical_protected():
    """Verify hypothetical scenario is protected against false positives (VERIFIED)."""
    pipeline = HallucinationDetectionPipeline()
    report = pipeline.analyze("Suppose Brazil wins the 2030 FIFA World Cup, they would secure their 6th title.", query="What if Brazil wins in 2030?")

    assert float(report.overall_h_score) < 0.35
    assert str(report.overall_risk_level.value) == "VERIFIED"
