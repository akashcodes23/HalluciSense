"""
Unit tests for HalluciSense Pillar 2 — Module 10.8: Unified H-Score.
"""

import pytest
from app.pillar2.contradiction_analysis.schemas import (
    ContradictionAnalysisResult,
    ContradictionGraphVisualization,
)
from app.pillar2.feature_generation.schemas import PillarTwoFeatures
from app.pillar2.unified_hscore.calculator import UnifiedHScoreCalculator
from app.pillar2.unified_hscore.schemas import RiskCategory


@pytest.fixture
def hscore_calculator():
    return UnifiedHScoreCalculator()


def test_very_low_risk_hscore(hscore_calculator):
    p1_prob = 0.10
    p2_feats = PillarTwoFeatures(
        support_ratio=1.0,
        contradiction_ratio=0.0,
        authority_score=0.95,
        source_diversity=0.8,
        evidence_coverage=1.0,
        evidence_density=2.0,
        citation_quality=0.9,
        consensus_confidence=0.95,
        recency_score=1.0,
        verification_completeness=1.0,
    )
    cnt_res = ContradictionAnalysisResult(
        contradictions=[],
        contradiction_count=0,
        fabrication_index=0.0,
        max_severity=0.0,
        graph_visualization=ContradictionGraphVisualization(
            nodes=[], edges=[], total_contradictions=0, high_severity_count=0
        ),
    )

    res = hscore_calculator.calculate_hscore(p1_prob, p2_feats, cnt_res)
    assert 0.0 <= res.hallucisense_score < 25.0
    assert res.risk_category in [RiskCategory.VERY_LOW, RiskCategory.LOW]
    assert res.pillar1_probability == 0.10


def test_critical_risk_hscore(hscore_calculator):
    p1_prob = 0.92
    p2_feats = PillarTwoFeatures(
        support_ratio=0.0,
        contradiction_ratio=1.0,
        authority_score=0.4,
        source_diversity=0.1,
        evidence_coverage=0.2,
        evidence_density=0.5,
        citation_quality=0.0,
        consensus_confidence=0.9,
        recency_score=0.2,
        verification_completeness=0.5,
    )
    cnt_res = ContradictionAnalysisResult(
        contradictions=[],
        contradiction_count=3,
        fabrication_index=1.0,
        max_severity=0.95,
        graph_visualization=ContradictionGraphVisualization(
            nodes=[], edges=[], total_contradictions=3, high_severity_count=3
        ),
    )

    res = hscore_calculator.calculate_hscore(p1_prob, p2_feats, cnt_res)
    assert res.hallucisense_score >= 75.0
    assert res.risk_category in [RiskCategory.HIGH, RiskCategory.CRITICAL]
