"""
Unit tests for HalluciSense Pillar 2 — Module 10.9: Explainability Engine.
"""

import pytest
from app.pillar2.claim_extraction.schemas import CharacterOffsets, ClaimType, ExtractedClaim
from app.pillar2.consensus_engine.schemas import ConsensusResult
from app.pillar2.contradiction_analysis.schemas import (
    ContradictionAnalysisResult,
    ContradictionGraphVisualization,
)
from app.pillar2.evidence_retrieval.schemas import CitationMetadata, EvidenceItem
from app.pillar2.explainability.engine import PillarTwoExplainabilityEngine
from app.pillar2.feature_generation.schemas import PillarTwoFeatures
from app.pillar2.multi_llm_verifier.schemas import VerificationLabel
from app.pillar2.unified_hscore.schemas import RiskCategory, UnifiedHScoreResult


@pytest.fixture
def explainability_engine():
    return PillarTwoExplainabilityEngine()


def test_generate_explanation(explainability_engine):
    claim = ExtractedClaim(
        claim_id="c1",
        claim_text="Quantum computing uses qubits.",
        claim_type=ClaimType.SCIENTIFIC,
        character_offsets=CharacterOffsets(start=0, end=29),
    )
    evidence = EvidenceItem(
        evidence_id="e1",
        title="Quantum Bits",
        source="Wikipedia",
        url="http://example.com/q",
        snippet="Quantum computers harness qubits.",
        authority_score=0.9,
    )
    consensus_map = {
        "c1": ConsensusResult(
            claim_id="c1",
            majority_label=VerificationLabel.SUPPORTED,
            weighted_label=VerificationLabel.SUPPORTED,
            consensus_confidence=0.95,
            label_distribution={"SUPPORTED": 3},
            label_weights={"SUPPORTED": 2.85},
            pairwise_agreement_score=1.0,
            shannon_entropy=0.0,
            confidence_variance=0.0,
            agreement_matrix={},
            disagreeing_verifiers=[],
            verdict_summary="Supported",
        )
    }
    cnt_res = ContradictionAnalysisResult(
        contradictions=[],
        contradiction_count=0,
        fabrication_index=0.0,
        max_severity=0.0,
        graph_visualization=ContradictionGraphVisualization(
            nodes=[], edges=[], total_contradictions=0, high_severity_count=0
        ),
    )
    p2_feats = PillarTwoFeatures(
        support_ratio=1.0,
        contradiction_ratio=0.0,
        authority_score=0.9,
        source_diversity=0.5,
        evidence_coverage=1.0,
        evidence_density=1.0,
        citation_quality=0.8,
        consensus_confidence=0.95,
        recency_score=1.0,
        verification_completeness=1.0,
    )
    hscore_res = UnifiedHScoreResult(
        hallucisense_score=12.5,
        risk_category=RiskCategory.VERY_LOW,
        overall_confidence=0.95,
        pillar1_probability=0.10,
        evidence_score=10.0,
        consensus_score=5.0,
        contradiction_score=0.0,
        component_weights={"p1": 0.4, "cnt": 0.3, "ev": 0.15, "cs": 0.15},
        explanation_summary="Low risk",
    )

    expl = explainability_engine.generate_explanation(
        [claim], [evidence], consensus_map, cnt_res, p2_feats, hscore_res
    )

    assert "HalluciSense Verification Report" in expl.executive_summary
    assert len(expl.claim_analysis) == 1
    assert expl.risk_category == RiskCategory.VERY_LOW
    assert len(expl.actionable_recommendations) >= 1
    assert "ACCEPT" in expl.actionable_recommendations[0]
