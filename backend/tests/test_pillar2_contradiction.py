"""
Unit tests for HalluciSense Pillar 2 — Module 10.6: Contradiction Analysis.
"""

import pytest
from app.pillar2.claim_extraction.schemas import CharacterOffsets, ClaimType, ExtractedClaim
from app.pillar2.consensus_engine.schemas import ConsensusResult
from app.pillar2.contradiction_analysis.analyzer import ContradictionAnalyzer
from app.pillar2.contradiction_analysis.schemas import ContradictionType
from app.pillar2.evidence_retrieval.schemas import CitationMetadata, EvidenceItem
from app.pillar2.multi_llm_verifier.schemas import VerificationLabel


@pytest.fixture
def contradiction_analyzer():
    return ContradictionAnalyzer()


def test_empty_contradictions(contradiction_analyzer):
    res = contradiction_analyzer.analyze_contradictions([], {}, [])
    assert res.contradiction_count == 0
    assert res.fabrication_index == 0.0
    assert res.max_severity == 0.0


def test_direct_contradiction(contradiction_analyzer):
    claim = ExtractedClaim(
        claim_id="c1",
        claim_text="The Earth is flat.",
        claim_type=ClaimType.DECLARATIVE,
        character_offsets=CharacterOffsets(start=0, end=17),
    )
    consensus = ConsensusResult(
        claim_id="c1",
        majority_label=VerificationLabel.CONTRADICTED,
        weighted_label=VerificationLabel.CONTRADICTED,
        consensus_confidence=0.95,
        label_distribution={"CONTRADICTED": 3},
        label_weights={"CONTRADICTED": 2.85},
        pairwise_agreement_score=1.0,
        shannon_entropy=0.0,
        confidence_variance=0.0,
        agreement_matrix={},
        disagreeing_verifiers=[],
        verdict_summary="Direct contradiction.",
    )
    ev = EvidenceItem(
        evidence_id="ev1",
        title="Earth Shape",
        source="Wikipedia",
        url="http://example.com",
        snippet="Earth is spherical.",
        authority_score=0.9,
    )

    res = contradiction_analyzer.analyze_contradictions([claim], {"c1": consensus}, [ev])
    assert res.contradiction_count == 1
    assert res.fabrication_index == 1.0
    assert res.max_severity >= 0.85
    cnt = res.contradictions[0]
    assert cnt.type == ContradictionType.DIRECT_CONTRADICTION
    assert len(res.graph_visualization.nodes) == 2
    assert len(res.graph_visualization.edges) >= 1


def test_speculation_detection(contradiction_analyzer):
    claim = ExtractedClaim(
        claim_id="c2",
        claim_text="Aliens allegedly built the pyramids.",
        claim_type=ClaimType.DECLARATIVE,
        character_offsets=CharacterOffsets(start=0, end=35),
    )
    consensus = ConsensusResult(
        claim_id="c2",
        majority_label=VerificationLabel.UNKNOWN,
        weighted_label=VerificationLabel.UNKNOWN,
        consensus_confidence=0.50,
        label_distribution={"UNKNOWN": 3},
        label_weights={"UNKNOWN": 1.50},
        pairwise_agreement_score=1.0,
        shannon_entropy=0.0,
        confidence_variance=0.0,
        agreement_matrix={},
        disagreeing_verifiers=[],
        verdict_summary="Unknown",
    )
    res = contradiction_analyzer.analyze_contradictions([claim], {"c2": consensus}, [])
    assert res.contradiction_count == 1
    assert res.contradictions[0].type == ContradictionType.SPECULATION
