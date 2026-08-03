"""
Unit tests for HalluciSense Pillar 2 — Module 10.7: Evidence Feature Generation.
"""

import pytest
from app.pillar2.claim_extraction.schemas import CharacterOffsets, ClaimType, ExtractedClaim
from app.pillar2.consensus_engine.schemas import ConsensusResult
from app.pillar2.evidence_retrieval.schemas import CitationMetadata, EvidenceItem
from app.pillar2.feature_generation.generator import EvidenceFeatureGenerator
from app.pillar2.multi_llm_verifier.schemas import VerificationLabel


@pytest.fixture
def feature_generator():
    return EvidenceFeatureGenerator()


def test_empty_features(feature_generator):
    feats = feature_generator.generate_features([], [], {})
    assert feats.support_ratio == 0.0
    assert feats.contradiction_ratio == 0.0
    assert feats.authority_score == 0.0


def test_feature_computation(feature_generator):
    claim1 = ExtractedClaim(
        claim_id="c1",
        claim_text="Claim 1",
        claim_type=ClaimType.DECLARATIVE,
        character_offsets=CharacterOffsets(start=0, end=7),
    )
    claim2 = ExtractedClaim(
        claim_id="c2",
        claim_text="Claim 2",
        claim_type=ClaimType.DECLARATIVE,
        character_offsets=CharacterOffsets(start=8, end=15),
    )

    consensus_map = {
        "c1": ConsensusResult(
            claim_id="c1",
            majority_label=VerificationLabel.SUPPORTED,
            weighted_label=VerificationLabel.SUPPORTED,
            consensus_confidence=0.90,
            label_distribution={"SUPPORTED": 3},
            label_weights={"SUPPORTED": 2.7},
            pairwise_agreement_score=1.0,
            shannon_entropy=0.0,
            confidence_variance=0.0,
            agreement_matrix={},
            disagreeing_verifiers=[],
            verdict_summary="Supported",
        ),
        "c2": ConsensusResult(
            claim_id="c2",
            majority_label=VerificationLabel.CONTRADICTED,
            weighted_label=VerificationLabel.CONTRADICTED,
            consensus_confidence=0.85,
            label_distribution={"CONTRADICTED": 3},
            label_weights={"CONTRADICTED": 2.55},
            pairwise_agreement_score=1.0,
            shannon_entropy=0.0,
            confidence_variance=0.0,
            agreement_matrix={},
            disagreeing_verifiers=[],
            verdict_summary="Contradicted",
        ),
    }

    evidence_items = [
        EvidenceItem(
            evidence_id="e1",
            title="T1",
            source="Wikipedia",
            url="u1",
            snippet="S1",
            publication_date="2025-01-01",
            authority_score=0.85,
            citation_metadata=CitationMetadata(journal="Wiki"),
        ),
        EvidenceItem(
            evidence_id="e2",
            title="T2",
            source="PubMed",
            url="u2",
            snippet="S2",
            publication_date="2026-02-01",
            authority_score=0.95,
            citation_metadata=CitationMetadata(doi="10.1000/1", journal="Nature"),
        ),
    ]

    feats = feature_generator.generate_features([claim1, claim2], evidence_items, consensus_map)
    assert feats.support_ratio == 0.5
    assert feats.contradiction_ratio == 0.5
    assert feats.evidence_coverage == 1.0
    assert feats.authority_score == 0.90
    assert feats.citation_quality == 1.0
    assert feats.source_diversity > 0.0
    assert feats.recency_score == 1.0
    assert feats.verification_completeness > 0.5
