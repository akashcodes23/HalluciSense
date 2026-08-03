"""
Unit tests for HalluciSense Pillar 2 — Module 10.5: Consensus Engine.
"""

import pytest
from app.pillar2.consensus_engine.engine import ConsensusEngine
from app.pillar2.multi_llm_verifier.schemas import SingleClaimVerification, VerificationLabel


@pytest.fixture
def consensus_engine():
    return ConsensusEngine()


def test_empty_verifications_consensus(consensus_engine):
    res = consensus_engine.compute_consensus("claim_000", [])
    assert res.majority_label == VerificationLabel.UNKNOWN
    assert res.consensus_confidence == 0.50
    assert res.shannon_entropy == 0.0


def test_unanimous_consensus(consensus_engine):
    verifications = [
        SingleClaimVerification(
            claim_id="c1", provider_name="Gemini", label=VerificationLabel.SUPPORTED, confidence=0.90, reasoning="OK"
        ),
        SingleClaimVerification(
            claim_id="c1", provider_name="GPT-4", label=VerificationLabel.SUPPORTED, confidence=0.95, reasoning="OK"
        ),
        SingleClaimVerification(
            claim_id="c1", provider_name="Claude", label=VerificationLabel.SUPPORTED, confidence=0.88, reasoning="OK"
        ),
    ]
    res = consensus_engine.compute_consensus("c1", verifications)
    assert res.majority_label == VerificationLabel.SUPPORTED
    assert res.weighted_label == VerificationLabel.SUPPORTED
    assert res.pairwise_agreement_score == 1.0
    assert res.shannon_entropy == 0.0
    assert len(res.disagreeing_verifiers) == 0


def test_split_vote_consensus(consensus_engine):
    verifications = [
        SingleClaimVerification(
            claim_id="c2", provider_name="Gemini", label=VerificationLabel.CONTRADICTED, confidence=0.90, reasoning="Fails"
        ),
        SingleClaimVerification(
            claim_id="c2", provider_name="GPT-4", label=VerificationLabel.CONTRADICTED, confidence=0.92, reasoning="Fails"
        ),
        SingleClaimVerification(
            claim_id="c2", provider_name="Claude", label=VerificationLabel.SUPPORTED, confidence=0.60, reasoning="Weak"
        ),
    ]
    res = consensus_engine.compute_consensus("c2", verifications)
    assert res.majority_label == VerificationLabel.CONTRADICTED
    assert res.weighted_label == VerificationLabel.CONTRADICTED
    assert res.shannon_entropy > 0.0
    assert len(res.disagreeing_verifiers) == 1
    assert res.disagreeing_verifiers[0].verifier_name == "Claude"
    assert "Gemini" in res.agreement_matrix
    assert res.agreement_matrix["Gemini"]["GPT-4"] == 1.0
    assert res.agreement_matrix["Gemini"]["Claude"] == 0.0
