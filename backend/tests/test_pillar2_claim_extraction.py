"""
Unit tests for HalluciSense Pillar 2 — Module 10.1: Claim Extraction Engine.
"""

import pytest
from app.pillar2.claim_extraction.extractor import ClaimExtractionEngine
from app.pillar2.claim_extraction.schemas import ClaimExtractionRequest, ClaimType


@pytest.fixture
def claim_engine():
    return ClaimExtractionEngine()


def test_empty_text_extraction(claim_engine):
    req = ClaimExtractionRequest(text="")
    res = claim_engine.extract_claims(req)
    assert res.total_claims == 0
    assert res.num_sentences == 0
    assert len(res.extracted_claims) == 0


def test_single_declarative_claim(claim_engine):
    text = "Albert Einstein was born in Ulm, Germany."
    req = ClaimExtractionRequest(text=text)
    res = claim_engine.extract_claims(req)
    assert res.total_claims == 1
    assert res.num_sentences == 1
    claim = res.extracted_claims[0]
    assert claim.claim_text == text
    assert claim.sentence_index == 0
    assert claim.character_offsets.start == 0
    assert claim.character_offsets.end == len(text)
    assert "Albert" in claim.entities or "Einstein" in claim.entities or "Ulm" in claim.entities


def test_numerical_and_temporal_claim(claim_engine):
    text = "In 1905, Einstein published 4 groundbreaking physics papers."
    req = ClaimExtractionRequest(text=text)
    res = claim_engine.extract_claims(req)
    assert res.total_claims == 1
    claim = res.extracted_claims[0]
    assert claim.claim_type in [ClaimType.TEMPORAL, ClaimType.NUMERICAL]
    assert "1905" in claim.dates
    assert "4" in claim.numbers


def test_scientific_claim(claim_engine):
    text = "CRISPR technology enables precise editing of DNA sequences."
    req = ClaimExtractionRequest(text=text)
    res = claim_engine.extract_claims(req)
    assert res.total_claims == 1
    claim = res.extracted_claims[0]
    assert claim.claim_type == ClaimType.SCIENTIFIC


def test_compound_sentence_decomposition(claim_engine):
    text = "Quantum computing uses qubits; additionally, classical computers use binary bits."
    req = ClaimExtractionRequest(text=text)
    res = claim_engine.extract_claims(req)
    assert res.total_claims == 2
    assert res.num_sentences == 1
    assert "Quantum computing" in res.extracted_claims[0].claim_text
    assert "classical computers" in res.extracted_claims[1].claim_text


def test_deterministic_claim_id(claim_engine):
    text = "Water boils at 100 degrees Celsius."
    req = ClaimExtractionRequest(text=text)
    res1 = claim_engine.extract_claims(req)
    res2 = claim_engine.extract_claims(req)
    assert res1.extracted_claims[0].claim_id == res2.extracted_claims[0].claim_id
