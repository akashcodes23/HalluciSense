"""Phase 46 — Verification State Semantics & Integrity Tests."""

import pytest
from app.core.engine.root_cause_classifier import RootCauseClassifier, RootCauseCategory
from app.core.verification.verification_state import VerificationStatus, EvidenceSufficiency

def test_root_cause_verified():
    res = RootCauseClassifier.classify(
        h_score=0.10,
        p1_res=None,
        p2_res=None,
        p3_res=None,
        evidence_items=[],
    )
    assert res == RootCauseCategory.VERIFIED

def test_root_cause_factual_contradiction():
    class MockP1:
        factual_error_score = 0.95
        claims = ["The capital of France is Berlin."]

    class MockEvidence:
        similarity_score = 0.85

    res = RootCauseClassifier.classify(
        h_score=0.85,
        p1_res=MockP1(),
        p2_res=None,
        p3_res=None,
        evidence_items=[MockEvidence()],
    )
    assert res == RootCauseCategory.FACTUAL_CONTRADICTION

def test_root_cause_evidence_missing():
    class MockP1:
        factual_error_score = 0.60
        claims = ["Unknown alien species live on Titan."]

    res = RootCauseClassifier.classify(
        h_score=0.60,
        p1_res=MockP1(),
        p2_res=None,
        p3_res=None,
        evidence_items=[],
    )
    assert res == RootCauseCategory.EVIDENCE_MISSING
