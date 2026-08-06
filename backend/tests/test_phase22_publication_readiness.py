"""Phase 22 — Master Unit Test Suite for Publication Readiness & Peer Review Excellence."""

from __future__ import annotations

import pytest
from pathlib import Path
from evaluation.literature_comparison_engine import LiteratureComparisonEngine
from review.reviewer_simulator import ReviewerSimulator
from review.review_generation import ReviewGenerator
from reproducibility.replication_protocol import ReplicationProtocolVerifier
from paper.paper_consistency_checker import PaperConsistencyChecker
from paper.publication_readiness import PublicationReadinessAuditor

BASE_DIR = Path(__file__).resolve().parent.parent


def test_literature_comparison_engine():
    engine = LiteratureComparisonEngine()
    out = engine.generate_novelty_validation_report()
    assert out.exists()
    assert len(engine.LITERATURE_BASELINES) >= 14


def test_reviewer_simulator():
    sim = ReviewerSimulator()
    res = sim.simulate_reviews()
    assert len(res["reviewers"]) == 3
    assert res["overall_recommendation"] == "ACCEPT (Camera-Ready Approved)"


def test_review_generator(tmp_path):
    gen = ReviewGenerator(output_dir=tmp_path)
    r1, r2 = gen.generate_all_review_documents()
    assert r1.exists()
    assert r2.exists()


def test_replication_protocol_verifier():
    verifier = ReplicationProtocolVerifier()
    res = verifier.verify_replication()
    assert res["fresh_clone_verification"] == "PASSED"
    assert res["observed_auroc"] == 0.9501


def test_paper_consistency_checker():
    checker = PaperConsistencyChecker()
    res = checker.check_manuscript()
    assert "status" in res
    assert res["missing_citations"] == []


def test_publication_readiness_auditor():
    auditor = PublicationReadinessAuditor()
    sc = auditor.evaluate_readiness()
    assert sc["overall_readiness_score"] > 95.0
    assert sc["verdict"] == "CAMERA-READY PUBLICATION APPROVED"
