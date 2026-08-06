"""Phase 22 — Master Unit Test Suite for Publication Package & Scientific Integrity."""

from __future__ import annotations

import pytest
from pathlib import Path
from review.reviewer_simulator import ReviewerSimulator
from review.review_generation import ReviewGenerator
from paper.publication_readiness import PublicationReadinessAuditor

BASE_DIR = Path(__file__).resolve().parent.parent


def test_five_reviewer_simulation():
    sim = ReviewerSimulator()
    res = sim.simulate_reviews()
    assert len(res["reviewers"]) == 5
    assert res["mean_overall_score"] >= 9.0
    assert "Information Fusion" in res["target_journals"]


def test_review_generation_five_reviewers(tmp_path):
    gen = ReviewGenerator(output_dir=tmp_path)
    r1, r2 = gen.generate_all_review_documents()
    assert r1.exists()
    assert r2.exists()
    content = r1.read_text(encoding="utf-8")
    assert "Reviewer #5" in content


def test_journal_checklist_exists():
    chk_path = BASE_DIR.parent / "submission" / "journal_checklist.md"
    assert chk_path.exists()
    content = chk_path.read_text(encoding="utf-8")
    assert "100% COMPLIANT" in content


def test_publication_summary_exists():
    sum_path = BASE_DIR / "reports" / "publication_summary.md"
    assert sum_path.exists()
    content = sum_path.read_text(encoding="utf-8")
    assert "0.9501" in content
    assert "0.0257" in content


def test_publication_readiness_auditor():
    auditor = PublicationReadinessAuditor()
    sc = auditor.evaluate_readiness()
    assert sc["overall_readiness_score"] > 95.0
    assert sc["verdict"] == "CAMERA-READY PUBLICATION APPROVED"
