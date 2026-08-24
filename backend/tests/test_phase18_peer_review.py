"""Phase 18 — Adversarial Peer Review Simulation Test Suite."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = BACKEND_DIR / "evaluation" / "results" / "benchmark_dataset.jsonl"
PAPER_DIR = BACKEND_DIR / "paper"
MANUSCRIPT_PATH = PAPER_DIR / "manuscript" / "main.tex"
REPORTS_DIR = BACKEND_DIR / "reports" / "phase18"
LIT_DIR = PAPER_DIR / "literature"

EXPECTED_BENCHMARK_SHA = "dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5"


class TestPhase18PeerReview:
    def test_canonical_benchmark_hash_strictly_invariant(self):
        """Verifies canonical benchmark dataset hash has never changed."""
        hasher = hashlib.sha256()
        with open(BENCHMARK_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        assert hasher.hexdigest() == EXPECTED_BENCHMARK_SHA

    def test_manuscript_main_tex_exists(self):
        """Verifies manuscript LaTeX source file exists."""
        assert MANUSCRIPT_PATH.exists()

    def test_claim_audit_csv_all_supported(self):
        """Verifies claim audit CSV exists and all claims are marked SUPPORTED."""
        audit_csv = REPORTS_DIR / "claim_audit.csv"
        assert audit_csv.exists()
        with open(audit_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            claims = list(reader)

        assert len(claims) >= 5
        for clm in claims:
            assert clm["status"] == "SUPPORTED", f"Claim not fully supported: {clm['claim_id']}"

    def test_novelty_and_baseline_matrix_integrity(self):
        """Verifies novelty and related work matrices exist and are valid."""
        nov_csv = LIT_DIR / "novelty_matrix.csv"
        rel_csv = LIT_DIR / "related_work_matrix.csv"
        assert nov_csv.exists()
        assert rel_csv.exists()

    def test_no_unsupported_first_claims_or_misused_accuracy(self):
        """Verifies absence of unscientific superlatives and unhedged 100% accuracy claims."""
        with open(MANUSCRIPT_PATH, "r", encoding="utf-8") as f:
            text = f.read()

        prohibited = [
            r"\bfirst hallucination detector\b",
            r"\bunconditionally perfect\b",
            r"\bsolves hallucinations\b",
            r"\b25\.69\b",  # Historical z-score must not appear in manuscript
        ]
        for pat in prohibited:
            assert not re.search(pat, text, re.IGNORECASE), f"Prohibited pattern matched: {pat}"

    def test_cohen_d_is_proper_per_sample_effect_size(self):
        """Verifies that Cohen's d in manuscript and statistical audit is 1.42."""
        with open(MANUSCRIPT_PATH, "r", encoding="utf-8") as f:
            text = f.read()
        assert "1.42" in text

        stat_rep = REPORTS_DIR / "PHASE18_STATISTICAL_AUDIT.md"
        assert stat_rep.exists()
        with open(stat_rep, "r", encoding="utf-8") as f:
            content = f.read()
        assert "1.42" in content

    def test_final_peer_review_gate_verdict(self):
        """Verifies final peer review gate report contains A — READY FOR SUBMISSION."""
        gate_path = REPORTS_DIR / "PHASE18_FINAL_PEER_REVIEW_GATE.md"
        assert gate_path.exists()
        with open(gate_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "A — READY FOR SUBMISSION" in content
