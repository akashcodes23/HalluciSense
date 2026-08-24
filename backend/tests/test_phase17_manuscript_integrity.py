"""Phase 17 — Manuscript Integrity & Submission Lock Test Suite."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = BACKEND_DIR / "evaluation" / "results" / "benchmark_dataset.jsonl"
PAPER_DIR = BACKEND_DIR / "paper"
MANUSCRIPT_PATH = PAPER_DIR / "manuscript" / "main.tex"
BIB_PATH = PAPER_DIR / "manuscript" / "references.bib"
TRACEABILITY_PATH = PAPER_DIR / "manuscript" / "claim_traceability.json"
TABLES_DIR = PAPER_DIR / "tables"
LIT_DIR = PAPER_DIR / "literature"
REPORTS_DIR = BACKEND_DIR / "reports" / "phase17"

EXPECTED_BENCHMARK_SHA = "dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5"


class TestPhase17ManuscriptIntegrity:
    def test_canonical_benchmark_hash_strictly_invariant(self):
        """Verifies canonical benchmark dataset hash has never changed."""
        hasher = hashlib.sha256()
        with open(BENCHMARK_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        assert hasher.hexdigest() == EXPECTED_BENCHMARK_SHA

    def test_manuscript_main_tex_structure_and_sections(self):
        """Verifies main.tex exists and contains core required manuscript sections."""
        assert MANUSCRIPT_PATH.exists()
        with open(MANUSCRIPT_PATH, "r", encoding="utf-8") as f:
            text = f.read()

        required_sections = [
            "\\section{Introduction}",
            "\\section{Related Work}",
            "\\section{Problem Formulation}",
            "\\section{HalluciSense Methodology}",
            "\\section{Experimental Methodology}",
            "\\section{Results}",
            "\\section{Ablation Studies}",
            "\\section{Availability Robustness}",
            "\\section{Calibration and Selective Prediction}",
            "\\section{Closed-Loop Correction}",
            "\\section{Failure Analysis}",
            "\\section{Threats to Validity}",
            "\\section{Reproducibility",
            "\\section{Discussion}",
            "\\section{Conclusion}",
        ]
        for sec in required_sections:
            assert sec in text, f"Missing section: {sec}"

    def test_no_prohibited_overclaims_in_manuscript(self):
        """Verifies absence of unscientific superlatives in manuscript text."""
        with open(MANUSCRIPT_PATH, "r", encoding="utf-8") as f:
            text = f.read()

        prohibited = [
            r"\bfirst hallucination detector\b",
            r"\bunconditionally perfect\b",
            r"\bstate-of-the-art across all\b",
            r"\bsolves hallucinations\b",
        ]
        for pattern in prohibited:
            assert not re.search(pattern, text, re.IGNORECASE), f"Prohibited phrase matched: {pattern}"

    def test_all_ten_latex_tables_exist_and_non_empty(self):
        """Verifies all 10 LaTeX tables exist and have valid content."""
        expected_tables = [
            "table1_system_architecture.tex",
            "table2_main_results.tex",
            "table3_external_generalization.tex",
            "table4_baseline_comparison.tex",
            "table5_ablation.tex",
            "table6_availability_robustness.tex",
            "table7_calibration.tex",
            "table8_selective_abstention.tex",
            "table9_closed_loop_correction.tex",
            "table10_failure_taxonomy.tex",
        ]
        for tbl in expected_tables:
            tbl_path = TABLES_DIR / tbl
            assert tbl_path.exists(), f"Missing table: {tbl}"
            assert tbl_path.stat().st_size > 50, f"Empty table: {tbl}"

    def test_citation_registry_and_bib_integrity(self):
        """Verifies citation registry and BibTeX file have matching entries."""
        reg_path = LIT_DIR / "citation_registry.json"
        assert reg_path.exists()
        assert BIB_PATH.exists()

        with open(reg_path, "r", encoding="utf-8") as f:
            citations = json.load(f)
        assert len(citations) >= 10

        with open(BIB_PATH, "r", encoding="utf-8") as f:
            bib_text = f.read()

        for cit in citations:
            assert cit["key"] in bib_text, f"Missing BibTeX entry for: {cit['key']}"

    def test_claim_traceability_ledger_integrity(self):
        """Verifies claim traceability ledger exists with verified status."""
        assert TRACEABILITY_PATH.exists()
        with open(TRACEABILITY_PATH, "r", encoding="utf-8") as f:
            ledger = json.load(f)
        assert len(ledger) >= 10
        for item in ledger:
            assert item["status"] == "VERIFIED"
            assert "metric" in item
            assert "value" in item

    def test_phase17_manuscript_readiness_verdict(self):
        """Verifies final manuscript readiness report contains A — MANUSCRIPT READY."""
        rep_path = REPORTS_DIR / "PHASE17_MANUSCRIPT_READINESS.md"
        assert rep_path.exists()
        with open(rep_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "A — MANUSCRIPT READY" in content
