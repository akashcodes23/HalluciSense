"""Phase 19 — Final Elsevier Submission Hardening & Integrity Test Suite."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = BACKEND_DIR / "evaluation" / "results" / "benchmark_dataset.jsonl"
PAPER_DIR = BACKEND_DIR / "paper"
SUBMISSION_DIR = PAPER_DIR / "submission"
REPORTS_DIR = BACKEND_DIR / "reports" / "phase19"

EXPECTED_BENCHMARK_SHA = "dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5"


class TestPhase19SubmissionIntegrity:
    def test_canonical_benchmark_hash_strictly_invariant(self):
        """Verifies canonical benchmark dataset hash has never changed."""
        hasher = hashlib.sha256()
        with open(BENCHMARK_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        assert hasher.hexdigest() == EXPECTED_BENCHMARK_SHA

    def test_submission_manifest_integrity(self):
        """Verifies master SUBMISSION_MANIFEST.json exists and is valid."""
        manifest_path = SUBMISSION_DIR / "SUBMISSION_MANIFEST.json"
        assert manifest_path.exists()
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["canonical_benchmark_sha256"] == EXPECTED_BENCHMARK_SHA
        assert len(manifest["tables"]) == 10
        assert len(manifest["figures"]) == 10
        assert len(manifest["target_journals"]) >= 3

    def test_graphical_abstract_and_highlights_exist(self):
        """Verifies graphical abstract and highlights documents exist."""
        ga_png = SUBMISSION_DIR / "graphical_abstract" / "graphical_abstract.png"
        ga_svg = SUBMISSION_DIR / "graphical_abstract" / "graphical_abstract.svg"
        highlights = SUBMISSION_DIR / "highlights.md"
        cover_letter = SUBMISSION_DIR / "cover_letter.md"
        journal_comp = SUBMISSION_DIR / "JOURNAL_TARGET_COMPARISON.md"

        assert ga_png.exists() and ga_png.stat().st_size > 1000
        assert ga_svg.exists() and ga_svg.stat().st_size > 1000
        assert highlights.exists() and highlights.stat().st_size > 100
        assert cover_letter.exists() and cover_letter.stat().st_size > 100
        assert journal_comp.exists() and journal_comp.stat().st_size > 100

    def test_all_six_mandatory_statements_exist(self):
        """Verifies all 6 mandatory submission statements exist."""
        statements = [
            "author_contributions.md",
            "data_availability.md",
            "code_availability.md",
            "conflict_of_interest.md",
            "funding_statement.md",
            "ethics_statement.md",
        ]
        for stmt in statements:
            stmt_path = SUBMISSION_DIR / "statements" / stmt
            assert stmt_path.exists(), f"Missing statement: {stmt}"
            assert stmt_path.stat().st_size > 30, f"Empty statement: {stmt}"

    def test_reproducibility_package_artifacts_exist(self):
        """Verifies reproducibility package contains manifests and executable reproduction script."""
        repro_dir = SUBMISSION_DIR / "reproducibility"
        repro_artifacts = [
            "README.md",
            "requirements.txt",
            "environment.yml",
            "RUN_REPRODUCTION.sh",
            "REPRODUCIBILITY_MANIFEST.json",
            "MODEL_MANIFEST.json",
            "DATASET_MANIFEST.json",
        ]
        for art in repro_artifacts:
            art_path = repro_dir / art
            assert art_path.exists(), f"Missing reproducibility artifact: {art}"

    def test_phase19_final_submission_verdict(self):
        """Verifies final submission audit report contains A — SUBMISSION READY."""
        rep_path = REPORTS_DIR / "PHASE19_FINAL_SUBMISSION_AUDIT.md"
        assert rep_path.exists()
        with open(rep_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "A — SUBMISSION READY" in content
