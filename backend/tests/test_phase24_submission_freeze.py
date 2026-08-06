"""Phase 24 — Master Unit Test Suite for Research Verification & Submission Freeze."""

from __future__ import annotations

import pytest
from pathlib import Path
from verification.experiment_verifier import ExperimentVerifier
from verification.metric_traceability import MetricTraceabilityEngine
from theory.theorem_verifier import TheoremVerifier
from verification.manuscript_code_sync import ManuscriptCodeSyncAuditor
from reproducibility.reproduction_audit import ReproductionEnvironmentAuditor

BASE_DIR = Path(__file__).resolve().parent.parent


def test_experiment_verifier():
    verifier = ExperimentVerifier()
    summary = verifier.verify_all_experiments()
    assert summary["discrepancy_count"] == 0
    assert "100%" in summary["status"]


def test_metric_traceability_engine():
    engine = MetricTraceabilityEngine()
    tr = engine.generate_traceability_matrix()
    assert tr["verified_count"] == 3
    assert tr["pending_count"] == 0


def test_theorem_verifier():
    verifier = TheoremVerifier()
    audit = verifier.verify_theorems()
    assert audit["unverified_conjectures"] == 0
    assert len(audit["theorems"]) == 3


def test_manuscript_code_sync_auditor():
    auditor = ManuscriptCodeSyncAuditor()
    res = auditor.run_full_sync_audit()
    assert res["status"] == "SUCCESS"
    assert res["reports_generated"] == 6


def test_reproduction_environment_auditor():
    auditor = ReproductionEnvironmentAuditor()
    info = auditor.audit_environment()
    assert info["status"] == "ENVIRONMENT_VERIFIED_SUITABLE"


def test_manifests_and_release_package_exist():
    fig_man = BASE_DIR.parent / "figure_manifest.json"
    tbl_man = BASE_DIR.parent / "table_manifest.json"
    art_man = BASE_DIR.parent / "artifact_manifest.json"
    dash = BASE_DIR.parent / "verification_dashboard.html"
    rel_notes = BASE_DIR.parent / "release" / "v1.0.0" / "RELEASE_NOTES.md"

    assert fig_man.exists()
    assert tbl_man.exists()
    assert art_man.exists()
    assert dash.exists()
    assert rel_notes.exists()
