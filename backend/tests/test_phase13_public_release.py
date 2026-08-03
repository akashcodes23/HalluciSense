"""
Unit tests for HalluciSense Phase 13 — Public Release, Production Deployment & Open Source Package (v1.0).
"""

from pathlib import Path
import pytest
import json
from app.saas.feedback_telemetry import FeedbackTelemetryService
from app.saas.public_analytics import PublicAnalyticsService
from documentation.doc_portal import DocumentationPortalGenerator
from website.playground import LivePlaygroundManager


def test_playground_file_parser():
    mgr = LivePlaygroundManager()
    res_txt = mgr.parse_uploaded_file("doc.txt", b"Albert Einstein was born in Ulm.")
    assert res_txt.character_count > 0
    assert "Einstein" in res_txt.extracted_text

    res_pdf = mgr.parse_uploaded_file("document.pdf", b"%PDF-1.4 sample")
    assert "PDF" in res_pdf.extracted_text

    viz = mgr.run_playground_verification("Quantum physics text")
    assert viz["hallucisense_score"] == 6.41
    assert "report_download_urls" in viz


def test_feedback_and_telemetry():
    svc = FeedbackTelemetryService()
    fb = svc.submit_feedback("BUG_REPORT", "Found typo in doc", rating=4, email="user@test.com")
    assert fb.feedback_id.startswith("fb_")
    assert fb.feedback_type == "BUG_REPORT"

    tel = svc.record_telemetry("VERIFICATION_EXECUTED", latency_ms=3.85, opt_in=True)
    assert tel is not None
    assert tel.latency_ms == 3.85

    assert svc.record_telemetry("VERIFICATION_EXECUTED", latency_ms=3.85, opt_in=False) is None


def test_doc_portal_generator(tmp_path):
    gen = DocumentationPortalGenerator()
    exported = gen.generate_portal(tmp_path / "docs")
    assert len(exported) == 7
    assert (tmp_path / "docs" / "GETTING_STARTED.md").exists()
    assert (tmp_path / "docs" / "API_REFERENCE.md").exists()


def test_public_analytics():
    analytics = PublicAnalyticsService()
    ov = analytics.get_public_analytics()
    assert ov.total_verifications_all_time > 1000
    assert ov.p95_latency_ms == 4.28
    assert ov.active_providers_count == 7


def test_open_source_files():
    root = Path(__file__).resolve().parents[1]
    assert (root / "README.md").exists()
    assert (root / "LICENSE").exists()
    assert (root / "CITATION.cff").exists()

    readme_txt = (root / "README.md").read_text()
    assert "HalluciSense" in readme_txt
    assert "0.8920 ROC-AUC" in readme_txt
