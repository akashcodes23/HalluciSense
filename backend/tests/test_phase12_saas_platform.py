"""
Unit tests for HalluciSense Phase 12 — Enterprise SaaS Platform & Deployment Infrastructure.
"""

from pathlib import Path
import pytest
from app.saas.admin_portal import AdminPortalService
from app.saas.api_platform import APIPlatformManager
from app.saas.auth import AuthenticationService, UserRole
from app.saas.claim_explorer import ClaimExplorerService
from app.saas.dashboard import DashboardService
from app.saas.report_generator import MultiFormatReportGenerator
from app.core.security_hardened import InputSanitizer
from app.core.observability import MetricsExporter
from app.core.cache_redis import RedisCacheManager
from sdk.python.hallucisense_sdk import HalluciSenseClient


def test_auth_service():
    auth = AuthenticationService()
    h_pwd = auth.hash_password("SuperSecretPass123!")
    assert auth.verify_password("SuperSecretPass123!", h_pwd) is True
    assert auth.verify_password("WrongPass", h_pwd) is False

    oauth_res = auth.authenticate_oauth("google", "code123", "user@test.com", "Test User")
    assert oauth_res.access_token != ""
    assert oauth_res.user.email == "user@test.com"

    decoded = auth.decode_token(oauth_res.access_token)
    assert decoded["email"] == "user@test.com"

    assert auth.verify_rbac(UserRole.USER, UserRole.ADMIN) is True
    assert auth.verify_rbac(UserRole.ADMIN, UserRole.USER) is False


def test_dashboard_service():
    dash_service = DashboardService()
    ov = dash_service.get_user_dashboard("usr_001")
    assert ov.user_id == "usr_001"
    assert ov.usage_stats.total_verifications_count > 0
    assert len(ov.recent_verifications) >= 1
    assert "VERY_LOW" in ov.risk_distribution_pct


def test_claim_explorer_service():
    explorer = ClaimExplorerService()
    detail = explorer.get_claim_details("claim_101", "Einstein was born in Ulm.")
    assert detail.claim_id == "claim_101"
    assert detail.consensus_label == "SUPPORTED"
    assert len(detail.reasoning_chain) >= 3


def test_report_generator():
    gen = MultiFormatReportGenerator()
    payload = {"verification_id": "v123", "text": "Sample text", "hallucisense_score": {"hallucisense_score": 10.0, "risk_category": "VERY_LOW"}}

    for fmt in ["html", "markdown", "json", "csv", "pdf"]:
        res = gen.generate_report(payload, output_format=fmt)
        assert res["format"] == fmt
        assert res["content"] != ""
        assert res["filename"].endswith(fmt if fmt != "markdown" else "md")


def test_api_platform():
    mgr = APIPlatformManager()
    raw_key, meta = mgr.generate_api_key("usr_001", "My App")
    assert raw_key.startswith("hs_live_")
    assert meta.user_id == "usr_001"

    val_meta = mgr.validate_api_key(raw_key)
    assert val_meta is not None
    assert val_meta.key_id == meta.key_id

    assert mgr.check_rate_limit(meta.key_id, limit_rpm=10) is True


def test_security_sanitizer():
    raw = "<script>alert('xss');</script><b>Hello</b>"
    clean = InputSanitizer.sanitize_text(raw)
    assert "<script>" not in clean
    assert "alert" in clean or "Hello" in clean


def test_observability_metrics():
    exporter = MetricsExporter()
    metrics_str = exporter.render_prometheus_metrics()
    assert "hallucisense_verifications_total" in metrics_str
    assert "hallucisense_latency_seconds" in metrics_str

    grafana = exporter.generate_grafana_dashboard()
    assert "panels" in grafana


def test_redis_cache():
    cache = RedisCacheManager()
    cache.set("ev_101", {"data": "test"}, ttl_seconds=10)
    val = cache.get("ev_101")
    assert val == {"data": "test"}

    assert cache.get("non_existent") is None
    cache.invalidate("ev_")
    assert cache.get("ev_101") is None


def test_admin_portal():
    admin = AdminPortalService()
    ov = admin.get_admin_overview()
    assert ov.system_health_status == "HEALTHY"
    assert "Wikipedia" in ov.provider_statuses

    updated_flags = admin.update_feature_flags({"ENABLE_ISOTONIC_CALIBRATION": True})
    assert updated_flags.ENABLE_ISOTONIC_CALIBRATION is True
