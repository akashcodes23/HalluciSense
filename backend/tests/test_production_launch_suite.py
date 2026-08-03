"""
Master Test Suite for HalluciSense v1.0 Production SaaS Launch (Sprints 1 - 14).
"""

from pathlib import Path
import json
import pytest
from app.saas.auth import AuthenticationService
from app.saas.stripe_payments import StripePaymentService
from app.core.cache_upstash import UpstashRedisManager
from app.core.performance_tuner import PerformanceTuner
from website.playground_engine import PlaygroundEngine, VerificationInputPayload
from scripts.security_audit import run_security_audit


def test_sprint_1_backend_deploy_config():
    root = Path(__file__).resolve().parents[1]
    assert (root / "deployment" / "railway.toml").exists()
    assert (root / "deployment" / "render.yaml").exists()


def test_sprint_2_frontend_seo():
    root = Path(__file__).resolve().parents[1]
    assert (root / "public" / "robots.txt").exists()
    assert (root / "public" / "sitemap.xml").exists()
    assert (root / "vercel.json").exists()


def test_sprint_3_database_alembic():
    root = Path(__file__).resolve().parents[1]
    assert (root / "alembic.ini").exists()
    assert (root / "alembic" / "versions" / "001_initial_schema.py").exists()


def test_sprint_4_upstash_redis():
    redis_mgr = UpstashRedisManager()
    redis_mgr.set("key_1", "val_1", ttl_seconds=10)
    assert redis_mgr.get("key_1") == "val_1"


def test_sprint_5_auth():
    auth = AuthenticationService()
    h_pwd = auth.hash_password("Pass123!")
    assert auth.verify_password("Pass123!", h_pwd) is True


def test_sprint_6_stripe_payments():
    stripe_svc = StripePaymentService()
    session = stripe_svc.create_checkout_session("user@test.com", plan_id="pro")
    assert "checkout_url" in session
    assert session["quota"] == 100000

    webhook_res = stripe_svc.process_webhook_event("mock_payload", "sig_header")
    assert webhook_res["status"] == "success"

    assert stripe_svc.check_quota_available(current_usage=500, plan_id="pro") is True
    assert stripe_svc.check_quota_available(current_usage=200000, plan_id="pro") is False


def test_sprint_8_security_audit():
    audit_res = run_security_audit()
    assert audit_res is True


def test_sprint_9_cicd():
    root = Path(__file__).resolve().parents[1]
    assert (root / ".github" / "workflows" / "production_deploy.yml").exists()


def test_sprint_11_playground_engine():
    engine = PlaygroundEngine()
    res = engine.process_input(VerificationInputPayload(input_type="text", content="Einstein born in 1879."))
    assert res["hallucisense_score"] == 6.41
    assert "report_urls" in res


def test_sprint_12_postman_collection():
    root = Path(__file__).resolve().parents[1]
    p_file = root / "hallucisense_postman_collection.json"
    assert p_file.exists()

    with open(p_file, "r") as f:
        data = json.load(f)
    assert "info" in data
    assert "item" in data


def test_sprint_13_performance_tuner():
    tuner = PerformanceTuner()
    res = tuner.optimize_pipeline_execution()
    assert res["target_status"] == "OPTIMIZED"


def test_sprint_14_deployment_scripts():
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "deploy.sh").exists()
    assert (root / "scripts" / "rollback.sh").exists()
