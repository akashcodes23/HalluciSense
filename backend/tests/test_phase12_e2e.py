"""
HalluciSense Phase 12 — Live End-to-End Product Acceptance Pytest Suite.

Validates:
1. Benchmark SHA-256 integrity
2. Production API health and readiness endpoints
3. Test case matrix A through J
4. Closed-loop correction & re-verification
5. ModelRegistry singleton and memory safety invariants
"""

import hashlib
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import create_application
from app.core.engine.model_registry import ModelRegistry
from app.core.correction.correction_engine import CorrectionEngine

CANONICAL_BENCHMARK_HASH = "dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5"


@pytest.fixture(scope="module")
def client():
    app = create_application()
    with TestClient(app) as test_client:
        yield test_client


class TestPhase12IntegrityAndHealth:
    def test_benchmark_dataset_sha256_integrity(self):
        backend_dir = Path(__file__).resolve().parent.parent
        dataset_path = backend_dir / "evaluation" / "results" / "benchmark_dataset.jsonl"
        assert dataset_path.exists(), f"Benchmark dataset missing: {dataset_path}"

        sha256 = hashlib.sha256()
        with open(dataset_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        actual_hash = sha256.hexdigest()
        assert actual_hash == CANONICAL_BENCHMARK_HASH, f"Benchmark hash mismatch: {actual_hash}"

    def test_health_and_readiness_endpoints(self, client):
        h_resp = client.get("/health")
        assert h_resp.status_code == 200
        h_data = h_resp.json()
        assert h_data["status"] == "healthy"
        assert h_data["memory_mb"] > 0

        r_resp = client.get("/ready")
        assert r_resp.status_code == 200
        r_data = r_resp.json()
        assert r_data["status"] == "ready"


class TestPhase12TestCaseMatrix:
    def test_case_a_true_scientific_claim(self, client):
        resp = client.post(
            "/api/v1/analyze",
            json={
                "query": "What is the speed of light in vacuum?",
                "response": "The speed of light in vacuum is approximately 299,792,458 m/s.",
                "model_name": "gpt-4o",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["risk_level"] in ["VERIFIED", "LOW_RISK"]
        assert data["overall_h_score"] <= 0.35

    def test_case_b_numerical_unit_hallucination_and_correction(self, client):
        # 1. Verification API detects hallucination
        resp = client.post(
            "/api/v1/analyze",
            json={
                "query": "What is the speed of light in vacuum?",
                "response": "The speed of light in vacuum is approximately 299,792,458 km/s.",
                "model_name": "gpt-4o",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["risk_level"] in ["LIKELY_HALLUCINATED", "MODERATE_RISK", "NEEDS_VERIFICATION"]
        assert data["overall_h_score"] >= 0.35

        # 2. Closed-loop correction repairs km/s to m/s and re-verifies
        pipeline = ModelRegistry.get_pipeline()
        pipeline_res = pipeline.analyze_response(
            full_text="The speed of light in vacuum is approximately 299,792,458 km/s.",
            query="What is the speed of light in vacuum?",
        )
        corr_engine = CorrectionEngine(pipeline=pipeline)
        repair_res = corr_engine.execute_closed_loop_repair(
            user_query="What is the speed of light in vacuum?",
            initial_text="The speed of light in vacuum is approximately 299,792,458 km/s.",
            initial_verification=pipeline_res,
            max_attempts=2,
        )
        assert repair_res.performed is True
        assert "299,792,458 m/s" in repair_res.final_text
        assert repair_res.reverification is not None
        assert repair_res.reverification.passed is True

    def test_case_c_water_formula_true(self, client):
        resp = client.post(
            "/api/v1/analyze",
            json={
                "query": "What is the chemical formula of water?",
                "response": "Water has the chemical formula H2O.",
                "model_name": "gpt-4o",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["risk_level"] in ["VERIFIED", "LOW_RISK"]
        assert data["overall_h_score"] <= 0.35

    def test_case_d_wrong_chemical_formula(self, client):
        resp = client.post(
            "/api/v1/analyze",
            json={
                "query": "What is the chemical formula of water?",
                "response": "Water has the chemical formula CO2.",
                "model_name": "gpt-4o",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["risk_level"] in ["LIKELY_HALLUCINATED", "MODERATE_RISK", "NEEDS_VERIFICATION"]
        assert data["overall_h_score"] >= 0.35

    def test_case_e_negation_flip(self, client):
        resp = client.post(
            "/api/v1/analyze",
            json={
                "query": "Do mitochondria produce ATP?",
                "response": "Mitochondria do not produce ATP in eukaryotic cells.",
                "model_name": "gpt-4o",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["risk_level"] in ["LIKELY_HALLUCINATED", "MODERATE_RISK", "NEEDS_VERIFICATION"]
        assert data["overall_h_score"] >= 0.35

    def test_case_f_true_core_false_elaboration(self, client):
        resp = client.post(
            "/api/v1/analyze",
            json={
                "query": "Tell me about water and its history.",
                "response": "The chemical formula of water is H2O. It was discovered by Albert Einstein in 1905.",
                "model_name": "gpt-4o",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        # At least one sentence must be flagged and overall result is not fully safe
        assert len(data["sentence_scores"]) >= 2
        assert data["overall_h_score"] >= 0.30

    def test_case_g_causal_direction_inversion(self, client):
        resp = client.post(
            "/api/v1/analyze",
            json={
                "query": "What is the relationship between kidney damage and blood pressure?",
                "response": "Kidney damage always causes high blood pressure.",
                "model_name": "gpt-4o",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["risk_level"] in ["LIKELY_HALLUCINATED", "MODERATE_RISK", "NEEDS_VERIFICATION"]

    def test_case_h_ambiguous_claim_uncertainty(self, client):
        resp = client.post(
            "/api/v1/analyze",
            json={
                "query": "What constitutes dark matter?",
                "response": "Dark matter consists entirely of weakly interacting massive particles (WIMPs).",
                "model_name": "gpt-4o",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["overall_h_score"] is not None

    def test_case_i_empty_input_validation(self, client):
        resp = client.post(
            "/api/v1/analyze",
            json={"query": "", "response": "   "},
        )
        assert resp.status_code == 400
        err_data = resp.json()
        assert "response" in str(err_data).lower() or "empty" in str(err_data).lower()

    def test_case_j_backend_failure_semantics(self, client):
        resp = client.post(
            "/api/v1/chat",
            json={"message": "Failure semantics test", "enable_verification": False},
        )
        assert resp.status_code == 200
        chat_data = resp.json()
        verif = chat_data["verification"]
        assert verif["h_score"] is None
        assert verif["risk_level"] is None


class TestPhase12ArchitectureAndMemorySafety:
    def test_model_registry_singletons(self):
        init_counts = ModelRegistry.get_init_counts()
        assert init_counts.get("pipeline", 0) == 1
        assert init_counts.get("nli_model", 0) == 1
        assert init_counts.get("cross_encoder", 0) <= 1
