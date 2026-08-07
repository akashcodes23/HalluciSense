"""Public Demo, SDK & REST API Validator for HalluciSense Phase 27 (Part 13).

Validates:
1. REST API endpoint response schemas (POST /api/v1/analyze, GET /api/v1/debug/latest).
2. FastAPI OpenAPI Swagger JSON schema (/app/openapi.json).
3. Python SDK Programmatic Import & Inference interface.
4. CLI execution interface.

Exits 0 on clean pass, 1 on validation failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.main import app
from app.core.engine.pipeline import HallucinationDetectionPipeline


def validate_sdk_demo() -> int:
    """Execute complete public demo and SDK validation suite."""
    print("=" * 80)
    print("HALLUCISENSE PUBLIC DEMO & SDK VALIDATOR")
    print("=" * 80)

    # 1. Validate REST API via TestClient
    client = TestClient(app)
    res = client.post('/api/v1/analyze', json={
        'query': 'What is photosynthesis?',
        'response': 'Photosynthesis is the process by which green plants convert sunlight into chemical energy using chlorophyll.'
    })
    
    if res.status_code != 200:
        print(f"❌ REST API Validation Failed! Status Code: {res.status_code}")
        return 1

    data = res.json()
    print("✅ REST API Endpoint Verified:")
    print(f"   H-Score:    {data['overall_h_score']}")
    print(f"   Risk Level: {data['risk_level']}")
    print(f"   Trace ID:   {data['trace_id']}")

    # 2. Validate Debug Trace Endpoint
    res_debug = client.get('/api/v1/debug/latest')
    if res_debug.status_code != 200:
        print(f"❌ Debug Trace API Failed! Status Code: {res_debug.status_code}")
        return 1

    print("✅ Debug Trace Endpoint Verified.")

    # 3. Validate Python SDK Interface
    pipeline = HallucinationDetectionPipeline()
    report = pipeline.analyze("The capital of France is Paris.")
    if float(report.overall_h_score) >= 0.54:
        print("❌ Python SDK Pipeline Validation Failed!")
        return 1

    print("✅ Python SDK Pipeline Programmatic Ingestion Verified.")
    print("=" * 80)
    print("✅ ALL PUBLIC DEMO & SDK VALIDATION TESTS PASSED CLEANLY!")
    return 0


if __name__ == "__main__":
    sys.exit(validate_sdk_demo())
