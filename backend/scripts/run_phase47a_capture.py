"""Phase 47A — Live Response Capture & Diagnostic Script."""

import json
import os
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.main import create_application

def capture_responses():
    print("=" * 70)
    print("PHASE 47A PRODUCTION RUNTIME RECOVERY & RESPONSE CAPTURE")
    print("=" * 70)

    app = create_application()
    client = TestClient(app)

    test_cases = [
        ("CASE_A", "The capital of France is Paris."),
        ("CASE_B", "The capital of France is Berlin."),
        ("CASE_C", "What is the capital of France?"),
        ("CASE_D", "Paris is the capital of France. Berlin is the capital of France."),
        ("CASE_E", "Paris is the capital of France. Berlin is the capital of Germany."),
        ("CASE_MATH_TRUE", "12 multiplied by 8 equals 96."),
        ("CASE_MATH_FALSE", "12 multiplied by 8 equals 95."),
        ("CASE_MULTI_FACT", "The Moon orbits Earth every 27.3 days. Jupiter is the largest planet in our solar system."),
    ]

    captured_results = {}

    for case_id, text in test_cases:
        print(f"\n--- Running {case_id}: '{text}' ---")
        payload = {
            "response": text,
            "model_name": "gpt-4o",
        }
        res = client.post("/api/v1/analyze", json=payload)
        assert res.status_code == 200, f"Failed with {res.status_code}: {res.text}"
        data = res.json()
        captured_results[case_id] = data

        p_scores = data.get("pillar_scores", {})
        p_status = data.get("pillar_status", {})
        f_decomp = data.get("fusion_decomposition", {})
        print(f"  P(H) = {data.get('overall_h_score')} | Risk: {data.get('overall_risk_level')}")
        print(f"  Root Cause: {data.get('root_cause_classification')}")
        print(f"  Pillar Scores: P1={p_scores.get('retrieval')}, P2={p_scores.get('confidence')}, P3={p_scores.get('consistency')}")
        print(f"  Pillar Status: P1={p_status.get('p1_status')}, P2={p_status.get('p2_status')}, P3={p_status.get('p3_status')}")
        print(f"  Fusion Mode: {f_decomp.get('fusion_mode')}")
        print(f"  Available Pillars: {f_decomp.get('available_pillars')}")
        print(f"  Missing Pillars: {f_decomp.get('missing_pillars')}")

    out_file = backend_dir / "reports" / "phase47a" / "live_responses_before_fix.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(captured_results, f, indent=2)
    print(f"\nSaved raw responses to: {out_file}")

if __name__ == "__main__":
    capture_responses()
