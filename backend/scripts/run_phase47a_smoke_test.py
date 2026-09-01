"""Phase 47A — Production Smoke Test Script.

Executes:
- 10 sequential single-claim requests
- 5 sequential 2-claim requests
- 5 sequential 5-claim requests

Tracks RSS, latencies, availability, and root causes.
"""

import os
import sys
import time
import psutil
import json
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.main import create_application

def run_smoke_test():
    print("=" * 70)
    print("PHASE 47A PRODUCTION SMOKE TEST")
    print("=" * 70)

    app = create_application()
    client = TestClient(app)
    proc = psutil.Process(os.getpid())

    # Pre-warm
    init_rss = proc.memory_info().rss / (1024 * 1024)
    print(f"Startup RSS: {init_rss:.2f} MB")

    records = []

    # Batch 1: 10 sequential single-claim requests
    print("\n--- Running 10 Sequential Single-Claim Requests ---")
    single_texts = [
        "The capital of France is Paris.",
        "The capital of France is Berlin.",
        "What is the capital of France?",
        "12 multiplied by 8 equals 96.",
        "12 multiplied by 8 equals 95.",
        "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        "Albert Einstein was awarded the Nobel Prize in Physics in 1921.",
        "The speed of light in vacuum is approximately 299792458 meters per second.",
        "The Great Wall of China is visible from Mars with the naked eye.",
        "Python is an interpreted, high-level programming language."
    ]

    for i, text in enumerate(single_texts):
        t0 = time.perf_counter()
        res = client.post("/api/v1/analyze", json={"response": text, "model_name": "gpt-4o"})
        lat_ms = (time.perf_counter() - t0) * 1000.0
        assert res.status_code == 200, f"Request failed: {res.text}"
        data = res.json()
        rss_mb = proc.memory_info().rss / (1024 * 1024)
        p_scores = data.get("pillar_scores", {})
        p_status = data.get("pillar_status", {})
        rec = {
            "req_index": i + 1,
            "type": "single_claim",
            "text": text,
            "latency_ms": round(lat_ms, 2),
            "rss_mb": round(rss_mb, 2),
            "h_score": data.get("overall_h_score"),
            "root_cause": data.get("root_cause_classification"),
            "p1_status": p_status.get("p1_status"),
            "p2_status": p_status.get("p2_status"),
            "p3_status": p_status.get("p3_status"),
        }
        records.append(rec)
        print(f"  Req #{i+1:02d} | RSS={rss_mb:6.2f}MB | Lat={lat_ms:6.1f}ms | H={data.get('overall_h_score')} | P1={p_scores.get('retrieval')} P2={p_scores.get('confidence')} P3={p_scores.get('consistency')}")

    # Batch 2: 5 sequential 2-claim requests
    print("\n--- Running 5 Sequential Two-Claim Requests ---")
    two_claim_texts = [
        "Paris is the capital of France. Berlin is the capital of Germany.",
        "Paris is the capital of France. Berlin is the capital of France.",
        "The Sun is a star. The Earth orbits the Sun.",
        "Mount Everest is the highest mountain on Earth. K2 is the second highest.",
        "Oxygen is a chemical element. Oxygen has atomic number 8."
    ]

    for i, text in enumerate(two_claim_texts):
        t0 = time.perf_counter()
        res = client.post("/api/v1/analyze", json={"response": text, "model_name": "gpt-4o"})
        lat_ms = (time.perf_counter() - t0) * 1000.0
        assert res.status_code == 200, f"Request failed: {res.text}"
        data = res.json()
        rss_mb = proc.memory_info().rss / (1024 * 1024)
        p_scores = data.get("pillar_scores", {})
        p_status = data.get("pillar_status", {})
        rec = {
            "req_index": 10 + i + 1,
            "type": "two_claims",
            "text": text,
            "latency_ms": round(lat_ms, 2),
            "rss_mb": round(rss_mb, 2),
            "h_score": data.get("overall_h_score"),
            "root_cause": data.get("root_cause_classification"),
            "p1_status": p_status.get("p1_status"),
            "p2_status": p_status.get("p2_status"),
            "p3_status": p_status.get("p3_status"),
        }
        records.append(rec)
        print(f"  Req #{10+i+1:02d} | RSS={rss_mb:6.2f}MB | Lat={lat_ms:6.1f}ms | H={data.get('overall_h_score')} | P1={p_scores.get('retrieval')} P2={p_scores.get('confidence')} P3={p_scores.get('consistency')}")

    # Batch 3: 5 sequential 5-claim requests
    print("\n--- Running 5 Sequential Five-Claim Requests ---")
    five_claim_texts = [
        "Mercury is the smallest planet. Venus is the second planet from the Sun. Earth is our home planet. Mars is the red planet. Jupiter is the largest planet.",
        "Hydrogen is the lightest element. Helium is a noble gas. Lithium is an alkali metal. Beryllium is an alkaline earth metal. Boron is a metalloid.",
        "Rome is in Italy. Madrid is in Spain. Lisbon is in Portugal. Athens is in Greece. Vienna is in Austria.",
        "Sharks are fish. Whales are mammals. Eagles are birds. Frogs are amphibians. Snakes are reptiles.",
        "Red is a primary color. Blue is a primary color. Yellow is a primary color. Green is a secondary color. Orange is a secondary color."
    ]

    for i, text in enumerate(five_claim_texts):
        t0 = time.perf_counter()
        res = client.post("/api/v1/analyze", json={"response": text, "model_name": "gpt-4o"})
        lat_ms = (time.perf_counter() - t0) * 1000.0
        assert res.status_code == 200, f"Request failed: {res.text}"
        data = res.json()
        rss_mb = proc.memory_info().rss / (1024 * 1024)
        p_scores = data.get("pillar_scores", {})
        p_status = data.get("pillar_status", {})
        rec = {
            "req_index": 15 + i + 1,
            "type": "five_claims",
            "text": text,
            "latency_ms": round(lat_ms, 2),
            "rss_mb": round(rss_mb, 2),
            "h_score": data.get("overall_h_score"),
            "root_cause": data.get("root_cause_classification"),
            "p1_status": p_status.get("p1_status"),
            "p2_status": p_status.get("p2_status"),
            "p3_status": p_status.get("p3_status"),
        }
        records.append(rec)
        print(f"  Req #{15+i+1:02d} | RSS={rss_mb:6.2f}MB | Lat={lat_ms:6.1f}ms | H={data.get('overall_h_score')} | P1={p_scores.get('retrieval')} P2={p_scores.get('confidence')} P3={p_scores.get('consistency')}")

    final_rss = proc.memory_info().rss / (1024 * 1024)
    print(f"\nFinal Steady RSS: {final_rss:.2f} MB")
    print(f"Total RSS Growth: {final_rss - init_rss:.2f} MB")

    out_file = backend_dir / "reports" / "phase47a" / "smoke_test_results.json"
    with open(out_file, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved smoke test report to {out_file}")

if __name__ == "__main__":
    run_smoke_test()
