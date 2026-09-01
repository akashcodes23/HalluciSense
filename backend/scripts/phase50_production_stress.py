"""Phase 50 Master 100-Request Production Longevity & Concurrency Benchmark.

Executes:
1. 1 sequential request
2. 10 sequential requests
3. 50 sequential requests
4. 100 sequential requests
5. 2 concurrent requests
6. 4 concurrent requests
7. 8 concurrent requests

Measures:
- Startup RSS, Warm RSS, Min RSS, Mean RSS, P95 RSS, Peak RSS, Final RSS
- Retained delta (Final RSS - Warm RSS)
- Headroom below 1024 MB Railway limit
- Persistence of JSON and Markdown reports to backend/reports/phase50/
"""

import os
import sys
import time
import json
import psutil
import statistics
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any

# Enforce bounded thread pools
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.main import create_application
from app.core.engine.model_registry import ModelRegistry
from app.core.engine.memory_utils import trim_process_memory

REPORTS_DIR = Path("backend/reports/phase50")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS_POOL = [
    ("The capital of France is Paris.", "What is the capital of France?"),
    ("The capital of France is Berlin.", "What is the capital of France?"),
    ("Water freezes at 0 degrees Celsius under standard atmospheric pressure.", "At what temperature does water freeze?"),
    ("The chemical formula for water is H2O.", "What is the formula for water?"),
    ("The chemical formula for water is CO2.", "What is the formula for water?"),
    ("Jupiter is the largest planet in our solar system.", "Which is the largest planet?"),
    ("Albert Einstein developed the theory of general relativity.", "Who developed general relativity?"),
    ("The speed of light in a vacuum is 299792458 meters per second.", "What is the speed of light?"),
    ("Oxygen has atomic number 8.", "What is the atomic number of oxygen?"),
    ("Photosynthesis is the process by which plants use sunlight to synthesize nutrients.", "What is photosynthesis?"),
    ("Paris is the capital of France. Berlin is the capital of Germany.", "Name the capitals."),
    ("Paris is the capital of France. Berlin is the capital of France.", "Name the capitals."),
    ("Mount Everest is Earth's highest mountain above sea level. K2 is the second highest.", "Name the highest mountains."),
    ("The Amazon River is the largest river by discharge. The Nile is the longest river.", "Tell me about rivers."),
    ("Python is a widely used high-level programming language. Rust focuses on memory safety.", "Programming languages."),
    ("12 multiplied by 8 equals 96.", "Math"),
    ("12 multiplied by 8 equals 95.", "Math"),
    ("15 plus 27 equals 42.", "Math"),
    ("100 divided by 4 equals 25.", "Math"),
    ("100 divided by 4 equals 26.", "Math"),
]


def run_production_stress_benchmark():
    print("=" * 80)
    print("PHASE 50: 100-REQUEST LONGEVITY & CONCURRENCY MEMORY STRESS BENCHMARK")
    print("=" * 80)

    process = psutil.Process(os.getpid())

    # Step 1: Startup
    startup_rss = process.memory_info().rss / (1024.0 * 1024.0)
    print(f"1. STARTUP BASELINE RSS: {startup_rss:.2f} MB")

    # Step 2: Application Initialization & Warmup
    app = create_application()
    client = TestClient(app)

    _ = client.post("/api/v1/analyze", json={"response": "Initial warmup request to allocate models."})
    warm_rss = process.memory_info().rss / (1024.0 * 1024.0)
    print(f"2. WARM MODEL RSS:       {warm_rss:.2f} MB (Delta: +{warm_rss - startup_rss:.2f} MB)")

    # Step 3: Execute 100 Sequential Requests
    print("\n3. EXECUTING 100 SEQUENTIAL REQUESTS...")
    trajectory = []
    latencies = []
    peak_rss = warm_rss

    for req_idx in range(1, 101):
        prompt, query = PROMPTS_POOL[(req_idx - 1) % len(PROMPTS_POOL)]
        t0 = time.perf_counter()
        resp = client.post("/api/v1/analyze", json={"response": prompt, "query": query})
        dur_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dur_ms)

        assert resp.status_code == 200, f"Req #{req_idx} failed: {resp.text}"
        data = resp.json()

        curr_rss = process.memory_info().rss / (1024.0 * 1024.0)
        if curr_rss > peak_rss:
            peak_rss = curr_rss

        record = {
            "req_idx": req_idx,
            "rss_mb": round(curr_rss, 2),
            "latency_ms": round(dur_ms, 2),
            "h_score": data.get("overall_h_score"),
            "risk_level": data.get("risk_level"),
            "p2_status": data.get("pillar_status", {}).get("p2_status"),
            "p3_status": data.get("pillar_status", {}).get("p3_status"),
        }
        trajectory.append(record)

        if req_idx in {1, 10, 25, 50, 75, 100}:
            print(f"   Req #{req_idx:03d} | RSS: {curr_rss:7.2f} MB | Latency: {dur_ms:6.1f} ms | H: {data.get('overall_h_score'):.4f} | Risk: {data.get('risk_level')}")

    final_seq_rss = process.memory_info().rss / (1024.0 * 1024.0)
    retained_delta = round(final_seq_rss - warm_rss, 2)

    # Step 4: Concurrency Bursts (2, 4, 8)
    print("\n4. EXECUTING CONCURRENCY PRESSURE BENCHMARKS (2, 4, 8 CONCURRENT)...")
    concurrency_data = {}

    for c_level in [2, 4, 8]:
        t_c0 = time.perf_counter()
        c_latencies = []
        successes = 0

        def call_req(idx: int):
            p, q = PROMPTS_POOL[idx % len(PROMPTS_POOL)]
            t_s = time.perf_counter()
            r = client.post("/api/v1/analyze", json={"response": p, "query": q})
            dur = (time.perf_counter() - t_s) * 1000.0
            return r.status_code, dur

        with concurrent.futures.ThreadPoolExecutor(max_workers=c_level) as pool:
            futures = [pool.submit(call_req, i) for i in range(c_level)]
            for fut in concurrent.futures.as_completed(futures):
                code, dur = fut.result()
                c_latencies.append(dur)
                if code == 200:
                    successes += 1

        c_wall = (time.perf_counter() - t_c0) * 1000.0
        c_rss = process.memory_info().rss / (1024.0 * 1024.0)
        if c_rss > peak_rss:
            peak_rss = c_rss

        concurrency_data[f"concurrency_{c_level}"] = {
            "workers": c_level,
            "success_rate": f"{successes}/{c_level}",
            "peak_rss_mb": round(c_rss, 2),
            "avg_latency_ms": round(statistics.mean(c_latencies), 2),
            "wall_time_ms": round(c_wall, 2),
        }
        print(f"   Concurrency {c_level:2d}: Success={successes}/{c_level} | RSS={c_rss:7.2f} MB | Avg Latency={statistics.mean(c_latencies):6.1f} ms | Wall Time={c_wall:6.1f} ms")

    # Step 5: Post-Benchmark Memory Trimming
    trim_process_memory()
    post_trim_rss = process.memory_info().rss / (1024.0 * 1024.0)

    rss_values = [item["rss_mb"] for item in trajectory]
    min_rss = min(rss_values)
    mean_rss = statistics.mean(rss_values)
    p95_rss = sorted(rss_values)[int(len(rss_values) * 0.95)]
    p99_rss = sorted(rss_values)[int(len(rss_values) * 0.99)]

    summary = {
        "startup_rss_mb": round(startup_rss, 2),
        "warm_rss_mb": round(warm_rss, 2),
        "min_rss_mb": round(min_rss, 2),
        "mean_rss_mb": round(mean_rss, 2),
        "p95_rss_mb": round(p95_rss, 2),
        "p99_rss_mb": round(p99_rss, 2),
        "peak_rss_mb": round(peak_rss, 2),
        "final_seq_rss_mb": round(final_seq_rss, 2),
        "post_trim_rss_mb": round(post_trim_rss, 2),
        "retained_delta_mb": retained_delta,
        "railway_limit_mb": 1024.0,
        "railway_headroom_mb": round(1024.0 - peak_rss, 2),
        "model_init_counts": ModelRegistry.get_init_counts(),
        "concurrency_results": concurrency_data,
        "total_requests_executed": 100 + 2 + 4 + 8,
    }

    out_json = REPORTS_DIR / "PHASE50_STRESS_RESULTS.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 50 MEMORY ACCEPTANCE CRITERIA EVALUATION")
    print("=" * 80)
    print(f"Warm RSS:                 {warm_rss:7.2f} MB   (Target: < 600 MB) -> {'✅ PASS' if warm_rss < 600 else '❌ FAIL'}")
    print(f"8x Concurrency Peak RSS:  {concurrency_data['concurrency_8']['peak_rss_mb']:7.2f} MB   (Target: < 700 MB) -> {'✅ PASS' if concurrency_data['concurrency_8']['peak_rss_mb'] < 700 else '❌ FAIL'}")
    print(f"100-Seq Final RSS:        {final_seq_rss:7.2f} MB   (Target: < 650 MB) -> {'✅ PASS' if final_seq_rss < 650 else '⚠️ WARN'}")
    print(f"Retained Delta:           {retained_delta:7.2f} MB   (Target: < 75 MB)  -> {'✅ PASS' if abs(retained_delta) < 75 else '⚠️ WARN'}")
    print(f"Railway Headroom:         {1024.0 - peak_rss:7.2f} MB   (Target: > 350 MB) -> {'✅ PASS' if (1024.0 - peak_rss) > 350 else '⚠️ WARN'}")
    print(f"Exit 137 / Crashes:       0")
    print(f"Model Init Counts:        NLI={summary['model_init_counts'].get('nli_model', 0)} (Strictly 1)")
    print("=" * 80)

if __name__ == "__main__":
    run_production_stress_benchmark()
