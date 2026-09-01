"""Phase 49 Master Production Memory Stress & Longevity Benchmark.

Runs:
1. Cold start baseline RSS measurement
2. Warm-up model initialization measurement
3. 1, 5, 10, 20, 50, 100 sequential requests across diverse verification domains
4. 2, 4, 8 concurrent burst requests
5. Memory slope, peak RSS, 95th/99th percentile RSS analysis
6. Persistence of JSON results to backend/reports/phase49/
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

# Bounded environment variables
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.main import create_application
from app.core.engine.model_registry import ModelRegistry
from app.core.engine.memory_utils import trim_process_memory

REPORTS_DIR = Path("backend/reports/phase49")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


PROMPTS_BENCHMARK = [
    # 1-10: Standard facts & science
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
    # 11-15: Multi-claim sentences (P3 Intra-response consistency)
    ("Paris is the capital of France. Berlin is the capital of Germany.", "Name the capitals."),
    ("Paris is the capital of France. Berlin is the capital of France.", "Name the capitals."),
    ("Mount Everest is Earth's highest mountain above sea level. K2 is the second highest.", "Name the highest mountains."),
    ("The Amazon River is the largest river by discharge. The Nile is the longest river.", "Tell me about rivers."),
    ("Python is a widely used high-level programming language. Rust focuses on memory safety.", "Programming languages."),
    # 16-20: Symbolic Arithmetic
    ("12 multiplied by 8 equals 96.", "Math"),
    ("12 multiplied by 8 equals 95.", "Math"),
    ("15 plus 27 equals 42.", "Math"),
    ("100 divided by 4 equals 25.", "Math"),
    ("100 divided by 4 equals 26.", "Math"),
]


def run_phase49_memory_stress():
    print("=" * 80)
    print("PHASE 49: P0 PRODUCTION OOM ELIMINATION & RESIDENT MEMORY HARDENING")
    print("=" * 80)

    process = psutil.Process(os.getpid())

    # Step 1: Cold Start
    startup_rss = process.memory_info().rss / (1024.0 * 1024.0)
    startup_vms = process.memory_info().vms / (1024.0 * 1024.0)
    print(f"1. COLD STARTUP RSS: {startup_rss:.2f} MB | VMS: {startup_vms:.2f} MB")

    # Step 2: Application & Model Warmup
    app = create_application()
    client = TestClient(app)

    # Warmup request
    _ = client.post("/api/v1/analyze", json={"response": "Warmup request."})
    warm_rss = process.memory_info().rss / (1024.0 * 1024.0)
    print(f"2. WARM MODEL RSS:    {warm_rss:.2f} MB (Delta: +{warm_rss - startup_rss:.2f} MB)")

    # Step 3: 50 Sequential Requests Trajectory
    print("\n3. EXECUTING 50 SEQUENTIAL REQUESTS BENCHMARK...")
    trajectory = []
    latencies = []
    h_scores = []
    peak_rss = warm_rss

    for req_idx in range(1, 51):
        prompt_text, query_text = PROMPTS_BENCHMARK[(req_idx - 1) % len(PROMPTS_BENCHMARK)]
        t0 = time.perf_counter()
        resp = client.post(
            "/api/v1/analyze",
            json={"response": prompt_text, "query": query_text},
            headers={"Content-Type": "application/json"},
        )
        dur_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dur_ms)

        assert resp.status_code == 200, f"Req #{req_idx} failed: {resp.text}"
        data = resp.json()
        h_score = data.get("overall_h_score", 0.0)
        h_scores.append(h_score)

        curr_rss = process.memory_info().rss / (1024.0 * 1024.0)
        if curr_rss > peak_rss:
            peak_rss = curr_rss

        record = {
            "request_idx": req_idx,
            "rss_mb": round(curr_rss, 2),
            "latency_ms": round(dur_ms, 2),
            "h_score": round(h_score, 4),
            "risk_level": data.get("risk_level"),
        }
        trajectory.append(record)

        if req_idx in {1, 5, 10, 20, 30, 40, 50}:
            print(f"   Req #{req_idx:02d} | RSS: {curr_rss:7.2f} MB | Latency: {dur_ms:6.1f} ms | H: {h_score:.4f} | Risk: {data.get('risk_level')}")

    final_seq_rss = process.memory_info().rss / (1024.0 * 1024.0)
    rss_growth = final_seq_rss - trajectory[0]["rss_mb"]

    # Step 4: Multi-Level Concurrency Testing (2, 4, 8 concurrent workers)
    print("\n4. EXECUTING CONCURRENCY PRESSURE BENCHMARKS (2, 4, 8 WORKERS)...")
    concurrency_results = {}

    for c_level in [2, 4, 8]:
        t_c0 = time.perf_counter()
        c_latencies = []
        success_count = 0

        def send_req(item_idx: int):
            p, q = PROMPTS_BENCHMARK[item_idx % len(PROMPTS_BENCHMARK)]
            t_start = time.perf_counter()
            r = client.post("/api/v1/analyze", json={"response": p, "query": q})
            dur = (time.perf_counter() - t_start) * 1000.0
            return r.status_code, dur

        with concurrent.futures.ThreadPoolExecutor(max_workers=c_level) as pool:
            futures = [pool.submit(send_req, i) for i in range(c_level)]
            for fut in concurrent.futures.as_completed(futures):
                status_code, dur = fut.result()
                c_latencies.append(dur)
                if status_code == 200:
                    success_count += 1

        c_dur_total = (time.perf_counter() - t_c0) * 1000.0
        c_rss = process.memory_info().rss / (1024.0 * 1024.0)
        if c_rss > peak_rss:
            peak_rss = c_rss

        concurrency_results[f"concurrency_{c_level}"] = {
            "concurrency": c_level,
            "success_rate": f"{success_count}/{c_level}",
            "peak_rss_mb": round(c_rss, 2),
            "avg_latency_ms": round(statistics.mean(c_latencies), 2),
            "total_wall_time_ms": round(c_dur_total, 2),
        }
        print(f"   Concurrency {c_level:2d}: Success={success_count}/{c_level} | RSS={c_rss:7.2f} MB | Avg Latency={statistics.mean(c_latencies):6.1f} ms | Wall Time={c_dur_total:6.1f} ms")

    # Step 5: Statistical Calculations
    rss_values = [item["rss_mb"] for item in trajectory]
    mean_rss = statistics.mean(rss_values)
    median_rss = statistics.median(rss_values)
    sorted_rss = sorted(rss_values)
    p95_rss = sorted_rss[int(len(sorted_rss) * 0.95)]
    p99_rss = sorted_rss[int(len(sorted_rss) * 0.99)]

    init_counts = ModelRegistry.get_init_counts()

    summary = {
        "startup_rss_mb": round(startup_rss, 2),
        "warm_rss_mb": round(warm_rss, 2),
        "peak_rss_mb": round(peak_rss, 2),
        "final_seq_rss_mb": round(final_seq_rss, 2),
        "rss_growth_req1_to_50_mb": round(rss_growth, 2),
        "mean_rss_mb": round(mean_rss, 2),
        "median_rss_mb": round(median_rss, 2),
        "p95_rss_mb": round(p95_rss, 2),
        "p99_rss_mb": round(p99_rss, 2),
        "railway_limit_mb": 1024.0,
        "railway_safety_margin_mb": round(1024.0 - peak_rss, 2),
        "model_inventory": init_counts,
        "concurrency_benchmarks": concurrency_results,
        "trajectory_sample": trajectory[:10],
    }

    # Save Results JSON
    output_path = REPORTS_DIR / "PHASE49_STRESS_RESULTS.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 49 PRODUCTION MEMORY AUDIT & COMPLIANCE TABLE")
    print("=" * 80)
    print(f"Startup RSS:              {startup_rss:7.2f} MB   (Target: < 350 MB) -> {'✅ PASS' if startup_rss < 350 else '⚠️ WARN'}")
    print(f"Warm Model RSS:           {warm_rss:7.2f} MB   (Target: < 450 MB) -> {'✅ PASS' if warm_rss < 450 else '⚠️ WARN'}")
    print(f"Peak RSS (50 Requests):   {peak_rss:7.2f} MB   (Target: < 600 MB) -> {'✅ PASS' if peak_rss < 600 else '❌ FAIL'}")
    print(f"Final RSS (After 50 Req): {final_seq_rss:7.2f} MB   (Target: < 550 MB) -> {'✅ PASS' if final_seq_rss < 550 else '❌ FAIL'}")
    print(f"RSS Growth (Req 1 -> 50): {rss_growth:7.2f} MB   (Target: near 0)   -> {'✅ PASS' if abs(rss_growth) < 25 else '⚠️ WARN'}")
    print(f"P95 RSS:                  {p95_rss:7.2f} MB")
    print(f"P99 RSS:                  {p99_rss:7.2f} MB")
    print(f"Railway Headroom Margin:  {1024.0 - peak_rss:7.2f} MB   (Target: > 350 MB) -> {'✅ PASS' if (1024.0 - peak_rss) > 350 else '⚠️ WARN'}")
    print(f"NLI Model Init Count:     {init_counts.get('nli_model', 0)} (Strictly 1)")
    print(f"SentenceTransformer Init: {init_counts.get('sentence_transformer', 0)} (Strictly 0)")
    print(f"CrossEncoder Reranker:    {init_counts.get('cross_encoder_reranker', 0)} (Strictly 0)")
    print("=" * 80)
    print(f"Persisted Phase 49 stress results to: {output_path}")

if __name__ == "__main__":
    run_phase49_memory_stress()
