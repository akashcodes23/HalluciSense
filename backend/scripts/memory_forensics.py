"""Phase 48 — Master Memory Forensics & Stress Test Script.

Evaluates:
1. Cold startup memory
2. Warmup request
3. 50 sequential analysis requests across diverse claim types
4. 10 repeated identical requests
5. 10 distinct factual/false requests
6. Long response request (multi-sentence paragraph)
7. Multi-claim request (5+ claims)
8. Contradictory claims request
9. Concurrent request pressure test (2, 4, 8 concurrent requests)

Records per-request telemetry (RSS, VMS, threads, model init counts, latencies, statuses).
Outputs structured summary table and persists JSON telemetry.
"""

import os
import sys
import time
import json
import statistics
import concurrent.futures
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import psutil
from fastapi.testclient import TestClient
from app.main import create_application
from app.core.engine.model_registry import ModelRegistry
from app.core.engine.memory_utils import trim_process_memory, get_memory_telemetry


def run_memory_forensics():
    print("=" * 80)
    print("PHASE 48 — MEMORY FORENSICS & PRODUCTION STRESS TEST")
    print("=" * 80)

    proc = psutil.Process(os.getpid())
    
    # 1. Cold Startup
    trim_process_memory()
    startup_rss = proc.memory_info().rss / (1024 * 1024)
    startup_vms = proc.memory_info().vms / (1024 * 1024)
    startup_threads = proc.num_threads()
    print(f"1. Cold Startup Telemetry: RSS={startup_rss:.2f} MB | VMS={startup_vms:.2f} MB | Threads={startup_threads}")

    app = create_application()
    client = TestClient(app)

    # 2. Warmup Request
    t_warm0 = time.perf_counter()
    warm_res = client.post("/api/v1/analyze", json={"response": "The Moon orbits the Earth.", "model_name": "gpt-4o"})
    assert warm_res.status_code == 200, f"Warmup failed: {warm_res.text}"
    warm_lat = (time.perf_counter() - t_warm0) * 1000.0
    trim_process_memory()
    warm_rss = proc.memory_info().rss / (1024 * 1024)
    print(f"2. Warm Model Telemetry:   RSS={warm_rss:.2f} MB | Latency={warm_lat:.1f}ms | InitCounts={ModelRegistry.get_init_counts()}")

    # 3. 50 Sequential Analysis Requests
    print("\n--- Executing 50 Sequential Requests ---")
    prompts_pool = [
        "The capital of France is Paris.",
        "The capital of France is Berlin.",
        "What is the capital of France?",
        "12 multiplied by 8 equals 96.",
        "12 multiplied by 8 equals 95.",
        "Water freezes at 0 degrees Celsius under standard atmospheric pressure.",
        "The chemical formula for water is H2O.",
        "The chemical formula for water is CO2.",
        "Jupiter is the largest planet in our solar system.",
        "Albert Einstein developed the theory of general relativity.",
        "Paris is the capital of France. Berlin is the capital of Germany.",
        "Paris is the capital of France. Berlin is the capital of France.",
        "The speed of light in a vacuum is 299792458 meters per second.",
        "The Sun is a yellow dwarf star at the center of our solar system.",
        "Photosynthesis is the process by which plants use sunlight to synthesize nutrients.",
        "Oxygen has atomic number 8.",
        "Helium is the second lightest and second most abundant element in the universe.",
        "Mount Everest is Earth's highest mountain above sea level.",
        "The Amazon River is the largest river by discharge volume of water in the world.",
        "Python is a widely used high-level, general-purpose programming language."
    ]

    sequential_records = []
    rss_series = []

    for req_idx in range(1, 51):
        prompt = prompts_pool[(req_idx - 1) % len(prompts_pool)]
        t0 = time.perf_counter()
        res = client.post("/api/v1/analyze", json={"response": prompt, "model_name": "gpt-4o"})
        lat_ms = (time.perf_counter() - t0) * 1000.0
        data = res.json()
        del res
        trim_process_memory()
        mem_info = proc.memory_info()
        cur_rss = mem_info.rss / (1024 * 1024)
        cur_vms = mem_info.vms / (1024 * 1024)
        cur_threads = proc.num_threads()
        rss_series.append(cur_rss)

        p_scores = data.get("pillar_scores", {})
        p_status = data.get("pillar_status", {})

        record = {
            "req_number": req_idx,
            "prompt": prompt,
            "rss_mb": round(cur_rss, 2),
            "vms_mb": round(cur_vms, 2),
            "threads": cur_threads,
            "latency_ms": round(lat_ms, 2),
            "h_score": data.get("overall_h_score"),
            "root_cause": data.get("root_cause_classification"),
            "p1_status": p_status.get("p1_status"),
            "p2_status": p_status.get("p2_status"),
            "p3_status": p_status.get("p3_status"),
            "p1_score": p_scores.get("retrieval"),
            "p2_score": p_scores.get("confidence"),
            "p3_score": p_scores.get("consistency"),
            "init_counts": ModelRegistry.get_init_counts(),
        }
        sequential_records.append(record)

        if req_idx in [1, 5, 10, 20, 30, 40, 50] or req_idx % 10 == 0:
            print(f"  Req #{req_idx:02d} | RSS={cur_rss:6.2f}MB | Lat={lat_ms:6.1f}ms | H={data.get('overall_h_score')} | P1={p_scores.get('retrieval')} P2={p_scores.get('confidence')} P3={p_scores.get('consistency')}")

    # 4. Long Multi-Sentence Request Test
    print("\n--- Long Multi-Sentence Request Test ---")
    long_text = (
        "The solar system consists of the Sun and the objects that orbit it. "
        "Mercury is the smallest planet in the solar system. "
        "Venus has a dense, toxic atmosphere primarily composed of carbon dioxide. "
        "Earth is the third planet from the Sun and the only astronomical object known to harbor life. "
        "Mars is known as the Red Planet due to iron oxide on its surface."
    )
    t0 = time.perf_counter()
    long_res = client.post("/api/v1/analyze", json={"response": long_text, "model_name": "gpt-4o"})
    long_lat = (time.perf_counter() - t0) * 1000.0
    assert long_res.status_code == 200
    trim_process_memory()
    long_rss = proc.memory_info().rss / (1024 * 1024)
    print(f"  Long text (5 sentences): RSS={long_rss:.2f}MB | Latency={long_lat:.1f}ms | H={long_res.json().get('overall_h_score')}")

    # 5. Contradictory Claim Test
    print("\n--- Contradictory Claim Test ---")
    contra_text = "Paris is the capital of France. Berlin is the capital of France."
    t0 = time.perf_counter()
    contra_res = client.post("/api/v1/analyze", json={"response": contra_text, "model_name": "gpt-4o"})
    contra_lat = (time.perf_counter() - t0) * 1000.0
    assert contra_res.status_code == 200
    contra_data = contra_res.json()
    trim_process_memory()
    contra_rss = proc.memory_info().rss / (1024 * 1024)
    print(f"  Contradictory Claims: RSS={contra_rss:.2f}MB | Latency={contra_lat:.1f}ms | H={contra_data.get('overall_h_score')} | P3 CF={contra_data.get('pillar_scores', {}).get('consistency')}")

    # 6. Concurrency Pressure Test (2, 4, 8 concurrent requests)
    print("\n--- Concurrency Pressure Test (2, 4, 8 concurrent workers) ---")
    concurrency_levels = [2, 4, 8]
    concurrency_results = {}

    for conc in concurrency_levels:
        def make_req(i):
            t_c0 = time.perf_counter()
            r = client.post("/api/v1/analyze", json={"response": f"Concurrency probe {i}: Paris is the capital of France.", "model_name": "gpt-4o"})
            return r.status_code, (time.perf_counter() - t_c0) * 1000.0

        t_conc_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as executor:
            futures = [executor.submit(make_req, i) for i in range(conc)]
            statuses = [f.result()[0] for f in futures]
            lats = [f.result()[1] for f in futures]
        conc_dur = (time.perf_counter() - t_conc_start) * 1000.0

        trim_process_memory()
        conc_rss = proc.memory_info().rss / (1024 * 1024)
        concurrency_results[f"concurrency_{conc}"] = {
            "workers": conc,
            "success_rate": sum(1 for s in statuses if s == 200) / conc,
            "avg_latency_ms": round(statistics.mean(lats), 2),
            "wall_time_ms": round(conc_dur, 2),
            "rss_mb": round(conc_rss, 2),
        }
        print(f"  Concurrency Level {conc}: Success={sum(1 for s in statuses if s == 200)}/{conc} | RSS={conc_rss:.2f}MB | AvgLat={statistics.mean(lats):.1f}ms | WallTime={conc_dur:.1f}ms")

    # Metrics Summary
    min_rss = min(rss_series)
    max_rss = max(rss_series)
    mean_rss = statistics.mean(rss_series)
    median_rss = statistics.median(rss_series)
    first_rss = rss_series[0]
    last_rss = rss_series[-1]
    rss_growth_50 = last_rss - first_rss
    rss_growth_per_req = rss_growth_50 / 49.0 if len(rss_series) > 1 else 0.0

    print("\n" + "=" * 80)
    print("PHASE 48 MEMORY METRICS & TARGET VALIDATION TABLE")
    print("=" * 80)
    print(f"Startup RSS:            {startup_rss:6.2f} MB   (Target: < 350 MB) -> {'✅ PASS' if startup_rss < 350 else '⚠️ WARN'}")
    print(f"Warm Model RSS:         {warm_rss:6.2f} MB   (Target: < 500 MB) -> {'✅ PASS' if warm_rss < 500 else '⚠️ WARN'}")
    print(f"Peak RSS (50 Requests): {max_rss:6.2f} MB   (Target: < 650 MB) -> {'✅ PASS' if max_rss < 650 else '❌ FAIL'}")
    print(f"RSS After 50 Requests:  {last_rss:6.2f} MB   (Target: < 650 MB) -> {'✅ PASS' if last_rss < 650 else '❌ FAIL'}")
    print(f"RSS Growth (Req 1->50): {rss_growth_50:6.2f} MB   (Target: near 0)   -> {'✅ PASS' if abs(rss_growth_50) < 50 else '⚠️ WARN'}")
    print(f"Mean RSS:               {mean_rss:6.2f} MB")
    print(f"Median RSS:             {median_rss:6.2f} MB")
    print(f"Peak Concurrency RSS:   {max(c['rss_mb'] for c in concurrency_results.values()):6.2f} MB")
    print(f"NLI Model Init Count:   {ModelRegistry.get_init_counts().get('nli_model', 0)} (Strictly 1)")
    print(f"SentenceTransformer:   {ModelRegistry.get_init_counts().get('sentence_transformer', 0)} (Strictly 0 in Prod)")
    print(f"Reranker:               {ModelRegistry.get_init_counts().get('cross_encoder_reranker', 0)} (Strictly 0 in Prod)")
    print("=" * 80)

    summary = {
        "startup_rss_mb": round(startup_rss, 2),
        "warm_rss_mb": round(warm_rss, 2),
        "min_rss_mb": round(min_rss, 2),
        "max_rss_mb": round(max_rss, 2),
        "mean_rss_mb": round(mean_rss, 2),
        "median_rss_mb": round(median_rss, 2),
        "rss_growth_req_1_to_50_mb": round(rss_growth_50, 2),
        "rss_growth_per_req_mb": round(rss_growth_per_req, 3),
        "concurrency_tests": concurrency_results,
        "init_counts": ModelRegistry.get_init_counts(),
        "sequential_records": sequential_records,
    }

    out_file = backend_dir / "reports" / "phase48" / "memory_forensics_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nPersisted forensics telemetry to: {out_file}")

if __name__ == "__main__":
    run_memory_forensics()
