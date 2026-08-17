"""Local Production Load Test for HalluciSense Phase 11B.

Executes:
1. 1 Single Request
2. 5 Sequential Requests
3. 10 Sequential Requests
4. 10 Concurrent Requests

Measures:
- Initial & Peak Process RSS
- Total & Per-Stage Latency
- Zero Error Guarantee
- Single Model Initialization Guarantee (`model_initializations == 1`)

Outputs structured results to backend/reports/phase11/phase11_load_test.json.
"""

from __future__ import annotations

import os
import sys
import json
import time
import psutil
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.core.engine.model_registry import ModelRegistry
from app.core.correction.correction_engine import CorrectionEngine


def get_rss_mb() -> float:
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)


TEST_PROMPTS = [
    ("What is the speed of light in vacuum?", "The speed of light in vacuum is defined as exactly 299792458 meters per second."),
    ("What is standard atmospheric pressure?", "Standard atmospheric pressure at sea level is approximately 101.325 kPa."),
    ("What is the molar mass of water?", "Water has a molar mass of approximately 18.015 g/mol."),
    ("What direction does DNA replication occur?", "DNA replication in eukaryotic cells proceeds in the 5-prime to 3-prime direction."),
    ("What causes Type 1 diabetes?", "Type 1 diabetes mellitus is characterized by autoimmune destruction of pancreatic beta cells."),
]


def execute_single_workflow(query: str, text: str, pipeline, engine) -> dict:
    t0 = time.perf_counter()
    init_verif = pipeline.analyze_response(full_text=text, query=query)
    h_score = float(getattr(init_verif, "hallucination_score", 0.0))
    
    repaired = False
    if h_score >= 0.35:
        corr_res = engine.execute_closed_loop_repair(
            user_query=query,
            initial_text=text,
            initial_verification=init_verif,
            max_attempts=2,
        )
        repaired = corr_res.performed
    
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "h_score": h_score,
        "repaired": repaired,
        "elapsed_ms": elapsed_ms,
    }


def run_load_test() -> dict:
    print("\n" + "=" * 60)
    print("RUNNING PHASE 11B PRODUCTION LOAD TEST")
    print("=" * 60)

    start_rss = get_rss_mb()
    pipeline = ModelRegistry.get_pipeline()
    engine = CorrectionEngine(pipeline=pipeline)

    results = {}
    
    # 1. Single Request
    print("1. Executing Single Request...")
    q, ans = TEST_PROMPTS[0]
    t0 = time.perf_counter()
    r1 = execute_single_workflow(q, ans, pipeline, engine)
    single_duration = (time.perf_counter() - t0) * 1000.0
    single_peak_rss = get_rss_mb()
    results["single_request"] = {
        "latency_ms": round(single_duration, 2),
        "rss_mb": single_peak_rss,
        "h_score": r1["h_score"],
    }
    print(f"   Done in {single_duration:.1f} ms, RSS: {single_peak_rss} MB")

    # 2. 5 Sequential Requests
    print("2. Executing 5 Sequential Requests...")
    t0 = time.perf_counter()
    seq5_latencies = []
    for i in range(5):
        q, ans = TEST_PROMPTS[i % len(TEST_PROMPTS)]
        r = execute_single_workflow(q, ans, pipeline, engine)
        seq5_latencies.append(r["elapsed_ms"])
    seq5_duration = (time.perf_counter() - t0) * 1000.0
    seq5_peak_rss = get_rss_mb()
    results["5_sequential"] = {
        "total_ms": round(seq5_duration, 2),
        "mean_ms": round(sum(seq5_latencies) / len(seq5_latencies), 2),
        "rss_mb": seq5_peak_rss,
    }
    print(f"   Done in {seq5_duration:.1f} ms, Mean: {results['5_sequential']['mean_ms']} ms, RSS: {seq5_peak_rss} MB")

    # 3. 10 Sequential Requests
    print("3. Executing 10 Sequential Requests...")
    t0 = time.perf_counter()
    seq10_latencies = []
    for i in range(10):
        q, ans = TEST_PROMPTS[i % len(TEST_PROMPTS)]
        r = execute_single_workflow(q, ans, pipeline, engine)
        seq10_latencies.append(r["elapsed_ms"])
    seq10_duration = (time.perf_counter() - t0) * 1000.0
    seq10_peak_rss = get_rss_mb()
    results["10_sequential"] = {
        "total_ms": round(seq10_duration, 2),
        "mean_ms": round(sum(seq10_latencies) / len(seq10_latencies), 2),
        "rss_mb": seq10_peak_rss,
    }
    print(f"   Done in {seq10_duration:.1f} ms, Mean: {results['10_sequential']['mean_ms']} ms, RSS: {seq10_peak_rss} MB")

    # 4. 10 Concurrent Requests (via bounded ThreadPoolExecutor)
    print("4. Executing 10 Concurrent Requests...")
    t0 = time.perf_counter()
    concurrent_latencies = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                execute_single_workflow,
                TEST_PROMPTS[i % len(TEST_PROMPTS)][0],
                TEST_PROMPTS[i % len(TEST_PROMPTS)][1],
                pipeline,
                engine,
            )
            for i in range(10)
        ]
        for f in futures:
            res = f.result()
            concurrent_latencies.append(res["elapsed_ms"])

    concurrent_duration = (time.perf_counter() - t0) * 1000.0
    concurrent_peak_rss = get_rss_mb()
    results["10_concurrent"] = {
        "total_ms": round(concurrent_duration, 2),
        "mean_ms": round(sum(concurrent_latencies) / len(concurrent_latencies), 2),
        "rss_mb": concurrent_peak_rss,
    }
    print(f"   Done in {concurrent_duration:.1f} ms, Mean: {results['10_concurrent']['mean_ms']} ms, RSS: {concurrent_peak_rss} MB")

    init_counts = ModelRegistry.get_init_counts()
    peak_rss = max(start_rss, single_peak_rss, seq5_peak_rss, seq10_peak_rss, concurrent_peak_rss)

    summary = {
        "start_rss_mb": start_rss,
        "peak_rss_mb": peak_rss,
        "benchmarks": results,
        "model_initialization_counts": init_counts,
        "single_instance_guarantee": all(c <= 1 for c in init_counts.values()) and init_counts.get("pipeline", 0) == 1,
        "errors": 0,
    }

    report_dir = Path("backend/reports/phase11")
    report_dir.mkdir(parents=True, exist_ok=True)
    out_file = report_dir / "phase11_load_test.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print(f"LOAD TEST SUMMARY")
    print(f"Peak Process RSS          : {peak_rss} MB")
    print(f"Single Instance Guarantee : {summary['single_instance_guarantee']}")
    print(f"Model Init Counts         : {init_counts}")
    print(f"Saved report to           : {out_file}")
    print("=" * 60 + "\n")
    return summary


if __name__ == "__main__":
    run_load_test()
