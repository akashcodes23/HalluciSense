"""
Phase 24 — Memory Profiling Harness.
Measures RSS, peak memory, model init counts, latency, and memory retention
across single, 5x, 10x, 20x repeated requests and concurrency.
"""

import os
import sys
import time
import gc
import psutil
import asyncio
from typing import Dict, Any, List

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app
from app.core.engine.model_registry import ModelRegistry


def get_rss_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def profile_endpoint(
    client: TestClient,
    method: str,
    path: str,
    payload: Dict[str, Any],
    label: str,
    iterations: int = 1,
) -> Dict[str, Any]:
    print(f"\n--- Profiling: {label} ({iterations} iterations) ---")
    gc.collect()
    rss_before = get_rss_mb()
    latencies: List[float] = []
    statuses: List[int] = []
    h_scores: List[float] = []
    rss_series: List[float] = []

    init_counts_before = ModelRegistry.get_init_counts()

    for i in range(iterations):
        t0 = time.perf_counter()
        if method == "POST":
            resp = client.post(path, json=payload)
        else:
            resp = client.get(path)
        t_req = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_req)
        statuses.append(resp.status_code)
        
        rss_current = get_rss_mb()
        rss_series.append(rss_current)
        
        if resp.status_code == 200:
            data = resp.json()
            score = (
                data.get("overall_h_score")
                or data.get("verification", {}).get("h_score")
                or 0.0
            )
            h_scores.append(score)
        else:
            print(f"  [Iter {i+1}] Error HTTP {resp.status_code}: {resp.text[:120]}")

        if (i + 1) in [1, 5, 10, 20] or i == iterations - 1:
            print(f"  Iter {i+1:02d}: RSS = {rss_current:.2f} MB | Latency = {t_req:.1f}ms | HTTP {resp.status_code}")

    gc.collect()
    rss_post_gc = get_rss_mb()
    init_counts_after = ModelRegistry.get_init_counts()

    summary = {
        "label": label,
        "iterations": iterations,
        "rss_before_mb": round(rss_before, 2),
        "rss_peak_mb": round(max(rss_series), 2),
        "rss_after_mb": round(rss_series[-1], 2),
        "rss_post_gc_mb": round(rss_post_gc, 2),
        "rss_growth_mb": round(rss_series[-1] - rss_before, 2),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        "min_latency_ms": round(min(latencies), 1),
        "max_latency_ms": round(max(latencies), 1),
        "init_counts_before": init_counts_before,
        "init_counts_after": init_counts_after,
        "all_200_ok": all(s == 200 for s in statuses),
        "avg_h_score": round(sum(h_scores) / len(h_scores), 4) if h_scores else None,
    }
    return summary


def run_full_memory_profile():
    print("============================================================")
    print("PHASE 24 MEMORY PROFILING — INITIAL DIAGNOSTIC RUN")
    print("============================================================")

    rss_initial = get_rss_mb()
    print(f"Initial Process RSS (Startup Baseline): {rss_initial:.2f} MB")

    with TestClient(app) as client:
        # 1. Warm up / Health check
        h_resp = client.get("/health")
        rss_after_health = get_rss_mb()
        print(f"RSS after /health: {rss_after_health:.2f} MB (HTTP {h_resp.status_code})")
        print(f"ModelRegistry counts at startup: {ModelRegistry.get_init_counts()}")

        # Test A: Analyze True Claim (single)
        res_a1 = profile_endpoint(
            client,
            "POST",
            "/api/v1/analyze",
            {"query": "What is the capital of Karnataka?", "response": "The capital of Karnataka is Bengaluru."},
            "Test A: Analyze True Claim (Karnataka=Bengaluru)",
            iterations=1,
        )

        # Test B: Analyze False Claim (single)
        res_b1 = profile_endpoint(
            client,
            "POST",
            "/api/v1/analyze",
            {"query": "What is the capital of Karnataka?", "response": "The capital of Karnataka is Mumbai."},
            "Test B: Analyze False Claim (Karnataka=Mumbai)",
            iterations=1,
        )

        # Test C: Analyze Molar Mass (single)
        res_c1 = profile_endpoint(
            client,
            "POST",
            "/api/v1/analyze",
            {"query": "What is the molar mass of water?", "response": "Water has a molar mass of approximately 18.015 g/mol."},
            "Test C: Analyze Molar Mass of Water",
            iterations=1,
        )

        # Test D: Chat Type 1 Diabetes (single)
        res_d1 = profile_endpoint(
            client,
            "POST",
            "/api/v1/chat",
            {"message": "What causes Type 1 diabetes mellitus?", "enable_verification": True, "auto_correct": True},
            "Test D: Chat Closed-Loop Type 1 Diabetes",
            iterations=1,
        )

        # Test E: Repeated Analyze Requests (20 iterations of Molar Mass)
        res_e20 = profile_endpoint(
            client,
            "POST",
            "/api/v1/analyze",
            {"query": "What is the molar mass of water?", "response": "Water has a molar mass of approximately 18.015 g/mol."},
            "Test E: Repeated Analyze (20 iterations)",
            iterations=20,
        )

        # Test F: Repeated Chat Requests (5 iterations of Molar Mass)
        res_f5 = profile_endpoint(
            client,
            "POST",
            "/api/v1/chat",
            {"message": "What is the molar mass of water?", "enable_verification": True, "auto_correct": True},
            "Test F: Repeated Chat (5 iterations)",
            iterations=5,
        )

        # Concurrency Test
        print("\n--- Concurrency Profile: 4 concurrent analyze requests ---")
        import concurrent.futures
        c_before = get_rss_mb()
        
        def send_req(q, r):
            return client.post("/api/v1/analyze", json={"query": q, "response": r})
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            queries = [
                ("What is the capital of Karnataka?", "The capital of Karnataka is Bengaluru."),
                ("What is the capital of Karnataka?", "The capital of Karnataka is Mumbai."),
                ("What is the molar mass of water?", "Water has a molar mass of approximately 18.015 g/mol."),
                ("What causes Type 1 diabetes mellitus?", "Type 1 diabetes is an autoimmune destruction of beta cells."),
            ]
            futs = [executor.submit(send_req, q, r) for q, r in queries]
            concur_statuses = [f.result().status_code for f in futs]
            
        c_after = get_rss_mb()
        print(f"4-Concurrent Requests Finished: Statuses = {concur_statuses}")
        print(f"RSS Before Concurrency: {c_before:.2f} MB | Peak/After: {c_after:.2f} MB")
        print(f"ModelRegistry counts after concurrency: {ModelRegistry.get_init_counts()}")

        print("\n============================================================")
        print("SUMMARY TABLE:")
        print("============================================================")
        for s in [res_a1, res_b1, res_c1, res_d1, res_e20, res_f5]:
            print(f"{s['label']:<45} | Start: {s['rss_before_mb']:>6.1f}MB | Peak: {s['rss_peak_mb']:>6.1f}MB | End: {s['rss_after_mb']:>6.1f}MB | Growth: {s['rss_growth_mb']:>+5.1f}MB | Latency: {s['avg_latency_ms']:>6.1f}ms | Counts: {s['init_counts_after']}")


if __name__ == "__main__":
    run_full_memory_profile()
