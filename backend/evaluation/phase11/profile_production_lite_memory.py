"""
Production Memory Profiler for HalluciSense Memory-Safe Profile.

Measures:
1. Startup RSS
2. Post-pipeline RSS
3. Post-first-verification RSS
4. Peak RSS across 10 closed-loop requests
5. Model initialization counts
6. Mean latency
"""
import os
import gc
import json
import time
import psutil
from typing import Dict, Any

from app.core.config import settings
from app.core.engine.model_registry import ModelRegistry
from app.core.engine.types import EvidenceItem


def get_current_rss_mb() -> float:
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024.0 * 1024.0), 2)


def main():
    print("=" * 70)
    print("HALLUCISENSE PRODUCTION MEMORY PROFILE")
    print("=" * 70)

    # 1. Startup RSS
    rss_startup = get_current_rss_mb()
    print(f"Startup RSS: {rss_startup} MB")

    # 2. Pipeline Initialization
    ModelRegistry.reset_for_testing()
    t0 = time.perf_counter()
    pipeline = ModelRegistry.get_pipeline()
    init_ms = (time.perf_counter() - t0) * 1000.0
    rss_post_pipeline = get_current_rss_mb()
    print(f"Post-Pipeline RSS: {rss_post_pipeline} MB (+{rss_post_pipeline - rss_startup:.2f} MB, {init_ms:.2f} ms)")

    # 3. First Verification Request
    ev = [
        EvidenceItem(
            claim="Speed of light",
            snippet="The speed of light in vacuum is defined as exactly 299792458 meters per second (m/s).",
            source_name="Wikipedia: Speed of light",
            similarity_score=0.95,
            is_supporting=True,
        )
    ]
    t0 = time.perf_counter()
    res1 = pipeline.analyze_response(
        full_text="The speed of light in vacuum is approximately 299,792,458 m/s.",
        query="What is the speed of light in vacuum?",
        evidence_items=ev,
        sample_responses=[],
    )
    t_first_ms = (time.perf_counter() - t0) * 1000.0
    rss_first_verif = get_current_rss_mb()
    print(f"Post-First Verification RSS: {rss_first_verif} MB (Latency: {t_first_ms:.2f} ms, H-Score: {res1.overall_h_score})")

    # 4. Multi-Request Load (10 requests)
    latencies = []
    peak_rss = rss_first_verif
    test_queries = [
        ("What is the speed of light in vacuum?", "The speed of light in vacuum is approximately 299,792,458 m/s."),
        ("What is the chemical formula of water?", "Water has the chemical formula H2O."),
        ("What is standard gravity?", "Standard acceleration due to gravity on Earth is 9.8 m/s²."),
        ("What is the speed of light in vacuum?", "The speed of light in vacuum is approximately 299,792,458 km/s."),
        ("What is the formula of water?", "The chemical formula of water is CO2."),
    ] * 2

    for i, (q, txt) in enumerate(test_queries, 1):
        t_req0 = time.perf_counter()
        _ = pipeline.analyze_response(full_text=txt, query=q, sample_responses=[])
        lat = (time.perf_counter() - t_req0) * 1000.0
        latencies.append(lat)
        curr_rss = get_current_rss_mb()
        if curr_rss > peak_rss:
            peak_rss = curr_rss

    mean_lat = round(sum(latencies) / len(latencies), 2)
    counts = ModelRegistry.get_init_counts()

    profile_data: Dict[str, Any] = {
        "startup_rss_mb": rss_startup,
        "post_pipeline_rss_mb": rss_post_pipeline,
        "post_first_verification_rss_mb": rss_first_verif,
        "peak_rss_mb": peak_rss,
        "model_counts": counts,
        "first_request_latency_ms": round(t_first_ms, 2),
        "mean_latency_ms": mean_lat,
        "phase11c_peak_rss_mb": 1049.98,
        "rss_reduction_pct": round((1049.98 - peak_rss) / 1049.98 * 100, 2),
    }

    print("-" * 70)
    print(f"Peak RSS: {peak_rss} MB (Phase 11C baseline: 1049.98 MB, reduction: {profile_data['rss_reduction_pct']}%)")
    print(f"Model Init Counts: {counts}")
    print(f"Mean Latency: {mean_lat} ms")
    print("=" * 70)

    out_path = "backend/reports/phase11/production_memory_profile_lite.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2)
    print(f"Saved memory profile to {out_path}")


if __name__ == "__main__":
    main()
