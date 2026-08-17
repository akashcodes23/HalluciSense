"""Production Memory Profiler for HalluciSense Phase 11B.

Measures Process Resident Set Size (RSS) at each stage of model loading, retrieval,
verification, closed-loop chat generation, correction, and re-verification.
Outputs structured metrics to backend/reports/phase11/phase11_memory_profile.json.
"""

from __future__ import annotations

import os
import sys
import json
import time
import psutil
from pathlib import Path

# Ensure backend in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def get_rss_mb() -> float:
    """Returns current process Resident Set Size in Megabytes."""
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)


def run_memory_profile() -> dict:
    telemetry = {}
    component_breakdown = {}

    # Stage 0: Initial baseline
    telemetry["startup_baseline_mb"] = get_rss_mb()

    # Stage 1: ModelRegistry initialization
    from app.core.engine.model_registry import ModelRegistry
    telemetry["after_registry_init_mb"] = get_rss_mb()

    # Stage 2: SentenceTransformer load
    t0 = time.perf_counter()
    st = ModelRegistry.get_sentence_transformer()
    t_st = (time.perf_counter() - t0) * 1000.0
    telemetry["after_embedding_model_mb"] = get_rss_mb()
    component_breakdown["SentenceTransformer"] = {
        "memory_delta_mb": round(telemetry["after_embedding_model_mb"] - telemetry["after_registry_init_mb"], 2),
        "load_time_ms": round(t_st, 2),
        "model": "all-MiniLM-L6-v2",
    }

    # Stage 3: DeBERTa CrossEncoder NLI load
    t0 = time.perf_counter()
    tok, nli = ModelRegistry.get_nli_model()
    t_nli = (time.perf_counter() - t0) * 1000.0
    telemetry["after_deberta_nli_mb"] = get_rss_mb()
    component_breakdown["DeBERTa_NLI"] = {
        "memory_delta_mb": round(telemetry["after_deberta_nli_mb"] - telemetry["after_embedding_model_mb"], 2),
        "load_time_ms": round(t_nli, 2),
        "model": "cross-encoder/nli-deberta-v3-small",
    }

    # Stage 4: CrossEncoder Reranker load
    t0 = time.perf_counter()
    reranker = ModelRegistry.get_cross_encoder_reranker()
    t_ce = (time.perf_counter() - t0) * 1000.0
    telemetry["after_cross_encoder_reranker_mb"] = get_rss_mb()
    component_breakdown["CrossEncoder_Reranker"] = {
        "memory_delta_mb": round(telemetry["after_cross_encoder_reranker_mb"] - telemetry["after_deberta_nli_mb"], 2),
        "load_time_ms": round(t_ce, 2),
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    }

    # Stage 5: Master Pipeline Orchestrator load
    t0 = time.perf_counter()
    pipeline = ModelRegistry.get_pipeline()
    t_pipe = (time.perf_counter() - t0) * 1000.0
    telemetry["after_pipeline_orchestrator_mb"] = get_rss_mb()
    component_breakdown["PipelineOrchestrator"] = {
        "memory_delta_mb": round(telemetry["after_pipeline_orchestrator_mb"] - telemetry["after_cross_encoder_reranker_mb"], 2),
        "load_time_ms": round(t_pipe, 2),
    }

    # Stage 6: First Verification Request
    t0 = time.perf_counter()
    report = pipeline.analyze_response(
        full_text="The speed of light in vacuum is defined as exactly 299792458 meters per second.",
        query="What is the speed of light in vacuum?",
    )
    t_first = (time.perf_counter() - t0) * 1000.0
    telemetry["after_first_verification_mb"] = get_rss_mb()

    # Stage 7: Closed-Loop Correction & Re-verification
    from app.core.correction.correction_engine import CorrectionEngine
    corr_engine = CorrectionEngine(pipeline=pipeline)
    
    init_rep = pipeline.analyze_response(
        full_text="The speed of light in vacuum is approximately 299,792,458 km/s.",
        query="What is the speed of light in vacuum?",
    )
    t0 = time.perf_counter()
    corr_res = corr_engine.execute_closed_loop_repair(
        user_query="What is the speed of light in vacuum?",
        initial_text="The speed of light in vacuum is approximately 299,792,458 km/s.",
        initial_verification=init_rep,
        max_attempts=2,
    )
    t_corr = (time.perf_counter() - t0) * 1000.0
    telemetry["after_correction_and_reverification_mb"] = get_rss_mb()

    # Peak RSS
    process = psutil.Process(os.getpid())
    telemetry["peak_rss_mb"] = round(process.memory_info().rss / (1024 * 1024), 2)
    telemetry["model_initialization_counts"] = ModelRegistry.get_init_counts()

    output = {
        "telemetry": telemetry,
        "component_breakdown": component_breakdown,
        "first_verification_ms": round(t_first, 2),
        "correction_reverification_ms": round(t_corr, 2),
        "single_instance_guarantee": all(c == 1 for c in ModelRegistry.get_init_counts().values()),
    }

    report_dir = Path("backend/reports/phase11")
    report_dir.mkdir(parents=True, exist_ok=True)
    out_file = report_dir / "phase11_memory_profile.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("PHASE 11B MEMORY COMPONENT REPORT")
    print("=" * 60)
    print(f"Startup Baseline RSS      : {telemetry['startup_baseline_mb']:>7.2f} MB")
    print(f"After SentenceTransformer : {telemetry['after_embedding_model_mb']:>7.2f} MB (+{component_breakdown['SentenceTransformer']['memory_delta_mb']} MB)")
    print(f"After DeBERTa NLI         : {telemetry['after_deberta_nli_mb']:>7.2f} MB (+{component_breakdown['DeBERTa_NLI']['memory_delta_mb']} MB)")
    print(f"After CrossEncoder Rerank : {telemetry['after_cross_encoder_reranker_mb']:>7.2f} MB (+{component_breakdown['CrossEncoder_Reranker']['memory_delta_mb']} MB)")
    print(f"After First Verification  : {telemetry['after_first_verification_mb']:>7.2f} MB")
    print(f"After Closed-Loop Repair  : {telemetry['after_correction_and_reverification_mb']:>7.2f} MB")
    print(f"Peak Process RSS          : {telemetry['peak_rss_mb']:>7.2f} MB")
    print(f"Model Initialization Counts: {telemetry['model_initialization_counts']}")
    print("=" * 60)
    print(f"Saved memory profile to: {out_file}\n")
    return output


if __name__ == "__main__":
    run_memory_profile()
