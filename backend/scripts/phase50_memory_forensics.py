"""Phase 50 Master Memory Forensics & Checkpoint Instrumentation Harness.

Instruments every stage:
- START
- IMPORT_COMPLETE
- APPLICATION_CREATED
- MODEL_REGISTRY_CREATED
- MODELS_WARM
- BEFORE_REQUEST
- AFTER_P1
- AFTER_P2
- AFTER_P3
- AFTER_FUSION
- AFTER_RESPONSE_SERIALIZATION
- AFTER_GC
- AFTER_MALLOC_TRIM

Measures:
- RSS, USS, PSS, VMS via psutil
- Python heap via tracemalloc (current & peak)
- gc.get_objects() count
- Thread count
- ModelRegistry instances
- Trace files count
"""

import gc
import os
import sys
import time
import json
import tracemalloc
import psutil
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure single-thread OpenMP/MKL
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

process = psutil.Process(os.getpid())
tracemalloc.start()

REPORTS_DIR = Path("backend/reports/phase50")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def capture_checkpoint(name: str) -> Dict[str, Any]:
    mem_info = process.memory_info()
    rss_mb = round(mem_info.rss / (1024.0 * 1024.0), 2)
    vms_mb = round(mem_info.vms / (1024.0 * 1024.0), 2)
    
    # Try capturing USS / PSS if available on OS
    uss_mb = None
    pss_mb = None
    try:
        full_mem = process.memory_full_info()
        uss_mb = round(getattr(full_mem, "uss", 0) / (1024.0 * 1024.0), 2)
        pss_mb = round(getattr(full_mem, "pss", 0) / (1024.0 * 1024.0), 2) if hasattr(full_mem, "pss") else None
    except Exception:
        pass

    tm_curr, tm_peak = tracemalloc.get_traced_memory()
    tm_curr_mb = round(tm_curr / (1024.0 * 1024.0), 2)
    tm_peak_mb = round(tm_peak / (1024.0 * 1024.0), 2)

    obj_count = len(gc.get_objects())
    num_threads = process.num_threads()

    checkpoint_data = {
        "checkpoint": name,
        "timestamp_unix": time.time(),
        "rss_mb": rss_mb,
        "uss_mb": uss_mb,
        "pss_mb": pss_mb,
        "vms_mb": vms_mb,
        "tracemalloc_current_mb": tm_curr_mb,
        "tracemalloc_peak_mb": tm_peak_mb,
        "gc_objects_count": obj_count,
        "thread_count": num_threads,
    }
    return checkpoint_data


def run_memory_forensics():
    print("=" * 80)
    print("PHASE 50: COMPREHENSIVE MEMORY FORENSICS & CHECKPOINT PROFILING")
    print("=" * 80)

    checkpoints: List[Dict[str, Any]] = []

    # Checkpoint 1: START
    checkpoints.append(capture_checkpoint("START"))
    print(f"[START]                        RSS: {checkpoints[-1]['rss_mb']} MB | Tracemalloc: {checkpoints[-1]['tracemalloc_current_mb']} MB")

    # Checkpoint 2: IMPORT_COMPLETE
    import torch
    import transformers
    import fastapi
    from app.core.config import settings
    from app.core.engine.model_registry import ModelRegistry
    from app.core.engine.pipeline import HallucinationDetectionPipeline
    from app.main import create_application
    from fastapi.testclient import TestClient

    checkpoints.append(capture_checkpoint("IMPORT_COMPLETE"))
    print(f"[IMPORT_COMPLETE]              RSS: {checkpoints[-1]['rss_mb']} MB | Tracemalloc: {checkpoints[-1]['tracemalloc_current_mb']} MB")

    # Checkpoint 3: APPLICATION_CREATED
    app = create_application()
    client = TestClient(app)
    checkpoints.append(capture_checkpoint("APPLICATION_CREATED"))
    print(f"[APPLICATION_CREATED]          RSS: {checkpoints[-1]['rss_mb']} MB | Tracemalloc: {checkpoints[-1]['tracemalloc_current_mb']} MB")

    # Checkpoint 4: MODEL_REGISTRY_CREATED
    tokenizer, nli_model = ModelRegistry.get_nli_model()
    checkpoints.append(capture_checkpoint("MODEL_REGISTRY_CREATED"))
    print(f"[MODEL_REGISTRY_CREATED]       RSS: {checkpoints[-1]['rss_mb']} MB | Tracemalloc: {checkpoints[-1]['tracemalloc_current_mb']} MB")

    # Checkpoint 5: MODELS_WARM
    pipeline = HallucinationDetectionPipeline()
    _ = pipeline.analyze("Warmup query to prime weights and buffers.")
    checkpoints.append(capture_checkpoint("MODELS_WARM"))
    print(f"[MODELS_WARM]                  RSS: {checkpoints[-1]['rss_mb']} MB | Tracemalloc: {checkpoints[-1]['tracemalloc_current_mb']} MB")

    # Canonical Test Claims A through G
    canonical_requests = [
        ("A", "The capital of France is Paris.", "What is the capital of France?"),
        ("B", "The capital of France is Berlin.", "What is the capital of France?"),
        ("C", "What is the capital of France?", "Question query"),
        ("D", "Paris is the capital of France. Berlin is the capital of France.", "Name the capitals."),
        ("E", "Paris is the capital of France. Berlin is the capital of Germany.", "Name the capitals."),
        ("F", "12 multiplied by 8 equals 96.", "Math"),
        ("G", "12 multiplied by 8 equals 95.", "Math"),
    ]

    request_profiles = []

    print("\n" + "-" * 80)
    print("STAGE-BY-STAGE CANONICAL REQUEST PROFILES (CLAIMS A -> G)")
    print("-" * 80)

    for req_id, text, query in canonical_requests:
        req_chk: Dict[str, Any] = {"request_id": req_id, "text": text}

        # Checkpoint: BEFORE_REQUEST
        c_before = capture_checkpoint(f"REQ_{req_id}_BEFORE")
        req_chk["before"] = c_before

        # Run P1
        t_p1_0 = time.perf_counter()
        ev_items = pipeline._retrieve_evidence(text, query=query)
        p1_res = pipeline.p1_engine.analyze(text, ev_items, query=query)
        c_p1 = capture_checkpoint(f"REQ_{req_id}_AFTER_P1")
        req_chk["after_p1"] = c_p1

        # Run P2
        t_p2_0 = time.perf_counter()
        p2_res = pipeline.p2_engine.analyze(text)
        c_p2 = capture_checkpoint(f"REQ_{req_id}_AFTER_P2")
        req_chk["after_p2"] = c_p2

        # Run P3
        t_p3_0 = time.perf_counter()
        p3_res = pipeline.p3_engine.analyze(text)
        c_p3 = capture_checkpoint(f"REQ_{req_id}_AFTER_P3")
        req_chk["after_p3"] = c_p3

        # Run Fusion
        report = pipeline.analyze(text, query=query, evidence_items=ev_items)
        c_fusion = capture_checkpoint(f"REQ_{req_id}_AFTER_FUSION")
        req_chk["after_fusion"] = c_fusion

        # Test API response serialization
        api_resp = client.post("/api/v1/analyze", json={"response": text, "query": query})
        c_ser = capture_checkpoint(f"REQ_{req_id}_AFTER_SERIALIZATION")
        req_chk["after_serialization"] = c_ser

        # GC & Malloc Trim
        gc.collect()
        c_gc = capture_checkpoint(f"REQ_{req_id}_AFTER_GC")
        req_chk["after_gc"] = c_gc

        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        c_trim = capture_checkpoint(f"REQ_{req_id}_AFTER_MALLOC_TRIM")
        req_chk["after_malloc_trim"] = c_trim

        req_chk["p1_status"] = "EXECUTED" if p1_res.factual_error_score is not None else "UNAVAILABLE"
        req_chk["p2_status"] = p2_res.status
        req_chk["p2_mode"] = p2_res.mode
        req_chk["p3_status"] = p3_res.status
        req_chk["p3_mode"] = p3_res.mode
        req_chk["p3_cf"] = p3_res.consistency_failure_score
        req_chk["h_score"] = report.overall_h_score
        req_chk["risk_level"] = report.overall_risk_level.value

        request_profiles.append(req_chk)
        print(f"Req {req_id}: RSS={c_ser['rss_mb']:6.2f} MB | Tracemalloc={c_ser['tracemalloc_current_mb']:5.2f} MB | H={report.overall_h_score:.4f} | P2={p2_res.mode} | P3={p3_res.mode} (CF={p3_res.consistency_failure_score})")

    # Persist Final Forensics Payload
    final_payload = {
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "pid": os.getpid(),
        },
        "lifecycle_checkpoints": checkpoints,
        "canonical_request_profiles": request_profiles,
        "model_registry_init_counts": ModelRegistry.get_init_counts(),
    }

    out_file = REPORTS_DIR / "PHASE50_MEMORY_FORENSICS.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Persisted Phase 50 Memory Forensics Data to: {out_file}")
    print("=" * 80)

if __name__ == "__main__":
    run_memory_forensics()
