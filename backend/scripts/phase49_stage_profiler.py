"""Phase 49 Fine-Grained Memory Stage Profiler.

Profiles RSS at every sub-operation:
1. Import Baseline
2. Pipeline Instantiation (HistGradientBoosting + Scaler)
3. ModelRegistry / DeBERTa NLI Loading (Model weights, Tokenizer, ONNX/PyTorch)
4. Wikipedia Retrieval & Page Parsing
5. BM25 / Vector Index Creation/Search
6. Tokenization & Tensor Allocation
7. NLI Inference Batching & Softmax
8. P2 Predictive Confidence Calculation
9. P3 Intra-response / Cross-generation Analysis
10. Adaptive Fusion & 19-Feature Assembly
11. Tracer Logging & JSON Persistence
12. Response Serialization
13. Post-Request Garbage Collection & Trimming
"""

import gc
import os
import sys
import time
from pathlib import Path
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force environment bounds before any ML imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

process = psutil.Process(os.getpid())

def get_rss() -> float:
    return process.memory_info().rss / (1024.0 * 1024.0)

def log_stage(name: str, prev_rss: float) -> float:
    curr = get_rss()
    delta = curr - prev_rss
    sign = "+" if delta >= 0 else ""
    print(f"[{name:<45}] RSS: {curr:7.2f} MB | Delta: {sign}{delta:6.2f} MB")
    return curr

def run_stage_profiling():
    print("=" * 80)
    print("PHASE 49: FINE-GRAINED RESIDENT SET SIZE (RSS) STAGE PROFILER")
    print("=" * 80)
    
    r0 = get_rss()
    print(f"Initial Process RSS (Python runtime): {r0:.2f} MB")

    # Step 1: Core imports
    import torch
    r = log_stage("After 'import torch'", r0)
    
    import transformers
    r = log_stage("After 'import transformers'", r)
    
    import sentence_transformers
    r = log_stage("After 'import sentence_transformers'", r)
    
    import joblib
    import sklearn
    r = log_stage("After 'import sklearn & joblib'", r)

    # Step 2: Frozen Model & Preprocessor loading
    from app.core.engine.model_registry import ModelRegistry
    from app.core.engine.pipeline import HallucinationDetectionPipeline
    
    pipeline = HallucinationDetectionPipeline()
    r = log_stage("Pipeline Object Instantiated (Classifier+Pillars)", r)

    # Step 3: DeBERTa NLI Model Loading
    tokenizer, nli_model = ModelRegistry.get_nli_model()
    r = log_stage("DeBERTa NLI Model Loaded in Memory", r)

    # Step 4: Test Requests Stage-by-Stage
    test_prompts = [
        ("Paris is the capital of France.", "What is the capital of France?"),
        ("The Amazon River is the largest river by discharge volume in the world. Berlin is the capital of France.", "Tell me about rivers and capitals."),
        ("12 multiplied by 8 equals 96.", "Math calculation"),
    ]

    for idx, (text, query) in enumerate(test_prompts, 1):
        print("\n" + "-" * 80)
        print(f"ANALYSIS RUN #{idx}: '{text[:60]}...'")
        print("-" * 80)
        
        r_start = get_rss()
        
        # Sub-stage A: Claim extraction
        claims = pipeline.p1_engine.extract_claims(text)
        r_claims = log_stage("  1. P1 Claim Extraction", r_start)
        
        # Sub-stage B: Retrieval
        ev_items = pipeline._retrieve_evidence(text, query=query)
        r_ret = log_stage(f"  2. Hybrid Retrieval ({len(ev_items)} items)", r_claims)
        
        # Sub-stage C: P1 NLI Evaluation
        p1_res = pipeline.p1_engine.analyze(text, ev_items, query=query)
        r_p1 = log_stage("  3. P1 NLI Verification", r_ret)
        
        # Sub-stage D: P2 Confidence
        p2_res = pipeline.p2_engine.analyze(text, query=query)
        r_p2 = log_stage("  4. P2 Predictive Confidence", r_p1)
        
        # Sub-stage E: P3 Consistency
        p3_res = pipeline.p3_engine.analyze(text)
        r_p3 = log_stage("  5. P3 Consistency Reasoning", r_p2)
        
        # Sub-stage F: Full Pipeline Execution (End-to-End)
        report = pipeline.analyze(text, query=query, evidence_items=ev_items)
        r_full = log_stage("  6. Full Pipeline Analyze & Feature Fusion", r_p3)
        
        # Sub-stage G: Post GC & Trim
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        r_clean = log_stage("  7. Post-Request GC & Trim", r_full)
        
        print(f"  Run #{idx} Net RSS Change: {r_clean - r_start:+.2f} MB (Peak in run: {r_full:.2f} MB)")

if __name__ == "__main__":
    run_stage_profiling()
