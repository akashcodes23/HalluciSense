"""Targeted Memory Stage Profiler for HalluciSense."""

import os
import sys
import psutil
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.engine.model_registry import ModelRegistry
from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.core.engine.memory_utils import trim_process_memory

def profile_stages():
    proc = psutil.Process(os.getpid())
    print("--- Detailed Stage-by-Stage Memory Profiling ---")
    
    trim_process_memory()
    m0 = proc.memory_info().rss / (1024 * 1024)
    print(f"Base RSS: {m0:.2f} MB")

    pipeline = ModelRegistry.get_pipeline()
    trim_process_memory()
    m1 = proc.memory_info().rss / (1024 * 1024)
    print(f"After ModelRegistry.get_pipeline(): {m1:.2f} MB (Delta: +{m1-m0:.2f} MB)")

    # Execute 10 requests and check each component
    for i in range(10):
        print(f"\n--- Request #{i+1} ---")
        text = f"Iteration {i}: The capital of France is Paris. The capital of Germany is Berlin."
        
        # 1. Retrieval
        t_m0 = proc.memory_info().rss / (1024 * 1024)
        evidence = pipeline._retrieve_evidence(text)
        t_m1 = proc.memory_info().rss / (1024 * 1024)
        print(f"  1. Retrieval:           RSS={t_m1:.2f} MB (Delta: +{t_m1 - t_m0:.2f} MB) [Evidence: {len(evidence)} items]")

        # 2. P1 Document-Level Factual Verification
        p1_res = pipeline.p1_engine.analyze(text, evidence)
        t_m2 = proc.memory_info().rss / (1024 * 1024)
        print(f"  2. P1 Factual NLI:      RSS={t_m2:.2f} MB (Delta: +{t_m2 - t_m1:.2f} MB)")

        # 3. P2 Confidence
        p2_res = pipeline.p2_engine.analyze(text.split(), None, evidence_items=evidence, p1_result=p1_res)
        t_m3 = proc.memory_info().rss / (1024 * 1024)
        print(f"  3. P2 Confidence:       RSS={t_m3:.2f} MB (Delta: +{t_m3 - t_m2:.2f} MB)")

        # 4. P3 Consistency
        p3_res = pipeline.p3_engine.analyze(text, [])
        t_m4 = proc.memory_info().rss / (1024 * 1024)
        print(f"  4. P3 Consistency:      RSS={t_m4:.2f} MB (Delta: +{t_m4 - t_m3:.2f} MB)")

        # 5. Fusion
        f_res = pipeline.fusion_engine.fuse(p1_res, p2_res, p3_res)
        t_m5 = proc.memory_info().rss / (1024 * 1024)
        print(f"  5. Fusion:              RSS={t_m5:.2f} MB (Delta: +{t_m5 - t_m4:.2f} MB)")

        # 6. Local Attribution (if in router)
        trim_process_memory()
        t_end = proc.memory_info().rss / (1024 * 1024)
        print(f"  --> Post-trim RSS:      RSS={t_end:.2f} MB (Delta vs Req Start: +{t_end - t_m0:.2f} MB)")

if __name__ == "__main__":
    profile_stages()
