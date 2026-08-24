"""
Comprehensive Memory Component Profiler for HalluciSense Phase 11.

Executes granular diagnostic profiling of:
1. Individual component RSS deltas (imports, tokenizer, model loading, engines, pipeline, inferences, gc).
2. Model parameter count, dtype, parameter memory in MB, and device.
3. Device availability (CUDA, MPS, CPU).
4. PyTorch thread count comparison (threads=1 vs default).
5. Batch size comparison (batch_size=4, 8, 16, 32) on peak RSS, latency, and outputs.
6. Verification of torch.inference_mode() and model.training == False.
7. Model object identity and memory duplication checks.
8. Tracemalloc and object retention analysis across 10 verifications.
9. Exports structured report to backend/reports/phase11/memory_component_profile.json.
"""
import os
import gc
import sys
import time
import json
import psutil
import tracemalloc
from typing import Dict, Any, List, Tuple


def get_rss_mb() -> float:
    """Returns exact current resident set size in megabytes."""
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0), 2)


def main():
    print("=" * 80)
    print("HALLUCISENSE MEMORY COMPONENT PROFILER & DIAGNOSTIC SUITE")
    print("=" * 80)

    tracemalloc.start()
    report: Dict[str, Any] = {}
    step_timings: List[Dict[str, Any]] = []

    def record_step(name: str, fn):
        nonlocal step_timings
        rss_before = get_rss_mb()
        t0 = time.perf_counter()
        result = fn()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        rss_after = get_rss_mb()
        delta_mb = round(rss_after - rss_before, 2)
        step_timings.append({
            "step": name,
            "rss_mb": rss_after,
            "delta_mb": delta_mb,
            "elapsed_ms": round(elapsed_ms, 2),
        })
        print(f"[{name:<45}] RSS: {rss_after:7.2f} MB | Delta: {delta_mb:+7.2f} MB | Time: {elapsed_ms:7.2f} ms")
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Step-by-Step Individual Import and Loading Profiling
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 1. Individual Component Profiling ---")

    # A. Initial baseline
    rss_base = get_rss_mb()
    step_timings.append({"step": "A. Baseline Process State", "rss_mb": rss_base, "delta_mb": 0.0, "elapsed_ms": 0.0})
    print(f"[{'A. Baseline Process State':<45}] RSS: {rss_base:7.2f} MB | Delta: +0.00 MB | Time:    0.00 ms")

    # B. import torch
    def step_import_torch():
        import torch
        return torch
    torch = record_step("B. import torch", step_import_torch)

    # C. import transformers
    def step_import_transformers():
        import transformers
        return transformers
    transformers = record_step("C. import transformers", step_import_transformers)

    # D. ModelRegistry before models
    def step_import_registry():
        from app.core.engine.model_registry import ModelRegistry
        ModelRegistry.reset_for_testing()
        return ModelRegistry
    ModelRegistry = record_step("D. import ModelRegistry", step_import_registry)

    # E. Load tokenizer only
    def step_load_tokenizer():
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-small")
        return tok
    tokenizer = record_step("E. load tokenizer only", step_load_tokenizer)

    # F. Load DeBERTa model only (raw from_pretrained)
    def step_load_model():
        from transformers import AutoModelForSequenceClassification
        m = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-small")
        return m
    raw_model = record_step("F. load DeBERTa model only", step_load_model)

    # G. model.eval()
    def step_eval():
        raw_model.eval()
        return raw_model
    record_step("G. model.eval()", step_eval)

    # H. model.to(device)
    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_name = "mps"
    device = torch.device(device_name)

    def step_to_device():
        raw_model.to(device)
        return raw_model
    record_step(f"H. model.to({device_name})", step_to_device)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Model Dtype, Parameters, and Memory Footprint
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 2. Model Dtype & Parameter Footprint ---")
    total_params = sum(p.numel() for p in raw_model.parameters())
    trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    param_bytes = sum(p.numel() * p.element_size() for p in raw_model.parameters())
    param_memory_mb = round(param_bytes / (1024.0 * 1024.0), 2)
    first_param = next(raw_model.parameters())
    model_dtype = str(first_param.dtype)
    model_device = str(first_param.device)

    model_metadata = {
        "model_name": "cross-encoder/nli-deberta-v3-small",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "parameter_bytes": param_bytes,
        "parameter_memory_mb": param_memory_mb,
        "dtype": model_dtype,
        "device": model_device,
        "training_mode": raw_model.training,
    }
    report["model_metadata"] = model_metadata
    print(f"Total Parameters: {total_params:,}")
    print(f"Parameter Memory: {param_memory_mb} MB")
    print(f"Parameter Dtype:  {model_dtype}")
    print(f"Model Device:     {model_device}")
    print(f"Model Training:   {raw_model.training}")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Device & PyTorch Runtime Diagnostic
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 3. Device & PyTorch Diagnostic ---")
    device_diagnostic = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "selected_device": str(device),
        "default_torch_threads": torch.get_num_threads(),
        "python_version": sys.version,
    }
    report["device_diagnostic"] = device_diagnostic
    print(f"CUDA Available:   {device_diagnostic['cuda_available']}")
    print(f"MPS Available:    {device_diagnostic['mps_available']}")
    print(f"Selected Device:  {device_diagnostic['selected_device']}")
    print(f"Default Threads:  {device_diagnostic['default_torch_threads']}")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Engine and Pipeline Construction Steps
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 4. Engine & Pipeline Construction ---")

    # Set ModelRegistry singletons so engine construction reuses raw_model
    ModelRegistry._nli_tokenizer = tokenizer
    ModelRegistry._nli_model = raw_model
    ModelRegistry._init_counts["nli_model"] = 1

    def step_entailment_engine():
        from app.core.engine.entailment import EvidenceEntailmentEngine
        return EvidenceEntailmentEngine()
    entailment_engine = record_step("I. construct EvidenceEntailmentEngine", step_entailment_engine)

    def step_p1_engine():
        from app.core.engine.pillar1_retrieval import Pillar1RetrievalEngine
        return Pillar1RetrievalEngine()
    p1_engine = record_step("J. construct Pillar1RetrievalEngine", step_p1_engine)

    def step_hybrid_retriever():
        from app.modules.knowledge.retriever import HybridRetriever
        return HybridRetriever()
    hybrid_retriever = record_step("K. construct HybridRetriever", step_hybrid_retriever)

    def step_pipeline():
        from app.core.engine.pipeline import HallucinationDetectionPipeline
        return HallucinationDetectionPipeline()
    pipeline = record_step("L. construct HallucinationDetectionPipeline", step_pipeline)

    # Check Model Duplication
    duplication_check = {
        "model_registry_nli_id": id(ModelRegistry._nli_model),
        "entailment_model_id": id(entailment_engine.model),
        "p1_model_id": id(p1_engine.entailment_engine.model),
        "are_identical_instances": (id(ModelRegistry._nli_model) == id(entailment_engine.model) == id(p1_engine.entailment_engine.model)),
        "data_pointer_identical": (raw_model.state_dict()["deberta.embeddings.word_embeddings.weight"].data_ptr() == entailment_engine.model.state_dict()["deberta.embeddings.word_embeddings.weight"].data_ptr()),
    }
    report["duplication_check"] = duplication_check
    print(f"\nDuplication Check -> Identical Instances: {duplication_check['are_identical_instances']}, Identical Data Pointers: {duplication_check['data_pointer_identical']}")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Inferences and Memory Cleanups
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 5. Inference & Cleanup Memory Profiling ---")

    sample_claim = "The speed of light in vacuum is approximately 299,792,458 m/s."
    sample_snippet = "The speed of light in vacuum is defined as exactly 299792458 meters per second (m/s)."

    def step_inf_1():
        return entailment_engine.classify(sample_claim, sample_snippet)
    record_step("M. first NLI inference", step_inf_1)

    def step_inf_2():
        return entailment_engine.classify(sample_claim, sample_snippet)
    record_step("N. second NLI inference", step_inf_2)

    def step_gc():
        gc.collect()
        return True
    record_step("O. after gc.collect()", step_gc)

    def step_torch_cleanup():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        return True
    record_step("P. after torch cache cleanup", step_torch_cleanup)

    report["individual_steps"] = step_timings

    # ─────────────────────────────────────────────────────────────────────────
    # 6. PyTorch Thread Count Memory & Latency Comparison
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 6. PyTorch Thread Count Impact ---")
    orig_threads = torch.get_num_threads()
    thread_results = []
    pairs_for_thread_test = [(sample_claim, sample_snippet)] * 8

    for th in [1, 2, 4, orig_threads]:
        torch.set_num_threads(th)
        t_th0 = time.perf_counter()
        rss_th_before = get_rss_mb()
        for c, s in pairs_for_thread_test:
            _ = entailment_engine.classify(c, s)
        elapsed_th = (time.perf_counter() - t_th0) * 1000.0
        rss_th_after = get_rss_mb()
        thread_results.append({
            "threads": th,
            "elapsed_ms": round(elapsed_th, 2),
            "rss_before_mb": rss_th_before,
            "rss_after_mb": rss_th_after,
        })
        print(f"Threads: {th:2d} | 8 Inferences: {elapsed_th:7.2f} ms | RSS: {rss_th_after:7.2f} MB")

    torch.set_num_threads(orig_threads)
    report["thread_count_comparison"] = thread_results

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Batch Size Temporary Memory & Latency Comparison
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 7. Batch Size Memory & Latency Comparison ---")
    batch_comparison = []
    test_pairs_pool = [
        ("Water is H2O.", "Water chemical formula is H2O."),
        ("Speed of light is 300,000 km/s.", "Speed of light is 299792458 m/s."),
        ("Mitochondria produce ATP.", "Mitochondria generate cellular ATP."),
        ("Gravity acceleration is 9.8 m/s².", "Standard gravity on Earth is 9.8 m/s²."),
    ] * 8  # 32 pairs total

    for bs in [4, 8, 16, 32]:
        rss_bs_before = get_rss_mb()
        t_bs0 = time.perf_counter()
        
        # Test batch inference with chunking
        pairs = test_pairs_pool[:32]
        all_results = []
        for i in range(0, len(pairs), bs):
            chunk = pairs[i:i + bs]
            with torch.inference_mode():
                features = tokenizer([c for c, s in chunk], [s for c, s in chunk], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
                logits = raw_model(**features).logits
                probs = torch.softmax(logits, dim=-1)
                all_results.extend(probs.cpu().tolist())

        elapsed_bs = (time.perf_counter() - t_bs0) * 1000.0
        rss_bs_after = get_rss_mb()
        batch_comparison.append({
            "batch_size": bs,
            "total_pairs": len(pairs),
            "elapsed_ms": round(elapsed_bs, 2),
            "rss_before_mb": rss_bs_before,
            "rss_after_mb": rss_bs_after,
            "delta_mb": round(rss_bs_after - rss_bs_before, 2),
        })
        print(f"Batch Size: {bs:2d} | 32 Pairs: {elapsed_bs:7.2f} ms | RSS: {rss_bs_after:7.2f} MB | Delta: {round(rss_bs_after - rss_bs_before, 2):+6.2f} MB")

    report["batch_size_comparison"] = batch_comparison

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Tracemalloc & Object Retention Across 10 Requests
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- 8. Tracemalloc & Retention Over 10 Pipeline Invocations ---")
    retention_rss = []
    test_queries = [
        ("What is the speed of light in vacuum?", "The speed of light in vacuum is approximately 299,792,458 m/s."),
        ("What is the chemical formula of water?", "Water has the chemical formula H2O."),
        ("What is standard gravity?", "Standard acceleration due to gravity on Earth is 9.8 m/s²."),
        ("What is the speed of light in vacuum?", "The speed of light in vacuum is approximately 299,792,458 km/s."),
        ("What is the formula of water?", "The chemical formula of water is CO2."),
    ] * 2

    for req_i, (q, txt) in enumerate(test_queries, 1):
        rss_pre = get_rss_mb()
        _ = pipeline.analyze_response(full_text=txt, query=q, sample_responses=[])
        rss_post = get_rss_mb()
        retention_rss.append({
            "request_index": req_i,
            "rss_mb": rss_post,
            "delta_mb": round(rss_post - rss_pre, 2),
        })

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")
    tracemalloc_top = []
    for stat in top_stats[:10]:
        tracemalloc_top.append({
            "size_kb": round(stat.size / 1024.0, 2),
            "count": stat.count,
            "trace": str(stat.traceback),
        })
    report["retention_over_10_requests"] = retention_rss
    report["tracemalloc_top_10"] = tracemalloc_top

    # ─────────────────────────────────────────────────────────────────────────
    # 9. Summary and Recommendations
    # ─────────────────────────────────────────────────────────────────────────
    peak_rss_total = get_rss_mb()
    report["peak_rss_mb"] = peak_rss_total

    # Find largest consumer in step deltas
    largest_step = max(step_timings, key=lambda x: x["delta_mb"])
    report["largest_memory_consumer"] = {
        "step": largest_step["step"],
        "delta_mb": largest_step["delta_mb"],
        "final_rss_mb": largest_step["rss_mb"],
    }

    out_file = "backend/reports/phase11/memory_component_profile.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print(f"Largest Memory Consumer: {largest_step['step']} (+{largest_step['delta_mb']} MB)")
    print(f"Peak Total RSS:          {peak_rss_total} MB")
    print(f"Saved Full Diagnostic:   {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
