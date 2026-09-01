"""Granular Step-by-Step Diagnostic of the +430 MB Warmup Jump."""

import os
import gc
import psutil

process = psutil.Process(os.getpid())

def p(step):
    gc.collect()
    rss = process.memory_info().rss / (1024 * 1024)
    print(f"[{step:<40}] RSS = {rss:7.2f} MB")

p("START")

import torch
p("AFTER_IMPORT_TORCH")

import transformers
p("AFTER_IMPORT_TRANSFORMERS")

from app.core.engine.model_registry import ModelRegistry
p("AFTER_IMPORT_MODEL_REGISTRY")

tok, nli = ModelRegistry.get_nli_model()
p("AFTER_LOAD_NLI_MODEL")

from app.core.engine.pipeline import HallucinationDetectionPipeline
pipeline = HallucinationDetectionPipeline()
p("AFTER_INIT_PIPELINE")

# 1. Claim extraction
claims = pipeline.p1_engine.extract_claims("The capital of France is Paris.")
p("AFTER_EXTRACT_CLAIMS")

# 2. Retrieval
ev = pipeline._retrieve_evidence("The capital of France is Paris.")
p("AFTER_RETRIEVE_EVIDENCE")

# 3. P1 Analyze
p1_res = pipeline.p1_engine.analyze("The capital of France is Paris.", ev)
p("AFTER_P1_ANALYZE")

# 4. P2 Analyze
p2_res = pipeline.p2_engine.analyze("The capital of France is Paris.")
p("AFTER_P2_ANALYZE")

# 5. P3 Analyze
p3_res = pipeline.p3_engine.analyze("The capital of France is Paris.")
p("AFTER_P3_ANALYZE")

# 6. Fusion
fusion_res = pipeline.fusion_engine.fuse(p1_res, p2_res, p3_res)
p("AFTER_FUSION")

# 7. Complete Pipeline analyze
rep = pipeline.analyze("The capital of France is Paris.")
p("AFTER_FULL_PIPELINE_ANALYZE")

# 8. ML Classifier in production_router (Shadow classifier / Hybrid meta classifier)
from app.modules.verification.production_router import _get_shadow_classifier, _get_scaler
clf = _get_shadow_classifier()
p("AFTER_LOAD_HYBRID_META_CLASSIFIER")

scaler = _get_scaler()
p("AFTER_LOAD_SCALER")
