"""Phase 50 — Strict Model Singleton & Memory Cleanliness Tests."""

import pytest
from app.core.engine.model_registry import ModelRegistry
from app.core.engine.pipeline import HallucinationDetectionPipeline


def test_model_registry_singleton_identity():
    """Verify exactly 1 NLI model is initialized in process memory."""
    pipeline = HallucinationDetectionPipeline()
    _ = pipeline.analyze("The capital of France is Paris.")
    
    counts = ModelRegistry.get_init_counts()
    assert counts.get("nli_model", 0) == 1
    assert counts.get("sentence_transformer", 0) == 0
    assert counts.get("cross_encoder_reranker", 0) == 0


def test_shared_tokenizer_and_nli_instances():
    """Verify tokenizer and NLI model are shared across all pipeline invocations."""
    t1, m1 = ModelRegistry.get_nli_model()
    t2, m2 = ModelRegistry.get_nli_model()

    assert id(t1) == id(t2)
    assert id(m1) == id(m2)
