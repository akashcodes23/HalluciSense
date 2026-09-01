"""Phase 49 — Model Singleton & Zero-Duplicate Transformer Integrity Tests."""

import pytest
from app.core.engine.model_registry import ModelRegistry
from app.core.engine.pipeline import HallucinationDetectionPipeline


def test_shared_nli_singleton_identity():
    """Verify Pillar 1 and Pillar 3 share the EXACT same NLI model object."""
    pipeline = HallucinationDetectionPipeline()
    p1_nli = pipeline.p1_engine.entailment_engine.model
    p3_nli = pipeline.p3_engine._get_nli_engine().model

    assert p1_nli is not None
    assert p3_nli is not None
    assert id(p1_nli) == id(p3_nli), "Pillar 1 and Pillar 3 must share the exact same DeBERTa NLI model instance!"


def test_model_registry_init_counts_strict():
    """Verify exactly 1 NLI model is loaded and ZERO duplicate transformers exist."""
    pipeline = HallucinationDetectionPipeline()
    _ = pipeline.analyze("The capital of France is Paris.")

    counts = ModelRegistry.get_init_counts()
    assert counts.get("nli_model", 0) == 1, f"Expected exactly 1 NLI model loaded, got {counts.get('nli_model')}"
    assert counts.get("sentence_transformer", 0) == 0, "SentenceTransformer must NEVER be initialized in production runtime!"
    assert counts.get("cross_encoder_reranker", 0) == 0, "CrossEncoderReranker must NEVER be initialized in production runtime!"
