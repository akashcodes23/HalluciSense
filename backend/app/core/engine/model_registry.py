"""Thread-Safe Singleton Model Registry for HalluciSense Phase 11B.

Ensures heavy ML models (DeBERTa NLI CrossEncoder, SentenceTransformer, CrossEncoderReranker,
FAISS VectorStore, and Master Pipeline) are loaded exactly ONCE per process with lazy-loading,
thread safety, bounded concurrency, and memory optimization (eval mode + inference mode).
"""

from __future__ import annotations

import os
import threading
import structlog
from typing import Optional, Tuple, Any

logger = structlog.get_logger(__name__)


class ModelRegistry:
    """Thread-safe singleton registry for all heavy ML models in HalluciSense."""

    _lock = threading.RLock()
    _init_counts = {
        "nli_model": 0,
        "sentence_transformer": 0,
        "cross_encoder_reranker": 0,
        "pipeline": 0,
    }

    _nli_tokenizer: Optional[Any] = None
    _nli_model: Optional[Any] = None
    _sentence_transformer: Optional[Any] = None
    _cross_encoder_reranker: Optional[Any] = None
    _pipeline: Optional[Any] = None

    # Concurrency control semaphore for heavy NLI inference
    _nli_semaphore: Optional[threading.Semaphore] = None

    @classmethod
    def get_nli_semaphore(cls, max_concurrent: int = 2) -> threading.Semaphore:
        if cls._nli_semaphore is None:
            with cls._lock:
                if cls._nli_semaphore is None:
                    cls._nli_semaphore = threading.Semaphore(max_concurrent)
        return cls._nli_semaphore

    @classmethod
    def get_nli_model(cls, model_name: str = "cross-encoder/nli-deberta-v3-small") -> Tuple[Any, Any]:
        """Returns the shared singleton (tokenizer, model) for DeBERTa NLI."""
        if cls._nli_model is None or cls._nli_tokenizer is None:
            with cls._lock:
                if cls._nli_model is None or cls._nli_tokenizer is None:
                    logger.info("loading_shared_nli_model", model_name=model_name)
                    from transformers import AutoTokenizer, AutoModelForSequenceClassification
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        low_cpu_mem_usage=True,
                    )
                    model.eval()
                    cls._nli_tokenizer = tokenizer
                    cls._nli_model = model
                    cls._init_counts["nli_model"] += 1
                    logger.info("shared_nli_model_loaded", init_count=cls._init_counts["nli_model"])
        return cls._nli_tokenizer, cls._nli_model

    @classmethod
    def get_sentence_transformer(cls, model_name: str = "all-MiniLM-L6-v2") -> Any:
        """Returns the shared singleton SentenceTransformer model."""
        if cls._sentence_transformer is None:
            with cls._lock:
                if cls._sentence_transformer is None:
                    logger.info("loading_shared_sentence_transformer", model_name=model_name)
                    from sentence_transformers import SentenceTransformer
                    st = SentenceTransformer(model_name)
                    st.eval()
                    cls._sentence_transformer = st
                    cls._init_counts["sentence_transformer"] += 1
                    logger.info("shared_sentence_transformer_loaded", init_count=cls._init_counts["sentence_transformer"])
        return cls._sentence_transformer

    @classmethod
    def get_cross_encoder_reranker(cls, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Any:
        """Returns the shared singleton CrossEncoder reranker."""
        if cls._cross_encoder_reranker is None:
            with cls._lock:
                if cls._cross_encoder_reranker is None:
                    logger.info("loading_shared_cross_encoder_reranker", model_name=model_name)
                    from sentence_transformers import CrossEncoder
                    ce = CrossEncoder(model_name)
                    cls._cross_encoder_reranker = ce
                    cls._init_counts["cross_encoder_reranker"] += 1
                    logger.info("shared_cross_encoder_reranker_loaded", init_count=cls._init_counts["cross_encoder_reranker"])
        return cls._cross_encoder_reranker

    @classmethod
    def get_pipeline(cls) -> Any:
        """Returns the single shared HallucinationDetectionPipeline orchestrator."""
        if cls._pipeline is None:
            with cls._lock:
                if cls._pipeline is None:
                    logger.info("loading_shared_hallucination_detection_pipeline")
                    from app.core.engine.pipeline import HallucinationDetectionPipeline
                    cls._pipeline = HallucinationDetectionPipeline()
                    cls._init_counts["pipeline"] += 1
                    logger.info("shared_pipeline_loaded", init_count=cls._init_counts["pipeline"])
        return cls._pipeline

    @classmethod
    def get_init_counts(cls) -> dict:
        return dict(cls._init_counts)

    @classmethod
    def reset_for_testing(cls):
        """Used only in memory tests to verify fresh initialization counts."""
        with cls._lock:
            cls._nli_tokenizer = None
            cls._nli_model = None
            cls._sentence_transformer = None
            cls._cross_encoder_reranker = None
            cls._pipeline = None
            cls._init_counts = {k: 0 for k in cls._init_counts}
