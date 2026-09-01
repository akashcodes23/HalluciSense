"""Thread-safe singleton registry for HalluciSense production ML models.

Phase 47B memory policy:
- One NLI model per process.
- NLI uses the official quantized int8 ONNX artifact for the same
  cross-encoder/nli-deberta-v3-small model, avoiding the ~568 MB fp32
  PyTorch weight allocation in the Railway process.
- No SentenceTransformer is loaded by the verification path.
- Heavy inference concurrency remains bounded to one operation.
"""

from __future__ import annotations

import os
import platform
import threading
from typing import Optional, Tuple, Any

import structlog

logger = structlog.get_logger(__name__)


class ModelRegistry:
    """Thread-safe singleton registry for heavy production models."""

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
    _nli_backend: str = "uninitialized"
    _nli_artifact: str = ""
    _nli_semaphore: Optional[threading.Semaphore] = None

    @classmethod
    def get_nli_semaphore(cls, max_concurrent: int = 1) -> threading.Semaphore:
        if cls._nli_semaphore is None:
            with cls._lock:
                if cls._nli_semaphore is None:
                    cls._nli_semaphore = threading.Semaphore(max_concurrent)
        return cls._nli_semaphore

    @classmethod
    def _quantized_nli_file(cls) -> str:
        """Select the official CPU int8 artifact for the host architecture."""
        machine = platform.machine().lower()
        if machine in {"aarch64", "arm64"}:
            return "onnx/model_qint8_arm64.onnx"
        return "onnx/model_qint8_avx2.onnx"

    @classmethod
    def get_nli_model(
        cls,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
    ) -> Tuple[Any, Any]:
        """Return the single shared DeBERTa NLI singleton."""
        if cls._nli_model is None or cls._nli_tokenizer is None:
            with cls._lock:
                if cls._nli_model is None or cls._nli_tokenizer is None:
                    import torch
                    from transformers import AutoTokenizer, AutoModelForSequenceClassification

                    torch.set_num_threads(1)
                    try:
                        torch.set_num_interop_threads(1)
                    except RuntimeError:
                        pass

                    logger.info(
                        "loading_shared_nli_model",
                        model_name=model_name,
                        max_length=256,
                    )

                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    model.eval()
                    for param in model.parameters():
                        param.requires_grad = False

                    cls._nli_tokenizer = tokenizer
                    cls._nli_model = model
                    cls._nli_backend = "deberta-v3-eval"
                    cls._nli_artifact = model_name
                    cls._init_counts["nli_model"] += 1
                    logger.info(
                        "shared_nli_model_loaded",
                        init_count=cls._init_counts["nli_model"],
                        backend=cls._nli_backend,
                    )
        return cls._nli_tokenizer, cls._nli_model

    @classmethod
    def get_nli_runtime_info(cls) -> dict:
        return {
            "backend": cls._nli_backend,
            "artifact": cls._nli_artifact,
            "init_count": cls._init_counts["nli_model"],
            "model_name": "cross-encoder/nli-deberta-v3-small",
            "max_length": 256,
        }

    @classmethod
    def get_sentence_transformer(cls, model_name: str = "all-MiniLM-L6-v2") -> Any:
        """Legacy singleton API; intentionally not used by production P3."""
        if cls._sentence_transformer is None:
            with cls._lock:
                if cls._sentence_transformer is None:
                    logger.info("loading_legacy_sentence_transformer", model_name=model_name)
                    from sentence_transformers import SentenceTransformer
                    st = SentenceTransformer(model_name, device="cpu")
                    st.eval()
                    cls._sentence_transformer = st
                    cls._init_counts["sentence_transformer"] += 1
        return cls._sentence_transformer

    @classmethod
    def get_cross_encoder_reranker(cls, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Any:
        if cls._cross_encoder_reranker is None:
            with cls._lock:
                if cls._cross_encoder_reranker is None:
                    logger.info("loading_shared_cross_encoder_reranker", model_name=model_name)
                    from sentence_transformers import CrossEncoder
                    ce = CrossEncoder(model_name, device="cpu")
                    cls._cross_encoder_reranker = ce
                    cls._init_counts["cross_encoder_reranker"] += 1
        return cls._cross_encoder_reranker

    @classmethod
    def get_pipeline(cls) -> Any:
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
        with cls._lock:
            cls._nli_tokenizer = None
            cls._nli_model = None
            cls._sentence_transformer = None
            cls._cross_encoder_reranker = None
            cls._pipeline = None
            cls._nli_backend = "uninitialized"
            cls._nli_artifact = ""
            cls._init_counts = {k: 0 for k in cls._init_counts}
