"""Thread-safe singleton registry for HalluciSense production ML models.

Phase 47B memory policy:
- One NLI model per process.
- NLI uses the official quantized ONNX artifact for the same
  cross-encoder/nli-deberta-v3-small model, avoiding the ~568 MB fp32
  PyTorch weight allocation in the Railway process.
- No SentenceTransformer is loaded by the verification path.
- Heavy inference concurrency remains bounded to one operation.
"""

from __future__ import annotations

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
        """Select an official 8-bit ONNX artifact supported by the host CPU.

        The upstream model publishes signed INT8 artifacts for ARM64 and
        AVX-512/AVX-512-VNNI, plus a UINT8 AVX2 artifact for the broad x86_64
        fallback.  Do not select an AVX-512 graph unless the host advertises
        the instruction set.
        """
        machine = platform.machine().lower()
        if machine in {"aarch64", "arm64"}:
            return "onnx/model_qint8_arm64.onnx"

        if machine in {"x86_64", "amd64", "x64"}:
            flags = set()
            try:
                with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.lower().startswith(("flags", "features")) and ":" in line:
                            flags.update(line.split(":", 1)[1].strip().lower().split())
                            break
            except OSError:
                pass

            if "avx512_vnni" in flags:
                return "onnx/model_qint8_avx512_vnni.onnx"
            if "avx512f" in flags:
                return "onnx/model_qint8_avx512.onnx"

            # The upstream repository does not publish a signed INT8 AVX2
            # graph; its AVX2 artifact is UINT8 and is the safe x86 fallback.
            return "onnx/model_quint8_avx2.onnx"

        raise RuntimeError(
            f"Unsupported CPU architecture for quantized NLI ONNX runtime: {machine}"
        )

    @classmethod
    def get_nli_model(
        cls,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
    ) -> Tuple[Any, Any]:
        """Return the single shared quantized ONNX DeBERTa NLI singleton."""
        if cls._nli_model is None or cls._nli_tokenizer is None:
            with cls._lock:
                if cls._nli_model is None or cls._nli_tokenizer is None:
                    import torch
                    from transformers import AutoTokenizer
                    from optimum.onnxruntime import ORTModelForSequenceClassification

                    torch.set_num_threads(1)
                    try:
                        torch.set_num_interop_threads(1)
                    except RuntimeError:
                        pass

                    artifact = cls._quantized_nli_file()
                    logger.info(
                        "loading_shared_quantized_nli_model",
                        model_name=model_name,
                        artifact=artifact,
                        max_length=256,
                    )

                    # This path intentionally requests an existing ONNX file
                    # from the model repository.  It must never use export=True,
                    # because exporting would instantiate the fp32 PyTorch model
                    # and recreate the Railway OOM condition during startup.
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = ORTModelForSequenceClassification.from_pretrained(
                        model_name,
                        file_name=artifact,
                        provider="CPUExecutionProvider",
                    )

                    cls._nli_tokenizer = tokenizer
                    cls._nli_model = model
                    cls._nli_backend = "onnxruntime-quantized"
                    cls._nli_artifact = artifact
                    cls._init_counts["nli_model"] += 1

                    from app.core.engine.memory_utils import trim_process_memory
                    post_load_rss = trim_process_memory()

                    logger.info(
                        "shared_quantized_nli_model_loaded",
                        init_count=cls._init_counts["nli_model"],
                        backend=cls._nli_backend,
                        artifact=cls._nli_artifact,
                        post_load_rss_mb=post_load_rss,
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
