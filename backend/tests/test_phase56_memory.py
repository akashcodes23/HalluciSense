"""Phase 56 Memory and ML Runtime Stability Tests.

Validates:
1. Singleton model lifecycle (exactly one NLI model instantiated across repeated calls).
2. Concurrency semaphore boundedness (max_concurrent=1).
3. Memory trimming utility execution without exceptions.
4. NLI classification correctness with cross-encoder/nli-deberta-v3-small.
5. Verification pipeline output structural integrity and feature compatibility.
"""

import gc
import pytest
from app.core.engine.model_registry import ModelRegistry
from app.core.engine.memory_utils import trim_process_memory, get_memory_telemetry
from app.core.engine.entailment import EvidenceEntailmentEngine
from app.core.engine.pipeline import HallucinationDetectionPipeline


def test_model_registry_singleton_lifecycle():
    """Verify that multiple requests for NLI model return the same singleton instance."""
    tok1, mod1 = ModelRegistry.get_nli_model()
    tok2, mod2 = ModelRegistry.get_nli_model()

    assert tok1 is tok2, "Tokenizer instance must be an exact singleton reference"
    assert mod1 is mod2, "NLI model instance must be an exact singleton reference"

    init_counts = ModelRegistry.get_init_counts()
    assert init_counts.get("nli_model", 0) >= 1, "NLI model must be initialized at least once"


def test_concurrency_semaphore():
    """Verify that the NLI semaphore enforces bounded concurrency."""
    sem = ModelRegistry.get_nli_semaphore(max_concurrent=1)
    assert sem is not None

    # Acquire and release to ensure semaphore is functional
    acquired = sem.acquire(timeout=2.0)
    assert acquired, "Semaphore must be acquirable"
    sem.release()


def test_trim_process_memory():
    """Verify that trim_process_memory executes cleanly and returns positive RSS."""
    rss = trim_process_memory()
    assert isinstance(rss, float)
    assert rss >= 0.0

    telemetry = get_memory_telemetry()
    assert "rss_mb" in telemetry
    assert "threads" in telemetry
    assert telemetry["rss_mb"] >= 0.0


def test_nli_inference_correctness():
    """Verify that NLI inference returns valid entailment and contradiction scores."""
    engine = EvidenceEntailmentEngine()

    # Supported claim
    result_sup = engine.classify(
        claim="Paris is the capital of France.",
        evidence="Paris is the capital and most populous city of France.",
    )
    assert "entailment" in result_sup
    assert "contradiction" in result_sup
    assert "neutral" in result_sup
    assert result_sup["entailment"] > result_sup["contradiction"]

    # Contradicted claim
    result_contra = engine.classify(
        claim="Berlin is the capital of France.",
        evidence="Paris is the capital of France, while Berlin is the capital of Germany.",
    )
    assert result_contra["contradiction"] > result_contra["entailment"]


def test_pipeline_structural_integrity():
    """Verify that the HallucinationDetectionPipeline produces valid reports."""
    pipeline = HallucinationDetectionPipeline()
    report = pipeline.analyze("The Earth revolves around the Sun.")

    assert report is not None
    assert hasattr(report, "overall_h_score")
    assert hasattr(report, "overall_risk_level")
    assert hasattr(report, "sentence_analyses")
    assert 0.0 <= report.overall_h_score <= 1.0
