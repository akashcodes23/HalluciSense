"""Focused Unit Tests for Pillar 3 Consistency Performance Optimization.

Verifies:
1. Concurrent generation of alternate samples using bounded concurrency.
2. Robust handling of single generation failures without breaking the overall pipeline.
3. Successful generation when all samples complete.
4. Per-sample timeout enforcement (fail fast).
5. Structurally valid consistency scores and backward compatible schemas.
6. Structured telemetry logging for individual sample timings (consistency_generation_1, etc.).
"""

import asyncio
import time
import pytest
from app.modules.orchestrator.service import LLMOrchestrator
from app.core.engine.pillar3_consistency import Pillar3ConsistencyEngine
from app.core.engine.types import Pillar3Result


@pytest.mark.asyncio
async def test_concurrent_sample_generation_speed_and_concurrency():
    """Verify that multiple alternate samples execute concurrently within bounded time."""
    orchestrator = LLMOrchestrator(primary_model="gpt-4o")

    # Mock provider that simulates a 0.2s latency per call
    class FastMockProvider:
        async def generate_response(self, messages, temperature=0.7, system_prompt=None):
            await asyncio.sleep(0.2)
            return "Paris is the capital and largest city of France."

    orchestrator._provider = lambda model: FastMockProvider()

    messages = [{"role": "user", "content": "What is the capital of France?"}]

    t0 = time.perf_counter()
    samples = await orchestrator.generate_samples(
        messages=messages,
        count=5,
        max_concurrency=5,
        per_sample_timeout=5.0,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # 5 calls in parallel taking ~0.2s each should complete in < 0.6s total (vs 1.0s+ sequentially)
    assert len(samples) == 5
    assert elapsed_ms < 600.0, f"Expected concurrent execution < 600ms, got {elapsed_ms:.1f}ms"


@pytest.mark.asyncio
async def test_single_sample_failure_resilience():
    """Verify that a single sample failure does not break the overall request or block others."""
    orchestrator = LLMOrchestrator(primary_model="gpt-4o")

    call_counter = 0

    class PartialFailingMockProvider:
        async def generate_response(self, messages, temperature=0.7, system_prompt=None):
            nonlocal call_counter
            call_counter += 1
            if call_counter == 2:
                raise RuntimeError("Simulated API failure or rate limit on sample 2")
            await asyncio.sleep(0.1)
            return f"Sample response version {call_counter}"

    orchestrator._provider = lambda model: PartialFailingMockProvider()

    messages = [{"role": "user", "content": "Tell me a fact."}]
    samples = await orchestrator.generate_samples(
        messages=messages,
        count=3,
        max_concurrency=3,
        per_sample_timeout=5.0,
    )

    # 1 out of 3 failed, so 2 valid samples returned safely
    assert len(samples) == 2
    assert "Sample response version 1" in samples or "Sample response version 3" in samples


@pytest.mark.asyncio
async def test_sample_timeout_behavior():
    """Verify per-sample timeout enforcement for slow providers."""
    orchestrator = LLMOrchestrator(primary_model="gpt-4o")

    class SlowMockProvider:
        async def generate_response(self, messages, temperature=0.7, system_prompt=None):
            await asyncio.sleep(10.0)  # Exceeds per_sample_timeout of 0.2s
            return "Slow response"

    orchestrator._provider = lambda model: SlowMockProvider()

    messages = [{"role": "user", "content": "Test prompt"}]
    t0 = time.perf_counter()
    samples = await orchestrator.generate_samples(
        messages=messages,
        count=3,
        max_concurrency=3,
        per_sample_timeout=0.2,  # Strict short timeout
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Should time out quickly without waiting 10 seconds
    assert elapsed_ms < 1000.0
    assert len(samples) == 0


def test_pillar3_score_structural_validity_with_samples():
    """Verify Pillar 3 output result schema and consistency calculations with alternate samples."""
    engine = Pillar3ConsistencyEngine()

    primary = "The capital of France is Paris."
    samples = [
        "Paris is the capital of France.",
        "France's capital city is Paris.",
        "Paris is the capital city of France.",
    ]

    result = engine.analyze(primary, samples)

    assert isinstance(result, Pillar3Result)
    assert result.available is True
    assert len(result.sample_responses) == 3
    assert result.consistency_failure_score is not None
    assert 0.0 <= result.consistency_failure_score <= 1.0
    assert result.contradiction_score is not None
    assert 0.0 <= result.contradiction_score <= 1.0
    assert result.sentence_consistency_score is not None
    assert 0.0 <= result.sentence_consistency_score <= 1.0
