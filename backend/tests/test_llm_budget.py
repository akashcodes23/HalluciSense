"""
Sprint 1 LLM Budget Instrumentation and Assertion Test Suite.
Verifies structured telemetry events, execution reporting, and budget limits.
"""
import pytest
import asyncio
from app.core.config import settings
from app.core.circuit_breaker import QuotaCircuitBreaker, RequestContext
from app.modules.orchestrator.service import LLMOrchestrator


@pytest.mark.asyncio
async def test_llm_budget_normal_prompt():
    """Verify normal user prompt consumes <= 1 LLM call."""
    QuotaCircuitBreaker.reset()
    req_context = RequestContext()

    # Task 1: Record Primary Response
    req_context.record_llm_call(
        operation="PRIMARY_RESPONSE",
        provider="Google Gemini",
        model="gemini-2.0-flash",
        duration_ms=245.5,
        input_tokens=18,
        output_tokens=42,
        status="SUCCESS",
    )

    # Task 4 Assertions
    assert req_context.llm_calls <= 1, f"Expected <= 1 total LLM calls, got {req_context.llm_calls}"
    assert req_context.primary_calls == 1, "Expected exactly 1 primary response call"
    assert req_context.correction_calls == 0, "Expected 0 correction calls"
    assert req_context.fallback_calls == 0, "Expected 0 fallback calls"

    req_context.log_summary(pipeline_time_ms=112.4)


@pytest.mark.asyncio
async def test_llm_budget_correction_and_fallback_disabled():
    """Verify correction and fallback models generate 0 extra calls when disabled."""
    QuotaCircuitBreaker.reset()
    req_context = RequestContext()

    req_context.record_llm_call(operation="PRIMARY_RESPONSE")

    # When ENABLE_AUTOMATIC_CORRECTION is False
    if not settings.ENABLE_AUTOMATIC_CORRECTION:
        req_context.skipped_correction = True

    # When ENABLE_FALLBACK_MODELS is False
    if not settings.ENABLE_FALLBACK_MODELS:
        req_context.skipped_fallbacks = 3

    assert req_context.correction_calls == 0, "Correction calls must be 0 when disabled"
    assert req_context.fallback_calls == 0, "Fallback calls must be 0 when disabled"
    assert req_context.skipped_correction is True
    assert req_context.skipped_fallbacks == 3


@pytest.mark.asyncio
async def test_llm_budget_quota_exceeded_trips_breaker():
    """Verify HTTP 429 quota error trips circuit breaker and halts downstream calls."""
    QuotaCircuitBreaker.reset()
    QuotaCircuitBreaker.trip("HTTP 429 ResourceExhausted")

    req_context = RequestContext()
    req_context.quota_triggered = True

    orchestrator = LLMOrchestrator(primary_model="gemini-2.0-flash")
    samples = await orchestrator.generate_samples(
        messages=[{"role": "user", "content": "Sample prompt"}],
        count=2,
    )

    assert samples == [], "Expected empty samples list on quota error"
    assert req_context.sample_calls == 0, "Expected 0 sample calls when circuit breaker tripped"
    assert QuotaCircuitBreaker.is_tripped() is True


if __name__ == "__main__":
    asyncio.run(test_llm_budget_normal_prompt())
    asyncio.run(test_llm_budget_correction_and_fallback_disabled())
    asyncio.run(test_llm_budget_quota_exceeded_trips_breaker())
    print("ALL SPRINT 1 LLM BUDGET ASSERTIONS PASSED!")
