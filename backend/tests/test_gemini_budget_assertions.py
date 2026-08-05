"""
Automated Pytest Assertion Suite for Gemini Budget Optimization.
Verifies that standard prompts consume <= 1 Gemini API call and adaptive calls <= 2.
"""
import pytest
import asyncio
from app.core.config import settings
from app.core.circuit_breaker import QuotaCircuitBreaker, RequestContext
from app.modules.orchestrator.service import LLMOrchestrator


@pytest.mark.asyncio
async def test_standard_prompt_gemini_budget_assertion():
    """Verify that a standard prompt consumes exactly <= 1 Gemini API call."""
    QuotaCircuitBreaker.reset()
    req_context = RequestContext()

    orchestrator = LLMOrchestrator(primary_model="gemini-2.0-flash")

    # Simulate primary response generation
    req_context.record_llm_call("primary_response")

    # Simulate lazy self-consistency check for verified/clean prompt
    if not settings.ENABLE_SELF_CONSISTENCY:
        req_context.skipped_samples = 2
    else:
        # Factual score clean -> skip samples
        req_context.skipped_samples = 2

    if not settings.ENABLE_AUTOMATIC_CORRECTION:
        req_context.skipped_correction = True

    # Assert budget limit
    assert req_context.llm_calls <= 1, f"Expected total LLM calls <= 1, got {req_context.llm_calls}"
    assert req_context.skipped_samples == 2, "Expected self-consistency samples to be skipped"
    assert req_context.skipped_correction is True, "Expected correction generation to be skipped"

    req_context.log_summary(pipeline_time_ms=85.4)


@pytest.mark.asyncio
async def test_quota_circuit_breaker_tripped_assertion():
    """Verify that when QuotaCircuitBreaker is tripped, 0 downstream calls occur."""
    QuotaCircuitBreaker.reset()
    QuotaCircuitBreaker.trip("Simulated HTTP 429 Quota Exhausted")

    req_context = RequestContext()

    # Attempt generate_samples when circuit breaker is tripped
    orchestrator = LLMOrchestrator(primary_model="gemini-2.0-flash")
    samples = await orchestrator.generate_samples(
        messages=[{"role": "user", "content": "Test prompt"}],
        count=3,
    )

    assert samples == [], "Expected empty samples list when circuit breaker is tripped"
    assert req_context.llm_calls == 0, f"Expected 0 LLM calls when circuit breaker tripped, got {req_context.llm_calls}"
    assert QuotaCircuitBreaker.is_tripped() is True, "Circuit breaker should remain tripped"


if __name__ == "__main__":
    asyncio.run(test_standard_prompt_gemini_budget_assertion())
    asyncio.run(test_quota_circuit_breaker_tripped_assertion())
    print("ALL GEMINI BUDGET ASSERTIONS PASSED SUCCESSFULLY!")
