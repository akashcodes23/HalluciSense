"""
LLM Orchestrator Service.
Handles provider routing, retries, streaming and intelligent fallback.
"""

import asyncio
from typing import AsyncGenerator, List, Optional

import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.circuit_breaker import QuotaCircuitBreaker
from app.core.exceptions import HalluciSenseError
from app.modules.providers.factory import get_provider

logger = structlog.get_logger(__name__)


class ProviderExhaustedError(HalluciSenseError):
    pass


class LLMOrchestrator:

    def __init__(
        self,
        primary_model: str,
        fallback_model: Optional[str] = None,
        enable_fallback: bool = False,   # <-- default OFF
    ):
        self.primary_model = primary_model
        self.enable_fallback = enable_fallback

        self.fallback_models: List[str] = []

        if enable_fallback:

            if fallback_model:
                self.fallback_models.append(fallback_model)

            # Gemini fallback only
            if settings.GEMINI_API_KEY:
                for model in (
                    "gemini-flash-latest",
                    "gemini-2.0-flash-lite",
                ):
                    if model != primary_model:
                        self.fallback_models.append(model)

            # OpenAI fallback only if explicitly enabled
            if settings.OPENAI_API_KEY:
                for model in (
                    "gpt-4o-mini",
                    "gpt-4o",
                ):
                    if model != primary_model:
                        self.fallback_models.append(model)

            self.fallback_models = list(dict.fromkeys(self.fallback_models))

        logger.info(
            "orchestrator_initialized",
            primary=self.primary_model,
            fallback=self.fallback_models,
        )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=3),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _provider(self, model: str):
        return get_provider(model)

    async def stream_chat(
        self,
        messages: List[dict],
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator:

        candidates = [self.primary_model] + self.fallback_models

        first_exception = None

        logger.info(
            "stream_started",
            candidates=candidates,
        )

        for model in candidates:

            try:

                logger.info("trying_provider", model=model)

                provider = self._provider(model)

                async for chunk in provider.stream_chat(
                    messages,
                    system_prompt=system_prompt,
                ):
                    yield chunk

                logger.info("provider_success", model=model)

                return

            except Exception as exc:

                if first_exception is None:
                    first_exception = exc

                logger.exception(
                    "provider_failed",
                    provider=model,
                )

                # Halt immediately on quota / rate limit errors regardless of fallback settings
                err_str = str(exc)
                if "429" in err_str or "Quota exceeded" in err_str or "ResourceExhausted" in err_str:
                    logger.warning("orchestrator_halting_fallback_on_quota_error", model=model, error=err_str)
                    raise exc

                # If fallback is disabled, stop immediately
                if not self.enable_fallback or not settings.ENABLE_FALLBACK_MODELS:
                    raise

                continue

        raise ProviderExhaustedError(
            f"Primary provider failed: {first_exception}"
        )

    async def generate_response(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> str:

        candidates = [self.primary_model] + (self.fallback_models if settings.ENABLE_FALLBACK_MODELS else [])

        first_exception = None

        for model in candidates:

            try:

                provider = self._provider(model)

                return await provider.generate_response(
                    messages,
                    temperature=temperature,
                    system_prompt=system_prompt,
                )

            except Exception as exc:

                if first_exception is None:
                    first_exception = exc

                logger.exception(
                    "generate_failed",
                    provider=model,
                )

                err_str = str(exc)
                if "429" in err_str or "Quota exceeded" in err_str or "ResourceExhausted" in err_str:
                    logger.warning("orchestrator_halting_fallback_on_quota_error", model=model, error=err_str)
                    raise exc

                if not self.enable_fallback or not settings.ENABLE_FALLBACK_MODELS:
                    raise

        raise ProviderExhaustedError(
            f"Primary provider failed: {first_exception}"
        )

    async def generate_samples(
        self,
        messages: List[dict],
        count: int = 3,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> List[str]:

        if QuotaCircuitBreaker.is_tripped():
            logger.warning("SELF_CONSISTENCY_SKIPPED", reason="circuit_breaker_tripped")
            return []

        if not settings.ENABLE_SELF_CONSISTENCY or count <= 0:
            logger.info("SELF_CONSISTENCY_SKIPPED", reason="disabled_in_config")
            return []

        effective_count = min(count, settings.MAX_SELF_CONSISTENCY_SAMPLES)
        provider = self._provider(self.primary_model)

        async def sample():
            if QuotaCircuitBreaker.is_tripped():
                return None
            try:
                return await asyncio.wait_for(
                    provider.generate_response(
                        messages,
                        temperature=temperature,
                        system_prompt=system_prompt,
                    ),
                    timeout=30,
                )
            except Exception as exc:
                err_str = str(exc)
                if "429" in err_str or "Quota exceeded" in err_str or "ResourceExhausted" in err_str:
                    QuotaCircuitBreaker.trip(err_str)
                    logger.warning("sample_generation_failed_quota_tripped", error=err_str)
                else:
                    logger.warning("sample_generation_failed", error=err_str)
                return None

        results = await asyncio.gather(*(sample() for _ in range(effective_count)))

        return [r.strip() for r in results if r and r.strip()]