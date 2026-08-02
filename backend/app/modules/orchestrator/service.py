"""
LLM Orchestrator Service.
Handles routing, retries, and provider fallback logic for LLM generation.
"""
import asyncio
from typing import AsyncGenerator, List, Optional
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.modules.providers.factory import get_provider
from app.core.exceptions import HalluciSenseError

logger = structlog.get_logger(__name__)

class ProviderExhaustedError(HalluciSenseError):
    def __init__(self, message: str):
        super().__init__(message=message)

class LLMOrchestrator:
    """
    Sits between the application logic and the raw AI providers.
    Implements robust connection handling and failover.
    """
    def __init__(self, primary_model: str, fallback_model: str = "gpt-4o-mini"):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _get_provider_with_retry(self, model: str):
        """Attempts to initialize provider with basic retry logic on setup failure."""
        return get_provider(model)

    async def stream_chat(self, messages: List[dict]) -> AsyncGenerator[dict, None]:
        """
        Streams a chat response, falling back to a secondary provider if the primary fails completely.
        """
        provider = self._get_provider_with_retry(self.primary_model)
        
        try:
            generator = provider.stream_chat(messages)
            # We yield from the generator. If it fails mid-stream, 
            # it's harder to fallback cleanly without confusing the user.
            # But if it fails on initialization/first token, we catch it.
            async for chunk in generator:
                yield chunk
        except Exception as e:
            logger.error("primary_provider_failed", model=self.primary_model, error=str(e))
            
            if self.fallback_model and self.fallback_model != self.primary_model:
                logger.info("attempting_fallback_provider", fallback=self.fallback_model)
                try:
                    fallback_provider = self._get_provider_with_retry(self.fallback_model)
                    fallback_gen = fallback_provider.stream_chat(messages)
                    async for chunk in fallback_gen:
                        yield chunk
                except Exception as fallback_err:
                    logger.error("fallback_provider_failed", error=str(fallback_err))
                    raise ProviderExhaustedError("All available LLM providers failed to generate a response.")
            else:
                raise ProviderExhaustedError("Primary LLM provider failed and no fallback available.")

    async def generate_response(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Non-streaming response generation with error handling.
        """
        provider = self._get_provider_with_retry(self.primary_model)
        return await provider.generate_response(messages, temperature=temperature, system_prompt=system_prompt)

    async def generate_samples(
        self,
        messages: List[dict],
        count: int = 3,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> List[str]:
        """
        Generates alternate responses concurrently for self-consistency testing.
        Fails safely per sample without raising an exception for the overall batch.
        """
        logger.info(
            "consistency_sampling_started",
            provider=self.primary_model,
            requested_samples=count
        )

        try:
            provider = self._get_provider_with_retry(self.primary_model)
        except Exception as exc:
            logger.warning("consistency_sampling_provider_init_failed", error=str(exc))
            return []

        async def _single_sample(index: int) -> Optional[str]:
            try:
                sample_text = await asyncio.wait_for(
                    provider.generate_response(messages, temperature=temperature, system_prompt=system_prompt),
                    timeout=30.0
                )
                sample_clean = sample_text.strip() if sample_text else ""
                if sample_clean:
                    logger.info("consistency_sample_generated", sample_index=index, length=len(sample_clean))
                    return sample_clean
                else:
                    logger.warning("consistency_sample_failed", sample_index=index, error="empty_response")
                    return None
            except Exception as e:
                logger.warning("consistency_sample_failed", sample_index=index, error=str(e))
                return None

        tasks = [_single_sample(i) for i in range(count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_samples: List[str] = []
        for res in results:
            if isinstance(res, str) and res.strip():
                valid_samples.append(res.strip())

        logger.info(
            "consistency_sampling_completed",
            requested_samples=count,
            successful_samples=len(valid_samples)
        )

        return valid_samples
