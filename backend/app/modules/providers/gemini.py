"""
Google Gemini provider implementation.
Production implementation featuring thread-safe model caching, queue sentinels,
bounded GenerationConfig caching, fast health check probes, and structured error logging.
"""
import asyncio
import os
import random
import threading
from dataclasses import dataclass, replace
from typing import AsyncGenerator, Dict, List, Optional, Set, TypedDict

# Enforce native DNS resolution for gRPC on macOS to prevent lookup deadlocks
os.environ["GRPC_DNS_RESOLVER"] = "native"

import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from google.api_core.exceptions import (
    DeadlineExceeded,
    GoogleAPICallError,
    InternalServerError,
    InvalidArgument,
    PermissionDenied,
    ResourceExhausted,
    ServiceUnavailable,
    Unauthenticated,
)
import structlog

from app.core.config import settings
from app.core.circuit_breaker import QuotaCircuitBreaker
from app.core.exceptions import HalluciSenseError
from app.modules.providers.base import AbstractLLMProvider
from app.modules.providers.schemas import ProviderResponseChunk

logger = structlog.get_logger(__name__)

# Queue completion sentinel
_STREAM_END = object()

TRANSIENT_EXCEPTIONS = (
    ResourceExhausted,
    ServiceUnavailable,
    DeadlineExceeded,
    InternalServerError,
)

NON_RETRYABLE_EXCEPTIONS = (
    InvalidArgument,
    PermissionDenied,
    Unauthenticated,
)


# ── Types & Dataclasses ────────────────────────────────────────────────────────

class ChatMessage(TypedDict):
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class GeminiDefaults:
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int = 4096


@dataclass(slots=True)
class ProviderMetrics:
    successful_requests: int = 0
    failed_requests: int = 0
    retries: int = 0
    quota_errors: int = 0
    fallbacks: int = 0
    stream_cancellations: int = 0
    timeouts: int = 0


# ── Typed Exceptions ───────────────────────────────────────────────────────────

class GeminiProviderError(HalluciSenseError):
    """Base exception for Gemini provider operations."""
    def __init__(self, message: str):
        super().__init__(message=message)


class GeminiQuotaExceededError(GeminiProviderError):
    """Raised when Gemini API quota or rate limits are exceeded."""
    pass


class GeminiStreamTimeoutError(GeminiProviderError):
    """Raised when Gemini stream queue consumption times out."""
    pass


class GeminiAuthenticationError(GeminiProviderError):
    """Raised when Gemini API key or authentication fails."""
    pass


class GeminiConfigurationError(GeminiProviderError):
    """Raised when Gemini model configuration is invalid."""
    pass


# ── Internal Helpers ───────────────────────────────────────────────────────────

def _calculate_backoff(attempt: int, max_backoff: float) -> float:
    """Calculates exponential backoff with random jitter."""
    return min(max_backoff, (2.0 ** (attempt - 1)) + random.uniform(0.0, 1.0))


def _extract_chunk_text(chunk: object) -> str:
    """Safely extracts text from a Gemini response chunk without throwing accessor exceptions."""
    if chunk is None:
        return ""

    try:
        if hasattr(chunk, "text") and chunk.text:
            return str(chunk.text)
    except (ValueError, AttributeError):
        pass

    try:
        if hasattr(chunk, "candidates") and chunk.candidates:
            cand = chunk.candidates[0]
            if hasattr(cand, "content") and cand.content and hasattr(cand.content, "parts"):
                parts_text: List[str] = []
                for part in cand.content.parts:
                    if hasattr(part, "text") and part.text:
                        parts_text.append(str(part.text))
                return "".join(parts_text)
    except (ValueError, AttributeError, IndexError):
        pass

    return ""


def _format_messages_for_gemini(messages: List[ChatMessage]) -> List[Dict[str, object]]:
    """Formats generic messages into alternating user/model turns required by Gemini."""
    raw_turns: List[Dict[str, str]] = []
    for msg in messages:
        role_raw = str(msg.get("role", "user")).lower()
        if role_raw in ("system", "user"):
            role = "user"
        elif role_raw == "assistant":
            role = "model"
        else:
            role = "user"

        content = str(msg.get("content", "") or "").strip()
        if not content:
            continue

        raw_turns.append({"role": role, "content": content})

    if not raw_turns:
        return [{"role": "user", "parts": ["Hello"]}]

    merged_turns: List[Dict[str, str]] = []
    for turn in raw_turns:
        if merged_turns and merged_turns[-1]["role"] == turn["role"]:
            merged_turns[-1]["content"] += f"\n\n{turn['content']}"
        else:
            merged_turns.append(dict(turn))

    if merged_turns[0]["role"] != "user":
        merged_turns.insert(0, {"role": "user", "content": "Context:"})

    formatted: List[Dict[str, object]] = []
    for turn in merged_turns:
        formatted.append({
            "role": turn["role"],
            "parts": [turn["content"]]
        })

    return formatted


# ── Gemini Provider Implementation ──────────────────────────────────────────────

class GeminiProvider(AbstractLLMProvider):
    """
    Enterprise-grade Google Gemini LLM Provider.
    """

    FALLBACK_MODELS: List[str] = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ]

    SUPPORTED_MODELS: Set[str] = {
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    }

    DEFAULTS = GeminiDefaults()
    MAX_CONFIG_CACHE_SIZE: int = 16

    def __init__(self, model_slug: str = "gemini-2.0-flash"):
        if model_slug not in self.SUPPORTED_MODELS:
            logger.warning(
                "gemini_unsupported_model_requested",
                requested=model_slug,
                fallback="gemini-2.0-flash",
            )
            self._model_slug = "gemini-2.0-flash"
        else:
            self._model_slug = model_slug

        self._internal_metrics = ProviderMetrics()
        self._generation_config_cache: Dict[float, GenerationConfig] = {}
        self._model_cache: Dict[str, genai.GenerativeModel] = {}
        self._cache_lock = threading.Lock()

        api_key = settings.GEMINI_API_KEY
        if not api_key:
            logger.warning("gemini_api_key_missing_in_settings")
        else:
            try:
                genai.configure(api_key=api_key)
                logger.info("gemini_configure_successful", model_slug=self._model_slug)
            except Exception as e:
                logger.exception("gemini_configure_failed", error_type=type(e).__name__, model_slug=self._model_slug)

    @property
    def provider_name(self) -> str:
        return "Google Gemini"

    @property
    def model_slug(self) -> str:
        return self._model_slug

    @property
    def metrics(self) -> ProviderMetrics:
        """Returns an immutable snapshot of metrics."""
        return replace(self._internal_metrics)

    async def health_check(self) -> bool:
        """Fast, lightweight API probe verifying connectivity without iterating all models."""
        if not settings.GEMINI_API_KEY:
            return False
        try:
            def _probe() -> bool:
                models_iter = genai.list_models()
                return next(iter(models_iter), None) is not None
            return await asyncio.to_thread(_probe)
        except Exception:
            return False

    def _get_generative_model(
        self, model_name: str, system_prompt: Optional[str] = None
    ) -> genai.GenerativeModel:
        """Thread-safe retrieval of base model instances, or temporary instance for system prompts."""
        if system_prompt and system_prompt.strip():
            return genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_prompt.strip(),
            )

        with self._cache_lock:
            if model_name not in self._model_cache:
                self._model_cache[model_name] = genai.GenerativeModel(model_name=model_name)
            return self._model_cache[model_name]

    def _get_generation_config(self, temperature: Optional[float] = None) -> GenerationConfig:
        """Retrieves or constructs cached immutable GenerationConfig instance."""
        temp = temperature if temperature is not None else self.DEFAULTS.temperature
        with self._cache_lock:
            if temp not in self._generation_config_cache:
                if len(self._generation_config_cache) >= self.MAX_CONFIG_CACHE_SIZE:
                    oldest = next(iter(self._generation_config_cache))
                    del self._generation_config_cache[oldest]

                self._generation_config_cache[temp] = GenerationConfig(
                    temperature=temp,
                    top_p=self.DEFAULTS.top_p,
                    top_k=self.DEFAULTS.top_k,
                    max_output_tokens=self.DEFAULTS.max_output_tokens,
                )
            return self._generation_config_cache[temp]

    async def _execute_with_retry(
        self,
        model_name: str,
        formatted_contents: List[Dict[str, object]],
        system_prompt: Optional[str],
        stream: bool,
        temperature: Optional[float] = None,
    ) -> object:
        """Executes generate_content with backoff retry policy for transient errors."""
        model_client = self._get_generative_model(model_name, system_prompt)
        gen_config = self._get_generation_config(temperature=temperature)

        max_retries = settings.GEMINI_MAX_RETRIES
        timeout_seconds = settings.GEMINI_GENERATION_TIMEOUT
        max_backoff = settings.GEMINI_MAX_BACKOFF

        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                def _do_generate() -> object:
                    return model_client.generate_content(
                        contents=formatted_contents,
                        generation_config=gen_config,
                        stream=stream,
                    )

                response = await asyncio.wait_for(
                    asyncio.to_thread(_do_generate),
                    timeout=timeout_seconds,
                )
                self._internal_metrics.successful_requests += 1
                return response
            except NON_RETRYABLE_EXCEPTIONS as non_retryable:
                self._internal_metrics.failed_requests += 1
                logger.exception(
                    "gemini_non_retryable_error",
                    provider=self.provider_name,
                    model=model_name,
                    error_type=type(non_retryable).__name__,
                    retry_number=attempt,
                )
                if isinstance(non_retryable, (PermissionDenied, Unauthenticated)):
                    raise GeminiAuthenticationError(str(non_retryable)) from non_retryable
                if isinstance(non_retryable, InvalidArgument):
                    raise GeminiConfigurationError(str(non_retryable)) from non_retryable
                raise non_retryable
            except ResourceExhausted as quota_err:
                self._internal_metrics.quota_errors += 1
                self._internal_metrics.failed_requests += 1
                logger.warning(
                    "gemini_quota_exhausted_immediate_halt",
                    provider=self.provider_name,
                    model=model_name,
                    error=str(quota_err),
                )
                raise GeminiQuotaExceededError(str(quota_err)) from quota_err
            except (TRANSIENT_EXCEPTIONS, asyncio.TimeoutError) as transient:
                last_error = transient
                self._internal_metrics.retries += 1
                if isinstance(transient, asyncio.TimeoutError):
                    self._internal_metrics.timeouts += 1

                backoff = _calculate_backoff(attempt, max_backoff)
                logger.warning(
                    "gemini_transient_error_retry",
                    provider=self.provider_name,
                    model=model_name,
                    error_type=type(transient).__name__,
                    retry_number=attempt,
                    backoff_seconds=backoff,
                )
                if attempt < max_retries:
                    await asyncio.sleep(backoff)
            except Exception as unhandled:
                last_error = unhandled
                err_str = str(unhandled)
                if "429" in err_str or "Quota exceeded" in err_str:
                    self._internal_metrics.quota_errors += 1
                    self._internal_metrics.failed_requests += 1
                    logger.warning(
                        "gemini_quota_string_matched_immediate_halt",
                        provider=self.provider_name,
                        model=model_name,
                        error=err_str,
                    )
                    raise GeminiQuotaExceededError(err_str) from unhandled
                self._internal_metrics.failed_requests += 1
                raise unhandled

        self._internal_metrics.failed_requests += 1
        if isinstance(last_error, ResourceExhausted):
            raise GeminiQuotaExceededError(str(last_error)) from last_error
        if last_error:
            raise last_error
        raise GeminiProviderError("Generation failed without exception details.")

    async def stream_chat(
        self, messages: List[ChatMessage], system_prompt: Optional[str] = None
    ) -> AsyncGenerator[ProviderResponseChunk, None]:
        """Stream chat responses from Gemini using bounded queue and thread-safe producer."""
        if QuotaCircuitBreaker.is_tripped():
            logger.warning("gemini_stream_chat_skipped_circuit_breaker_tripped")
            yield ProviderResponseChunk(
                text="⚠️ Google Gemini API Rate Limit Exceeded (HTTP 429). Operations paused until quota window resets.",
                logits=None,
                is_done=False,
            )
            yield ProviderResponseChunk(text="", logits=None, is_done=True)
            return

        formatted_contents = _format_messages_for_gemini(messages)
        logger.info(
            "gemini_stream_chat_started",
            provider=self.provider_name,
            model=self._model_slug,
            turns_count=len(formatted_contents),
            has_system_prompt=bool(system_prompt),
        )

        candidate_models: List[str] = [self._model_slug]
        if settings.ENABLE_FALLBACK_MODELS:
            for m in self.FALLBACK_MODELS:
                if m not in candidate_models:
                    candidate_models.append(m)

        active_model: Optional[str] = None
        response_stream: object = None
        last_error: Optional[Exception] = None

        for fallback_index, model_name in enumerate(candidate_models):
            try:
                if fallback_index > 0:
                    self._internal_metrics.fallbacks += 1

                logger.info(
                    "gemini_initializing_model",
                    provider=self.provider_name,
                    model=model_name,
                    fallback_number=fallback_index,
                )

                response_stream = await self._execute_with_retry(
                    model_name=model_name,
                    formatted_contents=formatted_contents,
                    system_prompt=system_prompt,
                    stream=True,
                )
                active_model = model_name
                logger.info(
                    "gemini_stream_initialized_successfully",
                    provider=self.provider_name,
                    model=active_model,
                    fallback_number=fallback_index,
                )
                break
            except Exception as e:
                last_error = e
                if isinstance(e, (ResourceExhausted, GeminiQuotaExceededError)) or "429" in str(e) or "Quota exceeded" in str(e):
                    QuotaCircuitBreaker.trip(str(e))
                    logger.warning("gemini_quota_error_halting_stream_fallbacks", model=model_name, error=str(e))
                    raise e
                logger.exception(
                    "gemini_model_attempt_failed",
                    provider=self.provider_name,
                    model=model_name,
                    error_type=type(e).__name__,
                    fallback_number=fallback_index,
                )
                continue

        if response_stream is None:
            err_str = str(last_error)
            if (
                isinstance(last_error, (ResourceExhausted, GeminiQuotaExceededError))
                or "429" in err_str
                or "Quota exceeded" in err_str
            ):
                self._internal_metrics.quota_errors += 1
                logger.warning(
                    "gemini_rate_limit_exceeded",
                    provider=self.provider_name,
                    model=self._model_slug,
                    error=err_str,
                )
                yield ProviderResponseChunk(
                    text="⚠️ Google Gemini API Free Tier Rate Limit Exceeded (HTTP 429). Please wait ~30 seconds before sending your next prompt.",
                    logits=None,
                    is_done=False,
                )
                yield ProviderResponseChunk(text="", logits=None, is_done=True)
                return

            err_msg = f"All Gemini models failed. Last error: {err_str}"
            logger.exception(
                "gemini_all_models_failed",
                provider=self.provider_name,
                model=self._model_slug,
                error=err_msg,
            )
            raise GeminiProviderError(err_msg) from last_error

        tokens_yielded = 0
        stop_event = threading.Event()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=settings.GEMINI_QUEUE_SIZE)

        def _producer() -> None:
            try:
                logger.info("gemini_producer_thread_started", provider=self.provider_name, model=active_model)
                if hasattr(response_stream, "__iter__"):
                    for chunk in response_stream:
                        if stop_event.is_set():
                            logger.info("gemini_producer_stop_signal_received", provider=self.provider_name, model=active_model)
                            break

                        text = _extract_chunk_text(chunk)
                        if text:
                            future = asyncio.run_coroutine_threadsafe(queue.put(text), loop)
                            try:
                                future.result(timeout=5.0)
                            except Exception as put_err:
                                logger.exception(
                                    "gemini_queue_put_failed",
                                    provider=self.provider_name,
                                    model=active_model,
                                    error_type=type(put_err).__name__,
                                )

                if not stop_event.is_set():
                    future = asyncio.run_coroutine_threadsafe(queue.put(_STREAM_END), loop)
                    try:
                        future.result(timeout=5.0)
                    except Exception:
                        pass
                logger.info("gemini_producer_thread_completed", provider=self.provider_name, model=active_model)
            except Exception as prod_err:
                logger.exception(
                    "gemini_producer_thread_error",
                    provider=self.provider_name,
                    model=active_model,
                    error_type=type(prod_err).__name__,
                )
                if not stop_event.is_set():
                    future = asyncio.run_coroutine_threadsafe(queue.put(prod_err), loop)
                    try:
                        future.result(timeout=5.0)
                    except Exception:
                        pass
            finally:
                stop_event.set()

        producer_future = loop.run_in_executor(None, _producer)

        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        queue.get(),
                        timeout=settings.GEMINI_STREAM_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    self._internal_metrics.timeouts += 1
                    logger.warning(
                        "gemini_queue_get_timeout",
                        provider=self.provider_name,
                        model=active_model,
                        timeout_seconds=settings.GEMINI_STREAM_TIMEOUT,
                    )
                    stop_event.set()
                    raise GeminiStreamTimeoutError(f"Gemini stream timeout after {settings.GEMINI_STREAM_TIMEOUT}s")
                except asyncio.CancelledError:
                    self._internal_metrics.stream_cancellations += 1
                    logger.info("gemini_stream_cancelled_by_client", provider=self.provider_name, model=active_model)
                    stop_event.set()
                    raise

                if item is _STREAM_END:
                    break
                if isinstance(item, Exception):
                    raise item

                tokens_yielded += 1
                logger.debug(
                    "gemini_chunk_yielded",
                    provider=self.provider_name,
                    model=active_model,
                    chunk_index=tokens_yielded,
                    length=len(str(item)),
                )
                yield ProviderResponseChunk(text=str(item), logits=None, is_done=False)

            logger.info(
                "gemini_stream_completed",
                provider=self.provider_name,
                model=active_model,
                tokens_yielded=tokens_yielded,
            )
            yield ProviderResponseChunk(text="", logits=None, is_done=True)

        except asyncio.CancelledError:
            stop_event.set()
            self._internal_metrics.stream_cancellations += 1
            logger.info("gemini_stream_cancelled_cleanup", provider=self.provider_name, model=active_model)
            raise
        except Exception as e:
            stop_event.set()
            err_str = str(e)
            if isinstance(e, (ResourceExhausted, GeminiQuotaExceededError)) or "429" in err_str or "Quota exceeded" in err_str:
                self._internal_metrics.quota_errors += 1
                logger.warning(
                    "gemini_stream_rate_limit_mid_stream",
                    provider=self.provider_name,
                    model=active_model,
                    error=err_str,
                )
                if tokens_yielded == 0:
                    yield ProviderResponseChunk(
                        text="⚠️ Google Gemini API Rate Limit Exceeded (HTTP 429). Please wait ~30 seconds and retry.",
                        logits=None,
                        is_done=False,
                    )
                    yield ProviderResponseChunk(text="", logits=None, is_done=True)
                    return

            logger.exception(
                "gemini_stream_iteration_failed",
                provider=self.provider_name,
                model=active_model,
                error_type=type(e).__name__,
            )
            raise e
        finally:
            stop_event.set()
            try:
                await asyncio.wrap_future(producer_future)
            except Exception:
                pass

    async def generate_response(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a complete non-streaming response text from Gemini."""
        if QuotaCircuitBreaker.is_tripped():
            logger.warning("gemini_generate_response_skipped_circuit_breaker_tripped")
            raise GeminiQuotaExceededError("Circuit breaker tripped: HTTP 429 Quota Exhausted")

        formatted_contents = _format_messages_for_gemini(messages)
        candidate_models: List[str] = [self._model_slug]
        if settings.ENABLE_FALLBACK_MODELS:
            for m in self.FALLBACK_MODELS:
                if m not in candidate_models:
                    candidate_models.append(m)

        last_error: Optional[Exception] = None
        for fallback_index, model_name in enumerate(candidate_models):
            try:
                if fallback_index > 0:
                    self._internal_metrics.fallbacks += 1

                response = await self._execute_with_retry(
                    model_name=model_name,
                    formatted_contents=formatted_contents,
                    system_prompt=system_prompt,
                    stream=False,
                    temperature=temperature,
                )

                text = _extract_chunk_text(response)
                if text:
                    logger.info(
                        "gemini_generate_response_success",
                        provider=self.provider_name,
                        model=model_name,
                        length=len(text),
                        fallback_number=fallback_index,
                    )
                    return text
            except Exception as e:
                last_error = e
                if isinstance(e, (ResourceExhausted, GeminiQuotaExceededError)) or "429" in str(e) or "Quota exceeded" in str(e):
                    QuotaCircuitBreaker.trip(str(e))
                    logger.warning("gemini_quota_error_halting_generate_fallbacks", model=model_name, error=str(e))
                    raise e
                logger.exception(
                    "gemini_generate_attempt_failed",
                    provider=self.provider_name,
                    model=model_name,
                    error_type=type(e).__name__,
                    fallback_number=fallback_index,
                )
                continue

        err_msg = f"Gemini non-streaming generation failed across all models. Last error: {str(last_error)}"
        logger.exception("gemini_generate_failed", provider=self.provider_name, error=err_msg)
        raise GeminiProviderError(err_msg) from last_error
