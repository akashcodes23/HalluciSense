"""
Quota Circuit Breaker & Request-Level Telemetry Context.
Prevents cascading LLM invocations after rate limits / HTTP 429 quota exhaustion events occur.
"""
import threading
import time
from dataclasses import dataclass, field
from typing import Optional
import structlog

logger = structlog.get_logger(__name__)


class QuotaCircuitBreaker:
    """
    Thread-safe global and request-level circuit breaker for LLM provider quota limits.
    When an HTTP 429 / ResourceExhausted exception occurs, the circuit breaker trips,
    preventing any downstream LLM operations (samples, corrections, fallbacks) from executing.
    """
    _tripped: bool = False
    _trip_reason: Optional[str] = None
    _tripped_at: Optional[float] = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def is_tripped(cls) -> bool:
        with cls._lock:
            return cls._tripped

    @classmethod
    def trip(cls, reason: str = "HTTP 429 ResourceExhausted") -> None:
        with cls._lock:
            if not cls._tripped:
                cls._tripped = True
                cls._trip_reason = reason
                cls._tripped_at = time.time()
                logger.warning(
                    "QUOTA_CIRCUIT_BREAKER_TRIPPED",
                    reason=reason,
                    timestamp=cls._tripped_at,
                )

    @classmethod
    def reset(cls) -> None:
        """Reset the circuit breaker (e.g. for testing or after quota window resets)."""
        with cls._lock:
            cls._tripped = False
            cls._trip_reason = None
            cls._tripped_at = None
            logger.info("QUOTA_CIRCUIT_BREAKER_RESET")


@dataclass
class RequestContext:
    """
    Request-level telemetry and LLM budget tracker.
    Tracks all LLM invocations, token counts, and skipped operations per request.
    """
    request_id: str = field(default_factory=lambda: "req-" + str(int(time.time() * 1000)))
    trace_id: str = field(default_factory=lambda: "tr-" + str(int(time.time() * 1000)))
    llm_calls: int = 0
    primary_calls: int = 0
    fallback_calls: int = 0
    correction_calls: int = 0
    sample_calls: int = 0
    skipped_samples: int = 0
    skipped_correction: bool = False
    skipped_fallbacks: int = 0
    quota_triggered: bool = False
    stream_duration_ms: float = 0.0
    verification_duration_ms: float = 0.0

    def record_llm_call(
        self,
        operation: str = "PRIMARY_RESPONSE",
        provider: str = "Google Gemini",
        model: str = "gemini-2.0-flash",
        duration_ms: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        status: str = "SUCCESS",
        retry_count: int = 0,
        fallback_used: bool = False,
    ) -> None:
        self.llm_calls += 1

        if operation == "PRIMARY_RESPONSE":
            self.primary_calls += 1
        elif operation == "SELF_CONSISTENCY":
            self.sample_calls += 1
        elif operation == "CORRECTION":
            self.correction_calls += 1
        elif operation == "FALLBACK":
            self.fallback_calls += 1

        logger.info(
            "STRUCTURED_LLM_INVOCATION_EVENT",
            request_id=self.request_id,
            trace_id=self.trace_id,
            provider=provider,
            model=model,
            operation=operation,
            timestamp=time.time(),
            duration_ms=round(duration_ms, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            status=status,
            retry_count=retry_count,
            fallback_used=fallback_used,
            total_llm_calls=self.llm_calls,
        )

    def log_summary(self, pipeline_time_ms: float = 0.0) -> None:
        self.verification_duration_ms = pipeline_time_ms
        logger.info(
            "LLM_EXECUTION_REPORT",
            request_id=self.request_id,
            trace_id=self.trace_id,
            total_llm_calls=self.llm_calls,
            primary_calls=self.primary_calls,
            sample_calls=self.sample_calls,
            correction_calls=self.correction_calls,
            fallback_calls=self.fallback_calls,
            skipped_samples=self.skipped_samples,
            skipped_correction=self.skipped_correction,
            skipped_fallbacks=self.skipped_fallbacks,
            quota_triggered=self.quota_triggered or QuotaCircuitBreaker.is_tripped(),
            stream_duration_ms=round(self.stream_duration_ms, 2),
            verification_duration_ms=round(self.verification_duration_ms, 2),
            total_pipeline_time_ms=round(self.stream_duration_ms + self.verification_duration_ms, 2),
        )
