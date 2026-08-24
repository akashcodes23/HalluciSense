"""
HalluciSense v1.0 — Lightweight In-Memory Rate Limiter.

Token-bucket rate limiter keyed by client IP address.
No external dependency (no Redis required).
Configured via settings.RATE_LIMIT_PER_MINUTE.
"""

import time
import threading
from collections import defaultdict
from typing import Dict, Tuple


class TokenBucket:
    """Thread-safe token bucket for a single client."""

    __slots__ = ("capacity", "refill_rate", "tokens", "last_refill", "lock")

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self.lock = threading.Lock()

    def consume(self) -> Tuple[bool, float]:
        """Try to consume one token. Returns (allowed, retry_after_seconds)."""
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True, 0.0
            else:
                wait = (1.0 - self.tokens) / self.refill_rate
                return False, round(wait, 1)


class InMemoryRateLimiter:
    """
    Per-IP in-memory rate limiter using token buckets.

    Automatically cleans up stale entries every `cleanup_interval` seconds
    to prevent unbounded memory growth.
    """

    def __init__(self, requests_per_minute: int = 100, cleanup_interval: float = 300.0):
        self.capacity = requests_per_minute
        self.refill_rate = requests_per_minute / 60.0
        self.cleanup_interval = cleanup_interval
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.monotonic()

    def is_allowed(self, client_ip: str) -> Tuple[bool, float]:
        """
        Check if a request from client_ip is allowed.

        Returns:
            (allowed: bool, retry_after: float)
        """
        self._maybe_cleanup()

        with self._lock:
            if client_ip not in self._buckets:
                self._buckets[client_ip] = TokenBucket(self.capacity, self.refill_rate)
            bucket = self._buckets[client_ip]

        return bucket.consume()

    def _maybe_cleanup(self):
        """Remove stale buckets that haven't been accessed recently."""
        now = time.monotonic()
        if now - self._last_cleanup < self.cleanup_interval:
            return

        with self._lock:
            self._last_cleanup = now
            stale_threshold = now - self.cleanup_interval
            stale_keys = [
                ip for ip, bucket in self._buckets.items()
                if bucket.last_refill < stale_threshold
            ]
            for key in stale_keys:
                del self._buckets[key]


# ── Module-level singleton ────────────────────────────────────────────────────

_rate_limiter: InMemoryRateLimiter | None = None
_init_lock = threading.Lock()


def get_rate_limiter(requests_per_minute: int = 100) -> InMemoryRateLimiter:
    """Get or create the singleton rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        with _init_lock:
            if _rate_limiter is None:
                _rate_limiter = InMemoryRateLimiter(requests_per_minute=requests_per_minute)
    return _rate_limiter


# ── Rate-limited paths ────────────────────────────────────────────────────────

RATE_LIMITED_PREFIXES = (
    "/api/v1/analyze",
    "/api/v1/chat",
    "/api/v1/explain",
)

# Never rate-limit health/readiness probes
EXEMPT_PATHS = (
    "/health",
    "/healthz",
    "/ready",
    "/readyz",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/",
)
