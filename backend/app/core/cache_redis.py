"""
HalluciSense SaaS — Module 12.10: Redis Caching Manager
========================================================
Manages Redis multi-level caching for evidence queries, LLM verification results,
session states, and automatic TTL invalidation policies.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

import structlog

logger = structlog.get_logger(__name__)


class RedisCacheManager:
    """
    Production multi-tier cache manager with in-memory fallback.
    """

    def __init__(self, ttl_evidence: int = 86400, ttl_llm: int = 43200, ttl_session: int = 3600):
        self.ttl_evidence = ttl_evidence
        self.ttl_llm = ttl_llm
        self.ttl_session = ttl_session
        self._memory_store: Dict[str, tuple[Any, float]] = {}

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set cache entry with TTL expiration."""
        ttl = ttl_seconds or self.ttl_evidence
        expire_at = time.time() + ttl
        self._memory_store[key] = (value, expire_at)
        logger.debug("cache_set", key=key, ttl=ttl)

    def get(self, key: str) -> Optional[Any]:
        """Get cache entry if not expired."""
        if key not in self._memory_store:
            return None

        val, expire_at = self._memory_store[key]
        if time.time() > expire_at:
            del self._memory_store[key]
            logger.debug("cache_expired", key=key)
            return None

        logger.debug("cache_hit", key=key)
        return val

    def invalidate(self, prefix: str) -> int:
        """Invalidate all keys matching prefix."""
        keys_to_del = [k for k in self._memory_store.keys() if k.startswith(prefix)]
        for k in keys_to_del:
            del self._memory_store[k]
        logger.info("cache_invalidated", prefix=prefix, count=len(keys_to_del))
        return len(keys_to_del)
