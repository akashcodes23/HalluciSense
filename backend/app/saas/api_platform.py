"""
HalluciSense SaaS — Module 12.7: Enterprise API Platform & Keys
================================================================
Manages API key generation, rate limiting, quota validation, and key revocation.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from typing import Dict, Optional
from pydantic import BaseModel, Field

import structlog

logger = structlog.get_logger(__name__)


class APIKeyMetadata(BaseModel):
    key_id: str
    key_prefix: str
    name: str
    user_id: str
    rate_limit_rpm: int = 600
    is_active: bool = True
    created_at_iso: str


class APIPlatformManager:
    """
    Manages API keys, quota policies, and rate-limiting counters.
    """

    def __init__(self):
        self._key_store: Dict[str, APIKeyMetadata] = {}
        self._rate_counters: Dict[str, list[float]] = {}

    def generate_api_key(self, user_id: str, key_name: str = "Production API Key") -> tuple[str, APIKeyMetadata]:
        """
        Generate a new enterprise API key.

        Returns
        -------
        tuple[raw_key, APIKeyMetadata]
        """
        raw_key = f"hs_live_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        key_id = f"key_{secrets.token_hex(6)}"

        meta = APIKeyMetadata(
            key_id=key_id,
            key_prefix=raw_key[:12],
            name=key_name,
            user_id=user_id,
            rate_limit_rpm=600,
            is_active=True,
            created_at_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        self._key_store[key_hash] = meta
        logger.info("api_key_generated", key_id=key_id, user_id=user_id)
        return raw_key, meta

    def validate_api_key(self, raw_key: str) -> Optional[APIKeyMetadata]:
        """Validate raw key string against store."""
        key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        meta = self._key_store.get(key_hash)
        if meta and meta.is_active:
            return meta
        return None

    def check_rate_limit(self, key_id: str, limit_rpm: int = 600) -> bool:
        """Check if key has exceeded rate limit RPM within a 60s sliding window."""
        now = time.time()
        timestamps = self._rate_counters.get(key_id, [])
        # Prune timestamps older than 60 seconds
        timestamps = [t for t in timestamps if now - t < 60.0]

        if len(timestamps) >= limit_rpm:
            return False  # Rate limit exceeded

        timestamps.append(now)
        self._rate_counters[key_id] = timestamps
        return True
