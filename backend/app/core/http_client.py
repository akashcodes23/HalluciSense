"""Shared Long-Lived Async HTTP Client for HalluciSense LLM Providers & API Clients.

Provides connection pooling, keepalive connection reuse, and bounded timeouts
to eliminate TCP/TLS handshake overhead across concurrent LLM generation calls.
"""

from __future__ import annotations

import httpx
from typing import Optional

_shared_async_client: Optional[httpx.AsyncClient] = None


def get_shared_async_client() -> httpx.AsyncClient:
    """Return long-lived thread-safe AsyncClient with connection pooling."""
    global _shared_async_client
    if _shared_async_client is None or _shared_async_client.is_closed:
        _shared_async_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=50,
                keepalive_expiry=30.0,
            ),
            timeout=httpx.Timeout(
                connect=5.0,
                read=15.0,
                write=15.0,
                pool=10.0,
            ),
        )
    return _shared_async_client


async def close_shared_async_client():
    """Cleanly close shared AsyncClient on app shutdown."""
    global _shared_async_client
    if _shared_async_client is not None and not _shared_async_client.is_closed:
        await _shared_async_client.aclose()
        _shared_async_client = None
