"""
HalluciSense SaaS — Sprint 7: Sentry Error Tracking & Tracing
==============================================================
Initializes Sentry SDK for error reporting and performance transaction monitoring.
"""

from __future__ import annotations

import os
import structlog

logger = structlog.get_logger(__name__)

SENTRY_DSN = os.getenv("SENTRY_DSN", "")


def init_sentry() -> bool:
    """Initialize Sentry error tracking."""
    if not SENTRY_DSN:
        logger.info("sentry_dsn_not_configured_skipping")
        return False

    try:
        import sentry_sdk
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            traces_sample_rate=0.1,
            profiles_sample_rate=0.1,
            environment=os.getenv("ENVIRONMENT", "production"),
        )
        logger.info("sentry_initialized_successfully")
        return True
    except Exception as e:
        logger.warning("sentry_init_failed", error=str(e))
        return False
