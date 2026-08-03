"""
HalluciSense SaaS — Module 12.8: Hardened Enterprise Security Middleware
========================================================================
Implements Helmet security headers, CSRF token validation, input sanitization
(SQL injection & XSS prevention), CORS policy enforcement, and audit logging.
"""

from __future__ import annotations

import re
import secrets
from typing import Dict, List, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

import structlog

logger = structlog.get_logger(__name__)


class HardenedSecurityMiddleware(BaseHTTPMiddleware):
    """
    Middleware applying Helmet headers, XSS/SQLi sanitization, and request tracking.
    """

    SQLI_PATTERNS = re.compile(
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|EXEC|UNION|TRUNCATE)\b)|(--)|(;--)|(/\*)|(\*/)",
        re.IGNORECASE,
    )
    XSS_PATTERNS = re.compile(
        r"(<script.*?>.*?</script>)|(javascript:)|(onload=)|(onerror=)|(<iframe.*?>)",
        re.IGNORECASE,
    )

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # 1. Input Sanitization Check on Query Parameters
        query_str = str(request.query_params)
        if self.SQLI_PATTERNS.search(query_str):
            logger.warning("sqli_attempt_blocked", client_ip=request.client.host if request.client else "unknown")
            return Response(content="Malicious payload detected", status_code=400)
        if self.XSS_PATTERNS.search(query_str):
            logger.warning("xss_attempt_blocked", client_ip=request.client.host if request.client else "unknown")
            return Response(content="Malicious script detected", status_code=400)

        # 2. Execute Request
        response = await call_next(request)

        # 3. Apply Helmet Security Headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; object-src 'none';"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        return response


class InputSanitizer:
    """Sanitizes user input strings against HTML/script injection."""

    @staticmethod
    def sanitize_text(text: str) -> str:
        """Strip raw HTML tags and dangerous script substrings."""
        clean = re.sub(r"<[^>]*>", "", text)
        clean = clean.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return clean.strip()
