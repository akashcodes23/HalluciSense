"""
Domain exception hierarchy.
All application errors inherit from HalluciSenseError so they can be
caught uniformly in the FastAPI exception handlers.
"""
from typing import Any, Optional


class HalluciSenseError(Exception):
    """Base exception for the HalluciSense application."""

    def __init__(self, message: str, detail: Optional[Any] = None) -> None:
        self.message = message
        self.detail = detail
        super().__init__(message)


# ── Authentication & Authorisation ────────────────────────────────────────────

class AuthenticationError(HalluciSenseError):
    """Raised when credentials are invalid or missing."""


class TokenExpiredError(AuthenticationError):
    """Raised when a JWT has expired."""


class TokenInvalidError(AuthenticationError):
    """Raised when a JWT signature or structure is invalid."""


class InsufficientPermissionsError(HalluciSenseError):
    """Raised when the authenticated user lacks required permissions."""


# ── Resource Errors ───────────────────────────────────────────────────────────

class NotFoundError(HalluciSenseError):
    """Raised when a requested resource does not exist."""

    def __init__(self, resource: str, identifier: Any) -> None:
        super().__init__(
            message=f"{resource} with identifier '{identifier}' was not found.",
            detail={"resource": resource, "identifier": str(identifier)},
        )


class ConflictError(HalluciSenseError):
    """Raised when a uniqueness constraint would be violated."""


class ValidationError(HalluciSenseError):
    """Raised when business-rule validation fails (distinct from Pydantic)."""


# ── AI Provider Errors ────────────────────────────────────────────────────────

class ProviderUnavailableError(HalluciSenseError):
    """Raised when an AI provider is unreachable or returns an error."""

    def __init__(self, provider: str, reason: str) -> None:
        super().__init__(
            message=f"AI provider '{provider}' is unavailable: {reason}",
            detail={"provider": provider, "reason": reason},
        )


class ProviderRateLimitError(ProviderUnavailableError):
    """Raised when the provider returns a 429 / quota-exceeded response."""


# ── Verification Engine Errors ────────────────────────────────────────────────

class VerificationError(HalluciSenseError):
    """Raised when the hallucination detection pipeline encounters an error."""


class KnowledgeSourceError(HalluciSenseError):
    """Raised when an external knowledge source fails to respond."""
