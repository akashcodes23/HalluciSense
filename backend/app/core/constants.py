"""
Application-wide constants and enumerations.
"""
from enum import Enum


class UserRole(str, Enum):
    USER = "USER"
    ADMIN = "ADMIN"


class VerificationStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class MessageRole(str, Enum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    SYSTEM = "SYSTEM"


class RiskLevel(str, Enum):
    VERIFIED = "VERIFIED"
    NEEDS_VERIFICATION = "NEEDS_VERIFICATION"
    LIKELY_HALLUCINATED = "LIKELY_HALLUCINATED"


class LLMProvider(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    OLLAMA = "ollama"


# ── Pagination Defaults ────────────────────────────────────────────────────────
DEFAULT_PAGE_SIZE: int = 20
MAX_PAGE_SIZE: int = 100

# ── Cache TTL (seconds) ────────────────────────────────────────────────────────
CACHE_TTL_USER_SESSION: int = 900        # 15 minutes
CACHE_TTL_CHAT_LIST: int = 300           # 5 minutes
CACHE_TTL_VERIFICATION: int = 3600       # 60 minutes
CACHE_TTL_LLM_RESPONSE: int = 86400      # 24 hours
CACHE_TTL_ANALYTICS: int = 600           # 10 minutes

# ── Token Prefixes for Redis ───────────────────────────────────────────────────
REDIS_PREFIX_USER: str = "user:"
REDIS_PREFIX_TOKEN_BLACKLIST: str = "blacklist:"
REDIS_PREFIX_CHAT_LIST: str = "chats:"
REDIS_PREFIX_VERIFICATION: str = "verification:"

# ── Colour codes for risk levels ──────────────────────────────────────────────
RISK_COLORS: dict[str, str] = {
    RiskLevel.VERIFIED: "#10B981",
    RiskLevel.NEEDS_VERIFICATION: "#F59E0B",
    RiskLevel.LIKELY_HALLUCINATED: "#EF4444",
}
