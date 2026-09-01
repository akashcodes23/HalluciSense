import os
os.environ["GRPC_DNS_RESOLVER"] = "native"

from datetime import timedelta
from pathlib import Path
from typing import List
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralised application configuration loaded from environment variables.
    Supports Railway deployment environment variables and volume paths (/data).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────────
    PROJECT_NAME: str = "HalluciSense"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ENABLE_TRACING: bool = True
    ENABLE_DEBUG_API: bool = True
    TOKENIZERS_PARALLELISM: bool = False

    # ── Volume & Directory Paths ──────────────────────────────────────────────
    # Supports Railway volume mount (/data) when deployed
    DATA_VOLUME_DIR: str = "/data"
    HF_HOME: str = "/data/cache/huggingface"
    TRANSFORMERS_CACHE: str = "/data/cache/transformers"
    TRACE_DIR: str = "/data/traces"
    MODEL_DIR: str = "/data/models"
    FAISS_DIR: str = "/data/faiss"

    # ── Memory & Concurrency Guard Settings ──────────────────────────────────
    HALLUCISENSE_MEMORY_LIMIT_MB: int = 2048
    HALLUCISENSE_MEMORY_GUARD_MB: int = 1500
    MAX_CONCURRENT_ANALYSES: int = 2

    # ── Security ──────────────────────────────────────────────────────────────
    SECRET_KEY: str = "dev-secret-key-change-in-production-must-be-32-chars"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 120
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ── CORS ──────────────────────────────────────────────────────────────────
    CORS_ORIGINS: str = "*"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str) -> str:
        return v

    def get_cors_origins(self) -> List[str]:
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = (
        "postgresql+asyncpg://hallucisense:hallucisense@localhost:5432/hallucisense_db"
    )
    DATABASE_URL_SYNC: str = (
        "postgresql+psycopg2://hallucisense:hallucisense@localhost:5432/hallucisense_db"
    )
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # ── AI Providers ─────────────────────────────────────────────────────────
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    DEFAULT_LLM_PROVIDER: str = "gemini"
    DEFAULT_LLM_MODEL: str = "gemini-2.0-flash"

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = 100
    AUTH_RATE_LIMIT_PER_MINUTE: int = 20

    # ── HalluciSense Engine ──────────────────────────────────────────────────
    ALPHA_FACTUAL_ERROR: float = 0.45
    BETA_CONFIDENCE_GAP: float = 0.30
    GAMMA_CONSISTENCY_FAILURE: float = 0.25
    VERIFIED_THRESHOLD: float = 0.35
    HALLUCINATED_THRESHOLD: float = 0.65
    MIN_TOKEN_PROBABILITY: float = 1e-5

    # ── Gemini Provider & Optimization Config ───────────────────────────────
    GEMINI_QUEUE_SIZE: int = 100
    GEMINI_STREAM_TIMEOUT: float = 15.0
    GEMINI_MAX_RETRIES: int = 2
    GEMINI_GENERATION_TIMEOUT: float = 30.0
    GEMINI_MAX_BACKOFF: float = 8.0

    # ── LLM Call Optimization & Architectural Controls ───────────────────────
    ENABLE_SELF_CONSISTENCY: bool = False
    MAX_SELF_CONSISTENCY_SAMPLES: int = 5
    ENABLE_AUTOMATIC_CORRECTION: bool = False
    H_SCORE_CORRECTION_THRESHOLD: float = 0.65
    ENABLE_FALLBACK_MODELS: bool = False
    ENABLE_REQUEST_CACHE: bool = True
    HALLUCISENSE_ENABLE_RERANKER: bool = False

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def access_token_delta(self) -> timedelta:
        return timedelta(minutes=self.ACCESS_TOKEN_EXPIRE_MINUTES)

    @property
    def refresh_token_delta(self) -> timedelta:
        return timedelta(days=self.REFRESH_TOKEN_EXPIRE_DAYS)

    def get_resolved_trace_dir(self) -> Path:
        """Resolve trace directory: check /data/traces or fallback to project traces/."""
        data_traces = Path("/data/traces")
        if data_traces.parent.exists():
            data_traces.mkdir(parents=True, exist_ok=True)
            return data_traces
        
        base_dir = Path(__file__).resolve().parent.parent.parent
        local_traces = base_dir / "traces"
        local_traces.mkdir(parents=True, exist_ok=True)
        return local_traces


settings = Settings()
