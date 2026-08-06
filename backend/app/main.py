"""
HalluciSense FastAPI Application Factory.
"""
import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.exceptions import (
    AuthenticationError,
    ConflictError,
    HalluciSenseError,
    InsufficientPermissionsError,
    NotFoundError,
)
from app.modules.auth.router import router as auth_router
from app.modules.users.router import router as users_router
from app.modules.chat.router import router as chat_router
from app.modules.messages.router import router as messages_router
from app.modules.verification.router import router as verification_router
from app.modules.notifications.router import router as notifications_router
from app.modules.analytics.router import router as analytics_router
from app.modules.export.router import router as export_router
from app.modules.admin.router import router as admin_router

# ---------------------------------------------------------------------------
# Structured Logging Setup
# ---------------------------------------------------------------------------

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer() if not settings.is_production
        else structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown events)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    logger.info(
        "HalluciSense starting",
        version=settings.VERSION,
        env=settings.APP_ENV,
    )
    yield
    logger.info("HalluciSense shutting down")


# ---------------------------------------------------------------------------
# Application Factory
# ---------------------------------------------------------------------------

def create_application() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=(
            "HalluciSense — A Confidence-Aware Hybrid Framework "
            "for Detecting and Quantifying Hallucinations in LLMs."
        ),
        # ✅ FIXED: Docs at root so they work on Railway
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS & Compression ───────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ── Request Tracing Middleware ─────────────────────────────────────────────
    @app.middleware("http")
    async def request_tracing_middleware(request: Request, call_next):
        request_id = str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(request_id=request_id)
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request_handled",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        response.headers["X-Request-ID"] = request_id
        structlog.contextvars.clear_contextvars()
        return response

    # ── Global Exception Handlers ─────────────────────────────────────────────
    @app.exception_handler(NotFoundError)
    async def not_found_handler(request: Request, exc: NotFoundError):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": exc.message, "extra": exc.detail},
        )

    @app.exception_handler(ConflictError)
    async def conflict_handler(request: Request, exc: ConflictError):
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": exc.message},
        )

    @app.exception_handler(AuthenticationError)
    async def auth_error_handler(request: Request, exc: AuthenticationError):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": exc.message},
        )

    @app.exception_handler(InsufficientPermissionsError)
    async def permission_error_handler(request: Request, exc: InsufficientPermissionsError):
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"detail": exc.message},
        )

    @app.exception_handler(HalluciSenseError)
    async def generic_domain_error_handler(request: Request, exc: HalluciSenseError):
        logger.error("unhandled_domain_error", message=exc.message, detail=exc.detail)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal error occurred."},
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(auth_router, prefix=settings.API_V1_STR)
    app.include_router(users_router, prefix=settings.API_V1_STR)
    app.include_router(chat_router, prefix=settings.API_V1_STR)
    app.include_router(messages_router, prefix=settings.API_V1_STR)
    app.include_router(verification_router, prefix=settings.API_V1_STR)
    app.include_router(notifications_router, prefix=settings.API_V1_STR)
    app.include_router(analytics_router, prefix=settings.API_V1_STR)
    app.include_router(export_router, prefix=settings.API_V1_STR)
    app.include_router(admin_router, prefix=settings.API_V1_STR)

    from app.modules.hallucisense.router import router as hallucisense_router
    app.include_router(hallucisense_router, prefix=settings.API_V1_STR)

    from app.modules.pillar2.router import router as pillar2_router
    app.include_router(pillar2_router, prefix=settings.API_V1_STR)

    from app.modules.mlops.router import router as mlops_router
    app.include_router(mlops_router, prefix=settings.API_V1_STR)

    # ── Health Check Probes ───────────────────────────────────────────────────
    @app.get("/health", tags=["System"], summary="Health check")
    @app.get("/healthz", tags=["System"], summary="Liveness check")
    @app.get("/ready", tags=["System"], summary="Readiness check")
    @app.get("/readyz", tags=["System"], summary="Readiness check")
    async def health_check():
        return {"status": "ok", "service": settings.PROJECT_NAME, "version": settings.VERSION}

    # ✅ ADDED: Root endpoint (fixes Railway "Not Found")
    @app.get("/", tags=["System"], summary="Root endpoint")
    async def root():
        return {
            "message": "HalluciSense API is running",
            "service": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "environment": settings.APP_ENV,
            "endpoints": {
                "docs": "/docs",
                "health": "/health",
                "api_prefix": settings.API_V1_STR,
            },
        }

    return app


# ---------------------------------------------------------------------------
# App Instance (used by uvicorn)
# ---------------------------------------------------------------------------

app = create_application()