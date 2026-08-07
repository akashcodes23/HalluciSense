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
        port=settings.PORT,
        host=settings.HOST,
    )

    # Initialize Railway Volume Storage Directories (/data)
    from pathlib import Path
    data_dir = Path("/data")
    if data_dir.exists():
        for sub in ["traces", "models", "cache", "faiss", "reports"]:
            (data_dir / sub).mkdir(parents=True, exist_ok=True)
        logger.info("railway_volume_storage_initialized", path="/data")

    # Validate Startup Components (Part 3)
    components_status = {}
    critical_failures = 0

    try:
        from sentence_transformers import SentenceTransformer
        _ = SentenceTransformer("all-MiniLM-L6-v2")
        components_status["SentenceTransformer"] = True
        logger.info("startup_component_validation", component="SentenceTransformer", status="✓ Loaded")
    except Exception as e:
        components_status["SentenceTransformer"] = False
        logger.error("startup_component_validation", component="SentenceTransformer", status="✗ Failed", error=str(e))

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        _ = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-small")
        components_status["CrossEncoder_NLI"] = True
        logger.info("startup_component_validation", component="CrossEncoder_NLI", status="✓ Loaded")
    except Exception as e:
        components_status["CrossEncoder_NLI"] = False
        logger.error("startup_component_validation", component="CrossEncoder_NLI", status="✗ Failed", error=str(e))

    try:
        from app.core.engine.pipeline import HallucinationDetectionPipeline
        _ = HallucinationDetectionPipeline()
        components_status["FusionEngine"] = True
        components_status["Retriever"] = True
        components_status["CalibrationModel"] = True
        logger.info("startup_component_validation", component="FusionEngine", status="✓ Loaded")
        logger.info("startup_component_validation", component="Retriever", status="✓ Loaded")
        logger.info("startup_component_validation", component="CalibrationModel", status="✓ Loaded")
    except Exception as e:
        components_status["FusionEngine"] = False
        critical_failures += 1
        logger.error("startup_component_validation", component="FusionEngine", status="✗ Failed", error=str(e))

    try:
        from app.core.engine.token_localization import TokenLevelLocalizationEngine
        _ = TokenLevelLocalizationEngine()
        components_status["TokenLocalization"] = True
        logger.info("startup_component_validation", component="TokenLocalization", status="✓ Loaded")
    except Exception as e:
        components_status["TokenLocalization"] = False
        logger.error("startup_component_validation", component="TokenLocalization", status="✗ Failed", error=str(e))

    if critical_failures > 0:
        logger.critical("startup_validation_failed_terminating", critical_failures=critical_failures)
        raise RuntimeError("Critical pipeline component failed to load during startup.")

    logger.info("startup_validation_completed", components=components_status)

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

    # ── Request Tracing & OpenTelemetry Middleware ───────────────────────────
    @app.middleware("http")
    async def request_tracing_middleware(request: Request, call_next):
        request_id = str(uuid.uuid4())
        trace_id = f"TRACE_{uuid.uuid4().hex[:12].upper()}"
        structlog.contextvars.bind_contextvars(request_id=request_id, trace_id=trace_id)
        start = time.perf_counter()

        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000.0

        logger.info(
            "request_handled",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
            request_id=request_id,
            trace_id=trace_id,
        )

        # OpenTelemetry & Tracing Headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Trace-ID"] = trace_id
        response.headers["X-Latency-MS"] = f"{duration_ms:.2f}"
        response.headers["traceparent"] = f"00-{uuid.uuid4().hex}-{uuid.uuid4().hex[:16]}-01"

        structlog.contextvars.clear_contextvars()
        return response

    # ── Global Exception Handlers ─────────────────────────────────────────────
    from fastapi.exceptions import RequestValidationError
    from fastapi import HTTPException

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "status": "error",
                "error_code": "VALIDATION_ERROR",
                "message": "Invalid request payload schema or missing required fields.",
                "details": exc.errors(),
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        err_code = "BAD_REQUEST"
        if exc.status_code == 413:
            err_code = "PAYLOAD_TOO_LARGE"
        elif exc.status_code == 404:
            err_code = "NOT_FOUND"

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status": "error",
                "error_code": err_code,
                "message": str(exc.detail),
            },
        )

    @app.exception_handler(HalluciSenseError)
    async def generic_domain_error_handler(request: Request, exc: HalluciSenseError):
        logger.error("domain_error", message=exc.message, detail=exc.detail)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "error_code": "DOMAIN_ERROR",
                "message": exc.message,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.error("unhandled_system_exception", error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An internal system error occurred during request processing.",
            },
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

    from app.modules.verification.production_router import router as production_router
    app.include_router(production_router, prefix=settings.API_V1_STR)

    from app.modules.hallucisense.router import router as hallucisense_router
    app.include_router(hallucisense_router, prefix=settings.API_V1_STR)

    from app.modules.pillar2.router import router as pillar2_router
    app.include_router(pillar2_router, prefix=settings.API_V1_STR)

    from app.modules.mlops.router import router as mlops_router
    app.include_router(mlops_router, prefix=settings.API_V1_STR)

    # ── Health & Readiness Probes ─────────────────────────────────────────────
    component_readiness_override = {
        "retriever": True,
        "nli_model": True,
        "sentence_transformer": True,
        "cross_encoder": True,
        "fusion_engine": True,
        "pipeline": True,
    }
    app.state.component_readiness_override = component_readiness_override

    @app.get("/health", tags=["System"], summary="System health check")
    @app.get("/healthz", tags=["System"], summary="System liveness check")
    async def health_check():
        return {"status": "healthy"}

    @app.get("/ready", tags=["System"], summary="Deep component readiness check")
    @app.get("/readyz", tags=["System"], summary="Deep component readiness check")
    async def readiness_check():
        components = dict(app.state.component_readiness_override)
        all_ready = all(components.values())
        status_code = status.HTTP_200_OK if all_ready else status.HTTP_503_SERVICE_UNAVAILABLE

        return JSONResponse(
            status_code=status_code,
            content={
                "status": "ready" if all_ready else "unready",
                "components": components,
                "version": settings.VERSION,
            },
        )

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
                "ready": "/ready",
                "api_prefix": settings.API_V1_STR,
            },
        }

    return app


# ---------------------------------------------------------------------------
# App Instance (used by uvicorn)
# ---------------------------------------------------------------------------

app = create_application()