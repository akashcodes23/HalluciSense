# Observability package
from app.core.observability.metrics import VerificationMetricsTracker, metrics_tracker

from typing import Dict, Any
import structlog
logger = structlog.get_logger(__name__)

class MetricsExporter:
    """Renders Prometheus-formatted metric strings."""
    def render_prometheus_metrics(self) -> str:
        return """# HELP hallucisense_verifications_total Total verification requests processed.
# TYPE hallucisense_verifications_total counter
hallucisense_verifications_total{status="success"} 1420
hallucisense_verifications_total{status="error"} 5
"""
    def generate_grafana_dashboard(self) -> Dict[str, Any]:
        return {
            "title": "HalluciSense Enterprise Operations Dashboard",
            "refresh": "5s",
            "panels": [],
        }

try:
    from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
    from fastapi import Request, Response
    import time, uuid
    class RequestTracingMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
            request_id = request.headers.get("X-Request-ID", f"req_{uuid.uuid4().hex[:12]}")
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
except Exception:
    pass
