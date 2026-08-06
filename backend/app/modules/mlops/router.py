"""Phase 18 — Enterprise MLOps Telemetry & Monitoring API Router.

Endpoints:
- GET /api/v1/mlops/metrics
- GET /api/v1/mlops/drift
- GET /api/v1/mlops/dashboard
- GET /api/v1/mlops/logs
"""

from __future__ import annotations

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, status

from app.modules.mlops.telemetry import mlops_telemetry

router = APIRouter(prefix="/mlops", tags=["MLOps & Telemetry"])


@router.get("/metrics", response_model=Dict[str, Any], summary="MLOps Performance & Latency Telemetry")
def get_mlops_metrics() -> Dict[str, Any]:
    """Return latency percentiles (P50, P90, P99), request rate, and memory stats."""
    return {
        "status": "ok",
        "latency": mlops_telemetry.get_latency_statistics(),
        "total_requests": len(mlops_telemetry.prediction_logs),
    }


@router.get("/drift", response_model=Dict[str, Any], summary="Feature & Prediction Drift Audit")
def audit_feature_drift() -> Dict[str, Any]:
    """Execute KS-test and Population Stability Index (PSI) drift audit on active predictions."""
    return {
        "status": "ok",
        "drift_audit": mlops_telemetry.compute_feature_drift(),
    }


@router.get("/dashboard", response_model=Dict[str, Any], summary="Unified MLOps Dashboard Payload")
def get_mlops_dashboard() -> Dict[str, Any]:
    """Return comprehensive MLOps telemetry dashboard payload."""
    return mlops_telemetry.get_dashboard_summary()


@router.get("/logs", response_model=List[Dict[str, Any]], summary="Recent Prediction Audit Logs")
def get_recent_prediction_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Return recent prediction audit logs."""
    logs = list(mlops_telemetry.prediction_logs)
    return logs[-limit:]
