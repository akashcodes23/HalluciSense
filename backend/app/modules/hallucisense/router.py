"""FastAPI Router for HalluciSense Phase 7 Endpoints.

Provides production REST endpoints:
- POST /api/v1/hallucisense/predict
- POST /api/v1/hallucisense/explain
- GET  /api/v1/hallucisense/health
- GET  /api/v1/hallucisense/version
- GET  /api/v1/hallucisense/metrics
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.pipeline import pipeline
from app.models.registry import registry

router = APIRouter(prefix="/hallucisense", tags=["HalluciSense Core"])


class PredictRequest(BaseModel):
    response_text: str = Field(..., description="LLM generated response text to analyze.")
    claims: Optional[List[str]] = Field(None, description="Optional pre-extracted claims list.")
    feature_vector: Optional[List[float]] = Field(None, description="Optional 19-dimensional hybrid feature vector.")


class ExplainRequest(BaseModel):
    response_text: str = Field(..., description="LLM generated response text to analyze.")
    claims: Optional[List[str]] = Field(None, description="Optional claims list.")


@router.post("/predict", response_model=Dict[str, Any], summary="Predict Hallucination Probability")
def predict_hallucination(req: PredictRequest) -> Dict[str, Any]:
    """Execute production hybrid inference pipeline on text input."""
    try:
        res = pipeline.predict(
            response_text=req.response_text,
            claims=req.claims,
            feature_vector=req.feature_vector,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/explain", response_model=Dict[str, Any], summary="Explain Hallucination Decision")
def explain_hallucination(req: ExplainRequest) -> Dict[str, Any]:
    """Generate detailed claim-level explanation for hallucination prediction."""
    try:
        res = pipeline.predict(response_text=req.response_text, claims=req.claims)
        return {
            "response_text": req.response_text,
            "prediction": res,
            "explanation_breakdown": res["explanation"],
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/health", response_model=Dict[str, Any], summary="System Health Audit")
def health_check() -> Dict[str, Any]:
    """Audit model registry and system health status."""
    checksums = registry.verify_checksums()
    all_healthy = all(checksums.values())
    return {
        "status": "healthy" if all_healthy else "degraded",
        "model_registry": checksums,
        "timestamp": time.time(),
    }


@router.get("/version", response_model=Dict[str, Any], summary="Version Metadata")
def version_metadata() -> Dict[str, Any]:
    """Return framework version metadata, git commit hash, and dataset SHA-256 fingerprints."""
    version_file = Path(__file__).resolve().parent.parent.parent.parent / "config" / "version.json"
    if version_file.exists():
        with open(version_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "framework": "HalluciSense",
        "version": "1.0.0",
        "status": "Production Packaged",
    }


@router.get("/metrics", response_model=Dict[str, Any], summary="System Performance Metrics")
def system_metrics() -> Dict[str, Any]:
    """Return frozen validation metrics and system benchmark metrics."""
    return {
        "pillar1_roc_auc": 0.6259,
        "pillar2_roc_auc": 0.5784,
        "hybrid_heldout_roc_auc": 0.6558,
        "hybrid_heldout_mcc": 0.1945,
        "operating_threshold": pipeline.threshold,
        "model_status": "FROZEN AND VALIDATED",
    }
