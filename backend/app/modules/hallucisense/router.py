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
    """Audit model registry and system health status with production diagnostic metadata."""
    checksums = registry.verify_checksums()
    detailed_status = registry.get_detailed_health_status()

    hybrid_available = detailed_status["loaded_successfully"]
    p1_available = checksums.get("pillar1_classifier_exists", True)

    if hybrid_available:
        active_model = "hybrid"
        fallback_active = False
        service_status = "ok"
    elif p1_available:
        active_model = "pillar1"
        fallback_active = True
        service_status = "ok"
    else:
        active_model = "none"
        fallback_active = False
        service_status = "degraded"

    return {
        "status": service_status,
        "active_model": active_model,
        "hybrid_available": hybrid_available,
        "fallback_active": fallback_active,
        "resolved_model_directory": detailed_status["resolved_results_directory"],
        "cwd": detailed_status["cwd"],
        "artifacts_found": detailed_status["artifacts_found"],
        "artifact_sizes": detailed_status["artifact_sizes"],
        "loaded_successfully": hybrid_available,
        "current_sklearn_version": detailed_status["current_sklearn_version"],
        "current_joblib_version": detailed_status["current_joblib_version"],
        "model_registry": checksums,
        "timestamp": time.time(),
    }


@router.get("/debug/filesystem", response_model=Dict[str, Any], summary="Filesystem Diagnostic Audit")
def filesystem_debug() -> Dict[str, Any]:
    """Temporary diagnostic endpoint inspecting runtime directory layout, joblib files, and git SHA."""
    import os
    import sklearn
    import joblib
    from pathlib import Path
    from app.models.registry import BASE_DIR, registry

    cwd = os.getcwd()
    cwd_path = Path(cwd)

    def scan_dir_recursive(root_dir: Path, ext_pattern: str, max_depth: int = 4) -> List[str]:
        if not root_dir.exists() or not root_dir.is_dir():
            return []
        matches = []
        try:
            for p in root_dir.rglob(ext_pattern):
                if len(p.relative_to(root_dir).parts) <= max_depth:
                    matches.append(str(p))
        except Exception:
            pass
        return matches[:50]

    joblib_files = scan_dir_recursive(cwd_path, "*.joblib")
    if not joblib_files and (cwd_path / "backend").exists():
        joblib_files = scan_dir_recursive(cwd_path / "backend", "*.joblib")

    json_files = scan_dir_recursive(cwd_path, "*.json")

    detailed_status = registry.get_detailed_health_status()
    git_sha = os.getenv("RAILWAY_GIT_COMMIT_SHA") or os.getenv("GIT_COMMIT_SHA") or "33c15b36f4801d253595c2999c5734b712f435ca"

    return {
        "cwd": cwd,
        "base_dir": str(BASE_DIR),
        "resolved_model_directory": str(registry.results_dir),
        "git_commit_sha": git_sha,
        "directory_tree_cwd": [f.name for f in cwd_path.iterdir()] if cwd_path.exists() else [],
        "all_joblib_files_discovered": joblib_files,
        "all_json_files_discovered": [j for j in json_files if "metadata" in j or "schema" in j or "config" in j],
        "artifacts_found": detailed_status["artifacts_found"],
        "artifact_sizes": detailed_status["artifact_sizes"],
        "loaded_successfully": detailed_status["loaded_successfully"],
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
    }


@router.get("/version", response_model=Dict[str, Any], summary="Version Metadata")
def version_metadata() -> Dict[str, Any]:
    """Return framework version metadata, git commit hash, and dependency versions."""
    import os
    import sklearn
    import joblib

    git_sha = os.getenv("RAILWAY_GIT_COMMIT_SHA") or os.getenv("GIT_COMMIT_SHA") or "33c15b36f4801d253595c2999c5734b712f435ca"
    return {
        "framework": "HalluciSense",
        "version": "1.0.0",
        "git_sha": git_sha,
        "build_timestamp": "2026-08-05T12:25:59Z",
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
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
