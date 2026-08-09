"""Pipeline Trace Engine for HalluciSense Phase 25.

Generates comprehensive, stage-by-stage diagnostic traces for every API request.
Traces capture timing (ms), memory deltas (MB), confidence metrics, intermediate payloads,
and root-cause classifications, saving TRACE_<uuid>.json artifacts to backend/traces/.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)

TRACES_DIR = settings.get_resolved_trace_dir()
TRACES_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class StageTrace:
    stage_name: str
    duration_ms: float
    memory_mb: float
    confidence: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


class PipelineTracer:
    """Stage-by-stage execution tracer for HalluciSense pipeline."""

    def __init__(self, trace_id: Optional[str] = None):
        self.trace_id = trace_id or f"TRACE_{uuid.uuid4().hex[:12].upper()}"
        self.start_time = time.time()
        self.process = psutil.Process()
        self.start_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        self.stages: List[StageTrace] = []
        self.trace_payload: Dict[str, Any] = {
            "trace_id": self.trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "start_time_unix": self.start_time,
            "environment": "production",
            "stages": {},
            "summary": {},
        }

    def record_stage(
        self,
        stage_name: str,
        duration_ms: float,
        details: Dict[str, Any],
        confidence: Optional[float] = None,
    ) -> None:
        """Record execution metrics for a specific pipeline stage."""
        current_mem = self.process.memory_info().rss / (1024 * 1024)
        mem_delta = round(current_mem - self.start_memory_mb, 2)

        stage_trace = StageTrace(
            stage_name=stage_name,
            duration_ms=round(duration_ms, 2),
            memory_mb=mem_delta,
            confidence=round(confidence, 4) if confidence is not None else None,
            details=details,
        )
        self.stages.append(stage_trace)

        self.trace_payload["stages"][stage_name] = {
            "duration_ms": round(duration_ms, 2),
            "memory_mb": mem_delta,
            "confidence": round(confidence, 4) if confidence is not None else None,
            "details": details,
        }

    def finalize(
        self,
        final_h_score: float,
        risk_level: str,
        root_cause: str = "VERIFIED",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Finalize trace recording and persist JSON to backend/traces/."""
        total_duration_ms = round((time.time() - self.start_time) * 1000, 2)
        end_mem = self.process.memory_info().rss / (1024 * 1024)

        summary = {
            "total_duration_ms": total_duration_ms,
            "total_memory_mb": round(end_mem - self.start_memory_mb, 2),
            "final_h_score": round(final_h_score, 4),
            "risk_level": risk_level,
            "root_cause_classification": root_cause,
            "stage_count": len(self.stages),
            "metadata": metadata or {},
        }
        if metadata and "performance_timings" in metadata:
            self.trace_payload["performance_timings"] = metadata["performance_timings"]
        self.trace_payload["summary"] = summary

        trace_file = TRACES_DIR / f"{self.trace_id}.json"
        try:
            with open(trace_file, "w", encoding="utf-8") as f:
                json.dump(self.trace_payload, f, indent=2, default=str)
            logger.info("trace_file_persisted", trace_id=self.trace_id, path=str(trace_file))
        except Exception as e:
            logger.error("trace_persist_failed", trace_id=self.trace_id, error=str(e))

        return self.trace_payload


def get_latest_trace() -> Optional[Dict[str, Any]]:
    """Retrieve the most recent trace payload from backend/traces/."""
    trace_files = sorted(TRACES_DIR.glob("TRACE_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not trace_files:
        return None
    try:
        with open(trace_files[0], "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_trace_by_id(trace_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a specific trace JSON payload by ID."""
    clean_id = trace_id.strip().upper()
    if not clean_id.startswith("TRACE_"):
        clean_id = f"TRACE_{clean_id}"
    path = TRACES_DIR / f"{clean_id}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
