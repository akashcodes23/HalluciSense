"""Production Metrics Tracker Singleton for HalluciSense.

Tracks runtime statistics:
- Total Requests
- Total Latency (ms)
- Total H-Score
- Successful Requests
- Error Requests
- Process Memory (MB)
"""

from __future__ import annotations

import os
import threading
from typing import Dict, Any

import psutil
import structlog

logger = structlog.get_logger(__name__)


class MetricsTracker:
    """Thread-safe runtime metrics tracker singleton."""

    _instance: MetricsTracker | None = None
    _lock = threading.Lock()

    def __new__(cls) -> MetricsTracker:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_tracker()
            return cls._instance

    def _init_tracker(self) -> None:
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._total_latency_ms = 0.0
        self._total_h_score = 0.0
        self._stage_latencies: Dict[str, float] = {
            "retrieval_ms": 0.0,
            "nli_ms": 0.0,
            "confidence_ms": 0.0,
            "consistency_ms": 0.0,
            "fusion_ms": 0.0,
            "risk_ms": 0.0,
            "localization_ms": 0.0,
            "serialization_ms": 0.0,
        }
        self._process = psutil.Process(os.getpid())

    def record_request(
        self,
        latency_ms: float,
        h_score: float,
        is_success: bool = True,
        stage_timings: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record request telemetry."""
        with self._lock:
            self._total_requests += 1
            if is_success:
                self._successful_requests += 1
                self._total_latency_ms += max(0.0, latency_ms)
                self._total_h_score += max(0.0, min(1.0, h_score))
                if stage_timings:
                    for k, v in stage_timings.items():
                        if k in self._stage_latencies:
                            self._stage_latencies[k] += max(0.0, float(v))
            else:
                self._failed_requests += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Return runtime statistics snapshot."""
        with self._lock:
            reqs = self._total_requests
            succ = self._successful_requests
            fails = self._failed_requests

            if reqs == 0:
                avg_lat = None
                avg_h = None
                succ_rate = None
                err_rate = None
                avg_stage_latencies = {k: None for k in self._stage_latencies}
                status_str = "AWAITING_FIRST_RUN"
            else:
                avg_lat = round(self._total_latency_ms / float(max(1, succ)), 2) if succ > 0 else None
                avg_h = round(self._total_h_score / float(max(1, succ)), 4) if succ > 0 else None
                succ_rate = round((succ / float(reqs)) * 100.0, 2)
                err_rate = round((fails / float(reqs)) * 100.0, 2)
                avg_stage_latencies = {
                    k: round(v / float(max(1, succ)), 2) if succ > 0 else None
                    for k, v in self._stage_latencies.items()
                }
                status_str = "READY"

            try:
                mem_mb = round(self._process.memory_info().rss / (1024 * 1024), 2)
            except Exception:
                mem_mb = 256.0

            return {
                "requests": reqs,
                "successful_requests": succ,
                "failed_requests": fails,
                "average_latency_ms": avg_lat,
                "average_h_score": avg_h,
                "success_rate": succ_rate,
                "error_rate": err_rate,
                "memory_mb": mem_mb,
                "status": status_str,
                "average_stage_latencies_ms": avg_stage_latencies,
            }


def get_metrics_tracker() -> MetricsTracker:
    """Return the global MetricsTracker singleton instance."""
    return MetricsTracker()
