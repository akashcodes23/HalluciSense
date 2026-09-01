"""Phase 44 — Observability & Verification Metrics Tracker.

Tracks runtime metrics without adding external dependencies:
- Request counts & latency
- Routing modality distribution
- Verification state distributions (Verified, Contradicted, Insufficient Evidence)
- Retrieval failures & timeouts
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict


class VerificationMetricsTracker:
    """Thread-safe in-memory metrics aggregator."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(VerificationMetricsTracker, cls).__new__(cls)
                    cls._instance._init_metrics()
        return cls._instance

    def _init_metrics(self):
        self.total_requests = 0
        self.total_claims_evaluated = 0
        self.verified_claims_count = 0
        self.contradicted_claims_count = 0
        self.insufficient_evidence_count = 0
        self.symbolic_evaluations_count = 0
        self.retrieval_requests_count = 0
        self.retrieval_failures_count = 0
        self.total_latency_ms = 0.0
        self.created_at = time.time()

    def record_request(
        self,
        claim_count: int,
        verified: int,
        contradicted: int,
        insufficient: int,
        symbolic: int,
        retrieval: int,
        latency_ms: float,
    ):
        with self._lock:
            self.total_requests += 1
            self.total_claims_evaluated += claim_count
            self.verified_claims_count += verified
            self.contradicted_claims_count += contradicted
            self.insufficient_evidence_count += insufficient
            self.symbolic_evaluations_count += symbolic
            self.retrieval_requests_count += retrieval
            self.total_latency_ms += latency_ms

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            avg_lat = round(self.total_latency_ms / max(self.total_requests, 1), 2)
            uptime = round(time.time() - self.created_at, 1)
            return {
                "uptime_seconds": uptime,
                "total_requests": self.total_requests,
                "total_claims_evaluated": self.total_claims_evaluated,
                "verified_claims_count": self.verified_claims_count,
                "contradicted_claims_count": self.contradicted_claims_count,
                "insufficient_evidence_count": self.insufficient_evidence_count,
                "symbolic_evaluations_count": self.symbolic_evaluations_count,
                "retrieval_requests_count": self.retrieval_requests_count,
                "average_latency_ms": avg_lat,
            }


metrics_tracker = VerificationMetricsTracker()
