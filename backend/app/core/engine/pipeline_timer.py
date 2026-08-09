"""Pipeline Timer Utility & Monotonic Stage Profiler for HalluciSense.

Provides a clean context manager and high-resolution timing utility
(using time.perf_counter()) to capture stage-by-stage latencies, emit
structured JSON logs without logging sensitive prompt contents, and rank
the slowest stages.
"""

from __future__ import annotations

import time
import logging
import json
from typing import Dict, List, Any, Optional
from contextlib import contextmanager

logger = logging.getLogger("hallucisense.perf")


class StageMeasurement:
    def __init__(self, stage: str, start_time: float):
        self.stage = stage
        self.start_timestamp = start_time
        self.end_timestamp: float = start_time
        self.duration_ms: float = 0.0

    def finish(self, end_time: float) -> float:
        self.end_timestamp = end_time
        self.duration_ms = round((end_time - self.start_timestamp) * 1000.0, 4)
        return self.duration_ms


class PipelineTimer:
    """Monotonic high-resolution timer for pipeline profiling."""

    def __init__(self, trace_id: str = "TRACE_INIT"):
        self.trace_id = trace_id
        self.start_timestamp = time.perf_counter()
        self.end_timestamp: float = self.start_timestamp
        self.total_ms: float = 0.0
        self.measurements: Dict[str, StageMeasurement] = {}

    @contextmanager
    def stage(self, stage_name: str):
        """Context manager to time a pipeline stage or sub-stage."""
        t_start = time.perf_counter()
        measurement = StageMeasurement(stage_name, t_start)
        try:
            yield measurement
        finally:
            t_end = time.perf_counter()
            measurement.finish(t_end)
            self.measurements[stage_name] = measurement
            
            # Emit structured stage log
            log_data = {
                "event": "pipeline_stage",
                "trace_id": self.trace_id,
                "stage": stage_name,
                "start_timestamp": round(measurement.start_timestamp, 6),
                "end_timestamp": round(measurement.end_timestamp, 6),
                "duration_ms": measurement.duration_ms,
            }
            logger.info(json.dumps(log_data))

    def record(self, stage_name: str, duration_ms: float, start_t: Optional[float] = None, end_t: Optional[float] = None):
        """Directly record pre-calculated timing for sub-components."""
        now = time.perf_counter()
        m = StageMeasurement(stage_name, start_t if start_t is not None else now)
        m.end_timestamp = end_t if end_t is not None else now
        m.duration_ms = round(max(0.0, duration_ms), 4)
        self.measurements[stage_name] = m
        
        log_data = {
            "event": "pipeline_stage",
            "trace_id": self.trace_id,
            "stage": stage_name,
            "start_timestamp": round(m.start_timestamp, 6),
            "end_timestamp": round(m.end_timestamp, 6),
            "duration_ms": m.duration_ms,
        }
        logger.info(json.dumps(log_data))

    def finish(self) -> float:
        """Finalize total timer and emit aggregate JSON summary."""
        self.end_timestamp = time.perf_counter()
        self.total_ms = round((self.end_timestamp - self.start_timestamp) * 1000.0, 4)

        stages_summary = {
            name: m.duration_ms
            for name, m in self.measurements.items()
        }

        # Structured aggregate log
        aggregate_log = {
            "event": "pipeline_timing",
            "trace_id": self.trace_id,
            "total_ms": self.total_ms,
            "stages": stages_summary,
        }
        logger.info(json.dumps(aggregate_log))

        # Identify top 3 slowest stages
        sorted_stages = sorted(
            [{"stage": name, "duration_ms": m.duration_ms} for name, m in self.measurements.items()],
            key=lambda x: x["duration_ms"],
            reverse=True
        )
        top_3 = sorted_stages[:3]

        top_slowest_log = {
            "event": "top_slowest_stages",
            "trace_id": self.trace_id,
            "slowest_stages": top_3,
        }
        logger.info(json.dumps(top_slowest_log))

        # Console print for live visibility
        print(f"\n[PERF_TIMING] Trace={self.trace_id} Total={self.total_ms:.2f}ms TopSlowest={[s['stage'] + ':' + str(s['duration_ms']) + 'ms' for s in top_3]}")

        return self.total_ms

    def get_summary(self) -> Dict[str, Any]:
        """Return dict of measured stage latencies."""
        return {
            "trace_id": self.trace_id,
            "total_ms": self.total_ms,
            "stages": {name: m.duration_ms for name, m in self.measurements.items()},
        }
