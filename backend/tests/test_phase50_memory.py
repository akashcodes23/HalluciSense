"""Phase 50 — Memory Boundedness & Retention Prevention Tests."""

import os
import psutil
import pytest
from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.core.engine.memory_utils import trim_process_memory


def test_sequential_requests_memory_stability():
    """Verify that multiple sequential requests do not exhibit runaway memory growth."""
    pipeline = HallucinationDetectionPipeline()
    process = psutil.Process(os.getpid())

    # Warmup
    _ = pipeline.analyze("The capital of France is Paris.")
    initial_rss = process.memory_info().rss / (1024 * 1024)

    test_queries = [
        "The capital of France is Paris.",
        "The capital of France is Berlin.",
        "Water freezes at 0 degrees Celsius.",
        "Paris is the capital of France. Berlin is the capital of Germany.",
        "12 multiplied by 8 equals 96.",
    ]

    for q in test_queries:
        _ = pipeline.analyze(q)

    trim_process_memory()
    final_rss = process.memory_info().rss / (1024 * 1024)
    growth = final_rss - initial_rss

    # Verify retained memory is bounded (< 85 MB)
    assert growth < 85.0, f"Unbounded memory growth detected: {growth:.2f} MB!"
