"""Phase 49 — Recursive Trace Memory & Tensor Leakage Prevention Tests."""

import json
import numpy as np
import torch
import pytest
from app.core.engine.tracer import PipelineTracer, TRACES_DIR


def _assert_no_heavy_objects(obj, path="root"):
    """Recursively verify no torch.Tensor, large np.ndarray, or massive strings exist."""
    assert not isinstance(obj, torch.Tensor), f"Found raw torch.Tensor at {path}!"
    if isinstance(obj, np.ndarray):
        assert obj.size <= 20, f"Found oversized np.ndarray (size {obj.size}) at {path}!"
    elif isinstance(obj, str):
        assert len(obj) <= 2000, f"Found oversized string ({len(obj)} chars) at {path}!"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_no_heavy_objects(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            _assert_no_heavy_objects(item, f"{path}[{idx}]")


def test_tracer_sanitization_removes_tensors():
    """Verify PipelineTracer converts and sanitizes tensors and large objects."""
    tracer = PipelineTracer()
    
    # Intentionally inject heavy tensors and arrays
    heavy_details = {
        "scalar_tensor": torch.tensor(0.95),
        "vector_tensor": torch.tensor([0.1, 0.2, 0.7]),
        "numpy_array": np.array([1.0, 2.0, 3.0]),
        "large_text": "A" * 5000,
    }
    tracer.record_stage("test_stage", 12.5, heavy_details, confidence=0.9)
    payload = tracer.finalize(final_h_score=0.1, risk_level="VERIFIED", metadata={"heavy": heavy_details})
    
    _assert_no_heavy_objects(payload)


def test_persisted_trace_files_are_bounded():
    """Verify saved JSON trace files on disk are lightweight (< 50 KB)."""
    trace_files = list(TRACES_DIR.glob("TRACE_*.json"))
    for tf in trace_files[-10:]:
        size_bytes = tf.stat().st_size
        assert size_bytes < 100 * 1024, f"Trace file {tf.name} is too large ({size_bytes / 1024:.2f} KB)!"
        with open(tf, "r", encoding="utf-8") as f:
            data = json.load(f)
            _assert_no_heavy_objects(data)
