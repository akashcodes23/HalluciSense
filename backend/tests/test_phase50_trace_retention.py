"""Phase 50 — Trace Object Retention & Heavy Structure Sanitization Tests."""

import torch
import numpy as np
import pytest
from app.core.engine.tracer import PipelineTracer, TRACES_DIR


def test_no_tensor_leak_in_tracer():
    """Verify PipelineTracer never leaks raw PyTorch tensors or unbounded ndarrays."""
    tracer = PipelineTracer()
    mock_payload = {
        "scalar_torch": torch.tensor(0.99),
        "array_np": np.array([0.1, 0.2, 0.3]),
        "large_text": "B" * 5000,
    }
    tracer.record_stage("test_stage", 10.0, mock_payload, confidence=0.95)
    result = tracer.finalize(final_h_score=0.05, risk_level="VERIFIED", metadata=mock_payload)

    # Check stage details
    st_details = result["stages"]["test_stage"]["details"]
    assert not isinstance(st_details["scalar_torch"], torch.Tensor)
    assert isinstance(st_details["scalar_torch"], float)
    assert not isinstance(st_details["array_np"], np.ndarray)
    assert isinstance(st_details["array_np"], list)
    assert len(st_details["large_text"]) <= 1100
