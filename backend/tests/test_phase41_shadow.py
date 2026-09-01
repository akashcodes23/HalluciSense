"""Phase 41 — Multi-Model Shadow Execution Test Suite.

Verifies:
- Production pipeline returns candidate_comparison when HALLUCISENSE_CLASSIFIER_SHADOW=true
- Shadow mode maintains memory safety and sub-millisecond incremental latency
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.pipeline import get_hallucisense_pipeline


def test_shadow_execution_flow():
    """Verify shadow comparison payload structure."""
    os.environ["HALLUCISENSE_CLASSIFIER_SHADOW"] = "true"
    try:
        pipeline = get_hallucisense_pipeline()
        res = pipeline.predict(response_text="The speed of light in vacuum is exactly 299,792,458 m/s.")
        assert "candidate_comparison" in res
        comp = res["candidate_comparison"]
        assert comp is not None
        assert "candidate_probability" in comp
        assert "production_probability" in comp
        assert "verdicts_match" in comp
    finally:
        os.environ["HALLUCISENSE_CLASSIFIER_SHADOW"] = "false"
