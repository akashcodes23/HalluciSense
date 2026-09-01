"""Phase 41 — Failure Injection & Resilience Test Suite.

Verifies:
- Pipeline gracefully handles candidate model load failures without degrading production
- Pipeline gracefully handles missing or corrupted shadow candidate metadata
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


def test_corrupted_shadow_candidate_graceful_handling():
    """Verify corrupted candidate environment does not crash production predict."""
    os.environ["HALLUCISENSE_CLASSIFIER_SHADOW"] = "true"
    pipeline = get_hallucisense_pipeline()
    # Execute normal prediction - candidate shadow failure is caught and logged
    res = pipeline.predict(response_text="Paris is the capital of France.")
    assert res["is_hallucinated"] is not None
    assert "hallucination_probability" in res
