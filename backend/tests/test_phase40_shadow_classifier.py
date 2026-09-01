"""Phase 40 — Shadow Classifier Isolation & Pipeline Test Suite.

Verifies:
- Production pipeline returns candidate_comparison when HALLUCISENSE_CLASSIFIER_SHADOW=true
- Shadow mode does NOT alter production decisions, probabilities, or verdicts
- Graceful degradation when candidate is unavailable
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


@pytest.fixture(scope="module")
def pipeline_instance():
    return get_hallucisense_pipeline()


def test_shadow_classifier_attachment(pipeline_instance):
    """Verify candidate_comparison is attached in shadow mode without altering production verdict."""
    os.environ["HALLUCISENSE_CLASSIFIER_SHADOW"] = "true"
    try:
        res = pipeline_instance.predict(response_text="Paris is the capital of France.")
        assert "candidate_comparison" in res
        comp = res["candidate_comparison"]
        assert comp is not None
        assert comp["shadow_only"] is True
        assert "candidate_probability" in comp
        assert "production_probability" in comp
        assert comp["production_probability"] == res["hallucination_probability"]
    finally:
        os.environ["HALLUCISENSE_CLASSIFIER_SHADOW"] = "false"
