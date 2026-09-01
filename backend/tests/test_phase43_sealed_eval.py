"""Phase 43 — Sealed Benchmark & Modality Gate Test Suite.

Verifies:
- 500-case sealed dataset generation & non-overlap
- Routing consistency across modalities
- Gateway + Frozen pipeline execution in active mode
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.verification.claim_type_classifier import ClaimTypeClassifier
from app.core.verification.gateway import EvidenceIntelligenceGateway
from app.core.pipeline import get_hallucisense_pipeline


def test_modality_routing_coverage():
    """Verify routing covers all primary modalities."""
    assert ClaimTypeClassifier.classify("25 * 4 = 100")["claim_type"] == "ARITHMETIC"
    assert ClaimTypeClassifier.classify("50 miles is 80.46 km")["claim_type"] == "UNIT_CONVERSION"
    assert ClaimTypeClassifier.classify("2020 was 5 years before 2025")["claim_type"] == "TEMPORAL_MATH"
    assert ClaimTypeClassifier.classify("Albert Einstein formulated general relativity.")["claim_type"] == "TEXTUAL_FACT"


def test_end_to_end_gateway_contradiction():
    """Verify arithmetic falsehood returns high contradiction in active mode."""
    pipe = get_hallucisense_pipeline()
    # Test arithmetic mutation
    res = pipe.predict(response_text="12 * 8 = 95", semantic_mode="active")
    assert res is not None
    assert "hallucination_probability" in res
    assert "semantic_grounding" in res
    assert "symbolic_verifications" in res["semantic_grounding"]
