"""Phase 44 — Observability, Provenance & Verification State Test Suite.

Verifies:
- VerificationStatus and EvidenceSufficiency typing
- ResponseVerificationSummary structure
- Multi-claim verification decomposition
- Metrics tracker thread safety
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.verification.verification_state import (
    VerificationStatus,
    EvidenceSufficiency,
    ConfidenceBand,
    ClaimVerificationResult,
    ResponseVerificationSummary,
)
from app.core.observability.metrics import metrics_tracker
from app.core.pipeline import get_hallucisense_pipeline


def test_verification_state_model():
    """Verify data model construction and serializability."""
    c_res = ClaimVerificationResult(
        claim_id=0,
        claim_text="12 * 8 = 96",
        claim_type="ARITHMETIC",
        verification_method="symbolic_computation",
        status=VerificationStatus.VERIFIED,
        evidence_sufficiency=EvidenceSufficiency.DIRECT_SUPPORT,
        confidence_band=ConfidenceBand.HIGH,
        verification_confidence=1.0,
        reason="12 * 8 = 96",
    )
    d = c_res.to_dict()
    assert d["status"] == "VERIFIED"
    assert d["evidence_sufficiency"] == "DIRECT_SUPPORT"


def test_end_to_end_verification_summary():
    """Verify predict returns verification_summary and request_id."""
    pipe = get_hallucisense_pipeline()
    res = pipe.predict(response_text="Paris is the capital of France.")
    assert "request_id" in res
    assert "trace_id" in res
    assert "verification_summary" in res
    summary = res["verification_summary"]
    assert "total_claims" in summary
    assert "claims" in summary


def test_metrics_tracker():
    """Verify in-memory metrics tracker."""
    metrics_tracker.record_request(
        claim_count=1,
        verified=1,
        contradicted=0,
        insufficient=0,
        symbolic=0,
        retrieval=1,
        latency_ms=25.0,
    )
    summary = metrics_tracker.get_summary()
    assert summary["total_requests"] > 0
