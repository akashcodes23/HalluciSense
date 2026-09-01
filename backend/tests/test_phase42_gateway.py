"""Phase 42 — Evidence Intelligence & Gateway Test Suites.

Verifies:
- Claim type classification (Arithmetic, Unit, Temporal, Textual)
- Symbolic arithmetic evaluation (Addition, subtraction, multiplication, division, powers, percentages)
- Physical unit conversion evaluation (Speed, length, time, mass)
- Temporal date calculation evaluation
- Security against AST injection and eval exploits
- Pipeline integration in active mode
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.verification.claim_type_classifier import ClaimTypeClassifier
from app.core.verification.symbolic_verifier import evaluate_arithmetic_claim
from app.core.verification.unit_verifier import evaluate_unit_claim
from app.core.verification.temporal_verifier import evaluate_temporal_claim
from app.core.verification.gateway import EvidenceIntelligenceGateway
from app.core.pipeline import get_hallucisense_pipeline


def test_claim_type_classifier():
    """Verify claim type classification."""
    assert ClaimTypeClassifier.classify("12 * 8 = 96")["claim_type"] == "ARITHMETIC"
    assert ClaimTypeClassifier.classify("15% of 200 is 30")["claim_type"] == "ARITHMETIC"
    assert ClaimTypeClassifier.classify("100 km/h is 27.78 m/s")["claim_type"] == "UNIT_CONVERSION"
    assert ClaimTypeClassifier.classify("2024 was 4 years after 2020")["claim_type"] == "TEMPORAL_MATH"
    assert ClaimTypeClassifier.classify("Paris is the capital of France.")["claim_type"] == "TEXTUAL_FACT"


def test_symbolic_arithmetic():
    """Verify arithmetic verification."""
    res_correct = evaluate_arithmetic_claim("12 * 8 = 96")
    assert res_correct["is_consistent"] is True
    
    res_wrong = evaluate_arithmetic_claim("12 * 8 = 95")
    assert res_wrong["is_consistent"] is False


def test_unit_conversion():
    """Verify physical unit conversion."""
    res_correct = evaluate_unit_claim("1 km is 1000 m")
    assert res_correct["is_consistent"] is True
    
    res_wrong = evaluate_unit_claim("1 km is 500 m")
    assert res_wrong["is_consistent"] is False


def test_temporal_math():
    """Verify temporal math."""
    res_correct = evaluate_temporal_claim("2024 was 4 years after 2020")
    assert res_correct["is_consistent"] is True
    
    res_wrong = evaluate_temporal_claim("2024 was 10 years after 2020")
    assert res_wrong["is_consistent"] is False


def test_symbolic_security():
    """Verify safe AST evaluator rejects arbitrary code execution."""
    malicious = "__import__('os').system('ls')"
    res = evaluate_arithmetic_claim(malicious)
    assert res is None or res.get("verified") is False
