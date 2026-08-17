"""Correction Engine Models and Schemas for HalluciSense Phase 11.

Defines claim-level states, error classifications, repair models, and re-verification gate payloads.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ClaimVerificationStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNSUPPORTED = "UNSUPPORTED"
    UNCERTAIN = "UNCERTAIN"


class ErrorClassification(str, Enum):
    NUMERICAL_PRECISION_ERROR = "NUMERICAL_PRECISION_ERROR"
    UNIT_SCALE_ERROR = "UNIT_SCALE_ERROR"
    NEGATION_CONFLICT = "NEGATION_CONFLICT"
    CAUSAL_DIRECTION_ERROR = "CAUSAL_DIRECTION_ERROR"
    TRUE_CORE_FALSE_ELABORATION = "TRUE_CORE_FALSE_ELABORATION"
    FABRICATED_DETAIL = "FABRICATED_DETAIL"
    UNSUPPORTED_SPECULATION = "UNSUPPORTED_SPECULATION"
    NONE = "NONE"


class AtomicClaimVerification(BaseModel):
    claim_id: str
    claim_text: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    status: ClaimVerificationStatus
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    nli_score: float = Field(default=0.0, ge=0.0, le=1.0)
    symbolic_checks: Dict[str, Any] = Field(default_factory=dict)
    correction_required: bool = False
    error_type: ErrorClassification = ErrorClassification.NONE


class ClaimRepairItem(BaseModel):
    claim_id: str
    original_claim: str
    corrected_claim: str
    error_type: ErrorClassification
    evidence_basis: str
    deterministic_repair: bool = False


class ReverificationResult(BaseModel):
    passed: bool
    h_score: float
    status: str
    attempt: int
    claims_analyzed: int
    claims_flagged: int


class CorrectionExecutionResult(BaseModel):
    performed: bool
    reason: str
    attempt_count: int
    claims_corrected: List[ClaimRepairItem] = Field(default_factory=list)
    original_to_corrected: List[Dict[str, str]] = Field(default_factory=list)
    reverification: Optional[ReverificationResult] = None
    final_text: str
