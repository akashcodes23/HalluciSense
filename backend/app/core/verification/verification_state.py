"""Phase 44 — Verification State & Provenance Data Models.

Defines canonical typed contracts for:
- VerificationStatus (VERIFIED, CONTRADICTED, INSUFFICIENT_EVIDENCE, NOT_APPLICABLE, ERROR)
- EvidenceSufficiency (DIRECT_SUPPORT, DIRECT_CONTRADICTION, PARTIAL_SUPPORT, AMBIGUOUS, NO_EVIDENCE)
- ClaimVerificationResult
- ResponseVerificationSummary
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    ERROR = "ERROR"


class EvidenceSufficiency(str, Enum):
    DIRECT_SUPPORT = "DIRECT_SUPPORT"
    DIRECT_CONTRADICTION = "DIRECT_CONTRADICTION"
    PARTIAL_SUPPORT = "PARTIAL_SUPPORT"
    AMBIGUOUS = "AMBIGUOUS"
    NO_EVIDENCE = "NO_EVIDENCE"


class ConfidenceBand(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class EvidenceProvenance:
    source_title: str
    source_url: Optional[str] = None
    retrieved_at_utc: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
    snippet: str = ""
    retrieval_score: float = 0.0
    nli_entailment: float = 0.0
    nli_contradiction: float = 0.0
    nli_neutral: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimVerificationResult:
    claim_id: int
    claim_text: str
    claim_type: str
    verification_method: str
    status: VerificationStatus
    evidence_sufficiency: EvidenceSufficiency
    confidence_band: ConfidenceBand
    verification_confidence: float
    evidence: List[EvidenceProvenance] = field(default_factory=list)
    symbolic_result: Optional[Dict[str, Any]] = None
    reason: str = ""
    limitations: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["evidence_sufficiency"] = self.evidence_sufficiency.value
        d["confidence_band"] = self.confidence_band.value
        return d


@dataclass
class ResponseVerificationSummary:
    request_id: str
    trace_id: str
    total_claims: int
    verified_claims: int
    contradicted_claims: int
    unsupported_claims: int
    error_claims: int
    primary_status: str
    model_score: float
    model_threshold: float
    is_hallucinated: bool
    claims: List[ClaimVerificationResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["claims"] = [c.to_dict() if hasattr(c, "to_dict") else c for c in self.claims]
        return d
