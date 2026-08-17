"""Closed-Loop Correction Engine for HalluciSense Phase 11.

Orchestrates draft verification, claim-level error detection, evidence-grounded repair,
and re-verification gating (max 2 attempts) with fallback to REVIEW status.
"""

from __future__ import annotations

import time
import structlog
from typing import Dict, Any, List, Optional, Tuple

from app.core.config import settings
from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.core.engine.claim_decomposition import ClaimDecomposer
from app.core.correction.correction_models import (
    ClaimVerificationStatus,
    ErrorClassification,
    AtomicClaimVerification,
    ClaimRepairItem,
    ReverificationResult,
    CorrectionExecutionResult,
)
from app.core.correction.correction_policy import CorrectionPolicy
from app.core.correction.correction_prompt import (
    SYSTEM_CORRECTION_PROMPT,
    build_claim_correction_prompt,
)

logger = structlog.get_logger(__name__)


class CorrectionEngine:
    """Production Closed-Loop Correction Engine with Re-Verification Gating."""

    def __init__(self, pipeline: Optional[HallucinationDetectionPipeline] = None):
        self.pipeline = pipeline or HallucinationDetectionPipeline()
        self.policy = CorrectionPolicy()
        self.decomposer = ClaimDecomposer()

    def execute_closed_loop_repair(
        self,
        user_query: str,
        initial_text: str,
        initial_verification: Any,
        max_attempts: int = 2,
    ) -> CorrectionExecutionResult:
        """Executes the closed-loop repair and re-verification gate."""
        start_time = time.perf_counter()
        
        # Check initial verification risk
        h_score = float(getattr(initial_verification, "hallucination_score", 0.0))
        is_verified = (h_score < 0.35) and not getattr(initial_verification, "requires_verification", False)

        if is_verified:
            return CorrectionExecutionResult(
                performed=False,
                reason="INITIAL_RESPONSE_VERIFIED_SAFE",
                attempt_count=0,
                final_text=initial_text,
            )

        # Decompose initial text and analyze claims
        evidence_items = getattr(initial_verification, "evidence", [])
        evidence_dicts = [
            {
                "source_name": getattr(e, "source_name", "Authoritative Source"),
                "snippet": getattr(e, "snippet", ""),
                "claim": getattr(e, "claim", ""),
            }
            for e in evidence_items
        ]

        sentence_analyses = getattr(initial_verification, "sentence_analyses", [])
        
        # Decompose into atomic claims (returns List[DecomposedClaim])
        decomposed_claims = self.decomposer.decompose(initial_text)
        atomic_claims: List[AtomicClaimVerification] = []
        
        for i, dc in enumerate(decomposed_claims):
            cid = f"claim_{i+1:02d}"
            c_text = dc.text if hasattr(dc, "text") else str(dc)
            
            # Match claim to sentence risk
            matching_s = next(
                (s for s in sentence_analyses if c_text in getattr(s, "text", "") or getattr(s, "text", "") in c_text),
                None
            )
            s_risk = float(getattr(matching_s, "hallucination_score", h_score)) if matching_s else h_score
            
            # Check best evidence snippet
            best_ev = evidence_dicts[0]["snippet"] if evidence_dicts else ""
            
            # Deterministic policy check
            repair_cand = self.policy.classify_and_repair_deterministic(cid, c_text, best_ev)
            
            if repair_cand:
                status = ClaimVerificationStatus.CONTRADICTED
                err_type = repair_cand.error_type
                req_corr = True
            elif s_risk >= 0.40:
                status = ClaimVerificationStatus.UNSUPPORTED
                err_type = ErrorClassification.UNSUPPORTED_SPECULATION
                req_corr = True
            else:
                status = ClaimVerificationStatus.SUPPORTED
                err_type = ErrorClassification.NONE
                req_corr = False

            atomic_claims.append(
                AtomicClaimVerification(
                    claim_id=cid,
                    claim_text=c_text,
                    risk_score=s_risk,
                    status=status,
                    evidence=evidence_dicts,
                    nli_score=s_risk,
                    correction_required=req_corr,
                    error_type=err_type,
                )
            )

        # Apply claim-level corrections
        repaired_claims: List[ClaimRepairItem] = []
        corrected_spans = []
        
        for c in atomic_claims:
            if not c.correction_required:
                continue
            
            # 1. Deterministic repair attempt
            best_ev = evidence_dicts[0]["snippet"] if evidence_dicts else ""
            rep = self.policy.classify_and_repair_deterministic(c.claim_id, c.claim_text, best_ev)
            
            if rep:
                repaired_claims.append(rep)
                corrected_spans.append((c.claim_text, rep.corrected_claim))
            else:
                # 2. Evidence-grounded substitution
                corrected_text = best_ev if best_ev else c.claim_text
                rep_item = ClaimRepairItem(
                    claim_id=c.claim_id,
                    original_claim=c.claim_text,
                    corrected_claim=corrected_text,
                    error_type=c.error_type,
                    evidence_basis=best_ev,
                    deterministic_repair=False,
                )
                repaired_claims.append(rep_item)
                corrected_spans.append((c.claim_text, corrected_text))

        # Synthesize candidate corrected text
        candidate_text = initial_text
        for orig, corr in corrected_spans:
            if orig in candidate_text:
                candidate_text = candidate_text.replace(orig, corr, 1)

        # RE-VERIFICATION GATE (Attempt 1 & 2)
        rever_passed = False
        final_h_score = h_score
        rever_status = "FAILED"

        for attempt in range(1, max_attempts + 1):
            try:
                rever_analysis = self.pipeline.analyze_response(candidate_text, user_query)
                rever_h_score = float(getattr(rever_analysis, "hallucination_score", 0.0))
                
                # Check if re-verification passed
                if rever_h_score < 0.35 and not getattr(rever_analysis, "requires_verification", False):
                    rever_passed = True
                    final_h_score = rever_h_score
                    rever_status = "PASSED"
                    break
                else:
                    final_h_score = rever_h_score
                    # Try secondary refined substitution
                    if evidence_dicts and attempt < max_attempts:
                        candidate_text = evidence_dicts[0]["snippet"]
            except Exception as e:
                logger.error("reverification_exception", error=str(e), attempt=attempt)
                break

        # If re-verification failed after max attempts, provide safe fallback
        if not rever_passed:
            candidate_text = (
                "HalluciSense could not produce a sufficiently verified correction. "
                "The available evidence was insufficient or conflicting."
            )
            rever_status = "REVIEW"

        orig_to_corr = [{"original": r.original_claim, "corrected": r.corrected_claim} for r in repaired_claims]

        return CorrectionExecutionResult(
            performed=True,
            reason="DETECTED_HALLUCINATIONS_AND_REPAIRED",
            attempt_count=len(repaired_claims),
            claims_corrected=repaired_claims,
            original_to_corrected=orig_to_corr,
            reverification=ReverificationResult(
                passed=rever_passed,
                h_score=final_h_score,
                status=rever_status,
                attempt=1 if rever_passed else max_attempts,
                claims_analyzed=len(atomic_claims),
                claims_flagged=len(repaired_claims),
            ),
            final_text=candidate_text,
        )
