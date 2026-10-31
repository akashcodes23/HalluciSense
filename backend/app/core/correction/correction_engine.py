"""HalluciSense Closed-Loop Correction & Re-Verification Engine.

Implements the DETECT → EXPLAIN → CORRECT → RE-VERIFY explainable AI research loop:
1. Explicit Correction State Machine (ORIGINAL_VERIFIED through CORRECTION_ACCEPTED/REJECTED)
2. Tier 1 — Deterministic / Symbolic Correction (Arithmetic, Unit Conversions, Temporal Math)
3. Tier 2 — Evidence-Directed Factual Correction (Cross-Encoder Entailed Facts & Knowledge Base)
4. Tier 3 — Ambiguous / Insufficient Evidence (Safe Abstention without Hallucination)
5. Independent Re-Verification Gate (Auditable Trace Linkage)
"""

from __future__ import annotations

import re
import uuid
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from app.core.verification.symbolic_verifier import evaluate_arithmetic_claim
from app.core.verification.unit_verifier import (
    evaluate_unit_claim,
    RECOGNIZED_UNITS_REGEX,
    are_quantities_equivalent,
    to_si_quantity,
)
from app.core.verification.temporal_verifier import evaluate_temporal_claim


class CorrectionState(str, Enum):
    """Explicit state machine states for HalluciSense closed-loop correction:
    DETECTED -> EXPLAINED -> CORRECTION_GENERATED -> REVERIFYING -> CORRECTION_ACCEPTED/REJECTED/ABSTAINED/NOT_REQUIRED/FAILED
    """
    DETECTED = "DETECTED"
    EXPLAINED = "EXPLAINED"
    CORRECTION_GENERATED = "CORRECTION_GENERATED"
    REVERIFYING = "REVERIFYING"
    CORRECTION_ACCEPTED = "CORRECTION_ACCEPTED"
    CORRECTION_REJECTED = "CORRECTION_REJECTED"
    ABSTAINED = "ABSTAINED"
    NOT_REQUIRED = "NOT_REQUIRED"
    FAILED = "FAILED"

    # Backward compatibility aliases
    ORIGINAL_VERIFIED = "NOT_REQUIRED"
    ORIGINAL_CONTRADICTED = "DETECTED"
    INSUFFICIENT_EVIDENCE = "ABSTAINED"
    REVERIFICATION_PENDING = "REVERIFYING"
    CORRECTION_FAILED = "FAILED"


class ReverificationInfo(BaseModel):
    passed: bool = True
    h_score: float = 0.0
    claims_analyzed: int = 1

    @property
    def status(self) -> str:
        return "PASSED" if self.passed else "REVIEW"


class ClaimCorrectionInfo(BaseModel):
    original_claim: str
    corrected_claim: str
    reason: str
    method: str = "evidence_grounded"


class ClosedLoopRepairResult(BaseModel):
    final_text: str
    performed: bool = False
    reason: str = "NO_CORRECTION_NEEDED"
    claims_corrected: List[ClaimCorrectionInfo] = Field(default_factory=list)
    original_to_corrected: List[Dict[str, str]] = Field(default_factory=list)
    reverification: Optional[ReverificationInfo] = None
    state: str = CorrectionState.ORIGINAL_VERIFIED.value


class HallucinationCorrectionEngine:
    """Orchestrates evidence-guided candidate generation and independent re-verification."""

    def __init__(self, pipeline: Optional[Any] = None) -> None:
        self.pipeline = pipeline

    @staticmethod
    def _generate_correction_id() -> str:
        return f"CORR_{uuid.uuid4().hex[:12].upper()}"

    def generate_candidate(
        self,
        query: str,
        original_response: str,
        retrieved_evidence: Optional[List[Dict[str, Any]]] = None,
        sentence_scores: Optional[List[Dict[str, Any]]] = None,
        overall_h_score: float = 0.0,
    ) -> Dict[str, Any]:
        """Generate a factual or symbolic correction candidate."""
        flagged_count = sum(1 for s in (sentence_scores or []) if s.get("score", 0.0) >= 0.35)

        # Rule: If original answer is verified, preserve it and do not correct
        if overall_h_score <= 0.35 and flagged_count == 0:
            return {
                "status": "not_needed",
                "state": CorrectionState.ORIGINAL_VERIFIED.value,
                "original_text": original_response,
                "corrected_text": original_response,
                "method": "none",
                "reason": "Original response is verified by reference evidence. No correction required.",
                "supporting_evidence": [],
                "confidence": 1.0,
            }

        # -------------------------------------------------------------
        # TIER 1: Symbolic Math / Unit / Temporal Verifier Correction
        # -------------------------------------------------------------
        # 1A. Arithmetic
        arith_res = evaluate_arithmetic_claim(original_response)
        if not arith_res and query:
            arith_res = evaluate_arithmetic_claim(f"{query} = {original_response}")

        if arith_res and arith_res.get("verified"):
            if not arith_res.get("is_consistent"):
                comp_val = arith_res.get("computed_value")
                if isinstance(comp_val, float) and comp_val.is_integer():
                    comp_val_str = str(int(comp_val))
                else:
                    comp_val_str = str(comp_val)

                if "=" in original_response:
                    parts = original_response.split("=")
                    lhs = parts[0].strip()
                    corrected_text = f"{lhs} = {comp_val_str}"
                else:
                    corrected_text = f"{query.strip()} is {comp_val_str}." if query else f"The correct calculated value is {comp_val_str}."

                return {
                    "status": "candidate_generated",
                    "state": CorrectionState.CORRECTION_GENERATED.value,
                    "original_text": original_response,
                    "corrected_text": corrected_text,
                    "method": "symbolic_arithmetic",
                    "reason": f"Mathematically corrected using AST evaluator: {arith_res.get('explanation')}",
                    "supporting_evidence": [
                        {
                            "source": "Symbolic Arithmetic Engine",
                            "snippet": f"Evaluated expression yielded verified result: {comp_val_str}",
                            "score": 1.0,
                        }
                    ],
                    "confidence": 1.0,
                }

        # 1B. Unit Conversion Statement
        unit_res = evaluate_unit_claim(original_response)
        if unit_res and unit_res.get("verified") and not unit_res.get("is_consistent"):
            claimed_v = unit_res.get("claimed_val")
            val_from = unit_res.get("val_from")
            unit_from = unit_res.get("unit_from")
            unit_to = unit_res.get("unit_to")
            # Calculate correct target value
            from app.core.verification.unit_verifier import to_si_quantity
            _, si_val = to_si_quantity(val_from, unit_from)
            if si_val is not None:
                # Convert SI to unit_to
                if unit_to.lower() in ("celsius", "degrees celsius", "°c", "deg c"):
                    expected_target = si_val - 273.15
                elif unit_to.lower() in ("fahrenheit", "degrees fahrenheit", "°f", "deg f"):
                    expected_target = (si_val - 273.15) * 1.8 + 32.0
                elif unit_to.lower() in ("kelvin", "k"):
                    expected_target = si_val
                elif unit_to.lower() == "km/s":
                    expected_target = si_val / 1000.0
                elif unit_to.lower() == "m/s":
                    expected_target = si_val
                else:
                    expected_target = claimed_v

                corrected_text = re.sub(
                    rf"\b{re.escape(str(claimed_v))}\b",
                    str(round(expected_target, 3) if not float(expected_target).is_integer() else int(expected_target)),
                    original_response,
                )
                return {
                    "status": "candidate_generated",
                    "state": CorrectionState.CORRECTION_GENERATED.value,
                    "original_text": original_response,
                    "corrected_text": corrected_text,
                    "method": "symbolic_unit",
                    "reason": f"Unit conversion statement corrected: {unit_res.get('explanation')}",
                    "supporting_evidence": [
                        {
                            "source": "Unit Conversion Engine",
                            "snippet": f"Expected {expected_target} {unit_to}",
                            "score": 1.0,
                        }
                    ],
                    "confidence": 1.0,
                }

        # 1C. Explicit Unit and Quantity Verification against Evidence
        num_unit_pattern = re.compile(
            rf"\b(\d{{1,3}}(?:,\d{{3}})+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*({RECOGNIZED_UNITS_REGEX})\b",
            re.IGNORECASE,
        )

        for ev in (retrieved_evidence or []):
            snip = ev.get("snippet", "") if isinstance(ev, dict) else getattr(ev, "snippet", "")
            if not snip:
                continue

            for match in num_unit_pattern.finditer(original_response):
                num_str = match.group(1)
                unit_str = match.group(2).strip()
                val_orig = float(num_str.replace(",", ""))

                # Search snippet for numerical quantities with recognized units
                for ev_match in num_unit_pattern.finditer(snip):
                    ev_num_str = ev_match.group(1)
                    ev_unit_str = ev_match.group(2).strip()
                    val_ev = float(ev_num_str.replace(",", ""))

                    # Check scientific equivalence before modifying
                    if are_quantities_equivalent(val_orig, unit_str, val_ev, ev_unit_str):
                        # Quantities are scientifically equivalent! No correction needed.
                        return {
                            "status": "not_needed",
                            "state": CorrectionState.ORIGINAL_VERIFIED.value,
                            "original_text": original_response,
                            "corrected_text": original_response,
                            "method": "none",
                            "reason": f"Original quantity {num_str} {unit_str} is scientifically equivalent to reference {ev_num_str} {ev_unit_str}.",
                            "supporting_evidence": [{"source": "Unit Equivalence Verifier", "snippet": snip, "score": 1.0}],
                            "confidence": 1.0,
                        }

                    # If not equivalent, and discussing same context (e.g. speed of light)
                    # replace the incorrect value/unit with verified reference value
                    if abs(val_orig - val_ev) > 1e-4 or unit_str.lower() != ev_unit_str.lower():
                        corrected_text = (
                            original_response[:match.start()]
                            + f"{ev_num_str} {ev_unit_str}"
                            + original_response[match.end():]
                        )
                        return {
                            "status": "candidate_generated",
                            "state": CorrectionState.CORRECTION_GENERATED.value,
                            "original_text": original_response,
                            "corrected_text": corrected_text,
                            "method": "unit_correction",
                            "reason": f"Corrected quantity from {num_str} {unit_str} to verified reference {ev_num_str} {ev_unit_str}.",
                            "supporting_evidence": [{"source": "Verified Reference Grounding", "snippet": snip, "score": 0.95}],
                            "confidence": 0.95,
                        }

        # -------------------------------------------------------------
        # TIER 2: Evidence-Directed Factual Correction
        # -------------------------------------------------------------
        evidence_list = retrieved_evidence or []
        subject_text = f"{query or ''} {original_response}".lower()
        substantive_keywords = [
            w for w in re.findall(r"\b[a-zA-Z]{4,}\b", subject_text)
            if w not in {"what", "when", "where", "which", "about", "tell", "this", "that", "with", "from", "were", "been", "have", "will", "would", "could", "should"}
        ]

        valid_evidence_passages: List[str] = []
        for ev in evidence_list:
            if isinstance(ev, str):
                snippet = ev.strip()
                score = 0.8
            elif isinstance(ev, dict):
                snippet = ev.get("snippet", "").strip()
                raw_score = ev.get("score", 0.8)
                try:
                    score = float(raw_score)
                except (TypeError, ValueError):
                    score = 0.8
            else:
                snippet = getattr(ev, "snippet", "").strip()
                raw_score = getattr(ev, "score", 0.8)
                try:
                    score = float(raw_score)
                except (TypeError, ValueError):
                    score = 0.8
            if snippet and score >= 0.40:
                snippet_lower = snippet.lower()
                if any(kw in snippet_lower for kw in substantive_keywords):
                    valid_evidence_passages.append(snippet)

        # Factual pattern 1: Capital city questions / claims
        q_lower = query.lower() if query else ""
        resp_lower = original_response.lower()
        if "capital" in q_lower or "capital" in resp_lower:
            from app.core.engine.pillar1_retrieval import extract_capital_relation, is_valid_geopolitical_entity

            # Extract the original claim's subject city and target entity
            orig_cap_rel = extract_capital_relation(original_response)
            if not orig_cap_rel and query:
                orig_cap_rel = extract_capital_relation(query)

            claimed_city = orig_cap_rel[0] if orig_cap_rel else None
            target_entity_to_match = orig_cap_rel[1] if orig_cap_rel else None

            # Fallback for questions without explicit relational syntax: extract target entity from keywords
            if not target_entity_to_match:
                for kw in substantive_keywords:
                    if is_valid_geopolitical_entity(kw) and kw.lower() not in {"capital", "what", "which", "name", "tell", "state", "city", "country"}:
                        target_entity_to_match = kw.lower()
                        break

            for passage in valid_evidence_passages:
                cap_rel = extract_capital_relation(passage)
                if cap_rel:
                    correct_capital, target_entity = cap_rel

                    # Strictly validate that the evidence identifies the same target entity as the original claim
                    if target_entity_to_match and target_entity.lower() != target_entity_to_match.lower():
                        continue

                    # Reject generic or ambiguous entities
                    if not is_valid_geopolitical_entity(target_entity) or not is_valid_geopolitical_entity(correct_capital):
                        continue

                    # Reject if the evidence asserts the exact same city that was claimed as false
                    if claimed_city and correct_capital.lower() == claimed_city.lower():
                        continue

                    correct_capital_str = correct_capital.title()
                    target_entity_str = (target_entity_to_match or target_entity).title()

                    # Preserve unrelated factual assertions in multi-claim responses
                    if claimed_city and re.search(rf"\b{re.escape(claimed_city)}\b", original_response, re.IGNORECASE):
                        corrected_text = re.sub(
                            rf"\b{re.escape(claimed_city)}\b",
                            correct_capital_str,
                            original_response,
                            count=1,
                            flags=re.IGNORECASE,
                        )
                    else:
                        corrected_text = f"{correct_capital_str} is the capital of {target_entity_str}."

                    return {
                        "status": "candidate_generated",
                        "state": CorrectionState.CORRECTION_GENERATED.value,
                        "original_text": original_response,
                        "corrected_text": corrected_text,
                        "method": "evidence_grounded",
                        "reason": f"Factual correction derived from verified reference evidence: '{passage[:120]}...'",
                        "supporting_evidence": [
                            {
                                "source": "Encyclopedic Reference Evidence",
                                "snippet": passage,
                                "score": 0.95,
                            }
                        ],
                        "confidence": 0.95,
                    }

        # Tier 2A: LLM-powered factual correction using verified ground truth
        if valid_evidence_passages:
            evidence_context = "\n".join([f"- {p[:350]}" for p in valid_evidence_passages[:3] if p])
            try:
                from app.core.config import settings
                import google.generativeai as genai

                if getattr(settings, "GEMINI_API_KEY", None):
                    genai.configure(api_key=settings.GEMINI_API_KEY)
                    model_name = getattr(settings, "DEFAULT_LLM_MODEL", "gemini-2.0-flash")
                    model = genai.GenerativeModel(model_name)
                    user_q_part = f'User Question / Context: "{query}"\n' if query else ''
                    ev_part = f'Verified Reference Evidence:\n{evidence_context}\n\n' if evidence_context else ''
                    correction_prompt = (
                        "You are the HalluciSense closed-loop factual correction engine.\n"
                        "A statement generated by an AI model has been flagged as containing a factual error or hallucination.\n\n"
                        f"Original Statement: \"{original_response}\"\n"
                        f"{user_q_part}"
                        f"{ev_part}"
                        "Task: Output a concise, factually accurate correction of the statement grounded in factual truth.\n"
                        "Guidelines:\n"
                        "- Correct the specific factual mistake, wrong number, wrong date, or false attribution directly.\n"
                        "- Preserve the sentence structure and tone of the original statement.\n"
                        "- Output ONLY the single corrected factual sentence.\n"
                        "- Do NOT include conversational framing, markdown headers, quotes, or preambles like 'Correction:'."
                    )
                    resp = model.generate_content(correction_prompt, request_options={"timeout": 3.0})
                    if resp and resp.text and resp.text.strip():
                        cand_text = resp.text.strip().strip('"*`')
                        if cand_text and cand_text.lower() != original_response.lower() and len(cand_text) > 10:
                            top_snip = valid_evidence_passages[0]
                            return {
                                "status": "candidate_generated",
                                "state": CorrectionState.CORRECTION_GENERATED.value,
                                "original_text": original_response,
                                "corrected_text": cand_text,
                                "method": "evidence_grounded",
                                "reason": "Factual correction generated and grounded in verified reference facts.",
                                "supporting_evidence": [
                                    {
                                        "source": "Verified Reference Grounding",
                                        "snippet": top_snip,
                                        "score": 0.95,
                                    }
                                ],
                                "confidence": 0.95,
                            }
            except Exception as exc:
                import structlog
                logger = structlog.get_logger(__name__)
                logger.warning("gemini_factual_correction_failed", error=str(exc))

        # -------------------------------------------------------------
        # TIER 3: Ambiguous / Insufficient Evidence -> Abstain
        # Note: Generic Wikipedia definition extraction has been REMOVED.
        # A Wikipedia definition must not replace the original claim unless it repairs it.
        # -------------------------------------------------------------
        return {
            "status": "abstained",
            "state": CorrectionState.INSUFFICIENT_EVIDENCE.value,
            "original_text": original_response,
            "corrected_text": None,
            "method": "abstained",
            "reason": "HalluciSense detected an unsupported or contradicted claim but safely abstained from generating a correction due to insufficient reference evidence. Original answer is preserved.",
            "missing_evidence_explanation": "No verified passage provided sufficient factual grounding to safely rewrite this claim.",
            "supporting_evidence": [],
            "confidence": 0.0,
        }

    def generate_explanation(
        self,
        query: str,
        original_response: str,
        overall_h_score: float,
        retrieved_evidence: Optional[List[Dict[str, Any]]] = None,
        sentence_scores: Optional[List[Dict[str, Any]]] = None,
        p1_score: Optional[float] = None,
        p2_score: Optional[float] = None,
        p3_score: Optional[float] = None,
        p2_details: Optional[Dict[str, Any]] = None,
        p3_details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate structured 9-point explainable AI diagnosis for DETECT -> EXPLAIN stage."""
        fe_val = p1_score if p1_score is not None else overall_h_score

        if fe_val >= 0.50:
            classification = "FACTUAL_CONTRADICTION"
        elif not retrieved_evidence:
            classification = "INSUFFICIENT_EVIDENCE"
        elif overall_h_score <= 0.35:
            classification = "VERIFIED"
        else:
            classification = "NEEDS_VERIFICATION"

        supporting = []
        contradicting = []
        for ev in (retrieved_evidence or []):
            snip = ev.get("snippet", "") if isinstance(ev, dict) else getattr(ev, "snippet", "")
            sc = ev.get("score", 0.8) if isinstance(ev, dict) else getattr(ev, "score", 0.8)
            c_p = ev.get("contradiction_prob", 0.0) if isinstance(ev, dict) else getattr(ev, "contradiction_prob", 0.0)
            if c_p >= 0.40 or (fe_val >= 0.50 and sc >= 0.50):
                contradicting.append(ev)
            elif sc >= 0.40:
                supporting.append(ev)

        if classification == "FACTUAL_CONTRADICTION":
            det_reason = f"Pillar 1 detected factual error or contradiction (FE={fe_val:.2f}) against reference corpus."
        elif classification == "VERIFIED":
            det_reason = f"Original response is factually supported by reference evidence (H={overall_h_score:.2f})."
        elif classification == "INSUFFICIENT_EVIDENCE":
            det_reason = "Reference evidence was insufficient to verify or refute this statement."
        else:
            det_reason = f"Response exhibits elevated uncertainty or partial evidence conflict (H={overall_h_score:.2f})."

        return {
            "original_claim": original_response,
            "claim_classification": classification,
            "supporting_evidence": supporting[:3],
            "contradicting_evidence": contradicting[:3],
            "p1_score": round(fe_val, 4),
            "p2_score_and_provenance": p2_details or {"confidence_gap": p2_score, "status": "UNAVAILABLE" if p2_score is None else "EXECUTED"},
            "p3_score_and_generation_comparisons": p3_details or {"consistency_failure": p3_score, "status": "UNAVAILABLE" if p3_score is None else "EXECUTED"},
            "detection_reason": det_reason,
            "confidence_and_limitations": (
                "Pillar 1 assesses factual entailment against retrieved reference documents. "
                "Pillar 2 reflects token-level generation predictability, not objective factual truth. "
                "Pillar 3 assesses sampling consistency across independent generations. "
                "Abstention safeguards are active when reference passages are inconclusive."
            ),
        }

    async def correct_and_reverify(
        self,
        query: str,
        original_response: str,
        original_trace_id: Optional[str],
        analyze_func: Callable[..., Any],
        retrieved_evidence: Optional[List[Dict[str, Any]]] = None,
        sentence_scores: Optional[List[Dict[str, Any]]] = None,
        overall_h_score: float = 0.0,
        model_name: str = "default",
        p1_score: Optional[float] = None,
        p2_score: Optional[float] = None,
        p3_score: Optional[float] = None,
        p2_details: Optional[Dict[str, Any]] = None,
        p3_details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate candidate and run independent re-verification with explicit state machine transitions:
        DETECTED -> EXPLAINED -> CORRECTION_GENERATED -> REVERIFYING -> CORRECTION_ACCEPTED / REJECTED / ABSTAINED
        """
        corr_id = f"CORR_{uuid.uuid4().hex[:12].upper()}"

        # Stage 2: EXPLAIN
        explanation = self.generate_explanation(
            query=query,
            original_response=original_response,
            overall_h_score=overall_h_score,
            retrieved_evidence=retrieved_evidence,
            sentence_scores=sentence_scores,
            p1_score=p1_score,
            p2_score=p2_score,
            p3_score=p3_score,
            p2_details=p2_details,
            p3_details=p3_details,
        )

        candidate = self.generate_candidate(
            query=query,
            original_response=original_response,
            retrieved_evidence=retrieved_evidence,
            sentence_scores=sentence_scores,
            overall_h_score=overall_h_score,
        )

        cand_status = candidate.get("status")

        # 1. If original response is verified or not needed
        if cand_status == "not_needed":
            return {
                "correction_id": corr_id,
                "original_trace_id": original_trace_id,
                "reverification_trace_id": None,
                "status": "not_needed",
                "state": CorrectionState.NOT_REQUIRED.value,
                "method": "none",
                "original_text": original_response,
                "corrected_text": original_response,
                "reason": candidate.get("reason", "Original response is verified."),
                "missing_evidence_explanation": None,
                "supporting_evidence": candidate.get("supporting_evidence", []),
                "confidence": 1.0,
                "explanation": explanation,
                "reverification": None,
            }

        # 2. If evidence is insufficient -> Abstain (preserve original)
        if cand_status == "abstained" or not candidate.get("corrected_text"):
            return {
                "correction_id": corr_id,
                "original_trace_id": original_trace_id,
                "reverification_trace_id": None,
                "status": "abstained",
                "state": CorrectionState.ABSTAINED.value,
                "method": "abstained",
                "original_text": original_response,
                "corrected_text": None,
                "reason": candidate.get("reason", "Insufficient evidence to safely correct claim."),
                "missing_evidence_explanation": candidate.get("missing_evidence_explanation"),
                "supporting_evidence": candidate.get("supporting_evidence", []),
                "confidence": 0.0,
                "explanation": explanation,
                "reverification": None,
            }

        corrected_text = candidate.get("corrected_text")

        # 3. Candidate generated -> Transition to REVERIFYING
        try:
            reverify_analysis = await analyze_func(
                query=query,
                response=corrected_text,
                model_name=model_name,
            )

            rev_h_score = getattr(reverify_analysis, "overall_h_score", 0.0)
            rev_risk = getattr(reverify_analysis, "risk_level", "NEEDS_VERIFICATION")
            rev_trace_id = getattr(reverify_analysis, "trace_id", f"TRACE_REV_{uuid.uuid4().hex[:8].upper()}")
            rev_pillars = getattr(reverify_analysis, "pillar_scores", None)

            p1_fe = getattr(rev_pillars, "retrieval", None) or getattr(rev_pillars, "pillar1_factual_error", 0.0) or 0.0
            p2_cg = getattr(rev_pillars, "confidence", None) or getattr(rev_pillars, "pillar2_confidence_gap", 0.0) or 0.0
            p3_cf = getattr(rev_pillars, "consistency", None) or getattr(rev_pillars, "pillar3_consistency_failure", 0.0) or 0.0

            is_verified = (rev_h_score <= 0.35 and p1_fe <= 0.30) or rev_risk == "VERIFIED"

            final_state = CorrectionState.CORRECTION_ACCEPTED if is_verified else CorrectionState.CORRECTION_REJECTED
            final_status = "verified" if is_verified else "rejected"

            # Rule: If re-verification fails, preserve original response! Never display failed draft as accepted
            final_corrected_text = corrected_text if is_verified else original_response

            return {
                "correction_id": corr_id,
                "original_trace_id": original_trace_id,
                "reverification_trace_id": rev_trace_id,
                "status": final_status,
                "state": final_state.value,
                "method": candidate.get("method"),
                "original_text": original_response,
                "corrected_text": final_corrected_text,
                "reason": "Correction successfully verified." if is_verified else "Failed independent re-verification; original text preserved.",
                "supporting_evidence": candidate.get("supporting_evidence", []),
                "confidence": candidate.get("confidence", 0.90) if is_verified else 0.0,
                "explanation": explanation,
                "reverification": {
                    "trace_id": rev_trace_id,
                    "status": "VERIFIED" if is_verified else "REJECTED",
                    "overall_h_score": round(rev_h_score, 4),
                    "risk_level": rev_risk,
                    "pillar_scores": {
                        "evidence_grounding": round(p1_fe, 4),
                        "confidence_gap": round(p2_cg, 4),
                        "consistency_failure": round(p3_cf, 4),
                    },
                },
            }

        except Exception as e:
            # Rule: Preserve original answer whenever correction / reverification fails
            return {
                "correction_id": corr_id,
                "original_trace_id": original_trace_id,
                "reverification_trace_id": None,
                "status": "failed",
                "state": CorrectionState.FAILED.value,
                "method": candidate.get("method"),
                "original_text": original_response,
                "corrected_text": original_response,
                "reason": f"Re-verification execution failed: {str(e)}. Original response preserved.",
                "supporting_evidence": candidate.get("supporting_evidence", []),
                "confidence": 0.0,
                "explanation": explanation,
                "reverification": None,
            }

    def execute_closed_loop_repair(
        self,
        user_query: str,
        initial_text: str,
        initial_verification: Any,
        max_attempts: int = 2,
    ) -> ClosedLoopRepairResult:
        """Synchronous chat-compatible closed-loop repair interface."""
        evidence_items = []
        if isinstance(initial_verification, dict):
            evidence_items = initial_verification.get("evidence") or initial_verification.get("evidence_items") or []
            h_score = float(initial_verification.get("overall_h_score") or initial_verification.get("factual_error_score") or initial_verification.get("hallucination_score") or 0.0)
        else:
            if hasattr(initial_verification, "evidence") and isinstance(getattr(initial_verification, "evidence"), list):
                evidence_items = initial_verification.evidence
            elif hasattr(initial_verification, "evidence_items") and isinstance(getattr(initial_verification, "evidence_items"), list):
                evidence_items = initial_verification.evidence_items
            elif hasattr(initial_verification, "pillar1_summary") and hasattr(getattr(initial_verification, "pillar1_summary"), "evidence"):
                evidence_items = initial_verification.pillar1_summary.evidence

            h_score = float(getattr(initial_verification, "overall_h_score", getattr(initial_verification, "factual_error_score", getattr(initial_verification, "hallucination_score", 0.0))))

        evidence_dicts = []
        for e in evidence_items:
            snippet = getattr(e, "snippet", "") if not isinstance(e, dict) else e.get("snippet", "")
            raw_score = getattr(e, "score", 0.8) if not isinstance(e, dict) else e.get("score", 0.8)
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = 0.8
            evidence_dicts.append({"snippet": str(snippet), "score": score})

        # Rule: If original is verified, do not correct
        if h_score <= 0.35:
            return ClosedLoopRepairResult(
                final_text=initial_text,
                performed=False,
                reason="ORIGINAL_ANSWER_VERIFIED",
                state=CorrectionState.ORIGINAL_VERIFIED.value,
                reverification=ReverificationInfo(passed=True, h_score=h_score, claims_analyzed=1),
            )

        cand = self.generate_candidate(
            query=user_query,
            original_response=initial_text,
            retrieved_evidence=evidence_dicts,
            overall_h_score=h_score,
        )

        corr_text = cand.get("corrected_text")
        if not corr_text or cand.get("status") in ("abstained", "not_needed"):
            if h_score > 0.35:
                return ClosedLoopRepairResult(
                    final_text=f"{initial_text}\n\n[Note: System could not produce a sufficiently verified correction.]",
                    performed=True,
                    reason=cand.get("reason", "SAFE_ABSTENTION_INSUFFICIENT_EVIDENCE"),
                    state=CorrectionState.INSUFFICIENT_EVIDENCE.value,
                    reverification=ReverificationInfo(passed=False, h_score=h_score, claims_analyzed=1),
                )
            return ClosedLoopRepairResult(
                final_text=initial_text,
                performed=False,
                reason=cand.get("reason", "SAFE_ABSTENTION_INSUFFICIENT_EVIDENCE"),
                state=CorrectionState.ORIGINAL_VERIFIED.value,
                reverification=ReverificationInfo(passed=False, h_score=h_score, claims_analyzed=1),
            )

        # Re-verification gate
        rev_h = h_score
        passed = False
        if self.pipeline:
            try:
                if hasattr(self.pipeline, "analyze_response"):
                    rev_res = self.pipeline.analyze_response(
                        full_text=corr_text,
                        query=user_query,
                        evidence_items=evidence_items,
                    )
                elif hasattr(self.pipeline, "analyze"):
                    rev_res = self.pipeline.analyze(
                        response=corr_text,
                        query=user_query,
                        evidence_items=evidence_items,
                    )
                else:
                    rev_res = None

                if rev_res is not None:
                    rev_h = float(getattr(rev_res, "overall_h_score", getattr(rev_res, "hallucination_score", 0.0)))
                    req_ver = getattr(rev_res, "requires_verification", None)
                    if req_ver is not None:
                        passed = not req_ver
                    else:
                        passed = rev_h <= 0.35
            except Exception:
                passed = False

        # On rejection, preserve original response text with unverified note
        final_out = corr_text if passed else f"{initial_text}\n\n[Note: System could not produce a sufficiently verified correction.]"
        final_state = CorrectionState.CORRECTION_ACCEPTED if passed else CorrectionState.CORRECTION_REJECTED

        return ClosedLoopRepairResult(
            final_text=final_out,
            performed=True,
            reason=cand.get("reason", "EVIDENCE_GROUNDED_REPAIR") if passed else "REVERIFICATION_FAILED_PRESERVED_ORIGINAL",
            state=final_state.value,
            claims_corrected=[
                ClaimCorrectionInfo(
                    original_claim=initial_text,
                    corrected_claim=corr_text,
                    reason=cand.get("reason", ""),
                    method=cand.get("method", "evidence_grounded"),
                )
            ] if passed else [],
            original_to_corrected=[{"original": initial_text, "corrected": corr_text}] if passed else [],
            reverification=ReverificationInfo(passed=passed, h_score=rev_h, claims_analyzed=1),
        )


# Backward-compatible alias
CorrectionEngine = HallucinationCorrectionEngine
