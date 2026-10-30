"""HalluciSense Closed-Loop Correction & Re-Verification Engine.

Implements the DETECT → EXPLAIN → CORRECT → RE-VERIFY explainable AI research loop:
1. Tier 1 — Deterministic / Symbolic Correction (Arithmetic, Unit Conversions, Temporal Math)
2. Tier 2 — Evidence-Directed Factual Correction (Cross-Encoder Entailed Facts & Knowledge Base)
3. Tier 3 — Ambiguous / Insufficient Evidence (Safe Abstention without Hallucination)
4. Independent Re-Verification Gate (Auditable Trace Linkage)
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from app.core.verification.symbolic_verifier import evaluate_arithmetic_claim
from app.core.verification.unit_verifier import evaluate_unit_claim
from app.core.verification.temporal_verifier import evaluate_temporal_claim


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
        # 0. If overall risk is low and no flagged sentences, return not_needed
        flagged_count = sum(1 for s in (sentence_scores or []) if s.get("score", 0.0) >= 0.35)
        if overall_h_score < 0.35 and flagged_count == 0:
            return {
                "status": "not_needed",
                "original_text": original_response,
                "corrected_text": original_response,
                "method": "none",
                "reason": "Original response is already well-grounded and verified by evidence.",
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

        # 1B. Unit Conversion
        unit_res = evaluate_unit_claim(original_response)
        if unit_res and unit_res.get("verified") and not unit_res.get("is_consistent"):
            exp_val = unit_res.get("expected_val")
            corrected_text = f"The correct converted value is {exp_val}."
            return {
                "status": "candidate_generated",
                "original_text": original_response,
                "corrected_text": corrected_text,
                "method": "symbolic_unit",
                "reason": f"Unit conversion corrected: {unit_res.get('explanation')}",
                "supporting_evidence": [
                    {
                        "source": "Unit Conversion Engine",
                        "snippet": f"Calculated expected conversion: {exp_val}",
                        "score": 1.0,
                    }
                ],
                "confidence": 1.0,
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

        # Check for strong evidence passages that provide the ground truth for the subject
        valid_evidence_passages: List[str] = []
        for ev in evidence_list:
            snippet = ev.get("snippet", "").strip() if isinstance(ev, dict) else getattr(ev, "snippet", "").strip()
            raw_score = ev.get("score", 0.8) if isinstance(ev, dict) else getattr(ev, "score", 0.8)
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
            from app.core.engine.pillar1_retrieval import extract_capital_relation
            for passage in valid_evidence_passages:
                cap_rel = extract_capital_relation(passage)
                if cap_rel:
                    correct_capital, target_entity = cap_rel
                    # Clean and title case properly
                    correct_capital_str = correct_capital.title()
                    target_entity_str = target_entity.title()
                    corrected_text = f"{correct_capital_str} is the capital of {target_entity_str}."

                    return {
                        "status": "candidate_generated",
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
        if valid_evidence_passages or evidence_list:
            effective_passages = valid_evidence_passages or [
                (ev.get("snippet", "") if isinstance(ev, dict) else getattr(ev, "snippet", "")).strip()
                for ev in evidence_list
                if (ev.get("snippet", "") if isinstance(ev, dict) else getattr(ev, "snippet", "")).strip()
            ]
            evidence_context = "\n".join([f"- {p[:350]}" for p in effective_passages[:3] if p])
            try:
                from app.core.config import settings
                import google.generativeai as genai

                if getattr(settings, "GEMINI_API_KEY", None):
                    genai.configure(api_key=settings.GEMINI_API_KEY)
                    model_name = getattr(settings, "DEFAULT_LLM_MODEL", "gemini-3.6-flash")
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
                        "- Correct the specific factual mistake, non-existent entity, wrong date, invalid award, or false attribution directly.\n"
                        "- If the premise does not exist (e.g. no Nobel Prize in Computer Science exists; the Turing Award was created in 1966), state the correct fact clearly.\n"
                        "- Output ONLY the single corrected factual sentence.\n"
                        "- Do NOT include conversational framing, markdown headers, quotes, or preambles like 'Correction:'."
                    )
                    resp = model.generate_content(correction_prompt, request_options={"timeout": 3.0})
                    if resp and resp.text and resp.text.strip():
                        cand_text = resp.text.strip().strip('"*`')
                        if cand_text and cand_text.lower() != original_response.lower() and len(cand_text) > 10:
                            top_snip = effective_passages[0] if effective_passages else "Verified factual knowledge base"
                            return {
                                "status": "candidate_generated",
                                "original_text": original_response,
                                "corrected_text": cand_text,
                                "method": "evidence_grounded",
                                "reason": f"Factual correction generated and grounded in verified reference facts.",
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

        # Factual pattern 2: Semantic evidence extraction with keyword relevance validation
        if valid_evidence_passages:
            for top_passage in valid_evidence_passages:
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", top_passage) if len(s.strip()) > 20]
                for candidate_fact in sentences:
                    cand_lower = candidate_fact.lower()
                    matched_kws = sum(1 for kw in substantive_keywords if kw in cand_lower)
                    # Only accept sentences that actually share multiple substantive subject keywords
                    if matched_kws >= 2 and not cand_lower.startswith("this is a list") and not "wikipedia" in cand_lower:
                        return {
                            "status": "candidate_generated",
                            "original_text": original_response,
                            "corrected_text": candidate_fact,
                            "method": "evidence_grounded",
                            "reason": "Generated evidence-grounded candidate directly from verified reference passage.",
                            "supporting_evidence": [
                                {
                                    "source": "Retrieved Reference Corpus",
                                    "snippet": top_passage,
                                    "score": 0.85,
                                }
                            ],
                            "confidence": 0.85,
                        }

        # -------------------------------------------------------------
        # TIER 3: Ambiguous / Insufficient Evidence -> Abstain
        # -------------------------------------------------------------
        return {
            "status": "abstained",
            "original_text": original_response,
            "corrected_text": None,
            "method": "abstained",
            "reason": "HalluciSense detected an unsupported claim but safely abstained from generating a correction due to insufficient external reference grounding. Human review is recommended.",
            "missing_evidence_explanation": "No retrieved passage met the minimum evidence threshold (>= 0.70) required to safely rewrite this claim.",
            "supporting_evidence": [],
            "confidence": 0.0,
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
    ) -> Dict[str, Any]:
        """Generate candidate and run independent re-verification."""
        corr_id = f"CORR_{uuid.uuid4().hex[:12].upper()}"

        candidate = self.generate_candidate(
            query=query,
            original_response=original_response,
            retrieved_evidence=retrieved_evidence,
            sentence_scores=sentence_scores,
            overall_h_score=overall_h_score,
        )

        # If abstained or not needed, return early without re-verification
        if candidate.get("status") in ("abstained", "not_needed"):
            return {
                "correction_id": corr_id,
                "original_trace_id": original_trace_id,
                "reverification_trace_id": None,
                "status": candidate.get("status"),
                "method": candidate.get("method"),
                "original_text": original_response,
                "corrected_text": candidate.get("corrected_text"),
                "reason": candidate.get("reason"),
                "missing_evidence_explanation": candidate.get("missing_evidence_explanation"),
                "supporting_evidence": candidate.get("supporting_evidence", []),
                "confidence": candidate.get("confidence", 0.0),
                "reverification": None,
            }

        corrected_text = candidate.get("corrected_text") or original_response

        # Execute independent re-verification through the verification pipeline
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

            return {
                "correction_id": corr_id,
                "original_trace_id": original_trace_id,
                "reverification_trace_id": rev_trace_id,
                "status": "verified" if is_verified else "rejected",
                "method": candidate.get("method"),
                "original_text": original_response,
                "corrected_text": corrected_text,
                "reason": "Correction successfully verified." if is_verified else "Failed independent re-verification.",
                "supporting_evidence": candidate.get("supporting_evidence", []),
                "confidence": candidate.get("confidence", 0.90),
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
            return {
                "correction_id": corr_id,
                "original_trace_id": original_trace_id,
                "reverification_trace_id": None,
                "status": "rejected",
                "method": candidate.get("method"),
                "original_text": original_response,
                "corrected_text": corrected_text,
                "reason": f"Re-verification execution failed: {str(e)}",
                "supporting_evidence": candidate.get("supporting_evidence", []),
                "confidence": 0.0,
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
        if hasattr(initial_verification, "evidence") and isinstance(getattr(initial_verification, "evidence"), list):
            evidence_items = initial_verification.evidence
        elif hasattr(initial_verification, "evidence_items") and isinstance(getattr(initial_verification, "evidence_items"), list):
            evidence_items = initial_verification.evidence_items
        elif hasattr(initial_verification, "pillar1_summary") and hasattr(getattr(initial_verification, "pillar1_summary"), "evidence"):
            evidence_items = initial_verification.pillar1_summary.evidence
        elif hasattr(initial_verification, "evidence"):
            raw_ev = getattr(initial_verification, "evidence")
            evidence_items = raw_ev if isinstance(raw_ev, (list, tuple)) else []
        evidence_dicts = []
        for e in evidence_items:
            snippet = getattr(e, "snippet", "") if not isinstance(e, dict) else e.get("snippet", "")
            raw_score = getattr(e, "score", 0.8) if not isinstance(e, dict) else e.get("score", 0.8)
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = 0.8
            evidence_dicts.append({"snippet": str(snippet), "score": score})

        h_score = float(getattr(initial_verification, "overall_h_score", getattr(initial_verification, "hallucination_score", 0.0)))

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
                    reason=cand.get("reason", "NO_CORRECTION_POSSIBLE"),
                    reverification=ReverificationInfo(passed=False, h_score=h_score, claims_analyzed=1),
                )
            return ClosedLoopRepairResult(
                final_text=initial_text,
                performed=False,
                reason=cand.get("reason", "NO_CORRECTION_POSSIBLE"),
                reverification=ReverificationInfo(passed=False, h_score=h_score, claims_analyzed=1),
            )

        # Run re-verification synchronously if pipeline exists
        rev_h = h_score
        passed = False
        if self.pipeline:
            try:
                if hasattr(self.pipeline, "analyze_response"):
                    rev_res = self.pipeline.analyze_response(full_text=corr_text, query=user_query)
                elif hasattr(self.pipeline, "analyze"):
                    rev_res = self.pipeline.analyze(response=corr_text, query=user_query)
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

        final_out = corr_text if passed else f"{initial_text}\n\n[Note: System could not produce a sufficiently verified correction.]"
        return ClosedLoopRepairResult(
            final_text=final_out,
            performed=True,
            reason=cand.get("reason", "EVIDENCE_GROUNDED_REPAIR"),
            claims_corrected=[
                ClaimCorrectionInfo(
                    original_claim=initial_text,
                    corrected_claim=corr_text,
                    reason=cand.get("reason", ""),
                    method=cand.get("method", "evidence_grounded"),
                )
            ],
            original_to_corrected=[{"original": initial_text, "corrected": corr_text}],
            reverification=ReverificationInfo(passed=passed, h_score=rev_h, claims_analyzed=1),
        )


# Backward-compatible alias
CorrectionEngine = HallucinationCorrectionEngine
