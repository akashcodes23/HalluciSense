import re
import structlog
import json
import numpy as np
import google.generativeai as genai

from typing import List, Optional, Tuple

from ..config import settings
from app.core.circuit_breaker import QuotaCircuitBreaker

logger = structlog.get_logger(__name__)

from .types import (
    HallucinationReport,
    SentenceAnalysis,
    TokenAnalysis,
    EvidenceItem,
    RiskLevel,
)

from .pillar1_retrieval import Pillar1RetrievalEngine
from .pillar2_confidence import Pillar2ConfidenceEngine
from .pillar3_consistency import Pillar3ConsistencyEngine
from .fusion import FusionEngine
from app.modules.knowledge.retriever import HybridRetriever


from app.core.engine.calibration import ProbabilityCalibrator, SelectiveAbstentionGate


class HallucinationDetectionPipeline:
    """
    Master Hybrid Hallucination Detection Pipeline Orchestrator.

    Executes:
        Pillar 1 -> Retrieval + factual verification
        Pillar 2 -> Token confidence analysis
        Pillar 3 -> Semantic consistency analysis
        Fusion   -> Sentence-level and document-level H-Score
        Correction -> Evidence-grounded correction when required
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
    ):
        self.p1_engine = Pillar1RetrievalEngine()
        self.p2_engine = Pillar2ConfidenceEngine()
        self.p3_engine = Pillar3ConsistencyEngine()
        self.retriever = HybridRetriever()

        self.fusion_engine = FusionEngine(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        self.calibrator = ProbabilityCalibrator(method="platt")
        self.abstention_gate = SelectiveAbstentionGate()

    def _retrieve_evidence(self, text: str, query: Optional[str] = None) -> List[EvidenceItem]:
        """Retrieve real evidence from Wikipedia + BM25 + FAISS + CrossEncoder.

        Converts raw retriever dicts into typed EvidenceItem objects
        that Pillar 1 can use for NLI-based scoring.
        """
        try:
            claims = self.p1_engine.extract_claims(text)
            if not claims:
                claims = [text]

            retrieval_queries = []
            if query and isinstance(query, str) and query.strip():
                retrieval_queries.append(query.strip())
            for c in claims:
                if c not in retrieval_queries:
                    retrieval_queries.append(c)

            raw_evidence = self.retriever.retrieve(retrieval_queries)

            evidence_items: List[EvidenceItem] = []
            for ev in raw_evidence:
                snippet = ev.get("snippet", "").strip()
                if not snippet:
                    continue
                evidence_items.append(
                    EvidenceItem(
                        claim=claims[0] if claims else text,
                        snippet=snippet,
                        source_name=ev.get("source_name", "Wikipedia"),
                        source_url=ev.get("source_url"),
                        similarity_score=max(0.0, min(1.0, float(ev.get("similarity_score", 0.5)))),
                        is_supporting=ev.get("is_supporting", True),
                    )
                )

            logger.info(
                "evidence_retrieval_completed",
                num_claims=len(claims),
                num_evidence=len(evidence_items),
            )
            return evidence_items

        except Exception as e:
            logger.warning("evidence_retrieval_failed", error=str(e))
            return []

    def _generate_p3_samples(self, text: str, query: Optional[str] = None, count: int = 3) -> List[str]:
        """Generate exactly `count` genuine stochastic alternate responses using active LLM provider.

        Used when client does not supply pre-generated candidate responses and self-consistency is enabled.
        """
        try:
            from app.core.config import settings
            from app.modules.orchestrator.service import LLMOrchestrator
            import asyncio

            if not getattr(settings, "ENABLE_SELF_CONSISTENCY", False):
                return []

            target_prompt = query.strip() if (query and isinstance(query, str) and query.strip()) else f"Provide a complete, factual response for: {text[:120]}"
            messages = [{"role": "user", "content": target_prompt}]

            orchestrator = LLMOrchestrator(
                primary_model=getattr(settings, "DEFAULT_LLM_MODEL", "gemini-2.0-flash")
            )

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        samples = pool.submit(
                            asyncio.run,
                            orchestrator.generate_samples(
                                messages=messages,
                                count=count,
                                temperature=0.7,
                            )
                        ).result()
                else:
                    samples = loop.run_until_complete(
                        orchestrator.generate_samples(
                            messages=messages,
                            count=count,
                            temperature=0.7,
                        )
                    )
            except Exception:
                samples = asyncio.run(
                    orchestrator.generate_samples(
                        messages=messages,
                        count=count,
                        temperature=0.7,
                    )
                )

            valid_samples = [s.strip() for s in (samples or []) if s and isinstance(s, str) and s.strip() and s.strip() != text.strip()]
            return valid_samples[:count]
        except Exception as exc:
            logger.info("p3_dynamic_sample_generation_unavailable", error=str(exc))
            return []

    # =========================================================
    # SENTENCE SPLITTING
    # =========================================================

    def _split_sentences(
        self,
        text: str,
    ) -> List[Tuple[str, int, int]]:
        """
        Split text into sentence strings while preserving
        character start/end offsets.
        """

        sentence_spans = []

        # Terminal punctuation followed by another sentence
        # or end-of-string.
        pattern = re.compile(
            r"[^.!?]+[.!?]+|\s*[^.!?]+$"
        )

        start = 0

        for match in pattern.finditer(text):
            sentence_text = match.group(0).strip()

            if not sentence_text:
                continue

            span_start = text.find(
                sentence_text,
                start,
            )

            # Defensive fallback in case find() fails.
            if span_start == -1:
                span_start = match.start()

                while (
                    span_start < len(text)
                    and text[span_start].isspace()
                ):
                    span_start += 1

            span_end = span_start + len(sentence_text)

            sentence_spans.append(
                (
                    sentence_text,
                    span_start,
                    span_end,
                )
            )

            start = span_end

        # Fallback for text without terminal punctuation.
        if not sentence_spans and text.strip():
            clean = text.strip()

            span_start = text.find(clean)

            if span_start == -1:
                span_start = 0

            sentence_spans.append(
                (
                    clean,
                    span_start,
                    span_start + len(clean),
                )
            )

        return sentence_spans

    # =========================================================
    # CORRECTION GENERATION
    # =========================================================

    def _generate_correction(
        self,
        full_text: str,
        sentence_analyses: List[SentenceAnalysis],
        evidence_items: List[EvidenceItem],
    ) -> Tuple[str, List[SentenceAnalysis]]:
        """
        Uses Gemini to generate an evidence-grounded corrected
        response and explanations for flagged sentences.
        """
        if QuotaCircuitBreaker.is_tripped():
            logger.warning("CORRECTION_SKIPPED", reason="circuit_breaker_tripped")
            return ("Correction generation skipped due to rate limits.", sentence_analyses)

        try:
            genai.configure(
                api_key=settings.GEMINI_API_KEY
            )

            model = genai.GenerativeModel(
                model_name=settings.DEFAULT_LLM_MODEL
            )

            evidence_text = "\n".join(
                [
                    (
                        f"- Claim: {e.claim}\n"
                        f"  Evidence: {e.snippet}\n"
                        f"  Source: {e.source_name}\n"
                        f"  Similarity: {e.similarity_score:.3f}\n"
                        f"  Supporting: {e.is_supporting}"
                    )
                    for e in evidence_items
                ]
            )

            if not evidence_text:
                evidence_text = (
                    "No external evidence was available."
                )

            flagged_sentences = "\n".join(
                [
                    (
                        f"[{s.sentence_id}] {s.text}\n"
                        f"Risk: {s.risk_level.value}\n"
                        f"H-Score: {s.hallucination_score:.4f}\n"
                        f"Factual Error: {s.factual_error:.4f}"
                    )
                    for s in sentence_analyses
                    if s.risk_level != RiskLevel.VERIFIED
                ]
            )

            if not flagged_sentences:
                flagged_sentences = (
                    "No individual sentence exceeded the "
                    "verification threshold, but the document-level "
                    "analysis requires verification."
                )

            prompt = f"""
You are the HalluciSense correction engine.

Your job is to correct factual errors in an AI-generated response
using ONLY the supplied evidence.

ORIGINAL RESPONSE:
{full_text}

RETRIEVED EVIDENCE:
{evidence_text}

FLAGGED SENTENCES:
{flagged_sentences}

TASK:

1. Produce a corrected version of the original response.
2. Preserve statements that are supported by the evidence.
3. Correct statements that are contradicted by the evidence.
4. Do not invent facts that are not supported by the supplied evidence.
5. For every flagged sentence, explain:
   - why it was flagged,
   - what evidence supports or contradicts it,
   - what the corrected sentence should be.
6. If the evidence is insufficient to establish the truth of a claim,
   explicitly describe it as unverified rather than inventing a correction.

Respond STRICTLY in JSON using this schema:

{{
    "corrected_full_text": "...",
    "sentence_corrections": [
        {{
            "sentence_id": 0,
            "explanation": "...",
            "correction": "..."
        }}
    ]
}}
"""

            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json"
                ),
                request_options={"timeout": 30},
            )

            result = json.loads(response.text)

            corrected_text = result.get(
                "corrected_full_text",
                full_text,
            )

            corrections = result.get(
                "sentence_corrections",
                [],
            )

            for correction in corrections:
                sentence_id = correction.get(
                    "sentence_id"
                )

                if (
                    sentence_id is not None
                    and isinstance(sentence_id, int)
                    and 0 <= sentence_id < len(sentence_analyses)
                ):
                    corrected_sentence = correction.get(
                        "correction"
                    )

                    explanation = correction.get(
                        "explanation"
                    )

                    sentence_analyses[
                        sentence_id
                    ].corrected_response = corrected_sentence

                    if explanation:
                        sentence_analyses[
                            sentence_id
                        ].reasoning += (
                            f"\nCorrection explanation: "
                            f"{explanation}"
                        )

            return (
                corrected_text,
                sentence_analyses,
            )

        except Exception as e:
            logger.error(
                "correction_generation_failed",
                error=str(e),
            )

            return (
                "Failed to generate correction.",
                sentence_analyses,
            )

    # =========================================================
    # MAIN ANALYSIS PIPELINE
    # =========================================================

    def analyze(
        self,
        text: str,
        token_probabilities: Optional[List[float]] = None,
        provided_evidence: Optional[List[EvidenceItem]] = None,
        sample_responses: Optional[List[str]] = None,
        query: Optional[str] = None,
        evidence_items: Optional[List[EvidenceItem]] = None,
    ) -> HallucinationReport:
        """Alias for analyze_response to support standard engine contract."""
        ev = evidence_items if evidence_items is not None else provided_evidence
        return self.analyze_response(
            full_text=text,
            token_probabilities=token_probabilities,
            evidence_items=ev,
            sample_responses=sample_responses,
            query=query,
        )

    def analyze_response(
        self,
        full_text: str,
        token_probabilities: Optional[List[float]] = None,
        evidence_items: Optional[List[EvidenceItem]] = None,
        sample_responses: Optional[List[str]] = None,
        query: Optional[str] = None,
        provided_evidence: Optional[List[EvidenceItem]] = None,
    ) -> HallucinationReport:
        """
        Run the complete hybrid hallucination detection pipeline
        on an LLM response.
        """
        if evidence_items is None and provided_evidence is not None:
            evidence_items = provided_evidence

        clean_text = full_text.strip()

        # =====================================================
        # EMPTY RESPONSE HANDLING
        # =====================================================

        if not clean_text:
            p1_res = self.p1_engine.analyze(
                "",
                [],
            )

            p2_res = self.p2_engine.analyze(
                [],
                [],
            )

            p3_res = self.p3_engine.analyze(
                "",
                [],
            )

            _, _, _, weights = self.fusion_engine.fuse(
                p1_res,
                p2_res,
                p3_res,
            )

            return HallucinationReport(
                full_text="",
                corrected_response=None,
                overall_h_score=0.0,
                overall_risk_level=RiskLevel.VERIFIED,
                sentence_analyses=[],
                token_analyses=[],
                pillar1_summary=p1_res,
                pillar2_summary=p2_res,
                pillar3_summary=p3_res,
                weights_used=weights,
            )

        # =====================================================
        # INPUT NORMALIZATION
        # =====================================================

        import time
        t_pipe_start = time.perf_counter()

        if evidence_items is None:
            evidence_items = []

        if sample_responses is None:
            sample_responses = []

        # =====================================================
        # EVIDENCE RETRIEVAL (when no external evidence provided)
        # =====================================================

        t_ret_start = time.perf_counter()
        if not evidence_items:
            evidence_items = self._retrieve_evidence(clean_text, query=query)
        ret_timings = getattr(self.retriever, "last_timings", {})

        # =====================================================
        # PILLAR 1 — DOCUMENT-LEVEL FACTUAL VERIFICATION
        # =====================================================

        p1_global = self.p1_engine.analyze(
            clean_text,
            evidence_items,
            query=query,
        )
        t_ret_end = time.perf_counter()
        retrieval_duration_ms = round((t_ret_end - t_ret_start) * 1000.0, 2)
        claim_ext_ms = getattr(self.p1_engine, "last_claim_extraction_ms", 0.0)
        nli_ms = getattr(self.p1_engine, "last_nli_ms", 0.0)

        # =====================================================
        # PILLAR 2 — DOCUMENT-LEVEL TOKEN CONFIDENCE
        # =====================================================

        t_p2_start = time.perf_counter()
        raw_tokens = re.findall(
            r"\S+",
            clean_text,
        )

        token_analyses, _, _, _ = (
            self.p2_engine.evaluate_tokens(
                raw_tokens,
                token_probabilities,
            )
        )

        p2_global = self.p2_engine.analyze(
            raw_tokens,
            token_probabilities,
            evidence_items=evidence_items,
            p1_result=p1_global,
        )
        confidence_duration_ms = round((time.perf_counter() - t_p2_start) * 1000.0, 2)
        tok_proc_ms = getattr(self.p2_engine, "last_token_processing_ms", 0.0)
        ent_calc_ms = getattr(self.p2_engine, "last_entropy_calculation_ms", 0.0)

        # =====================================================
        # PILLAR 3 — DOCUMENT-LEVEL CONSISTENCY
        # =====================================================

        t_p3_start = time.perf_counter()
        has_llm_key = bool(getattr(settings, "GEMINI_API_KEY", "") or getattr(settings, "OPENAI_API_KEY", ""))
        if (not sample_responses or len(sample_responses) == 0) and getattr(settings, "ENABLE_SELF_CONSISTENCY", False) and has_llm_key:
            sample_responses = self._generate_p3_samples(clean_text, query=query, count=3)

        p3_global = self.p3_engine.analyze(
            clean_text,
            sample_responses,
        )
        consistency_duration_ms = round((time.perf_counter() - t_p3_start) * 1000.0, 2)
        p3_sub_timings = getattr(p3_global, "last_timings", {})

        # =====================================================
        # SENTENCE EXTRACTION
        # =====================================================

        sentence_spans = self._split_sentences(
            clean_text
        )

        logger.info(
            "sentence_extraction_completed",
            num_sentences=len(sentence_spans),
        )

        sentence_analyses: List[
            SentenceAnalysis
        ] = []

        # IMPORTANT:
        # Tracks where each sentence lies inside the global
        # token probability sequence.
        token_cursor = 0

        # =====================================================
        # SENTENCE-LEVEL THREE-PILLAR ANALYSIS
        # =====================================================

        for (
            idx,
            (sent_text, s_start, s_end),
        ) in enumerate(sentence_spans):

            # -------------------------------------------------
            # PILLAR 1
            # -------------------------------------------------

            sent_p1 = self.p1_engine.analyze(
                sent_text,
                evidence_items,
            )

            # -------------------------------------------------
            # PILLAR 2
            # -------------------------------------------------

            sent_tokens = re.findall(
                r"\S+",
                sent_text,
            )

            if token_probabilities is not None:
                sentence_token_count = len(
                    sent_tokens
                )

                sent_probs = token_probabilities[
                    token_cursor:
                    token_cursor + sentence_token_count
                ]

                token_cursor += sentence_token_count

            else:
                sent_probs = None

            sent_p2 = self.p2_engine.analyze(
                sent_tokens,
                sent_probs,
                evidence_items=evidence_items,
                p1_result=sent_p1,
            )

            # -------------------------------------------------
            # PILLAR 3
            # -------------------------------------------------

            sent_p3 = self.p3_engine.analyze(
                sent_text,
                sample_responses,
            )

            # -------------------------------------------------
            # SENTENCE-LEVEL FUSION
            # -------------------------------------------------

            (
                s_h_score,
                s_risk,
                s_color,
                _,
            ) = self.fusion_engine.fuse(
                sent_p1,
                sent_p2,
                sent_p3,
            )

            reasoning = (
                f"H-Score: {s_h_score:.4f}. "
                f"Pillar 1: {sent_p1.reasoning} "
                f"Pillar 2: {sent_p2.reasoning} "
                f"Pillar 3: {sent_p3.reasoning}"
            )
            # Propagate sentence H-Score into token attribution & span localization
            span_localization = {
                "start_char": s_start,
                "end_char": s_end,
                "token_count": len(sent_tokens),
                "sentence_h_score": s_h_score,
                "risk_tier": s_risk.value,
                "color": s_color,
            }

            confidence_decomp_sent = {
                "evidence_grounding": round(sent_p1.factual_error_score, 4),
                "confidence_gap": round(sent_p2.confidence_gap_score or 0.0, 4) if sent_p2.available else 0.0,
                "consistency_failure": round(sent_p3.consistency_failure_score or 0.0, 4) if sent_p3.available else 0.0,
            }

            sentence_analyses.append(
                SentenceAnalysis(
                    sentence_id=idx,
                    text=sent_text,
                    start_char=s_start,
                    end_char=s_end,

                    factual_error=(
                        sent_p1.factual_error_score
                    ),

                    confidence_gap=(
                        sent_p2.confidence_gap_score
                    ),

                    consistency_failure=(
                        sent_p3.consistency_failure_score
                    ),

                    hallucination_score=s_h_score,
                    risk_level=s_risk,
                    color_code=s_color,

                    evidence=sent_p1.evidence,

                    reasoning=reasoning,
                    span_localization=span_localization,
                    confidence_decomposition=confidence_decomp_sent,
                )
            )

        # =====================================================
        # DOCUMENT-LEVEL FUSION & SENSITIVITY ANALYSIS
        # =====================================================

        (
            overall_h_score,
            overall_risk,
            overall_color,
            weights,
        ) = self.fusion_engine.fuse(
            p1_global,
            p2_global,
            p3_global,
        )

        sensitivity_diag = self.fusion_engine.compute_sensitivity_analysis(
            fe=p1_global.factual_error_score,
            cg=p2_global.confidence_gap_score if p2_global.available else None,
            cf=p3_global.consistency_failure_score if p3_global.available else None,
        )

        if overall_h_score is not None:
            overall_h_score = float(np.nan_to_num(overall_h_score, nan=0.0))
        if p1_global.factual_error_score is not None:
            p1_global.factual_error_score = float(np.nan_to_num(p1_global.factual_error_score, nan=0.0))
        if p2_global.confidence_gap_score is not None:
            p2_global.confidence_gap_score = float(np.nan_to_num(p2_global.confidence_gap_score, nan=0.0))
        else:
            p2_global.status = "UNAVAILABLE"
        if p3_global.consistency_failure_score is not None:
            p3_global.consistency_failure_score = float(np.nan_to_num(p3_global.consistency_failure_score, nan=0.0))
        else:
            p3_global.status = "UNAVAILABLE"

        # =====================================================
        # SENTENCE-LEVEL SAFETY CHECK
        # =====================================================

        has_flagged_sentence = any(
            sentence.risk_level
            != RiskLevel.VERIFIED
            for sentence in sentence_analyses
        )

        flagged_sentence_count = sum(
            1
            for sentence in sentence_analyses
            if sentence.risk_level
            != RiskLevel.VERIFIED
        )

        logger.info(
            "pipeline_analysis_completed",
            overall_h_score=overall_h_score,
            risk_level=overall_risk.value,
            flagged_sentences=flagged_sentence_count,
            factual_error=(
                p1_global.factual_error_score
            ),
            confidence_gap=(
                p2_global.confidence_gap_score
            ),
            consistency_failure=(
                p3_global.consistency_failure_score
            ),
        )

        # =====================================================
        # CORRECTION DECISION
        # =====================================================

        corrected_text = None

        # Correction is required when:
        # 1. Automatic correction is enabled in config AND overall H-Score exceeds threshold
        # OR
        # 2. Document overall H-Score is at high risk (>= H_SCORE_CORRECTION_THRESHOLD)

        requires_correction = (
            settings.ENABLE_AUTOMATIC_CORRECTION
            and (
                overall_h_score >= settings.H_SCORE_CORRECTION_THRESHOLD
                or overall_risk == RiskLevel.HALLUCINATED
            )
        )

        if requires_correction:
            try:
                (
                    corrected_text,
                    sentence_analyses,
                ) = self._generate_correction(
                    clean_text,
                    sentence_analyses,
                    evidence_items,
                )
            except Exception as correction_exc:
                logger.error(
                    "correction_generation_failed_in_pipeline",
                    error=str(correction_exc),
                )
                corrected_text = (
                    "Correction generation failed. "
                    "Detection results are still valid."
                )

        else:
            corrected_text = (
                "This response is factually consistent.\n"
                "No correction required."
            )

        # =====================================================
        # RESEARCH EXPLAINABILITY & CALIBRATION DECOMPOSITION
        # =====================================================
        w_p1 = weights.get("alpha", 0.40)
        w_p2 = weights.get("beta", 0.30)
        w_p3 = weights.get("gamma", 0.30)

        confidence_decomp = {
            "pillar1_evidence_grounding": round(w_p1 * float(p1_global.factual_error_score or 0.0), 4),
            "pillar2_predictive_uncertainty": round(w_p2 * float(p2_global.confidence_gap_score or 0.0), 4) if p2_global.available else 0.0,
            "pillar3_structural_consistency": round(w_p3 * float(p3_global.consistency_failure_score or 0.0), 4) if p3_global.available else 0.0,
        }

        # Epistemic vs Aleatoric Uncertainty Decomposition
        p2_ent = float(p2_global.avg_entropy or 0.0) if p2_global.available else 0.0
        p3_diff = float(p3_global.consistency_failure_score or 0.0) if p3_global.available else 0.0
        epistemic_unc = round(float(np.clip(p3_diff * 0.7 + (1.0 - w_p1) * 0.3, 0.0, 1.0)), 4)
        aleatoric_unc = round(float(np.clip(p2_ent * 0.6 + (1.0 - (p1_global.factual_error_score or 0.0)) * 0.4, 0.0, 1.0)), 4)
        predictive_entropy = round(float(-overall_h_score * np.log2(overall_h_score + 1e-9) - (1.0 - overall_h_score) * np.log2(1.0 - overall_h_score + 1e-9)), 4)

        uncertainty_analysis = {
            "epistemic_uncertainty": epistemic_unc,
            "aleatoric_uncertainty": aleatoric_unc,
            "predictive_entropy": predictive_entropy,
            "total_uncertainty": round((epistemic_unc + aleatoric_unc) / 2.0, 4),
        }

        citations = [
            {
                "claim": item.claim,
                "snippet": item.snippet,
                "source_name": item.source_name,
                "source_url": item.source_url,
                "similarity_score": round(item.similarity_score, 4),
                "is_supporting": item.is_supporting,
            }
            for item in evidence_items
        ]

        # Platt / Isotonic probability calibration
        calib_res = self.calibrator.calibrate(overall_h_score)
        calibrated_p = calib_res.calibrated_probability
        fusion_duration_ms = getattr(self.fusion_engine, "last_fusion_ms", 0.0)

        perf_timings = {
            "retrieval": {
                "duration_ms": retrieval_duration_ms,
                "claim_extraction_ms": claim_ext_ms,
                "external_retrieval_ms": ret_timings.get("external_retrieval_ms", 0.0),
                "wikipedia_ms": ret_timings.get("wikipedia_ms", 0.0),
                "faiss_ms": ret_timings.get("faiss_ms", 0.0),
                "bm25_ms": ret_timings.get("bm25_ms", 0.0),
                "reranker_ms": ret_timings.get("reranker_ms", 0.0),
                "nli_ms": nli_ms,
            },
            "confidence": {
                "duration_ms": confidence_duration_ms,
                "token_processing_ms": tok_proc_ms,
                "entropy_calculation_ms": ent_calc_ms,
            },
            "consistency": {
                "duration_ms": consistency_duration_ms,
                "sanitization_ms": p3_sub_timings.get("sanitization_ms", 0.0),
                "jaccard_ms": p3_sub_timings.get("jaccard_ms", 0.0),
                "semantic_ms": p3_sub_timings.get("semantic_ms", 0.0),
                "nli_ms": p3_sub_timings.get("nli_ms", 0.0),
            },
            "fusion": {
                "duration_ms": fusion_duration_ms,
            },
            "pipeline_total_ms": round((time.perf_counter() - t_pipe_start) * 1000.0, 2),
        }

        # =====================================================
        # FINAL REPORT
        # =====================================================

        return HallucinationReport(
            full_text=clean_text,

            corrected_response=corrected_text,

            overall_h_score=overall_h_score,
            overall_risk_level=overall_risk,

            sentence_analyses=sentence_analyses,
            token_analyses=token_analyses,

            pillar1_summary=p1_global,
            pillar2_summary=p2_global,
            pillar3_summary=p3_global,

            weights_used=weights,
            confidence_decomposition=confidence_decomp,
            uncertainty_analysis=uncertainty_analysis,
            evidence_citations=citations,
            calibrated_probability=calibrated_p,
            fusion_mode="ADAPTIVE",
            sensitivity_analysis=sensitivity_diag,
            performance_timings=perf_timings,
        )