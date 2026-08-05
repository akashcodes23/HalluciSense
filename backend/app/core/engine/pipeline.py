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

        self.fusion_engine = FusionEngine(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

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

    def analyze_response(
        self,
        full_text: str,
        token_probabilities: Optional[List[float]] = None,
        evidence_items: Optional[List[EvidenceItem]] = None,
        sample_responses: Optional[List[str]] = None,
    ) -> HallucinationReport:
        """
        Run the complete hybrid hallucination detection pipeline
        on an LLM response.
        """

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

        if evidence_items is None:
            evidence_items = []

        if sample_responses is None:
            sample_responses = []

        # =====================================================
        # PILLAR 1 — DOCUMENT-LEVEL FACTUAL VERIFICATION
        # =====================================================

        p1_global = self.p1_engine.analyze(
            clean_text,
            evidence_items,
        )

        # =====================================================
        # PILLAR 2 — DOCUMENT-LEVEL TOKEN CONFIDENCE
        # =====================================================

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
        )

        # =====================================================
        # PILLAR 3 — DOCUMENT-LEVEL CONSISTENCY
        # =====================================================

        p3_global = self.p3_engine.analyze(
            clean_text,
            sample_responses,
        )

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
                )
            )

        # =====================================================
        # DOCUMENT-LEVEL FUSION
        # =====================================================

        (
            overall_h_score,
            overall_risk,
            _,
            weights,
        ) = self.fusion_engine.fuse(
            p1_global,
            p2_global,
            p3_global,
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
        )