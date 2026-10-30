import re
import structlog
import json
import numpy as np
import google.generativeai as genai

from typing import List, Optional, Tuple, Any, Dict

from ..config import settings
from app.core.circuit_breaker import QuotaCircuitBreaker

logger = structlog.get_logger(__name__)

from .types import (
    HallucinationReport,
    SentenceAnalysis,
    TokenAnalysis,
    EvidenceItem,
    RiskLevel,
    Pillar1Result,
    Pillar2Result,
    Pillar3Result,
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
            sentence_spans = self._split_sentences(text)
            claims = [
                s[0] for s in sentence_spans
                if not self._is_conversational_filler(s[0])
            ]
            if not claims:
                p1_claims = self.p1_engine.extract_claims(text)
                claims = p1_claims if p1_claims else [text]

            retrieval_queries = []
            if query and isinstance(query, str) and query.strip():
                clean_q = self.clean_conversational_prefix(query)
                if clean_q and clean_q not in retrieval_queries:
                    retrieval_queries.append(clean_q)
                if query.strip() not in retrieval_queries:
                    retrieval_queries.append(query.strip())

            for c in claims:
                clean_c = self.clean_conversational_prefix(c)
                target_claim = clean_c if (clean_c and not self._is_conversational_filler(clean_c)) else c
                if target_claim not in retrieval_queries:
                    retrieval_queries.append(target_claim)

                # Phase 16C: Targeted scientific queries from SRO structure
                from app.core.engine.scientific_claim_reasoner import ScientificClaimReasoner
                sro = ScientificClaimReasoner.extract_sro(target_claim)
                targeted = ScientificClaimReasoner.generate_targeted_queries(sro)
                for t_q in targeted:
                    if t_q not in retrieval_queries:
                        retrieval_queries.append(t_q)

                # Expand relation-specific queries for factual assertions
                from .pillar1_retrieval import extract_capital_relation
                cap_rel = extract_capital_relation(target_claim)
                if cap_rel:
                    city, entity = cap_rel
                    alt1 = f"capital of {entity}"
                    alt2 = entity
                    if alt1 not in retrieval_queries:
                        retrieval_queries.append(alt1)
                    if alt2 not in retrieval_queries:
                        retrieval_queries.append(alt2)

                # Key content terms expansion for factual assertions
                c_words = [w for w in re.findall(r'[A-Za-z0-9]+', target_claim) if w.lower() not in {"a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "at", "to", "for", "with", "by", "from", "that", "this", "it", "and", "or", "as", "composed", "consists"}]
                if len(c_words) >= 2:
                    k_query = " ".join(c_words)
                    if k_query not in retrieval_queries:
                        retrieval_queries.append(k_query)

            raw_evidence = self.retriever.retrieve(retrieval_queries)

            evidence_items: List[EvidenceItem] = []
            for ev in raw_evidence:
                snippet = ev.get("snippet", "").strip()
                if not snippet:
                    continue
                evidence_items.append(
                    EvidenceItem(
                        claim=ev.get("claim", ""),
                        snippet=snippet,
                        source_name=ev.get("source_name", "Wikipedia"),
                        source_url=ev.get("source_url"),
                        similarity_score=max(0.0, min(1.0, float(ev.get("similarity_score", 0.5)))),
                        is_supporting=ev.get("is_supporting", True),
                        retrieval_query=ev.get("retrieval_query", ""),
                        claim_subject=ev.get("claim_subject"),
                        claim_relation=ev.get("claim_relation"),
                        claim_object=ev.get("claim_object"),
                        relevance_score=ev.get("relevance_score"),
                        entailment_prob=ev.get("entailment_prob"),
                        contradiction_prob=ev.get("contradiction_prob"),
                        neutral_prob=ev.get("neutral_prob"),
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

    @staticmethod
    def clean_conversational_prefix(text: str) -> str:
        cleaned = re.sub(
            r"^(?:sure|certainly|of course|absolutely|gladly|happy to help|to answer your question)[,!:]?\s*(?:here (?:is|are) (?:the|some|what|your)[^:]*[:.]\s*)?",
            "",
            text.strip(),
            flags=re.IGNORECASE,
        )
        return cleaned.strip()

    @staticmethod
    def _is_conversational_filler(text: str) -> bool:
        t = text.strip().lower()
        cleaned = re.sub(
            r"^(?:sure|certainly|of course|absolutely|gladly|happy to help|to answer your question)[,!:]?\s*(?:here (?:is|are) (?:the|some|what|your)[^:]*[:.]\s*)?",
            "",
            t,
            flags=re.IGNORECASE,
        ).strip()
        if not cleaned or len(cleaned) < 5:
            return True
        filler_exact = [
            r"^(?:sure|certainly|of course|absolutely|gladly|happy to help)[,!.]?$",
            r"^here (?:is|are) (?:the|your) (?:answer|information)[.!:]?$",
            r"^i hope this (?:helps|information is helpful|is useful)[.!]?$",
            r"^let me know if you (?:have any|need|would like)[^.]*[.!]?$",
            r"^feel free to ask[^.]*[.!]?$",
            r"^please let me know[^.]*[.!]?$",
            r"^thanks for asking[.!]?$",
            r"^as an ai language model[,.]?.*",
        ]
        return any(re.match(pat, t) for pat in filler_exact)

    def _split_sentences(
        self,
        text: str,
    ) -> List[Tuple[str, int, int]]:
        """
        Split text into sentence strings while preserving character start/end offsets.
        Correctly handles decimal numbers (e.g. 18.015), scientific notation (e.g. 3.0e8),
        abbreviations (e.g. Dr., U.S.), chemical formulas, units, dates, and initials.
        """
        if not text or not text.strip():
            return []

        abbreviations = [
            r'Dr\.', r'Mr\.', r'Mrs\.', r'Ms\.', r'Prof\.', r'Sr\.', r'Jr\.',
            r'U\.S\.', r'U\.K\.', r'E\.U\.', r'i\.e\.', r'e\.g\.', r'a\.m\.', r'p\.m\.',
            r'vs\.', r'etc\.', r'St\.', r'Inc\.', r'Ltd\.', r'Co\.', r'Corp\.',
            r'approx\.', r'fig\.', r'eq\.', r'no\.', r'vol\.', r'dept\.', r'et al\.'
        ]

        protected_ranges: List[Tuple[int, int]] = []

        # 1. Abbreviations
        for abbr in abbreviations:
            for m in re.finditer(r'\b' + abbr, text, flags=re.IGNORECASE):
                protected_ranges.append((m.start(), m.end()))

        # 2. Decimals and scientific notation: e.g. 18.015, 3.0e8, 1.5e-3
        for m in re.finditer(r'\d+\.\d+(?:[eE][+-]?\d+)?', text):
            protected_ranges.append((m.start(), m.end()))

        # 3. Person initials: single capital letter preceded by space/start and followed by capital
        for m in re.finditer(r'(?:^|\s)[A-Z]\.(?=\s+[A-Z])', text):
            protected_ranges.append((m.end() - 1, m.end()))

        # 4. Ellipses
        for m in re.finditer(r'\.{3,}', text):
            protected_ranges.append((m.start(), m.end()))

        # 5. URLs
        for m in re.finditer(r'https?://\S+', text):
            protected_ranges.append((m.start(), m.end()))

        def is_protected(pos: int) -> bool:
            return any(start <= pos < end for start, end in protected_ranges)

        sentence_spans: List[Tuple[str, int, int]] = []
        current_start = 0
        while current_start < len(text) and text[current_start].isspace():
            current_start += 1

        for m in re.finditer(r'[.!?]+["\'”’\)\]]*(?=\s+|$)', text):
            end_pos = m.end()
            punct_pos = m.start()
            if not is_protected(punct_pos):
                sent = text[current_start:end_pos].strip()
                if sent:
                    # Find exact start of non-whitespace content in text[current_start:end_pos]
                    sent_offset = text.find(sent, current_start)
                    if sent_offset != -1:
                        sentence_spans.append((sent, sent_offset, sent_offset + len(sent)))
                    else:
                        sentence_spans.append((sent, current_start, end_pos))

                next_start = end_pos
                while next_start < len(text) and text[next_start].isspace():
                    next_start += 1
                current_start = next_start

        if current_start < len(text):
            remaining = text[current_start:].strip()
            if remaining:
                sent_offset = text.find(remaining, current_start)
                if sent_offset != -1:
                    sentence_spans.append((remaining, sent_offset, sent_offset + len(remaining)))
                else:
                    sentence_spans.append((remaining, current_start, len(text)))

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
                request_options={"timeout": 3.0},
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

    _UNSET = object()

    def analyze(
        self,
        text: str,
        token_probabilities: Any = _UNSET,
        provided_evidence: Optional[List[EvidenceItem]] = None,
        sample_responses: Any = _UNSET,
        query: Optional[str] = None,
        evidence_items: Optional[List[EvidenceItem]] = None,
        is_logprobs: Optional[bool] = None,
        token_measurements: Optional[List[Any]] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> HallucinationReport:
        """Alias for analyze_response to support standard engine contract."""
        ev = evidence_items if evidence_items is not None else provided_evidence
        return self.analyze_response(
            full_text=text,
            token_probabilities=token_probabilities,
            evidence_items=ev,
            sample_responses=sample_responses,
            query=query,
            is_logprobs=is_logprobs,
            token_measurements=token_measurements,
            provider=provider,
            model_name=model_name,
        )

    def analyze_response(
        self,
        full_text: str,
        token_probabilities: Any = _UNSET,
        evidence_items: Optional[List[EvidenceItem]] = None,
        sample_responses: Any = _UNSET,
        query: Optional[str] = None,
        provided_evidence: Optional[List[EvidenceItem]] = None,
        is_logprobs: Optional[bool] = None,
        token_measurements: Optional[List[Any]] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
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
                overall_risk_level=RiskLevel.INSUFFICIENT_EVIDENCE,
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

        if sample_responses is self._UNSET:
            sample_responses = None

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

        temp_spans = self._split_sentences(clean_text)
        factual_parts = [
            self.clean_conversational_prefix(s[0])
            for s in temp_spans
            if not self._is_conversational_filler(s[0])
        ]
        factual_parts = [p for p in factual_parts if p and not self._is_conversational_filler(p)]
        p1_eval_text = " ".join(factual_parts) if factual_parts else clean_text

        p1_global = self.p1_engine.analyze(
            p1_eval_text,
            [EvidenceItem(**e.model_dump()) for e in evidence_items],
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

        has_measurements = token_measurements is not None and len(token_measurements) > 0
        has_probs = token_probabilities is not self._UNSET and token_probabilities is not None and len(token_probabilities) > 0

        if not has_probs and not has_measurements:
            token_analyses = []
            p2_global = Pillar2Result(
                avg_probability=None,
                avg_entropy=None,
                confidence_gap_score=None,
                available=False,
                status="UNAVAILABLE",
                mode="UNAVAILABLE",
                is_proxy=False,
                reasoning=(
                    "Generation-time token log-probabilities were not captured. "
                    "For manually pasted text or providers without logprob support, "
                    "Pillar 2 uncertainty analysis is excluded from fusion (m2=0)."
                ),
            )
        else:
            actual_probs = None if token_probabilities is self._UNSET else token_probabilities
            token_analyses, _, _, _ = (
                self.p2_engine.evaluate_tokens(
                    raw_tokens,
                    actual_probs,
                    is_logprobs=is_logprobs,
                    token_measurements=token_measurements,
                )
            )

            p2_global = self.p2_engine.analyze(
                raw_tokens,
                actual_probs,
                evidence_items=evidence_items,
                p1_result=p1_global,
                is_logprobs=is_logprobs,
                token_measurements=token_measurements,
                provider=provider,
                model=model_name,
                allow_proxy=False,
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

        if not sample_responses:
            p3_global = Pillar3Result(
                sample_responses=[],
                pairwise_similarities=[],
                consistency_failure_score=None,
                similarity_method="unavailable",
                mode="MULTI_SAMPLE_REQUIRED",
                nli_analyses=[],
                contradiction_score=None,
                nli_available=False,
                alignment_method="unavailable",
                reasoning="Multiple independent generations not available. Consistency analysis excluded from fusion.",
                available=False,
                status="UNAVAILABLE",
            )
        else:
            p3_global = self.p3_engine.analyze(
                clean_text,
                sample_responses,
            )
        consistency_duration_ms = round((time.perf_counter() - t_p3_start) * 1000.0, 2)
        p3_sub_timings = getattr(p3_global, "last_timings", None) or {}

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

        all_sent_p1_statuses: List[str] = []
        for (
            idx,
            (sent_text, s_start, s_end),
        ) in enumerate(sentence_spans):

            # -------------------------------------------------
            # PILLAR 1
            # -------------------------------------------------

            if self._is_conversational_filler(sent_text):
                sent_p1 = Pillar1Result(
                    claims=[sent_text],
                    factual_error_score=0.05,
                    evidence=[],
                    claim_analyses=[],
                    available=True,
                    status="VERIFIED",
                    reasoning="Conversational discourse framing; non-assertive phrase.",
                )
                all_sent_p1_statuses.append("VERIFIED")
            else:
                clean_sent = self.clean_conversational_prefix(sent_text)
                target_sent = clean_sent if (clean_sent and not self._is_conversational_filler(clean_sent)) else sent_text
                sent_p1 = self.p1_engine.analyze(
                    target_sent,
                    [EvidenceItem(**e.model_dump()) for e in evidence_items],
                )
                statuses = getattr(self.p1_engine, "last_claim_statuses", [])
                if statuses:
                    all_sent_p1_statuses.extend(statuses)
                elif sent_p1.factual_error_score is not None:
                    all_sent_p1_statuses.append("VERIFIED" if sent_p1.factual_error_score <= 0.35 else "CONTRADICTED")

            # -------------------------------------------------
            # PILLAR 2
            # -------------------------------------------------

            if token_measurements and len(token_measurements) > 0:
                # Genuine BPE token alignment using character offsets
                matching_tms = [
                    tm for tm in token_measurements
                    if (
                        isinstance(tm, dict)
                        and tm.get("char_start") is not None
                        and tm.get("char_end") is not None
                        and tm["char_start"] < s_end
                        and tm["char_end"] > s_start
                    )
                ]
                if matching_tms:
                    sent_tokens = [tm.get("token", "") for tm in matching_tms]
                    sent_probs = [
                        tm.get("prob")
                        if tm.get("prob") is not None
                        else (math.exp(tm["logprob"]) if tm.get("logprob") is not None else 1.0)
                        for tm in matching_tms
                    ]
                    sent_p2 = self.p2_engine.analyze(
                        sent_tokens,
                        sent_probs,
                        evidence_items=evidence_items,
                        p1_result=sent_p1,
                        is_logprobs=False,
                    )
                else:
                    sent_tokens = re.findall(r"\S+", sent_text)
                    sent_p2 = Pillar2Result(
                        avg_probability=None,
                        avg_entropy=None,
                        confidence_gap_score=None,
                        available=False,
                        status="UNAVAILABLE",
                        mode="WHITE_BOX_LOGPROBS_REQUIRED",
                        is_proxy=False,
                        reasoning="No token measurements in sentence span.",
                    )
            elif token_probabilities is None or (isinstance(token_probabilities, list) and len(token_probabilities) == 0):
                sent_tokens = re.findall(r"\S+", sent_text)
                sent_p2 = Pillar2Result(
                    avg_probability=None,
                    avg_entropy=None,
                    confidence_gap_score=None,
                    available=False,
                    status="UNAVAILABLE",
                    mode="WHITE_BOX_LOGPROBS_REQUIRED",
                    is_proxy=False,
                    reasoning="Token-level generation probabilities not provided.",
                )
            elif token_probabilities is not self._UNSET and token_probabilities is not None:
                sent_tokens = re.findall(r"\S+", sent_text)
                sentence_token_count = len(sent_tokens)
                sent_probs = token_probabilities[
                    token_cursor:
                    token_cursor + sentence_token_count
                ]
                token_cursor += sentence_token_count
                sent_p2 = self.p2_engine.analyze(
                    sent_tokens,
                    sent_probs,
                    evidence_items=evidence_items,
                    p1_result=sent_p1,
                    is_logprobs=is_logprobs,
                )
            else:
                sent_tokens = re.findall(r"\S+", sent_text)
                sent_p2 = self.p2_engine.analyze(
                    sent_tokens,
                    None,
                    evidence_items=evidence_items,
                    p1_result=sent_p1,
                    is_logprobs=is_logprobs,
                )

            # -------------------------------------------------
            # PILLAR 3
            # -------------------------------------------------

            if not sample_responses:
                sent_p3 = Pillar3Result(
                    sample_responses=[],
                    pairwise_similarities=[],
                    consistency_failure_score=None,
                    similarity_method="unavailable",
                    mode="MULTI_SAMPLE_REQUIRED",
                    nli_analyses=[],
                    contradiction_score=None,
                    nli_available=False,
                    alignment_method="unavailable",
                    reasoning="Multiple independent generations not available.",
                    available=False,
                    status="UNAVAILABLE",
                )
            else:
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

        # Authoritative Verdict Integrity Stage (Phase F & Blocker 3/4 Governance)
        raw_fusion_score = float(overall_h_score)
        decision_override_reason = None
        p1_statuses = all_sent_p1_statuses if all_sent_p1_statuses else getattr(self.p1_engine, "last_claim_statuses", [])
        p1_fe = float(p1_global.factual_error_score or 0.0) if p1_global else 0.0
        p3_cf = float(p3_global.consistency_failure_score or 0.0) if p3_global and p3_global.available else 0.0
        p1_has_evidence = bool(evidence_items and p1_global and getattr(p1_global, "available", True) and p1_global.factual_error_score is not None)

        if not p1_has_evidence:
            # Pillar 2/3 Factuality Governance: Without independent factual reference grounding,
            # internal token uncertainty (P2) or semantic inconsistency (P3) cannot establish factual truth or falsity.
            # The system preserves all pillar measurements while issuing an explicit abstention.
            overall_risk = RiskLevel.INSUFFICIENT_EVIDENCE
            overall_color = "#6B7280"
            decision_override_reason = "Selective prediction abstention: factual verification requires independent external evidence (Pillar 1 unavailable)"
        elif not p1_statuses:
            # Empty claim list invariant: VERIFIED is allowed only when all required claims are SUPPORTED
            overall_risk = RiskLevel.INSUFFICIENT_EVIDENCE
            overall_color = "#6B7280"
            decision_override_reason = "Selective prediction abstention: no atomic factual claims to verify"
        else:
            # Filter out conversational filler sentences from whole-response gating
            factual_p1_statuses = [
                s for idx, s in enumerate(p1_statuses)
                if not self._is_conversational_filler(sentence_spans[idx][0] if idx < len(sentence_spans) else "")
            ]
            if not factual_p1_statuses:
                factual_p1_statuses = p1_statuses

            all_supported = all(s in ("VERIFIED", "SUPPORTED") for s in factual_p1_statuses)
            has_insufficient = any(s == "INSUFFICIENT_EVIDENCE" for s in factual_p1_statuses)
            supported_count = sum(1 for s in factual_p1_statuses if s in ("VERIFIED", "SUPPORTED"))
            total_factual = len(factual_p1_statuses)

            factual_sentences = [
                s for s in sentence_analyses
                if not self._is_conversational_filler(s.text)
            ]
            if factual_sentences:
                factual_sent_fe = sum(s.factual_error for s in factual_sentences) / len(factual_sentences)
                if all_supported and not any(s == "CONTRADICTED" for s in factual_p1_statuses):
                    p1_fe = min(p1_fe, factual_sent_fe)

            has_contradiction = any(s == "CONTRADICTED" for s in factual_p1_statuses) or p1_fe >= 0.70

            if has_contradiction:
                overall_risk = RiskLevel.LIKELY_HALLUCINATED
                overall_color = "#EF4444"
                if raw_fusion_score < 0.50:
                    decision_override_reason = "Claim-level factual contradiction detected in atomic claims (Pillar 1)"
            elif all_supported and p1_fe <= 0.30 and p3_cf < 0.50 and raw_fusion_score < 0.35:
                overall_risk = RiskLevel.VERIFIED
                overall_color = "#10B981"
                if raw_fusion_score >= 0.30:
                    decision_override_reason = "All atomic claims verified by grounded reference evidence"
            elif total_factual > 1 and (supported_count / total_factual) >= 0.70 and not has_contradiction and p1_fe <= 0.35:
                # When majority of claims are supported and none contradicted, whole response is verified
                # while ungrounded claims are cleanly flagged at sentence level
                overall_risk = RiskLevel.VERIFIED if raw_fusion_score < 0.30 else RiskLevel.NEEDS_VERIFICATION
                overall_color = "#10B981" if overall_risk == RiskLevel.VERIFIED else "#F59E0B"
                decision_override_reason = f"{supported_count}/{total_factual} claims verified by reference evidence; ungrounded claims flagged at sentence level"
            elif has_insufficient:
                overall_risk = RiskLevel.INSUFFICIENT_EVIDENCE
                overall_color = "#6B7280"
                decision_override_reason = "Selective prediction abstention: atomic claims lack verified reference grounding"
            else:
                overall_risk = RiskLevel.NEEDS_VERIFICATION
                overall_color = "#F59E0B"
                decision_override_reason = "Claims require verification: numerical criteria or evidence confidence not satisfied"

        # Synchronize sentence analysis if single-sentence
        if len(sentence_analyses) == 1:
            sentence_analyses[0].hallucination_score = overall_h_score
            sentence_analyses[0].risk_level = overall_risk
            sentence_analyses[0].color_code = overall_color
            if sentence_analyses[0].span_localization:
                sentence_analyses[0].span_localization["sentence_h_score"] = overall_h_score
                sentence_analyses[0].span_localization["risk_tier"] = overall_risk.value
                sentence_analyses[0].span_localization["color"] = overall_color

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
                # Closed-Loop Downstream Re-Verification Gate (Contribution N5)
                # Verify that the repaired text actually resolves factual errors without introducing new ones
                if corrected_text and corrected_text != clean_text and evidence_items:
                    try:
                        reverify_p1 = self.p1_engine.analyze(corrected_text, evidence_items, query=query)
                        post_fe = float(getattr(reverify_p1, "factual_error_score", 0.0) or 0.0)
                        if post_fe >= 0.35:
                            logger.warning("reverification_gate_rejected_correction", post_fe=post_fe)
                            corrected_text = (
                                f"Correction generated but failed downstream re-verification "
                                f"(post-repair FE={post_fe:.2f} >= 0.35). Reverted to original."
                            )
                    except Exception as rev_err:
                        logger.warning("reverification_exception", error=str(rev_err))
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
            if overall_risk in (RiskLevel.VERIFIED, RiskLevel.LOW_RISK):
                corrected_text = (
                    "This response is factually consistent.\n"
                    "No correction required."
                )
            else:
                corrected_text = None

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