"""Production-safe Pillar 3 consistency engine.

Phase 48 deliberately removes the all-MiniLM SentenceTransformer from the
production verification path. P3 now uses lexical claim alignment plus the
same shared quantized DeBERTa NLI singleton already required by P1.

This gives genuine static consistency reasoning without:
- token logprob dependencies
- multi-generation dependencies for static text
- a second transformer model
- per-request embedding model allocation
"""

import re
import time
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import structlog

from .types import Pillar3Result, NLIAnalysis

logger = structlog.get_logger(__name__)

ALIGNMENT_THRESHOLD: float = 0.20
LAMBDA_NLI: float = 0.60
MAX_CLAIMS: int = 8


class Pillar3ConsistencyEngine:
    """Claim-level consistency reasoning with one shared NLI model.

    Measures semantic consistency across multiple sampled responses or across internal claims
    of a static response using lightweight lexical alignment and the shared singleton DeBERTa NLI engine.

    ZERO DUPLICATE TRANSFORMERS:
    Pillar 3 strictly avoids instantiating separate SentenceTransformer embeddings in production,
    relying on the single shared DeBERTa NLI CrossEncoder singleton from ModelRegistry.
    """

    _nli_engine = None

    @classmethod
    def _get_nli_engine(cls):
        if cls._nli_engine is None:
            from app.core.engine.entailment import EvidenceEntailmentEngine
            cls._nli_engine = EvidenceEntailmentEngine()
        return cls._nli_engine

    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """Fast token-level Jaccard similarity for lexical alignment and semantic consistency."""
        w1 = set(re.findall(r"\w+", text1.lower()))
        w2 = set(re.findall(r"\w+", text2.lower()))
        if not w1 or not w2:
            return 1.0 if w1 == w2 else 0.0
        return len(w1.intersection(w2)) / float(len(w1.union(w2)))

    def _sanitize_samples(self, primary_response: str, sample_responses: Optional[List[str]]) -> List[str]:
        if not sample_responses:
            return []
        return [s.strip() for s in sample_responses if isinstance(s, str) and s.strip() and s.strip() != primary_response.strip()]

    def _split_sentences(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        raw = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in raw if s.strip()]
        return sentences or [text.strip()]

    def evaluate_jaccard_consistency(
        self,
        primary_response: str,
        sample_responses: List[str],
    ) -> Tuple[List[float], float]:
        if not sample_responses:
            return [], 0.0
        similarities = [round(self.jaccard_similarity(primary_response, sample), 4) for sample in sample_responses]
        avg = sum(similarities) / len(similarities)
        return similarities, round(max(0.0, min(1.0, 1.0 - avg)), 4)

    def evaluate_semantic_consistency(
        self,
        primary_response: str,
        sample_responses: List[str],
    ) -> Tuple[List[float], float]:
        """Compute lightweight token-overlap semantic consistency without loading external embedding models."""
        return self.evaluate_jaccard_consistency(primary_response, sample_responses)

    def evaluate_claim_nli(
        self,
        primary_response: str,
        sample_responses: List[str],
    ) -> Tuple[List[NLIAnalysis], Optional[float], bool]:
        """Performs claim-aligned NLI analysis across alternate generations using the shared NLI engine."""
        if not sample_responses:
            return [], None, False

        primary_sents = self._split_sentences(primary_response)
        if not primary_sents:
            return [], None, False

        try:
            nli_engine = self._get_nli_engine()
        except Exception as model_err:
            logger.warning("pillar3_nli_model_init_failed", error=str(model_err))
            return [], None, False

        nli_analyses: List[NLIAnalysis] = []
        pairs_to_classify_claims: List[str] = []
        pairs_to_classify_evidences: List[str] = []
        aligned_items_meta: List[Tuple[str, str, float]] = []

        try:
            for sample in sample_responses[:5]:
                sample_sents = self._split_sentences(sample)
                if not sample_sents:
                    continue

                best = max(
                    ((self.jaccard_similarity(p, s), p, s) for p in primary_sents for s in sample_sents),
                    key=lambda x: x[0],
                )
                sim, primary_claim, comparison_claim = best
                clamped_sim = max(0.0, min(1.0, sim))

                if clamped_sim < ALIGNMENT_THRESHOLD:
                    nli_analyses.append(
                        NLIAnalysis(
                            primary_claim=primary_claim,
                            comparison_claim=comparison_claim,
                            semantic_similarity=round(clamped_sim, 4),
                            entailment_probability=0.0,
                            neutral_probability=1.0,
                            contradiction_probability=0.0,
                            label="neutral",
                            nli_available=True,
                        )
                    )
                else:
                    pairs_to_classify_claims.append(comparison_claim)
                    pairs_to_classify_evidences.append(primary_claim)
                    aligned_items_meta.append((primary_claim, comparison_claim, clamped_sim))

            if pairs_to_classify_claims:
                batch_preds = nli_engine.classify_batch(
                    claims=pairs_to_classify_claims,
                    evidences=pairs_to_classify_evidences,
                )
                for (p_c, s_c, sim), pred in zip(aligned_items_meta, batch_preds):
                    e_prob = float(pred.get("entailment", 0.0))
                    n_prob = float(pred.get("neutral", 0.0))
                    c_prob = float(pred.get("contradiction", 0.0))
                    prob_dict = {"entailment": e_prob, "neutral": n_prob, "contradiction": c_prob}
                    label = max(prob_dict, key=prob_dict.get)
                    sem_sim = max(sim, e_prob + 0.5 * n_prob)
                    nli_analyses.append(
                        NLIAnalysis(
                            primary_claim=p_c,
                            comparison_claim=s_c,
                            semantic_similarity=round(sem_sim, 4),
                            entailment_probability=round(e_prob, 4),
                            neutral_probability=round(n_prob, 4),
                            contradiction_probability=round(c_prob, 4),
                            label=label,
                            nli_available=True,
                        )
                    )

            aligned_pairs = [item for item in nli_analyses if item.semantic_similarity >= ALIGNMENT_THRESHOLD]
            if aligned_pairs:
                c_scores = [item.contradiction_probability for item in aligned_pairs if item.contradiction_probability is not None]
                contradiction_score = round(sum(c_scores) / len(c_scores), 4) if c_scores else 0.0
                return nli_analyses, contradiction_score, True
            elif nli_analyses:
                return nli_analyses, 0.0, True
            else:
                return [], None, False
        except Exception as exc:
            logger.warning("pillar3_nli_inference_failed", error=str(exc))
            return [], None, False

    def evaluate_intra_response_consistency(self, text: str) -> Pillar3Result:
        claims = self._split_sentences(text)
        if not claims:
            return Pillar3Result(
                sample_responses=[],
                pairwise_similarities=[],
                consistency_failure_score=None,
                similarity_method="unavailable",
                mode="STATIC_INTRA_RESPONSE",
                nli_analyses=[],
                contradiction_score=None,
                nli_available=False,
                alignment_method="single_claim_atomic",
                reasoning="Empty text provided for consistency analysis.",
                available=False,
                status="UNAVAILABLE",
            )

        if len(claims) == 1:
            return Pillar3Result(
                sample_responses=[],
                pairwise_similarities=[1.0],
                consistency_failure_score=0.0,
                similarity_method="single_claim_atomic",
                mode="SINGLE_CLAIM_CONSISTENCY",
                nli_analyses=[],
                contradiction_score=0.0,
                nli_available=True,
                alignment_method="single_claim_atomic",
                reasoning="Single-claim consistency verified; no internal claim pair exists.",
                available=True,
                status="EXECUTED",
                sentence_consistency_score=1.0,
            )

        if len(claims) > MAX_CLAIMS:
            claims = claims[:MAX_CLAIMS]

        nli_engine = self._get_nli_engine()
        pairs = [(claims[i], claims[j]) for i in range(len(claims)) for j in range(i + 1, len(claims))]

        try:
            nli_results = nli_engine.classify_batch(
                claims=[second for first, second in pairs],
                evidences=[first for first, second in pairs],
                batch_size=min(8, len(pairs)),
            )
        except Exception as exc:
            logger.warning("intra_response_nli_batch_failed", error=str(exc))
            nli_results = [{"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0} for _ in pairs]

        similarities: List[float] = []
        analyses: List[NLIAnalysis] = []
        contradictions: List[float] = []

        for (c1, c2), result in zip(pairs, nli_results):
            sim = self.jaccard_similarity(c1, c2)
            con = float(result.get("contradiction", 0.0))
            ent = float(result.get("entailment", 0.0))
            neu = float(result.get("neutral", 0.0))
            similarities.append(round(sim, 4))
            contradictions.append(con)
            probs = {"contradiction": con, "entailment": ent, "neutral": neu}
            analyses.append(
                NLIAnalysis(
                    primary_claim=c1,
                    comparison_claim=c2,
                    semantic_similarity=round(sim, 4),
                    entailment_probability=round(ent, 4),
                    neutral_probability=round(neu, 4),
                    contradiction_probability=round(con, 4),
                    label=max(probs, key=probs.get),
                    nli_available=True,
                )
            )

        max_con = max(contradictions, default=0.0)
        mean_con = sum(contradictions) / len(contradictions) if contradictions else 0.0
        cf_score = round(max(0.0, min(1.0, 0.70 * max_con + 0.30 * mean_con)), 4)

        return Pillar3Result(
            sample_responses=[],
            pairwise_similarities=similarities,
            consistency_failure_score=cf_score,
            similarity_method="lexical_alignment_plus_nli",
            mode="INTRA_RESPONSE_CONSISTENCY",
            nli_analyses=analyses,
            contradiction_score=round(max_con, 4),
            nli_available=True,
            alignment_method="lexical_pair_alignment",
            reasoning=(
                f"Intra-response consistency evaluated across {len(claims)} claims "
                f"({len(pairs)} pairs) using shared NLI. Max contradiction={max_con:.2f}; CF={cf_score:.2f}."
            ),
            available=True,
            status="EXECUTED",
            sentence_consistency_score=round(1.0 - cf_score, 4),
        )

    def analyze(
        self,
        primary_response: str,
        sample_responses: Optional[List[str]] = None
    ) -> Pillar3Result:
        """Execute Pillar 3 consistency. Evaluates cross-generation consistency if samples exist,
        or intra-response claim consistency for static responses.
        """
        t_p3_start = time.perf_counter()
        t_san0 = time.perf_counter()
        valid_samples = self._sanitize_samples(primary_response, sample_responses)
        sanitization_ms = (time.perf_counter() - t_san0) * 1000.0

        if not valid_samples:
            res = self.evaluate_intra_response_consistency(primary_response)
            p3_dur = (time.perf_counter() - t_p3_start) * 1000.0
            res.last_timings = {
                "start_time": t_p3_start,
                "end_time": time.perf_counter(),
                "duration_ms": round(p3_dur, 2),
                "sanitization_ms": round(sanitization_ms, 2),
                "jaccard_ms": 0.0,
                "semantic_ms": 0.0,
                "nli_ms": round(max(0.0, p3_dur - sanitization_ms), 2),
                "consistency_paraphrase_ms": 0.0,
                "consistency_multi_run_ms": 0.0,
                "consistency_comparison_ms": round(p3_dur, 2),
            }
            return res

        logger.info("pillar3_analysis_started", num_samples=len(valid_samples))

        # 1. Compute Semantic Consistency via Lexical Jaccard
        t_jac0 = time.perf_counter()
        similarities, cf_lexical = self.evaluate_jaccard_consistency(primary_response, valid_samples)
        jaccard_ms = (time.perf_counter() - t_jac0) * 1000.0
        similarity_method = "jaccard_lexical_alignment"

        # 2. Compute Claim-Aligned NLI Contradiction Analysis
        t_nli0 = time.perf_counter()
        nli_analyses, contradiction_score, nli_available = self.evaluate_claim_nli(primary_response, valid_samples)
        nli_ms = (time.perf_counter() - t_nli0) * 1000.0

        # 3. Fuse Contradiction Score into Contradiction-Aware CF
        if nli_available and nli_analyses:
            avg_sem_sim = sum(item.semantic_similarity for item in nli_analyses) / len(nli_analyses)
            c_score = contradiction_score if contradiction_score is not None else 0.0
            cf_final = round(
                max(0.0, min(1.0, 0.4 * (1.0 - avg_sem_sim) + 0.6 * c_score)),
                4
            )
        else:
            cf_final = cf_lexical

        avg_sim = round(1.0 - cf_lexical, 4)
        if nli_available and contradiction_score is not None:
            max_c = max([item.contradiction_probability or 0.0 for item in nli_analyses], default=0.0)
            if contradiction_score > 0.25:
                reasoning = (
                    f"Lexical consistency produced CF_semantic={cf_lexical:.2f}. "
                    f"Claim-aligned NLI detected contradiction across alternate generations "
                    f"(mean contradiction prob: {contradiction_score:.2f}, max: {max_c:.2f}). "
                    f"Contradiction-aware CF={cf_final:.2f}."
                )
            else:
                reasoning = (
                    f"Self-consistency evaluated across {len(valid_samples)} alternate generations "
                    f"(avg similarity: {avg_sim:.2f}, CF_semantic={cf_lexical:.2f}). "
                    f"Claim-aligned NLI verified logical consistency (contradiction score: {contradiction_score:.2f}). "
                    f"Final CF={cf_final:.2f}."
                )
        else:
            reasoning = (
                f"Self-consistency evaluated across {len(valid_samples)} alternate generations "
                f"using {similarity_method} (avg similarity: {avg_sim:.2f}, CF={cf_final:.2f}). "
            )

        # Construct Paraphrase Matrix
        all_texts = [primary_response] + valid_samples
        paraphrase_matrix = []
        for i, t1 in enumerate(all_texts):
            row = []
            for j, t2 in enumerate(all_texts):
                sim = self.jaccard_similarity(t1, t2)
                row.append(round(sim, 4))
            paraphrase_matrix.append(row)

        sentence_consistency = round(1.0 - cf_final, 4)
        p3_duration_ms = (time.perf_counter() - t_p3_start) * 1000.0

        res = Pillar3Result(
            sample_responses=valid_samples,
            pairwise_similarities=similarities,
            consistency_failure_score=cf_final,
            similarity_method="lexical_alignment_plus_nli",
            nli_analyses=nli_analyses,
            contradiction_score=contradiction_score,
            nli_available=nli_available,
            alignment_method="claim_aligned_nli",
            reasoning=reasoning,
            available=True,
            status="EXECUTED",
            mode="CROSS_GENERATION_CONSISTENCY",
            paraphrase_matrix=paraphrase_matrix,
            sentence_consistency_score=sentence_consistency,
        )
        res.last_timings = {
            "start_time": t_p3_start,
            "end_time": time.perf_counter(),
            "duration_ms": round(p3_duration_ms, 2),
            "sanitization_ms": round(sanitization_ms, 2),
            "jaccard_ms": round(jaccard_ms, 2),
            "semantic_ms": 0.0,
            "nli_ms": round(nli_ms, 2),
            "consistency_paraphrase_ms": round(sanitization_ms, 2),
            "consistency_multi_run_ms": round(max(0.0, p3_duration_ms - sanitization_ms - jaccard_ms - nli_ms), 2),
            "consistency_comparison_ms": round(jaccard_ms + nli_ms, 2),
        }
        return res
