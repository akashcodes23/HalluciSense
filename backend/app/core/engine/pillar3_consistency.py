"""Production-safe Pillar 3 consistency engine.

Phase 47B deliberately removes the all-MiniLM SentenceTransformer from the
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
from typing import List, Tuple, Optional

import structlog

from .types import Pillar3Result, NLIAnalysis

logger = structlog.get_logger(__name__)

ALIGNMENT_THRESHOLD: float = 0.20
LAMBDA_NLI: float = 0.70
MAX_CLAIMS: int = 15


class Pillar3ConsistencyEngine:
    """Claim-level consistency reasoning with one shared NLI model."""

    _nli_engine = None

    @classmethod
    def _get_nli_engine(cls):
        if cls._nli_engine is None:
            from app.core.engine.entailment import EvidenceEntailmentEngine
            cls._nli_engine = EvidenceEntailmentEngine()
        return cls._nli_engine

    def jaccard_similarity(self, text1: str, text2: str) -> float:
        w1 = set(re.findall(r"\w+", text1.lower()))
        w2 = set(re.findall(r"\w+", text2.lower()))
        if not w1 or not w2:
            return 1.0 if w1 == w2 else 0.0
        return len(w1.intersection(w2)) / float(len(w1.union(w2)))

    def _sanitize_samples(self, primary_response: str, sample_responses: Optional[List[str]]) -> List[str]:
        if not sample_responses:
            return []
        return [s.strip() for s in sample_responses if isinstance(s, str) and s.strip()]

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

    # Backwards-compatible API. It intentionally does not load an embedding model.
    def evaluate_semantic_consistency(
        self,
        primary_response: str,
        sample_responses: List[str],
    ) -> Tuple[List[float], float]:
        return self.evaluate_jaccard_consistency(primary_response, sample_responses)

    def _nli_pair(self, first: str, second: str) -> NLIAnalysis:
        nli = self._get_nli_engine().classify(claim=second, evidence=first)
        e_prob = float(nli.get("entailment", 0.0))
        n_prob = float(nli.get("neutral", 0.0))
        c_prob = float(nli.get("contradiction", 0.0))
        probs = {"entailment": e_prob, "neutral": n_prob, "contradiction": c_prob}
        label = max(probs, key=probs.get)
        return NLIAnalysis(
            primary_claim=first,
            comparison_claim=second,
            semantic_similarity=round(self.jaccard_similarity(first, second), 4),
            entailment_probability=round(e_prob, 4),
            neutral_probability=round(n_prob, 4),
            contradiction_probability=round(c_prob, 4),
            label=label,
            nli_available=True,
        )

    def evaluate_claim_nli(
        self,
        primary_response: str,
        sample_responses: List[str],
    ) -> Tuple[List[NLIAnalysis], Optional[float], bool]:
        if not sample_responses:
            return [], None, False

        primary_sents = self._split_sentences(primary_response)
        if not primary_sents:
            return [], None, False

        analyses: List[NLIAnalysis] = []
        try:
            for sample in sample_responses:
                sample_sents = self._split_sentences(sample)
                if not sample_sents:
                    continue

                # Cheap lexical alignment chooses the candidate claim; the actual
                # contradiction/entailment decision is still made by NLI.
                best = max(
                    ((self.jaccard_similarity(p, s), p, s) for p in primary_sents for s in sample_sents),
                    key=lambda x: x[0],
                )
                sim, primary_claim, comparison_claim = best
                if sim < ALIGNMENT_THRESHOLD:
                    analyses.append(
                        NLIAnalysis(
                            primary_claim=primary_claim,
                            comparison_claim=comparison_claim,
                            semantic_similarity=round(sim, 4),
                            entailment_probability=0.0,
                            neutral_probability=1.0,
                            contradiction_probability=0.0,
                            label="neutral",
                            nli_available=True,
                        )
                    )
                else:
                    analyses.append(self._nli_pair(primary_claim, comparison_claim))

            if not analyses:
                return [], None, False
            aligned = [a for a in analyses if (a.semantic_similarity or 0.0) >= ALIGNMENT_THRESHOLD]
            contradiction = [a.contradiction_probability or 0.0 for a in aligned]
            score = sum(contradiction) / len(contradiction) if contradiction else 0.0
            return analyses, round(score, 4), True
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

        claims = claims[:MAX_CLAIMS]
        nli_engine = self._get_nli_engine()
        pairs = [(claims[i], claims[j]) for i in range(len(claims)) for j in range(i + 1, len(claims))]

        # Batch all pairs through the shared NLI model. This is materially safer
        # than loading a separate embedding model and running one inference at a time.
        nli_results = nli_engine.classify_batch(
            [second for first, second in pairs],
            [first for first, second in pairs],
            batch_size=min(8, len(pairs)),
        )

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

    def analyze(self, primary_response: str, sample_responses: Optional[List[str]] = None) -> Pillar3Result:
        t0 = time.perf_counter()
        valid_samples = self._sanitize_samples(primary_response, sample_responses)

        if not valid_samples:
            result = self.evaluate_intra_response_consistency(primary_response)
            result.last_timings = {
                "start_time": t0,
                "end_time": time.perf_counter(),
                "duration_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "semantic_ms": 0.0,
                "nli_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }
            return result

        t_nli = time.perf_counter()
        similarities, cf_lexical = self.evaluate_jaccard_consistency(primary_response, valid_samples)
        nli_analyses, contradiction_score, nli_available = self.evaluate_claim_nli(primary_response, valid_samples)
        nli_ms = (time.perf_counter() - t_nli) * 1000.0

        if nli_available and contradiction_score is not None:
            cf_final = round((1.0 - LAMBDA_NLI) * cf_lexical + LAMBDA_NLI * contradiction_score, 4)
        else:
            cf_final = cf_lexical

        max_con = max((a.contradiction_probability or 0.0 for a in nli_analyses), default=0.0)
        reasoning = (
            f"Static/cross-generation consistency evaluated with lexical alignment and shared NLI. "
            f"Samples={len(valid_samples)}, contradiction={max_con:.2f}, CF={cf_final:.2f}."
        )

        result = Pillar3Result(
            sample_responses=valid_samples,
            pairwise_similarities=similarities,
            consistency_failure_score=cf_final,
            similarity_method="lexical_alignment_plus_nli",
            nli_analyses=nli_analyses,
            contradiction_score=contradiction_score,
            nli_available=nli_available,
            alignment_method="lexical_pair_alignment",
            reasoning=reasoning,
            available=True,
            status="EXECUTED",
            mode="CROSS_GENERATION_CONSISTENCY",
            sentence_consistency_score=round(1.0 - cf_final, 4),
        )
        result.last_timings = {
            "start_time": t0,
            "end_time": time.perf_counter(),
            "duration_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "semantic_ms": 0.0,
            "nli_ms": round(nli_ms, 2),
        }
        return result
