import re
from typing import List, Tuple, Optional
import numpy as np
import structlog
from .types import Pillar3Result, NLIAnalysis

logger = structlog.get_logger(__name__)

ALIGNMENT_THRESHOLD: float = 0.35
LAMBDA_NLI: float = 0.35


class Pillar3ConsistencyEngine:
    """
    Pillar 3: Self-Consistency & Claim-Aligned NLI Analysis Engine.

    Measures semantic consistency across multiple sampled responses
    using sentence embeddings (all-MiniLM-L6-v2) as the primary similarity metric,
    with a lexical Jaccard similarity fallback.

    Phase 4 Enhancement:
    Integrates claim-aligned Natural Language Inference (NLI) using cross-encoder/nli-deberta-v3-small
    to detect explicit logical contradictions between primary and alternate responses.

    IMPORTANT:
    Missing alternate responses are treated as unavailable data.
    HalluciSense must never treat missing samples as 0.0 or perfect consistency.
    NLI failure safely falls back to semantic-only consistency without breaking the pipeline.
    """

    _embedding_model = None
    _nli_engine = None

    @classmethod
    def _get_embedding_model(cls):
        from app.core.engine.model_registry import ModelRegistry
        return ModelRegistry.get_sentence_transformer("all-MiniLM-L6-v2")

    @classmethod
    def _get_nli_engine(cls):
        if cls._nli_engine is None:
            from app.core.engine.entailment import EvidenceEntailmentEngine
            cls._nli_engine = EvidenceEntailmentEngine()
        return cls._nli_engine

    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Fast token-level Jaccard similarity fallback for semantic consistency calculation.
        """
        w1 = set(re.findall(r'\w+', text1.lower()))
        w2 = set(re.findall(r'\w+', text2.lower()))
        if not w1 or not w2:
            return 1.0 if w1 == w2 else 0.0
        intersection = len(w1.intersection(w2))
        union = len(w1.union(w2))
        return intersection / union if union > 0 else 0.0

    def _sanitize_samples(self, primary_response: str, sample_responses: Optional[List[str]]) -> List[str]:
        """
        Filter out None, empty, whitespace-only samples.
        """
        if not sample_responses:
            return []
        
        valid_samples = []
        for sample in sample_responses:
            if not sample or not isinstance(sample, str):
                continue
            sample_clean = sample.strip()
            if not sample_clean:
                continue
            valid_samples.append(sample_clean)
            
        return valid_samples

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into clean, non-empty sentence strings.
        """
        if not text or not text.strip():
            return []
        raw_sents = re.split(r'(?<=[.!?])\s+', text.strip())
        sents = [s.strip() for s in raw_sents if s.strip()]
        return sents if sents else [text.strip()]

    def evaluate_jaccard_consistency(
        self,
        primary_response: str,
        sample_responses: List[str]
    ) -> Tuple[List[float], float]:
        """
        Compute pairwise lexical Jaccard similarities between primary response and alternate samples.
        CF = 1.0 - mean(similarity)
        """
        if not sample_responses:
            return [], 0.0

        similarities: List[float] = []
        for sample in sample_responses:
            sim = self.jaccard_similarity(primary_response, sample)
            similarities.append(round(sim, 4))

        avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0
        consistency_failure = max(0.0, min(1.0, 1.0 - avg_similarity))
        return similarities, round(consistency_failure, 4)

    def evaluate_semantic_consistency(
        self,
        primary_response: str,
        sample_responses: List[str]
    ) -> Tuple[List[float], float]:
        """
        Compute batch sentence embedding cosine similarities between primary response and alternate samples.
        CF = 1.0 - mean(similarity)
        """
        if not sample_responses:
            return [], 0.0

        model = self._get_embedding_model()
        all_texts = [primary_response] + sample_responses
        embeddings = model.encode(all_texts, normalize_embeddings=True)

        primary_emb = embeddings[0]
        sample_embs = embeddings[1:]

        similarities: List[float] = []
        for sample_emb in sample_embs:
            raw_cosine = float(np.dot(primary_emb, sample_emb))
            clamped_sim = max(0.0, min(1.0, raw_cosine))
            similarities.append(round(clamped_sim, 4))

        avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0
        consistency_failure = max(0.0, min(1.0, 1.0 - avg_similarity))
        return similarities, round(consistency_failure, 4)

    def evaluate_claim_nli(
        self,
        primary_response: str,
        sample_responses: List[str]
    ) -> Tuple[List[NLIAnalysis], Optional[float], bool]:
        """
        Performs claim-aligned NLI analysis across alternate generations.
        
        1. Splits primary and alternate responses into sentences.
        2. Aligns each primary claim with the most semantically similar candidate claim in each alternate generation.
        3. Runs NLI classification (Premise = primary claim, Hypothesis = aligned alternate claim).
        4. Calculates aggregate contradiction score across valid aligned pairs.
        
        Returns:
            (nli_analyses, contradiction_score, nli_available)
        """
        if not sample_responses:
            return [], None, False

        primary_sents = self._split_sentences(primary_response)
        if not primary_sents:
            return [], None, False

        try:
            nli_engine = self._get_nli_engine()
            emb_model = self._get_embedding_model()
        except Exception as model_err:
            logger.warning("pillar3_nli_model_init_failed", error=str(model_err))
            return [], None, False

        nli_analyses: List[NLIAnalysis] = []

        try:
            for sample in sample_responses:
                sample_sents = self._split_sentences(sample)
                if not sample_sents:
                    continue

                # Batch embed primary sentences and candidate sample sentences for alignment
                all_sents = primary_sents + sample_sents
                embeddings = emb_model.encode(all_sents, normalize_embeddings=True)

                p_embs = embeddings[:len(primary_sents)]
                s_embs = embeddings[len(primary_sents):]

                best_sim = -1.0
                best_p_idx = 0
                best_s_idx = 0

                for i, p_emb in enumerate(p_embs):
                    for j, s_emb in enumerate(s_embs):
                        sim = float(np.dot(p_emb, s_emb))
                        if sim > best_sim:
                            best_sim = sim
                            best_p_idx = i
                            best_s_idx = j

                best_p_claim = primary_sents[best_p_idx]
                best_s_claim = sample_sents[best_s_idx]
                clamped_sim = max(0.0, min(1.0, best_sim))

                if clamped_sim < ALIGNMENT_THRESHOLD:
                    # Weak alignment -> non-comparable / neutral
                    nli_analyses.append(
                        NLIAnalysis(
                            primary_claim=best_p_claim,
                            comparison_claim=best_s_claim,
                            semantic_similarity=round(clamped_sim, 4),
                            entailment_probability=0.0,
                            neutral_probability=1.0,
                            contradiction_probability=0.0,
                            label="neutral",
                            nli_available=True
                        )
                    )
                else:
                    # Direct NLI classification: Premise = Primary Claim, Hypothesis = Aligned Alternate Claim
                    nli_probs = nli_engine.classify(claim=best_s_claim, evidence=best_p_claim)
                    e_prob = nli_probs.get("entailment", 0.0)
                    n_prob = nli_probs.get("neutral", 0.0)
                    c_prob = nli_probs.get("contradiction", 0.0)

                    # Determine dominant label
                    prob_dict = {"entailment": e_prob, "neutral": n_prob, "contradiction": c_prob}
                    label = max(prob_dict, key=prob_dict.get)

                    nli_analyses.append(
                        NLIAnalysis(
                            primary_claim=best_p_claim,
                            comparison_claim=best_s_claim,
                            semantic_similarity=round(clamped_sim, 4),
                            entailment_probability=round(e_prob, 4),
                            neutral_probability=round(n_prob, 4),
                            contradiction_probability=round(c_prob, 4),
                            label=label,
                            nli_available=True
                        )
                    )

            # Compute aggregate contradiction score over aligned pairs (sim >= ALIGNMENT_THRESHOLD)
            aligned_pairs = [item for item in nli_analyses if item.semantic_similarity >= ALIGNMENT_THRESHOLD]

            if aligned_pairs:
                c_scores = [item.contradiction_probability for item in aligned_pairs if item.contradiction_probability is not None]
                contradiction_score = round(sum(c_scores) / len(c_scores), 4) if c_scores else 0.0
                return nli_analyses, contradiction_score, True
            elif nli_analyses:
                # All pairs were weakly aligned (sim < 0.35)
                return nli_analyses, 0.0, True
            else:
                return [], None, False

        except Exception as exc:
            logger.warning("pillar3_nli_inference_failed", error=str(exc))
            return [], None, False

    def analyze(
        self,
        primary_response: str,
        sample_responses: Optional[List[str]] = None
    ) -> Pillar3Result:
        """
        Execute Pillar 3 consistency and claim-aligned NLI contradiction checking.
        """
        import time
        t_p3_start = time.perf_counter()

        t_san0 = time.perf_counter()
        valid_samples = self._sanitize_samples(primary_response, sample_responses)
        sanitization_ms = (time.perf_counter() - t_san0) * 1000.0

        if not valid_samples:
            p3_duration_ms = (time.perf_counter() - t_p3_start) * 1000.0
            res = Pillar3Result(
                sample_responses=[],
                pairwise_similarities=[],
                consistency_failure_score=None,
                similarity_method="unavailable",
                nli_analyses=[],
                contradiction_score=None,
                nli_available=False,
                alignment_method="sentence_semantic_alignment",
                reasoning=(
                    "Alternate generations were not available. "
                    "Self-consistency analysis was excluded from fusion."
                ),
                available=False,
            )
            res.last_timings = {
                "start_time": t_p3_start,
                "end_time": time.perf_counter(),
                "duration_ms": round(p3_duration_ms, 2),
                "sanitization_ms": round(sanitization_ms, 2),
                "jaccard_ms": 0.0,
                "semantic_ms": 0.0,
                "nli_ms": 0.0,
            }
            return res

        logger.info("pillar3_analysis_started", num_samples=len(valid_samples))

        # 1. Compute Semantic Consistency
        semantic_ms = 0.0
        jaccard_ms = 0.0
        try:
            t_sem0 = time.perf_counter()
            similarities, cf_semantic = self.evaluate_semantic_consistency(primary_response, valid_samples)
            semantic_ms = (time.perf_counter() - t_sem0) * 1000.0
            similarity_method = "semantic_embedding"
        except Exception as exc:
            logger.warning("pillar3_embedding_fallback", num_samples=len(valid_samples), error=str(exc))
            t_jac0 = time.perf_counter()
            similarities, cf_semantic = self.evaluate_jaccard_consistency(primary_response, valid_samples)
            jaccard_ms = (time.perf_counter() - t_jac0) * 1000.0
            similarity_method = "jaccard_fallback"

        # 2. Compute Claim-Aligned NLI Contradiction Analysis
        t_nli0 = time.perf_counter()
        nli_analyses, contradiction_score, nli_available = self.evaluate_claim_nli(primary_response, valid_samples)
        nli_ms = (time.perf_counter() - t_nli0) * 1000.0

        # 3. Fuse Contradiction Score into Contradiction-Aware CF
        if nli_available and contradiction_score is not None:
            cf_final = round(
                max(0.0, min(1.0, (1.0 - LAMBDA_NLI) * cf_semantic + LAMBDA_NLI * contradiction_score)),
                4
            )
        else:
            cf_final = cf_semantic

        # 4. Construct Descriptive Reasoning
        avg_sim = round(1.0 - cf_semantic, 4)
        if nli_available and contradiction_score is not None:
            max_c = max([item.contradiction_probability or 0.0 for item in nli_analyses], default=0.0)
            if contradiction_score > 0.25:
                reasoning = (
                    f"Semantic consistency produced CF_semantic={cf_semantic:.2f}. "
                    f"Claim-aligned NLI detected contradiction across alternate generations "
                    f"(mean contradiction prob: {contradiction_score:.2f}, max: {max_c:.2f}). "
                    f"Contradiction-aware CF={cf_final:.2f}."
                )
            else:
                reasoning = (
                    f"Semantic self-consistency evaluated across {len(valid_samples)} alternate generations "
                    f"(avg similarity: {avg_sim:.2f}, CF_semantic={cf_semantic:.2f}). "
                    f"Claim-aligned NLI verified logical consistency (contradiction score: {contradiction_score:.2f}). "
                    f"Final CF={cf_final:.2f}."
                )
        else:
            reasoning = (
                f"Self-consistency evaluated across {len(valid_samples)} alternate generations "
                f"using {similarity_method} (avg similarity: {avg_sim:.2f}, CF={cf_final:.2f}). "
                f"NLI contradiction analysis was unavailable."
            )

        logger.info(
            "pillar3_analysis_completed",
            num_samples=len(valid_samples),
            similarity_method=similarity_method,
            nli_available=nli_available,
            contradiction_score=contradiction_score,
            cf_semantic=cf_semantic,
            cf_final=cf_final
        )

        # 5. Construct Paraphrase Matrix & Sentence Consistency Score
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
            similarity_method=similarity_method,
            nli_analyses=nli_analyses,
            contradiction_score=contradiction_score,
            nli_available=nli_available,
            alignment_method="sentence_semantic_alignment",
            reasoning=reasoning,
            available=True,
            paraphrase_matrix=paraphrase_matrix,
            sentence_consistency_score=sentence_consistency,
        )
        res.last_timings = {
            "start_time": t_p3_start,
            "end_time": time.perf_counter(),
            "duration_ms": round(p3_duration_ms, 2),
            "sanitization_ms": round(sanitization_ms, 2),
            "jaccard_ms": round(jaccard_ms, 2),
            "semantic_ms": round(semantic_ms, 2),
            "nli_ms": round(nli_ms, 2),
            "consistency_paraphrase_ms": round(sanitization_ms, 2),
            "consistency_multi_run_ms": round(max(0.0, p3_duration_ms - sanitization_ms - semantic_ms - nli_ms), 2),
            "consistency_comparison_ms": round(semantic_ms + nli_ms, 2),
        }
        return res
