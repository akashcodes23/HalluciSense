import re
from typing import List, Tuple
from .types import Pillar3Result

class Pillar3ConsistencyEngine:
    """
    Pillar 3: Consistency Checking Engine.
    - Compares target response against alternate sample generations / paraphrases.
    - Measures pairwise semantic consistency.
    - Computes Consistency Failure Score (CF) in range [0.0, 1.0].
    """

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

    def evaluate_consistency(
        self,
        primary_response: str,
        sample_responses: List[str]
    ) -> Tuple[List[float], float]:
        """
        Compute pairwise similarities between primary response and alternate samples.
        CF = 1.0 - mean(similarity)
        """
        if not sample_responses:
            # Self-consistency defaults to perfect consistency if no samples provided
            return [1.0], 0.0

        similarities: List[float] = []
        for sample in sample_responses:
            sim = self.jaccard_similarity(primary_response, sample)
            similarities.append(round(sim, 4))

        avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0
        consistency_failure = max(0.0, min(1.0, 1.0 - avg_similarity))
        return similarities, round(consistency_failure, 4)

    def analyze(
        self,
        primary_response: str,
        sample_responses: List[str] = None
    ) -> Pillar3Result:
        """
        Execute Pillar 3 consistency checking flow.
        """
        if sample_responses is None:
            sample_responses = []

        similarities, cf_score = self.evaluate_consistency(primary_response, sample_responses)

        if not sample_responses:
            reasoning = "Single sample evaluated; consistency baseline assumed."
        elif cf_score < 0.2:
            reasoning = f"High self-consistency across {len(sample_responses)} sample outputs (Avg similarity: {1.0 - cf_score:.2f})."
        elif cf_score < 0.5:
            reasoning = f"Moderate semantic variance across sample outputs."
        else:
            reasoning = f"Significant self-contradiction or variation across generated samples."

        return Pillar3Result(
            sample_responses=sample_responses,
            pairwise_similarities=similarities,
            consistency_failure_score=cf_score,
            reasoning=reasoning
        )
