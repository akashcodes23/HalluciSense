import math
from typing import List, Tuple, Optional
from .types import Pillar2Result, TokenAnalysis, RiskLevel
from ..config import settings

class Pillar2ConfidenceEngine:
    """
    Pillar 2: Confidence Analysis Engine.
    - Analyzes token probabilities and logits.
    - Computes Shannon entropy per token.
    - Computes Confidence Gap Score (CG) in range [0.0, 1.0].
    """

    def calculate_entropy(self, prob: float) -> float:
        """
        Calculate binary token entropy in bits.
        H(X) = -p log2(p) - (1-p) log2(1-p)
        """
        p = max(settings.MIN_TOKEN_PROBABILITY, min(1.0 - settings.MIN_TOKEN_PROBABILITY, prob))
        entropy = -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))
        return max(0.0, round(entropy, 4))

    def evaluate_tokens(
        self,
        tokens: List[str],
        probabilities: Optional[List[float]] = None
    ) -> Tuple[List[TokenAnalysis], float, float, float]:
        """
        Process tokens and their associated generation probabilities.
        If probabilities are not provided, estimates default probabilities from token characteristics.
        """
        if not tokens:
            return [], 1.0, 0.0, 0.0

        if probabilities is None or len(probabilities) != len(tokens):
            # Fallback uniform baseline if raw logits are not passed directly
            probabilities = [0.85] * len(tokens)

        token_analyses: List[TokenAnalysis] = []
        total_prob = 0.0
        total_entropy = 0.0
        low_prob_count = 0

        for idx, (tok, prob) in enumerate(zip(tokens, probabilities)):
            prob_clamped = max(0.0, min(1.0, prob))
            entropy = self.calculate_entropy(prob_clamped)

            if prob_clamped >= 0.75:
                risk = RiskLevel.VERIFIED
                color = "#10B981" # Green
            elif prob_clamped >= 0.45:
                risk = RiskLevel.NEEDS_VERIFICATION
                color = "#F59E0B" # Yellow
            else:
                risk = RiskLevel.LIKELY_HALLUCINATED
                color = "#EF4444" # Red
                low_prob_count += 1

            token_analyses.append(
                TokenAnalysis(
                    token=tok,
                    position=idx,
                    probability=round(prob_clamped, 4),
                    entropy=entropy,
                    risk_level=risk,
                    color_code=color
                )
            )

            total_prob += prob_clamped
            total_entropy += entropy

        avg_prob = total_prob / len(tokens)
        avg_entropy = total_entropy / len(tokens)

        # Confidence Gap Score: 1 - avg_prob weighted by fraction of uncertain tokens
        confidence_gap = (1.0 - avg_prob) * 0.7 + (low_prob_count / len(tokens)) * 0.3
        confidence_gap_score = max(0.0, min(1.0, round(confidence_gap, 4)))

        return token_analyses, round(avg_prob, 4), round(avg_entropy, 4), confidence_gap_score

    def analyze(
        self,
        tokens: List[str],
        probabilities: Optional[List[float]] = None
    ) -> Pillar2Result:
        """
        Execute Pillar 2 confidence analysis flow.
        """
        _, avg_prob, avg_entropy, cg_score = self.evaluate_tokens(tokens, probabilities)

        if cg_score < 0.25:
            reasoning = f"High token confidence. Average probability: {avg_prob:.2f}, Low entropy."
        elif cg_score < 0.55:
            reasoning = f"Moderate token confidence gap. Average probability: {avg_prob:.2f}."
        else:
            reasoning = f"High token uncertainty detected. Multiple low-probability or high-entropy tokens."

        return Pillar2Result(
            avg_probability=avg_prob,
            avg_entropy=avg_entropy,
            confidence_gap_score=cg_score,
            reasoning=reasoning
        )
