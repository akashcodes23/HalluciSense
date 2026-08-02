import math
from typing import List, Tuple, Optional

from .types import Pillar2Result, TokenAnalysis, RiskLevel
from ..config import settings


class Pillar2ConfidenceEngine:
    """
    Pillar 2: Model Confidence Analysis.

    Uses real token generation probabilities when supplied by
    the underlying LLM provider.

    IMPORTANT:
    Missing probabilities are treated as unavailable data.
    HalluciSense must never manufacture confidence values.
    """

    def calculate_entropy(self, prob: float) -> float:
        """
        Binary entropy:

        H(p) = -p log2(p) - (1-p) log2(1-p)
        """

        p = max(
            settings.MIN_TOKEN_PROBABILITY,
            min(
                1.0 - settings.MIN_TOKEN_PROBABILITY,
                prob,
            ),
        )

        entropy = -(
            p * math.log2(p)
            + (1.0 - p) * math.log2(1.0 - p)
        )

        return max(
            0.0,
            round(entropy, 4),
        )

    def evaluate_tokens(
        self,
        tokens: List[str],
        probabilities: Optional[List[float]] = None,
    ) -> Tuple[
        List[TokenAnalysis],
        Optional[float],
        Optional[float],
        Optional[float],
    ]:

        if not tokens:
            return [], None, None, None

        # --------------------------------------------------
        # No real probability measurements available.
        # --------------------------------------------------

        if probabilities is None:
            return [], None, None, None

        # Token/probability mismatch means confidence cannot
        # be measured reliably.
        if len(probabilities) != len(tokens):
            return [], None, None, None

        token_analyses: List[TokenAnalysis] = []

        total_prob = 0.0
        total_entropy = 0.0
        low_prob_count = 0

        for idx, (tok, prob) in enumerate(
            zip(tokens, probabilities)
        ):

            prob_clamped = max(
                0.0,
                min(1.0, prob),
            )

            entropy = self.calculate_entropy(
                prob_clamped
            )

            if prob_clamped >= 0.75:

                risk = RiskLevel.VERIFIED
                color = "#10B981"

            elif prob_clamped >= 0.45:

                risk = RiskLevel.NEEDS_VERIFICATION
                color = "#F59E0B"

            else:

                risk = RiskLevel.LIKELY_HALLUCINATED
                color = "#EF4444"
                low_prob_count += 1

            token_analyses.append(
                TokenAnalysis(
                    token=tok,
                    position=idx,
                    probability=round(
                        prob_clamped,
                        4,
                    ),
                    entropy=entropy,
                    risk_level=risk,
                    color_code=color,
                )
            )

            total_prob += prob_clamped
            total_entropy += entropy

        avg_prob = (
            total_prob / len(tokens)
        )

        avg_entropy = (
            total_entropy / len(tokens)
        )

        uncertain_fraction = (
            low_prob_count / len(tokens)
        )

        confidence_gap = (
            (1.0 - avg_prob) * 0.7
            + uncertain_fraction * 0.3
        )

        confidence_gap_score = max(
            0.0,
            min(
                1.0,
                round(confidence_gap, 4),
            ),
        )

        return (
            token_analyses,
            round(avg_prob, 4),
            round(avg_entropy, 4),
            confidence_gap_score,
        )

    def analyze(
        self,
        tokens: List[str],
        probabilities: Optional[List[float]] = None,
    ) -> Pillar2Result:

        (
            _,
            avg_prob,
            avg_entropy,
            cg_score,
        ) = self.evaluate_tokens(
            tokens,
            probabilities,
        )

        if cg_score is None:

            return Pillar2Result(
                avg_probability=None,
                avg_entropy=None,
                confidence_gap_score=None,
                available=False,
                reasoning=(
                    "Token-level generation probabilities "
                    "were not available from the model provider. "
                    "Confidence analysis was excluded from fusion."
                ),
            )

        if cg_score < 0.25:

            reasoning = (
                f"High model confidence. "
                f"Average token probability: "
                f"{avg_prob:.2f}."
            )

        elif cg_score < 0.55:

            reasoning = (
                f"Moderate model confidence gap. "
                f"Average token probability: "
                f"{avg_prob:.2f}."
            )

        else:

            reasoning = (
                "High token uncertainty detected. "
                "Multiple low-probability tokens were observed."
            )

        return Pillar2Result(
            avg_probability=avg_prob,
            avg_entropy=avg_entropy,
            confidence_gap_score=cg_score,
            available=True,
            reasoning=reasoning,
        )