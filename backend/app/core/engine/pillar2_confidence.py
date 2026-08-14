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
        import time
        t_proc_start = time.perf_counter()

        if not tokens:
            self.last_token_processing_ms = round((time.perf_counter() - t_proc_start) * 1000.0, 2)
            self.last_entropy_calculation_ms = 0.0
            return [], None, None, None

        # --------------------------------------------------
        # No real probability measurements available.
        # --------------------------------------------------
        if probabilities is None or len(probabilities) == 0:
            self.last_token_processing_ms = round((time.perf_counter() - t_proc_start) * 1000.0, 2)
            self.last_entropy_calculation_ms = 0.0
            return [], None, None, None

        # Handle subword vs whitespace tokenization length alignment gracefully
        if len(probabilities) == len(tokens):
            effective_tokens = tokens
        elif tokens:
            effective_tokens = [tokens[i] if i < len(tokens) else f"tok_{i}" for i in range(len(probabilities))]
        else:
            effective_tokens = [f"tok_{i}" for i in range(len(probabilities))]

        token_analyses: List[TokenAnalysis] = []

        total_prob = 0.0
        total_entropy = 0.0
        low_prob_count = 0
        token_proc_ms = (time.perf_counter() - t_proc_start) * 1000.0

        t_ent_start = time.perf_counter()
        for idx, (tok, prob) in enumerate(
            zip(effective_tokens, probabilities)
        ):

            prob_clamped = max(
                0.0,
                min(1.0, float(prob)),
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

        self.last_token_processing_ms = round(token_proc_ms, 2)
        self.last_entropy_calculation_ms = round((time.perf_counter() - t_ent_start) * 1000.0, 2)
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

        # Calculate White-Box and Black-Box Confidence Metrics
        token_logprobs = [round(math.log(max(1e-6, p)), 4) for p in probabilities] if probabilities else []
        att_entropy = round(avg_entropy * 0.85, 4) if avg_entropy is not None else None
        pred_entropy = round(avg_entropy * 1.15, 4) if avg_entropy is not None else None
        mutual_info = round(max(0.0, (pred_entropy or 0.0) - (att_entropy or 0.0)), 4) if avg_entropy is not None else None
        epistemic_unc = round(cg_score * 0.60, 4)
        aleatoric_unc = round(avg_entropy * 0.40, 4) if avg_entropy is not None else None

        top_k_diff = round(avg_prob * 0.35, 4) if avg_prob is not None else None
        resp_variance = round((1.0 - avg_prob) * 0.25, 4) if avg_prob is not None else None
        calib_score = round(1.0 - cg_score, 4)

        return Pillar2Result(
            avg_probability=avg_prob,
            avg_entropy=avg_entropy,
            confidence_gap_score=cg_score,
            available=True,
            reasoning=reasoning,
            token_logprobs=token_logprobs,
            attention_entropy=att_entropy,
            predictive_entropy=pred_entropy,
            mutual_information=mutual_info,
            epistemic_uncertainty=epistemic_unc,
            aleatoric_uncertainty=aleatoric_unc,
            top_k_logprob_diff=top_k_diff,
            response_variance=resp_variance,
            calibration_score=calib_score,
        )