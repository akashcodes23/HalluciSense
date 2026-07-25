from typing import Dict, Tuple
from .types import RiskLevel, Pillar1Result, Pillar2Result, Pillar3Result
from ..config import settings

class FusionEngine:
    """
    Hybrid Fusion Engine.
    Combines Factual Error (FE), Confidence Gap (CG), and Consistency Failure (CF)
    using formula: H = alpha * FE + beta * CG + gamma * CF
    """

    def __init__(
        self,
        alpha: float = None,
        beta: float = None,
        gamma: float = None
    ):
        self.alpha = alpha if alpha is not None else settings.ALPHA_FACTUAL_ERROR
        self.beta = beta if beta is not None else settings.BETA_CONFIDENCE_GAP
        self.gamma = gamma if gamma is not None else settings.GAMMA_CONSISTENCY_FAILURE

        # Normalize weights to ensure sum = 1.0
        total_weight = self.alpha + self.beta + self.gamma
        if total_weight > 0:
            self.alpha = round(self.alpha / total_weight, 4)
            self.beta = round(self.beta / total_weight, 4)
            self.gamma = round(self.gamma / total_weight, 4)

    def compute_h_score(
        self,
        fe: float,
        cg: float,
        cf: float
    ) -> float:
        """
        Compute aggregate H-Score in range [0.0, 1.0]
        H = alpha * FE + beta * CG + gamma * CF
        """
        fe_clamped = max(0.0, min(1.0, fe))
        cg_clamped = max(0.0, min(1.0, cg))
        cf_clamped = max(0.0, min(1.0, cf))

        h_score = (self.alpha * fe_clamped) + (self.beta * cg_clamped) + (self.gamma * cf_clamped)
        return round(max(0.0, min(1.0, h_score)), 4)

    def determine_risk_level(self, h_score: float) -> Tuple[RiskLevel, str]:
        """
        Assign RiskLevel enum and hexadecimal color indicator.
        - VERIFIED (< 0.35): Green (#10B981)
        - NEEDS_VERIFICATION (0.35 to 0.65): Yellow (#F59E0B)
        - LIKELY_HALLUCINATED (>= 0.65): Red (#EF4444)
        """
        if h_score < settings.VERIFIED_THRESHOLD:
            return RiskLevel.VERIFIED, "#10B981"
        elif h_score < settings.HALLUCINATED_THRESHOLD:
            return RiskLevel.NEEDS_VERIFICATION, "#F59E0B"
        else:
            return RiskLevel.LIKELY_HALLUCINATED, "#EF4444"

    def fuse(
        self,
        p1: Pillar1Result,
        p2: Pillar2Result,
        p3: Pillar3Result
    ) -> Tuple[float, RiskLevel, str, Dict[str, float]]:
        """
        Fuse individual pillar results into final metrics.
        """
        h_score = self.compute_h_score(
            fe=p1.factual_error_score,
            cg=p2.confidence_gap_score,
            cf=p3.consistency_failure_score
        )

        risk_level, color_code = self.determine_risk_level(h_score)
        weights_used = {
            "alpha_factual_error": self.alpha,
            "beta_confidence_gap": self.beta,
            "gamma_consistency_failure": self.gamma
        }

        return h_score, risk_level, color_code, weights_used
