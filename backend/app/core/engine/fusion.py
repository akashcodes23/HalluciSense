from typing import Dict, Tuple, Optional, Any
from .types import RiskLevel, Pillar1Result, Pillar2Result, Pillar3Result
from ..config import settings

class FusionEngine:
    """
    Hybrid Fusion Engine.
    Combines Factual Error (FE), Confidence Gap (CG), and Consistency Failure (CF)
    using formula: H = alpha * FE + beta * CG + gamma * CF

    Dynamically renormalizes weights if CG or CF metrics are unavailable (None).
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None
    ):
        self.alpha = alpha if alpha is not None else settings.ALPHA_FACTUAL_ERROR
        self.beta = beta if beta is not None else settings.BETA_CONFIDENCE_GAP
        self.gamma = gamma if gamma is not None else settings.GAMMA_CONSISTENCY_FAILURE

        # Normalize configured base weights to ensure sum = 1.0
        total_weight = self.alpha + self.beta + self.gamma
        if total_weight > 0:
            self.alpha = round(self.alpha / total_weight, 4)
            self.beta = round(self.beta / total_weight, 4)
            self.gamma = round(self.gamma / total_weight, 4)

    def get_effective_weights(self, cg_available: bool, cf_available: bool) -> Dict[str, float]:
        """
        Compute effective weights for fusion dynamically without mutating base configured weights.
        """
        w_alpha = self.alpha
        w_beta = self.beta if cg_available else 0.0
        w_gamma = self.gamma if cf_available else 0.0

        total = w_alpha + w_beta + w_gamma
        if total > 0:
            eff_alpha = round(w_alpha / total, 4)
            eff_beta = round(w_beta / total, 4)
            eff_gamma = round(w_gamma / total, 4)
        else:
            eff_alpha = 1.0
            eff_beta = 0.0
            eff_gamma = 0.0

        return {
            "alpha_factual_error": eff_alpha,
            "beta_confidence_gap": eff_beta,
            "gamma_consistency_failure": eff_gamma,
        }

    def compute_h_score(
        self,
        fe: float,
        cg: Optional[float],
        cf: Optional[float]
    ) -> float:
        """
        Compute aggregate H-Score in range [0.0, 1.0].
        Dynamically renormalizes weights for available metrics (where metric is not None).
        """
        fe_clamped = max(0.0, min(1.0, fe))
        cg_available = cg is not None
        cf_available = cf is not None

        eff_weights = self.get_effective_weights(cg_available=cg_available, cf_available=cf_available)

        h_score = (eff_weights["alpha_factual_error"] * fe_clamped)
        if cg_available and cg is not None:
            cg_clamped = max(0.0, min(1.0, cg))
            h_score += (eff_weights["beta_confidence_gap"] * cg_clamped)
        if cf_available and cf is not None:
            cf_clamped = max(0.0, min(1.0, cf))
            h_score += (eff_weights["gamma_consistency_failure"] * cf_clamped)

        return round(max(0.0, min(1.0, h_score)), 4)

    def determine_risk_level(self, h_score: float) -> Tuple[RiskLevel, str]:
        """
        Assign RiskLevel enum and hexadecimal color indicator across 4 risk tiers.
        - VERIFIED (< 0.35): Green (#10B981)
        - NEEDS_VERIFICATION (0.35 to 0.50): Yellow (#F59E0B)
        - MODERATE_RISK (0.50 to 0.65): Orange (#F97316)
        - LIKELY_HALLUCINATED (>= 0.65): Red (#EF4444)
        """
        if h_score < 0.35:
            return RiskLevel.VERIFIED, "#10B981"
        elif h_score < 0.50:
            return RiskLevel.NEEDS_VERIFICATION, "#F59E0B"
        elif h_score < 0.65:
            return RiskLevel.MODERATE_RISK, "#F97316"
        else:
            return RiskLevel.LIKELY_HALLUCINATED, "#EF4444"

    def compute_sensitivity_analysis(
        self,
        fe: float,
        cg: Optional[float],
        cf: Optional[float]
    ) -> Dict[str, Any]:
        """
        Computes 1D/2D parameter sensitivity grid for alpha, beta, gamma weight perturbation.
        """
        cg_val = cg if cg is not None else 0.5
        cf_val = cf if cf is not None else 0.5

        sensitivity_grid = []
        for a_step in [0.2, 0.4, 0.6, 0.8]:
            for b_step in [0.1, 0.2, 0.3, 0.4]:
                g_step = round(max(0.0, 1.0 - a_step - b_step), 2)
                sim_h = round(a_step * fe + b_step * cg_val + g_step * cf_val, 4)
                sensitivity_grid.append({
                    "alpha": a_step,
                    "beta": b_step,
                    "gamma": g_step,
                    "h_score": sim_h,
                })

        weight_importance = {
            "evidence_grounding_importance": round(self.alpha * fe, 4),
            "confidence_estimation_importance": round(self.beta * cg_val, 4),
            "consistency_reasoning_importance": round(self.gamma * cf_val, 4),
        }

        return {
            "weight_importance": weight_importance,
            "sensitivity_grid": sensitivity_grid[:8],
        }

    def fuse(
        self,
        p1: Pillar1Result,
        p2: Pillar2Result,
        p3: Pillar3Result,
        mode: str = "ADAPTIVE",
    ) -> Tuple[float, RiskLevel, str, Dict[str, float]]:
        """
        Fuse individual pillar results into final metrics with mode support (STATIC, ADAPTIVE, GRADIENT).
        Handles p2.confidence_gap_score being None and p3.consistency_failure_score being None safely.
        """
        import time
        t_fus0 = time.perf_counter()
        cg_score = p2.confidence_gap_score if (p2 and getattr(p2, 'available', False)) else None
        if p2 and p2.confidence_gap_score is None:
            cg_score = None

        cf_score = p3.consistency_failure_score if (p3 and getattr(p3, 'available', False)) else None
        if p3 and p3.consistency_failure_score is None:
            cf_score = None

        h_score = self.compute_h_score(
            fe=p1.factual_error_score,
            cg=cg_score,
            cf=cf_score
        )

        risk_level, color_code = self.determine_risk_level(h_score)
        weights_used = self.get_effective_weights(
            cg_available=(cg_score is not None),
            cf_available=(cf_score is not None)
        )

        self.last_fusion_ms = round((time.perf_counter() - t_fus0) * 1000.0, 2)
        return h_score, risk_level, color_code, weights_used
