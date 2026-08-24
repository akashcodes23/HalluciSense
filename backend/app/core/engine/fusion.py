"""HalluciSense Hybrid Fusion Engine.

Implements two complementary fusion formulations:
1. Mode A: Canonical Fixed-Weight Baseline
   H_canonical = alpha * FE + beta * CG + gamma * CF
   where alpha + beta + gamma = 1.0 (configured / learned from validation).

2. Mode B: Availability-Aware Adaptive Fusion (Extension)
   H_adaptive = sum(m_i * r_i * w_i * S_i) / sum(m_i * r_i * w_i)
   where:
   - S = [FE, CG, CF] is the pillar hallucination signal vector
   - m in {0, 1}^3 is the signal availability mask
   - r in (0, 1]^3 is the empirical signal reliability vector
   - w = [alpha, beta, gamma] are base feature importance weights
"""

from __future__ import annotations

import time
from typing import Dict, Tuple, Optional, Any, List
from .types import RiskLevel, Pillar1Result, Pillar2Result, Pillar3Result
from ..config import settings


class FusionEngine:
    """Hybrid Fusion Engine supporting Canonical Baseline & Availability-Aware Adaptive Weighting."""

    def __init__(
        self,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
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

        self.last_fusion_ms: float = 0.0

    def compute_canonical_h_score(
        self,
        fe: float,
        cg: Optional[float],
        cf: Optional[float],
    ) -> Tuple[Optional[float], bool]:
        """Computes Mode A: Canonical three-pillar score H = alpha*FE + beta*CG + gamma*CF.

        Returns (score, is_complete). If CG or CF are absent, returns (None, False).
        """
        if cg is None or cf is None:
            return None, False

        fe_c = max(0.0, min(1.0, float(fe)))
        cg_c = max(0.0, min(1.0, float(cg)))
        cf_c = max(0.0, min(1.0, float(cf)))
        score = self.alpha * fe_c + self.beta * cg_c + self.gamma * cf_c
        return round(max(0.0, min(1.0, score)), 4), True

    def compute_adaptive_h_score(
        self,
        fe: Optional[float],
        cg: Optional[float],
        cf: Optional[float],
        reliabilities: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[float, Dict[str, float], List[int]]:
        """Computes Mode B: Availability-Aware & Reliability-Modulated Adaptive Fusion."""
        m_fe = 1 if (fe is not None and not (isinstance(fe, float) and fe != fe)) else 0
        m_cg = 1 if (cg is not None and not (isinstance(cg, float) and cg != cg)) else 0
        m_cf = 1 if (cf is not None and not (isinstance(cf, float) and cf != cf)) else 0

        fe_c = max(0.0, min(1.0, float(fe if fe is not None else 0.0)))

        r_fe, r_cg, r_cf = reliabilities if reliabilities is not None else (1.0, 1.0, 1.0)
        r_fe = max(0.05, min(1.0, r_fe))
        r_cg = max(0.05, min(1.0, r_cg))
        r_cf = max(0.05, min(1.0, r_cf))

        w1 = m_fe * r_fe * self.alpha
        w2 = m_cg * r_cg * self.beta
        w3 = m_cf * r_cf * self.gamma

        total_denom = w1 + w2 + w3
        if total_denom > 0:
            eff_w1 = w1 / total_denom
            eff_w2 = w2 / total_denom
            eff_w3 = w3 / total_denom
        else:
            eff_w1, eff_w2, eff_w3 = 1.0, 0.0, 0.0

        h = eff_w1 * fe_c
        if m_cg == 1 and cg is not None:
            h += eff_w2 * max(0.0, min(1.0, float(cg)))
        if m_cf == 1 and cf is not None:
            h += eff_w3 * max(0.0, min(1.0, float(cf)))

        effective_weights = {
            "alpha_factual_error": round(eff_w1, 4),
            "beta_confidence_gap": round(eff_w2, 4),
            "gamma_consistency_failure": round(eff_w3, 4),
        }
        mask = [m_fe, m_cg, m_cf]
        return round(max(0.0, min(1.0, h)), 4), effective_weights, mask

    def get_effective_weights(self, cg_available: bool, cf_available: bool) -> Dict[str, float]:
        """Compute effective weights dynamically without mutating base configured weights."""
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
        cf: Optional[float],
    ) -> float:
        """Compute aggregate H-Score in range [0.0, 1.0] with dynamic re-normalization."""
        h_adaptive, _, _ = self.compute_adaptive_h_score(fe, cg, cf)
        return h_adaptive

    def determine_risk_level(self, h_score: float) -> Tuple[RiskLevel, str]:
        """Assign RiskLevel enum and hexadecimal color indicator across risk tiers:

        - VERIFIED (< 0.20): Green (#10B981)
        - LOW_RISK (0.20 to 0.35): Green (#10B981)
        - NEEDS_VERIFICATION (0.35 to 0.50): Yellow (#F59E0B)
        - MODERATE_RISK (0.50 to 0.65): Orange (#F97316)
        - LIKELY_HALLUCINATED (>= 0.65): Red (#EF4444)
        """
        if h_score < 0.20:
            return RiskLevel.VERIFIED, "#10B981"
        elif h_score < 0.35:
            return RiskLevel.LOW_RISK, "#10B981"
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
        cf: Optional[float],
    ) -> Dict[str, Any]:
        """Computes parameter sensitivity grid for alpha, beta, gamma weight perturbation."""
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
        """Fuse individual pillar results into final metrics with mode support (CANONICAL, ADAPTIVE)."""
        t_fus0 = time.perf_counter()

        cg_score = p2.confidence_gap_score if (p2 and getattr(p2, 'available', False)) else None
        if p2 and p2.confidence_gap_score is None:
            cg_score = None

        cf_score = p3.consistency_failure_score if (p3 and getattr(p3, 'available', False)) else None
        if p3 and p3.consistency_failure_score is None:
            cf_score = None

        # Determine signal reliabilities
        r_p1 = getattr(p1, "dense_retrieval_score", None) or 1.0
        r_p2 = getattr(p2, "calibration_score", None) or 1.0
        r_p3 = 1.0 if cf_score is not None else 0.5
        reliabilities = (float(r_p1), float(r_p2), float(r_p3))

        if mode.upper() == "CANONICAL":
            canon_score, is_complete = self.compute_canonical_h_score(p1.factual_error_score, cg_score, cf_score)
            if is_complete and canon_score is not None:
                h_score = canon_score
                weights_used = {
                    "alpha_factual_error": self.alpha,
                    "beta_confidence_gap": self.beta,
                    "gamma_consistency_failure": self.gamma,
                }
            else:
                h_score, weights_used, _ = self.compute_adaptive_h_score(p1.factual_error_score, cg_score, cf_score, reliabilities)
        else:
            h_score, weights_used, _ = self.compute_adaptive_h_score(
                p1.factual_error_score, cg_score, cf_score, reliabilities
            )

        risk_level, color_code = self.determine_risk_level(h_score)
        self.last_fusion_ms = round((time.perf_counter() - t_fus0) * 1000.0, 2)
        return h_score, risk_level, color_code, weights_used
