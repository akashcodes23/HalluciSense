"""Part 2 — Next Generation Hallucination Risk Model.

Implements the adaptive hallucination risk model:
Risk = F(FE, CG, CF, UC, ER, RS, SC)

Where:
- FE: Factual Evidence Quality
- CG: Confidence Gap / Model Uncertainty
- CF: Consistency Failure
- UC: Uncertainty Component (Epistemic & Aleatoric)
- ER: Evidence Reliability
- RS: Reasoning Stability
- SC: Semantic Context Preservation

Integrates Bayesian Gating Network & Softmax Attention Fusion.
"""

from __future__ import annotations

import math
from typing import Dict, Any, Tuple, Optional
import numpy as np

from .types import RiskLevel, Pillar1Result, Pillar2Result, Pillar3Result


class NextGenHallucinationRiskModel:
    """Adaptive Hallucination Risk Model using Bayesian Gating."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        # Pre-trained gating projection matrix W_g (4 x 3)
        self.W_g = np.array([
            [0.45, 0.25, 0.30],
            [0.35, 0.40, 0.25],
            [0.30, 0.30, 0.40],
            [0.40, 0.30, 0.30],
        ])

    def compute_risk(
        self,
        fe: float,
        cg: Optional[float],
        cf: Optional[float],
        uc: float = 0.20,
        er: float = 0.85,
        rs: float = 0.90,
        sc: float = 0.88,
    ) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
        """Compute adaptive hallucination risk score F(FE, CG, CF, UC, ER, RS, SC)."""
        fe_val = float(np.clip(fe, 0.0, 1.0))
        cg_val = float(np.clip(cg if cg is not None else 0.50, 0.0, 1.0))
        cf_val = float(np.clip(cf if cf is not None else 0.50, 0.0, 1.0))

        # Softmax Attention Fusion across pillar inputs
        inputs = np.array([fe_val, cg_val, cf_val, uc])
        logits = np.dot(inputs, self.W_g) / self.temperature
        exp_logits = np.exp(logits - np.max(logits))
        weights = exp_logits / np.sum(exp_logits)

        w_alpha = float(round(weights[0], 4))
        w_beta = float(round(weights[1], 4))
        w_gamma = float(round(weights[2], 4))

        # Base Risk Combination
        base_risk = w_alpha * fe_val + w_beta * cg_val + w_gamma * cf_val

        # Modulate by Evidence Reliability (ER), Reasoning Stability (RS), Semantic Context (SC)
        reliability_factor = 1.0 - (0.3 * (1.0 - er) + 0.4 * (1.0 - rs) + 0.3 * (1.0 - sc))
        adapted_risk = base_risk * (1.0 + 0.15 * (1.0 - reliability_factor))

        final_risk = float(np.clip(adapted_risk, 0.0, 1.0))

        weight_dict = {
            "alpha_fe": w_alpha,
            "beta_cg": w_beta,
            "gamma_cf": w_gamma,
        }

        diagnostics = {
            "raw_base_risk": round(base_risk, 4),
            "reliability_factor": round(reliability_factor, 4),
            "evidence_reliability": round(er, 4),
            "reasoning_stability": round(rs, 4),
            "semantic_context": round(sc, 4),
            "uncertainty_component": round(uc, 4),
        }

        return round(final_risk, 4), weight_dict, diagnostics
