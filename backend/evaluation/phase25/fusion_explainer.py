"""Adaptive Fusion Explainer for HalluciSense Phase 25 (Part 7).

Logs FE, CG, CF, UC, effective dynamic weights (alpha, beta, gamma),
fusion formula, partial derivatives, and percentage component contributions.
Saves fusion_trace.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import structlog

from app.core.engine.fusion import FusionEngine

logger = structlog.get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "evaluation_results" / "phase25"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def explain_fusion(
    fe_score: float,
    cg_score: float,
    cf_score: float,
    alpha: float = 0.40,
    beta: float = 0.30,
    gamma: float = 0.30,
) -> Dict[str, Any]:
    """Compute mathematical fusion explanation, partial derivatives, and contributions."""
    fusion_engine = FusionEngine(alpha=alpha, beta=beta, gamma=gamma)
    eff_weights = fusion_engine.get_effective_weights(cg_available=True, cf_available=True)

    w_a = eff_weights["alpha_factual_error"]
    w_b = eff_weights["beta_confidence_gap"]
    w_g = eff_weights["gamma_consistency_failure"]

    h_score = round(w_a * fe_score + w_b * cg_score + w_g * cf_score, 4)
    denom = max(1e-9, h_score)

    contrib_fe = round((w_a * fe_score / denom) * 100.0, 2)
    contrib_cg = round((w_b * cg_score / denom) * 100.0, 2)
    contrib_cf = round((w_g * cf_score / denom) * 100.0, 2)

    explanation = {
        "final_h_score": h_score,
        "input_scores": {
            "factual_error_FE": fe_score,
            "confidence_gap_CG": cg_score,
            "consistency_failure_CF": cf_score,
        },
        "effective_weights": {
            "alpha": w_a,
            "beta": w_b,
            "gamma": w_g,
        },
        "fusion_equation": f"H = {w_a:.2f} * FE ({fe_score:.2f}) + {w_b:.2f} * CG ({cg_score:.2f}) + {w_g:.2f} * CF ({cf_score:.2f})",
        "partial_derivatives": {
            "dH_dFE": w_a,
            "dH_dCG": w_b,
            "dH_dCF": w_g,
        },
        "contribution_percentages": {
            "factual_error_contribution_pct": contrib_fe,
            "confidence_gap_contribution_pct": contrib_cg,
            "consistency_failure_contribution_pct": contrib_cf,
        },
    }

    with open(RESULTS_DIR / "fusion_trace.json", "w", encoding="utf-8") as f:
        json.dump(explanation, f, indent=2)

    return explanation
