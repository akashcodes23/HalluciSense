"""Pillar Ablation Study Module for HalluciSense Phase 6A Evaluation.

Evaluates performance across 7 pillar configurations:
1. P1_ONLY
2. P2_ONLY
3. P3_ONLY
4. P1_P2
5. P1_P3
6. P2_P3
7. P1_P2_P3

Measures incremental accuracy, precision, recall, F1, and ROC-AUC of individual pillars and combinations.
"""

from typing import Any, Dict, List, Optional
from app.core.engine.fusion import FusionEngine
from evaluation.metrics import compute_all_metrics


class AblationConfig:
    CONFIGS = {
        "P1_ONLY": (True, False, False),
        "P2_ONLY": (False, True, False),
        "P3_ONLY": (False, False, True),
        "P1_P2": (True, True, False),
        "P1_P3": (True, False, True),
        "P2_P3": (False, True, True),
        "P1_P2_P3": (True, True, True),
    }


def compute_ablation_score(
    fe: Optional[float],
    cg: Optional[float],
    cf: Optional[float],
    use_p1: bool,
    use_p2: bool,
    use_p3: bool,
    fusion_engine: Optional[FusionEngine] = None,
) -> Optional[float]:
    """Computes an ablated H-Score by masking inactive or unavailable pillars and renormalizing weights."""
    if fusion_engine is None:
        fusion_engine = FusionEngine()

    eff_fe = fe if (use_p1 and fe is not None) else None
    eff_cg = cg if (use_p2 and cg is not None) else None
    eff_cf = cf if (use_p3 and cf is not None) else None

    # Determine which of the 3 are active and available
    avail_fe = eff_fe is not None
    avail_cg = eff_cg is not None
    avail_cf = eff_cf is not None

    if not avail_fe and not avail_cg and not avail_cf:
        return None

    w_alpha = fusion_engine.alpha if avail_fe else 0.0
    w_beta = fusion_engine.beta if avail_cg else 0.0
    w_gamma = fusion_engine.gamma if avail_cf else 0.0

    total_weight = w_alpha + w_beta + w_gamma
    if total_weight <= 0:
        return None

    norm_alpha = w_alpha / total_weight
    norm_beta = w_beta / total_weight
    norm_gamma = w_gamma / total_weight

    score = 0.0
    if avail_fe and eff_fe is not None:
        score += norm_alpha * max(0.0, min(1.0, eff_fe))
    if avail_cg and eff_cg is not None:
        score += norm_beta * max(0.0, min(1.0, eff_cg))
    if avail_cf and eff_cf is not None:
        score += norm_gamma * max(0.0, min(1.0, eff_cf))

    return round(max(0.0, min(1.0, score)), 4)


def run_ablation_study(
    y_true: List[int],
    samples_p1_fe: List[Optional[float]],
    samples_p2_cg: List[Optional[float]],
    samples_p3_cf: List[Optional[float]],
    threshold: float = 0.35,
) -> Dict[str, Dict[str, Any]]:
    """Runs pillar ablation analysis across all 7 pillar configurations."""
    fusion_engine = FusionEngine()
    results = {}

    for config_name, (use_p1, use_p2, use_p3) in AblationConfig.CONFIGS.items():
        ablated_scores: List[Optional[float]] = []
        valid_y_true: List[int] = []
        valid_scores: List[float] = []
        valid_preds: List[int] = []

        for i in range(len(y_true)):
            fe = samples_p1_fe[i]
            cg = samples_p2_cg[i]
            cf = samples_p3_cf[i]

            score = compute_ablation_score(
                fe, cg, cf, use_p1, use_p2, use_p3, fusion_engine
            )
            if score is not None:
                ablated_scores.append(score)
                valid_y_true.append(y_true[i])
                valid_scores.append(score)
                valid_preds.append(1 if score >= threshold else 0)

        if not valid_y_true:
            results[config_name] = {
                "sample_count": 0,
                "metrics": None,
            }
        else:
            metrics = compute_all_metrics(valid_y_true, valid_preds, valid_scores)
            results[config_name] = {
                "sample_count": len(valid_y_true),
                "metrics": metrics,
            }

    return results
