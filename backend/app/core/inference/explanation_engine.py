"""Production Real Explanation Engine Module."""

from __future__ import annotations

from typing import Any, Dict, List

import structlog

logger = structlog.get_logger(__name__)


def generate_rich_explanation(
    prob_hybrid: float,
    threshold: float,
    is_hallucinated: bool,
    claims: List[Dict[str, Any]],
    p1_prob: float,
    p2_prob: float,
    evidence_attribution: List[Dict[str, Any]],
    structural_diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate comprehensive, non-placeholder explanation breakdown for production inference.

    Args:
        prob_hybrid: Final hybrid hallucination probability.
        threshold: Operating decision threshold (tau* = 0.54).
        is_hallucinated: Boolean verdict.
        claims: List of claim dicts.
        p1_prob: Base Pillar-1 probability.
        p2_prob: Base Pillar-2 probability.
        evidence_attribution: Per-claim evidence passages and entailment.
        structural_diagnostics: Entity/numeric/temporal conflicts and graph stats.

    Returns:
        Structured explanation dict.
    """
    severity = "HIGH" if prob_hybrid >= 0.75 else ("MODERATE" if prob_hybrid >= 0.54 else "LOW")
    verdict_str = "HALLUCINATED" if is_hallucinated else "FACTUAL"

    disagreement = abs(p1_prob - p2_prob)

    if p1_prob > p2_prob and p1_prob > threshold:
        driver = f"Primary driver: Evidence Grounding (Pillar 1 risk = {p1_prob*100:.1f}%)"
    elif p2_prob > p1_prob and p2_prob > threshold:
        driver = f"Primary driver: Structural Contradiction (Pillar 2 risk = {p2_prob*100:.1f}%)"
    else:
        driver = "Primary driver: Unified cross-pillar agreement"

    summary = (
        f"The response was classified as {verdict_str} with {prob_hybrid*100:.1f}% hallucination probability "
        f"(Operating threshold τ* = {threshold}). Pillar 1 risk is {p1_prob*100:.1f}% and Pillar 2 risk is {p2_prob*100:.1f}%."
    )

    graph_stats = structural_diagnostics.get("graph_stats", {})

    return {
        "verdict": verdict_str,
        "risk_severity": severity,
        "summary": summary,
        "primary_driver": driver,
        "pillar_contributions": {
            "pillar_1_probability": round(p1_prob, 4),
            "pillar_2_probability": round(p2_prob, 4),
            "probability_disagreement": round(disagreement, 4),
        },
        "claim_analysis": evidence_attribution,
        "structural_analysis": {
            "entity_features": structural_diagnostics.get("entity_features", {}),
            "numeric_features": structural_diagnostics.get("numeric_features", {}),
            "temporal_features": structural_diagnostics.get("temporal_features", {}),
            "graph_statistics": {
                "contradiction_pair_count": graph_stats.get("contradiction_pair_count", 0),
                "graph_density": round(float(graph_stats.get("graph_density", 0.0)), 4),
            },
        },
        "recommendation": "Flag for human verification and source cross-checking." if is_hallucinated else "Verified factual against reference knowledge base.",
    }
