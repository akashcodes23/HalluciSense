"""Phase 16 — Advanced Explainability & Topological Claim Graph Engine.

Provides:
- Local counterfactual feature attributions for 19-feature hybrid vectors
  (one-feature-at-a-time counterfactuals against training-median baseline;
   NOT SHAP — does not use Shapley value marginalisation)
- Topological claim-evidence support & contradiction graph construction
- Enriched interactive explanation JSON generator
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import math
import numpy as np
import structlog

from evaluation.phase6m.config import HYBRID_FEATURE_SCHEMA

logger = structlog.get_logger(__name__)


def compute_local_feature_attributions(
    X_raw: np.ndarray,
    scaler: Any,
    clf: Any,
    feature_names: List[str] = HYBRID_FEATURE_SCHEMA,
) -> Dict[str, float]:
    """Compute local counterfactual feature attributions for an input vector.

    Attribution method: one-feature-at-a-time counterfactual in RAW (unscaled) space.

    For each feature i:
        a_i = P(H | X) - P(H | X_i)
        where X_i is X with feature i replaced by its training-median value.

    The training-median baseline is sourced from RobustScaler.center_ (not zero).
    This is local counterfactual attribution, NOT SHAP. SHAP requires marginalisation
    over all feature coalitions via Shapley values.

    Args:
        X_raw:         Unscaled 1x19 raw feature array.
        scaler:        Frozen RobustScaler instance.
        clf:           Frozen HistGradientBoostingClassifier instance.
        feature_names: Canonical 19-feature schema list.

    Returns:
        Dict mapping feature name to local probability delta (a_i).
    """
    # Delegate to the canonical attribution engine to avoid duplication
    from app.core.inference.local_attribution import compute_local_attribution, get_training_medians

    training_medians = get_training_medians()
    X = np.atleast_2d(np.array(X_raw, dtype=np.float64))
    if X.shape != (1, 19):
        X = X.reshape(1, 19)

    X_original_scaled = scaler.transform(X)
    P_original = float(clf.predict_proba(X_original_scaled)[0, 1])

    attributions: Dict[str, float] = {}
    for i in range(min(len(feature_names), 19)):
        fname = feature_names[i]
        X_i = X.copy()
        X_i[0, i] = training_medians[i]  # replace with training median, in raw space
        X_i_scaled = scaler.transform(X_i)
        P_i = float(clf.predict_proba(X_i_scaled)[0, 1])
        attributions[fname] = round(P_original - P_i, 4)

    return attributions


def build_topological_claim_graph(
    claims: List[Dict[str, Any]],
    evidence_attribution: List[Dict[str, Any]],
    structural_diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    """Construct D3/Mermaid compatible claim-evidence topological graph.

    Nodes: Claims, Evidence Passages, Entities.
    Edges: Entails, Contradicts, Mentions.
    """
    nodes = []
    edges = []

    for i, c in enumerate(claims):
        cid = f"claim_{c.get('claim_id', i)}"
        ctext = c.get("text", f"Claim {i+1}")
        nodes.append({
            "id": cid,
            "label": ctext[:50] + ("..." if len(ctext) > 50 else ""),
            "type": "claim",
            "full_text": ctext,
        })

    # Evidence nodes & edges
    for item in evidence_attribution:
        cid = f"claim_{item.get('claim_id', 0)}"
        passages = item.get("evidence_passages", [])
        top_ent = item.get("top_entailment", 0.5)

        for j, ptext in enumerate(passages[:2]):  # Top 2 passages per claim
            eid = f"ev_{item.get('claim_id', 0)}_{j}"
            nodes.append({
                "id": eid,
                "label": ptext[:40] + "...",
                "type": "evidence",
                "full_text": ptext,
                "entailment_score": top_ent,
            })

            edge_type = "supports" if top_ent >= 0.20 else "unsupported"
            edges.append({
                "source": eid,
                "target": cid,
                "relation": edge_type,
                "weight": round(top_ent, 4),
            })

    # Contradiction edges from pairwise evaluated pairs
    evaluated_pairs = structural_diagnostics.get("evaluated_pairs", [])
    for pair in evaluated_pairs:
        if isinstance(pair, dict) and pair.get("is_contradictory", False):
            c1_idx = pair.get("claim_1_index", 0)
            c2_idx = pair.get("claim_2_index", 1)
            edges.append({
                "source": f"claim_{c1_idx}",
                "target": f"claim_{c2_idx}",
                "relation": "contradicts",
                "weight": round(float(pair.get("contradiction_score", 0.8)), 4),
            })

    return {"nodes": nodes, "edges": edges}


def generate_interactive_explanation(
    prob_hybrid: float,
    threshold: float,
    is_hallucinated: bool,
    claims: List[Dict[str, Any]],
    p1_prob: float,
    p2_prob: float,
    evidence_attribution: List[Dict[str, Any]],
    structural_diagnostics: Dict[str, Any],
    X_raw: Optional[np.ndarray] = None,
    scaler: Optional[Any] = None,
    clf: Optional[Any] = None,
) -> Dict[str, Any]:
    """Generate publication-grade interactive explanation JSON payload for frontend visualization."""
    severity = "HIGH" if prob_hybrid >= 0.75 else ("MODERATE" if prob_hybrid >= 0.54 else "LOW")
    verdict_str = "HALLUCINATED" if is_hallucinated else "FACTUAL"
    confidence = round(abs(prob_hybrid - 0.5) * 2.0, 4)

    # Feature importances
    feature_importance = {}
    if X_raw is not None and scaler is not None and clf is not None:
        try:
            feature_importance = compute_local_feature_attributions(X_raw, scaler, clf)
        except Exception as e:
            logger.warning("local_attribution_computation_exception", error=str(e))

    # Topological graph
    claim_graph = build_topological_claim_graph(claims, evidence_attribution, structural_diagnostics)

    # Extracted sources & evidence scores
    retrieved_sources = []
    evidence_scores = []
    for item in evidence_attribution:
        for p in item.get("evidence_passages", []):
            retrieved_sources.append({"claim_id": item.get("claim_id", 0), "passage": p})
        evidence_scores.append({
            "claim_id": item.get("claim_id", 0),
            "top_entailment": item.get("top_entailment", 0.5),
        })

    # Contradictions list
    contradictions = []
    graph_stats = structural_diagnostics.get("graph_stats", {})
    if graph_stats.get("contradiction_pair_count", 0) > 0:
        for pair in structural_diagnostics.get("evaluated_pairs", []):
            if isinstance(pair, dict) and pair.get("is_contradictory", False):
                contradictions.append({
                    "pair": [pair.get("claim_1_text", ""), pair.get("claim_2_text", "")],
                    "score": round(float(pair.get("contradiction_score", 0.8)), 4),
                })

    rec = (
        "Flag for human verification and source cross-checking."
        if is_hallucinated
        else "Verified factual against reference knowledge base."
    )

    return {
        "hallucination_score": round(prob_hybrid, 4),
        "confidence": confidence,
        "risk": severity,
        "verdict": verdict_str,
        "pillar_1": round(p1_prob, 4),
        "pillar_2": round(p2_prob, 4),
        "feature_importance": feature_importance,
        "claim_graph": claim_graph,
        "retrieved_sources": retrieved_sources,
        "evidence_scores": evidence_scores,
        "contradictions": contradictions,
        "recommendation": rec,
    }
