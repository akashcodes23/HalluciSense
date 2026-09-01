"""Phase 37 — Explainable AI & Topological Claim Graph Engine.

Provides:
- faithful local leave-one-feature-at-baseline attribution for the frozen 19-feature hybrid model
- compatibility feature_importance output for existing consumers
- topological claim-evidence support / contradiction graph construction
- enriched interactive explanation JSON

Important: the local attribution is deliberately NOT labelled SHAP. Each
feature is replaced independently by the training median represented by
RobustScaler.center_, the exact frozen classifier is re-evaluated, and the
probability delta is reported. Nonlinear interaction residual is explicit.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import structlog

from evaluation.phase6m.config import HYBRID_FEATURE_SCHEMA

logger = structlog.get_logger(__name__)


def compute_local_feature_attributions(
    X_raw: np.ndarray,
    scaler: Any,
    clf: Any,
    feature_names: Optional[Sequence[str]] = None,
    threshold: Optional[float] = None,
    top_k: int = 7,
) -> Dict[str, Any]:
    """Compute exact local leave-one-feature-at-training-median effects."""
    names = list(feature_names or HYBRID_FEATURE_SCHEMA)
    raw = np.asarray(X_raw, dtype=np.float64).reshape(-1)

    if raw.size != len(names):
        raise ValueError(f"Expected {len(names)} features, received {raw.size}")
    if not np.all(np.isfinite(raw)):
        raise ValueError("Feature vector contains non-finite values")

    center = getattr(scaler, "center_", None)
    if center is None:
        baseline = np.zeros(raw.size, dtype=np.float64)
        baseline_method = "zero_raw_feature_baseline"
    else:
        baseline = np.asarray(center, dtype=np.float64).reshape(-1)
        if baseline.size != raw.size or not np.all(np.isfinite(baseline)):
            raise ValueError("RobustScaler center_ is missing or incompatible with the feature vector")
        baseline_method = "training_median_from_RobustScaler_center"

    def predict(row: np.ndarray) -> float:
        scaled = scaler.transform(row.reshape(1, -1))
        return float(clf.predict_proba(scaled)[0, 1])

    observed_probability = predict(raw)
    baseline_probability = predict(baseline)
    attributions: List[Dict[str, Any]] = []

    for index, feature_name in enumerate(names):
        counterfactual = raw.copy()
        counterfactual[index] = baseline[index]
        counterfactual_probability = predict(counterfactual)
        delta = observed_probability - counterfactual_probability
        attributions.append({
            "index": index,
            "feature": feature_name,
            "value": round(float(raw[index]), 8),
            "baseline_value": round(float(baseline[index]), 8),
            "counterfactual_probability": round(float(counterfactual_probability), 8),
            "delta": round(float(delta), 8),
            "direction": (
                "increases_hallucination" if delta > 1e-9 else
                "decreases_hallucination" if delta < -1e-9 else
                "neutral"
            ),
        })

    attributions.sort(key=lambda item: abs(float(item["delta"])), reverse=True)
    abs_total = sum(abs(float(item["delta"])) for item in attributions)
    for item in attributions:
        item["relative_strength"] = round(abs(float(item["delta"])) / abs_total, 6) if abs_total else 0.0

    interaction_gap = observed_probability - (
        baseline_probability + sum(float(item["delta"]) for item in attributions)
    )

    return {
        "available": True,
        "method": "LOCAL_LEAVE_ONE_FEATURE_AT_BASELINE",
        "methodology": (
            "Each feature is replaced independently by its training median "
            "(RobustScaler.center_) and the exact frozen classifier is re-evaluated. "
            "Delta = P(observed) - P(counterfactual)."
        ),
        "baseline_method": baseline_method,
        "baseline_probability": round(float(baseline_probability), 8),
        "observed_probability": round(float(observed_probability), 8),
        "decision_threshold": round(float(threshold), 8) if threshold is not None else None,
        "decision_margin": round(float(observed_probability - threshold), 8) if threshold is not None else None,
        "interaction_gap": round(float(interaction_gap), 8),
        "non_additivity_note": (
            "These are local perturbation effects, not SHAP values or global feature importance. "
            "The interaction_gap captures nonlinear interaction effects."
        ),
        "feature_count": len(names),
        "features": attributions,
        "top_positive_drivers": [a for a in attributions if a["delta"] > 0][:top_k],
        "top_negative_drivers": [a for a in attributions if a["delta"] < 0][:top_k],
    }


def compute_shap_feature_attributions(
    X_raw: np.ndarray,
    scaler: Any,
    clf: Any,
    feature_names: List[str] = HYBRID_FEATURE_SCHEMA,
) -> Dict[str, float]:
    """Backward-compatible mapping of feature name -> local probability delta.

    Existing callers may still use this function name. New code should use
    compute_local_feature_attributions() and its explicit methodology.
    """
    result = compute_local_feature_attributions(X_raw, scaler, clf, feature_names)
    return {
        item["feature"]: round(float(item["delta"]), 4)
        for item in result["features"]
    }


def build_topological_claim_graph(
    claims: List[Dict[str, Any]],
    evidence_attribution: List[Dict[str, Any]],
    structural_diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    """Construct D3/Mermaid compatible claim-evidence topological graph."""
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

    for item in evidence_attribution:
        cid = f"claim_{item.get('claim_id', 0)}"
        passages = item.get("evidence_passages", [])
        top_ent = item.get("top_entailment", 0.5)

        for j, ptext in enumerate(passages[:2]):
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
    """Generate publication-grade interactive explanation JSON payload."""
    severity = "HIGH" if prob_hybrid >= 0.75 else ("MODERATE" if prob_hybrid >= 0.54 else "LOW")
    verdict_str = "HALLUCINATED" if is_hallucinated else "FACTUAL"
    confidence = round(abs(prob_hybrid - 0.5) * 2.0, 4)

    local_attribution: Dict[str, Any] = {
        "available": False,
        "method": "UNAVAILABLE",
        "reason": "Hybrid feature vector or frozen model was not supplied.",
    }
    if X_raw is not None and scaler is not None and clf is not None:
        try:
            local_attribution = compute_local_feature_attributions(
                X_raw, scaler, clf, HYBRID_FEATURE_SCHEMA, threshold=threshold
            )
        except Exception as exc:
            logger.warning("local_explainability_failed", error=str(exc))
            local_attribution["reason"] = str(exc)

    feature_importance = {
        item["feature"]: round(float(item["delta"]), 4)
        for item in local_attribution.get("features", [])
    }
    claim_graph = build_topological_claim_graph(claims, evidence_attribution, structural_diagnostics)

    retrieved_sources = []
    evidence_scores = []
    for item in evidence_attribution:
        for p in item.get("evidence_passages", []):
            retrieved_sources.append({"claim_id": item.get("claim_id", 0), "passage": p})
        evidence_scores.append({
            "claim_id": item.get("claim_id", 0),
            "top_entailment": item.get("top_entailment", 0.5),
        })

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
        "local_feature_attribution": local_attribution,
        "claim_graph": claim_graph,
        "retrieved_sources": retrieved_sources,
        "evidence_scores": evidence_scores,
        "contradictions": contradictions,
        "recommendation": rec,
    }
