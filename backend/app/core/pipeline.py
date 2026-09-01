"""HalluciSense Unified Production Inference Pipeline.

Executes end-to-end real production inference pipeline:
Input text -> Claim Extraction -> Knowledge Retrieval -> Pillar 1 Grounding -> Pillar 2 Structure -> Hybrid Fusion -> Calibration -> Real Explanation.

Phase 37 adds a faithful local explanation layer around the frozen hybrid
classifier. The research model and preprocessing artifacts are not changed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import structlog

from app.core.inference.claim_extractor import extract_claims
from app.core.inference.explanation_engine import generate_rich_explanation
from app.core.inference.pillar1_engine import Pillar1Engine
from app.core.inference.pillar2_engine import Pillar2Engine
from app.models.registry import registry
from evaluation.phase6m.dataset import compute_logit
from evaluation.phase6m.config import EPSILON

logger = structlog.get_logger(__name__)


HYBRID_FEATURE_NAMES: List[str] = [
    "p1_mean_entailment",
    "p1_max_entailment",
    "p1_mean_contradiction",
    "p1_min_support_margin",
    "p1_num_claims",
    "p2_max_pairwise_contradiction",
    "p2_mean_pairwise_contradiction",
    "p2_max_pairwise_similarity",
    "p2_fraction_contradictory_pairs",
    "p2_num_claims",
    "prob_p1",
    "prob_p2",
    "logit_p1",
    "logit_p2",
    "prob_disagreement_abs",
    "prob_mean",
    "prob_max",
    "prob_min",
    "prob_ratio",
]


def _predict_probability(clf: Any, scaler: Any, raw_row: np.ndarray) -> float:
    """Run the exact frozen preprocessing + classifier path."""
    scaled = scaler.transform(raw_row.reshape(1, -1))
    return float(clf.predict_proba(scaled)[0, 1])


def _compute_local_feature_explanation(
    clf: Any,
    scaler: Any,
    raw_features: Sequence[float],
    observed_probability: float,
    threshold: float,
    top_k: int = 7,
) -> Dict[str, Any]:
    """Compute local leave-one-feature-at-baseline attribution.

    Each feature is replaced independently by the training median represented
    by RobustScaler.center_. The exact frozen classifier is then evaluated.
    Positive delta means the observed feature increases hallucination
    probability relative to its training-median counterfactual.

    This is intentionally *not* called SHAP or feature_importances_. Because
    HistGradientBoosting is nonlinear, the deltas need not be additive; the
    residual is exposed as ``interaction_gap``.
    """
    raw = np.asarray(list(raw_features), dtype=np.float64).reshape(-1)
    if raw.size != len(HYBRID_FEATURE_NAMES):
        return {
            "available": False,
            "method": "UNAVAILABLE",
            "reason": f"Hybrid attribution requires 19 features; received {raw.size}.",
        }
    if not np.all(np.isfinite(raw)):
        return {
            "available": False,
            "method": "UNAVAILABLE",
            "reason": "Hybrid feature vector contains non-finite values.",
        }

    center = getattr(scaler, "center_", None)
    if center is None:
        baseline = np.zeros(raw.size, dtype=np.float64)
        baseline_method = "zero_raw_feature_baseline"
    else:
        baseline = np.asarray(center, dtype=np.float64).reshape(-1)
        if baseline.size != raw.size or not np.all(np.isfinite(baseline)):
            return {
                "available": False,
                "method": "UNAVAILABLE",
                "reason": "Preprocessor does not expose a compatible finite RobustScaler center_.",
            }
        baseline_method = "training_median_from_RobustScaler_center"

    baseline_probability = _predict_probability(clf, scaler, baseline)
    attributions: List[Dict[str, Any]] = []

    for index, feature_name in enumerate(HYBRID_FEATURE_NAMES):
        counterfactual = raw.copy()
        counterfactual[index] = baseline[index]
        counterfactual_probability = _predict_probability(clf, scaler, counterfactual)
        delta = float(observed_probability - counterfactual_probability)
        attributions.append({
            "index": index,
            "feature": feature_name,
            "value": round(float(raw[index]), 8),
            "baseline_value": round(float(baseline[index]), 8),
            "counterfactual_probability": round(float(counterfactual_probability), 8),
            "delta": round(delta, 8),
            "direction": (
                "increases_hallucination" if delta > 1e-9 else
                "decreases_hallucination" if delta < -1e-9 else
                "neutral"
            ),
        })

    attributions.sort(key=lambda item: abs(float(item["delta"])), reverse=True)
    abs_total = sum(abs(float(item["delta"])) for item in attributions)
    for item in attributions:
        item["relative_strength"] = round(
            abs(float(item["delta"])) / abs_total, 6
        ) if abs_total > 0 else 0.0

    positive = [item for item in attributions if item["delta"] > 0][:top_k]
    negative = [item for item in attributions if item["delta"] < 0][:top_k]
    interaction_gap = float(
        observed_probability - (
            baseline_probability + sum(float(item["delta"]) for item in attributions)
        )
    )

    return {
        "available": True,
        "method": "LOCAL_LEAVE_ONE_FEATURE_AT_BASELINE",
        "methodology": (
            "Each feature is replaced independently by its training median "
            "(RobustScaler.center_) and the exact frozen HistGradientBoostingClassifier "
            "is re-evaluated. Delta = P(observed) - P(counterfactual)."
        ),
        "baseline_method": baseline_method,
        "baseline_probability": round(float(baseline_probability), 8),
        "observed_probability": round(float(observed_probability), 8),
        "decision_threshold": round(float(threshold), 8),
        "decision_margin": round(float(observed_probability - threshold), 8),
        "interaction_gap": round(interaction_gap, 8),
        "non_additivity_note": (
            "These are local perturbation effects, not SHAP values or global feature importance. "
            "The interaction_gap explicitly captures nonlinear interaction effects."
        ),
        "feature_count": len(HYBRID_FEATURE_NAMES),
        "features": attributions,
        "top_positive_drivers": positive,
        "top_negative_drivers": negative,
    }


class HalluciSensePipeline:
    """Production Real Inference Pipeline for HalluciSense."""

    def __init__(self):
        logger.info("initializing_hallucisense_production_pipeline")
        self.scaler, self.clf, self.metadata = registry.load_hybrid_model()
        self.threshold = float(self.metadata.get("protocol", {}).get("decision_threshold", 0.54))

        self.pillar1_engine = Pillar1Engine()
        self.pillar2_engine = Pillar2Engine()

    def predict(
        self,
        response_text: str,
        claims: Optional[List[str]] = None,
        evidence_passages: Optional[List[str]] = None,
        feature_vector: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Predict hallucination probability executing full real research pipeline."""
        if not response_text or not response_text.strip():
            response_text = "Empty prompt."

        # Task 1: Claim Extraction
        if claims is not None and len(claims) > 0:
            claims_struct = [{"claim_id": i, "text": c.strip()} for i, c in enumerate(claims) if c.strip()]
        else:
            claims_struct = extract_claims(response_text)

        if not claims_struct:
            claims_struct = [{"claim_id": 0, "text": response_text}]

        # Task 3 & 5: Pillar 1 Execution
        p1_feats, prob_p1, evidence_attribution = self.pillar1_engine.extract_features_and_predict(claims_struct)

        # Task 4 & 5: Pillar 2 Execution
        p2_feats, prob_p2, structural_diagnostics = self.pillar2_engine.extract_features_and_predict(claims_struct)

        # Task 6: 19-Feature Hybrid Assembly (SET_A_FULL_HYBRID)
        if feature_vector is not None and len(feature_vector) == 19:
            X_raw = np.array(feature_vector, dtype=np.float64).reshape(1, -1)
        else:
            l1 = compute_logit(prob_p1)
            l2 = compute_logit(prob_p2)
            disagg_abs = float(abs(prob_p1 - prob_p2))
            p_mean = float((prob_p1 + prob_p2) / 2.0)
            p_max = float(max(prob_p1, prob_p2))
            p_min = float(min(prob_p1, prob_p2))
            p_ratio = float((prob_p1 + EPSILON) / (prob_p2 + EPSILON))

            hybrid_vector = [
                p1_feats[0], p1_feats[1], p1_feats[2], p1_feats[3], p1_feats[4],
                p2_feats[0], p2_feats[1], p2_feats[2], p2_feats[3], p2_feats[4],
                prob_p1, prob_p2, l1, l2,
                disagg_abs, p_mean, p_max, p_min, p_ratio,
            ]
            X_raw = np.array(hybrid_vector, dtype=np.float64).reshape(1, -1)

        # Task 7: Hybrid Fusion Inference
        expected_features = getattr(self.scaler, "n_features_in_", 19)
        if expected_features == 5:
            X_input = np.array(p1_feats, dtype=np.float64).reshape(1, -1)
        else:
            X_input = X_raw

        X_scaled = self.scaler.transform(X_input)
        prob_hybrid = float(self.clf.predict_proba(X_scaled)[0, 1])
        is_hallucinated = bool(prob_hybrid >= self.threshold)

        # Task 8: Faithful local explanation of the frozen hybrid decision.
        if expected_features == 19 and X_raw.shape[1] == 19:
            explainability = _compute_local_feature_explanation(
                clf=self.clf,
                scaler=self.scaler,
                raw_features=X_raw[0],
                observed_probability=prob_hybrid,
                threshold=self.threshold,
            )
        else:
            explainability = {
                "available": False,
                "method": "UNAVAILABLE",
                "reason": "Active production model is not the 19-feature hybrid classifier.",
            }

        # Task 9: Real Explanation Generation
        explanation = generate_rich_explanation(
            prob_hybrid=prob_hybrid,
            threshold=self.threshold,
            is_hallucinated=is_hallucinated,
            claims=claims_struct,
            p1_prob=prob_p1,
            p2_prob=prob_p2,
            evidence_attribution=evidence_attribution,
            structural_diagnostics=structural_diagnostics,
        )

        # Attach explainability to the existing explanation object without
        # changing the frozen model's decision path.
        explanation["model_explainability"] = explainability
        explanation["decision_rule"] = {
            "threshold": self.threshold,
            "comparison": f"P(H) {'>=' if is_hallucinated else '<'} τ*",
            "margin": round(prob_hybrid - self.threshold, 8),
        }

        return {
            "is_hallucinated": is_hallucinated,
            "hallucination_probability": round(prob_hybrid, 4),
            "operating_threshold": self.threshold,
            "claim_count": len(claims_struct),
            "claims": [c["text"] for c in claims_struct],
            "explanation": explanation,
            "explainability": explainability,
            "feature_vector": [round(float(v), 8) for v in X_raw[0].tolist()] if X_raw.shape[1] == 19 else None,
            "feature_schema": HYBRID_FEATURE_NAMES if X_raw.shape[1] == 19 else None,
            "confidence_score": round(abs(prob_hybrid - 0.5) * 2.0, 4),
        }

    def generate_explanation(self, prob: float, claims: List[str], is_hallucinated: bool) -> Dict[str, Any]:
        """Backward-compatible helper method for explanation generation."""
        claims_struct = [{"claim_id": i, "text": c} for i, c in enumerate(claims)]
        return generate_rich_explanation(
            prob_hybrid=prob,
            threshold=self.threshold,
            is_hallucinated=is_hallucinated,
            claims=claims_struct,
            p1_prob=prob,
            p2_prob=prob,
            evidence_attribution=[],
            structural_diagnostics={},
        )


# Lazy Singleton pipeline instance
_pipeline_instance: Optional[HalluciSensePipeline] = None


def get_hallucisense_pipeline() -> HalluciSensePipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = HalluciSensePipeline()
    return _pipeline_instance


class _LazyPipelineProxy:
    def __getattr__(self, name):
        return getattr(get_hallucisense_pipeline(), name)


pipeline = _LazyPipelineProxy()
