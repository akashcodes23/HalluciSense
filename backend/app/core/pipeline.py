"""HalluciSense Unified Production Inference Pipeline.

Executes end-to-end real production inference pipeline:
Input text -> Claim Extraction -> Knowledge Retrieval -> Pillar 1 Grounding -> Pillar 2 Structure -> Hybrid Fusion -> Calibration -> Real Explanation.

Zero synthetic defaults. Zero retraining or modification of frozen research models.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import structlog

from app.core.inference.claim_extractor import extract_claims
from app.core.inference.explanation_engine import generate_rich_explanation
from app.core.inference.local_attribution import compute_local_attribution
from app.core.inference.pillar1_engine import Pillar1Engine
from app.core.inference.pillar2_engine import Pillar2Engine
from app.models.registry import registry
from evaluation.phase6m.dataset import compute_logit
from evaluation.phase6m.config import EPSILON

logger = structlog.get_logger(__name__)


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
        p1_feats, prob_p1, evidence_attribution, semantic_grounding = self.pillar1_engine.extract_features_and_predict(claims_struct)

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
                p1_feats[0], p1_feats[1], p1_feats[2], p1_feats[3], p1_feats[4],  # P1 features (5)
                p2_feats[0], p2_feats[1], p2_feats[2], p2_feats[3], p2_feats[4],  # P2 features (5)
                prob_p1, prob_p2, l1, l2,                                          # Probabilities & Logits (4)
                disagg_abs, p_mean, p_max, p_min, p_ratio,                        # Meta signals (5)
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

        # Task 8: Real Explanation Generation
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

        # Task 9: Local Counterfactual Attribution
        local_attribution_dict: dict = {}
        try:
            attr_result = compute_local_attribution(
                X_raw=X_input,  # unscaled raw vector used for inference
                scaler=self.scaler,
                clf=self.clf,
                threshold=self.threshold,
            )
            local_attribution_dict = attr_result.to_dict()
        except Exception as attr_exc:
            logger.warning("local_attribution_failed", error=str(attr_exc))
        # Task 10: Phase 40 Shadow Classifier Evaluation
        candidate_comparison = None
        if os.getenv("HALLUCISENSE_CLASSIFIER_SHADOW", "").lower() in ("true", "1", "yes"):
            try:
                cand_dir = Path(__file__).resolve().parent.parent.parent / "evaluation_results" / "phase40_candidate"
                candidate_model_path = cand_dir / "hybrid_meta_classifier_phase40_candidate.joblib"
                candidate_scaler_path = cand_dir / "preprocessing_phase40_candidate.joblib"
                if candidate_model_path.exists() and candidate_scaler_path.exists():
                    c_clf = joblib.load(candidate_model_path)
                    c_scaler = joblib.load(candidate_scaler_path)
                    X_cand_scaled = c_scaler.transform(X_input)
                    cand_prob = float(c_clf.predict_proba(X_cand_scaled)[0, 1])
                    cand_verdict = bool(cand_prob >= self.threshold)
                    candidate_comparison = {
                        "candidate_model_version": "phase40_candidate_v1",
                        "shadow_only": True,
                        "candidate_probability": round(cand_prob, 4),
                        "candidate_verdict": "hallucinated" if cand_verdict else "factual",
                        "production_probability": round(prob_hybrid, 4),
                        "production_verdict": "hallucinated" if is_hallucinated else "factual",
                        "decision_delta": round(cand_prob - prob_hybrid, 4),
                        "verdicts_match": bool(cand_verdict == is_hallucinated),
                    }
            except Exception as cand_exc:
                logger.warning("candidate_shadow_evaluation_failed", error=str(cand_exc))

        return {
            "is_hallucinated": is_hallucinated,
            "hallucination_probability": round(prob_hybrid, 4),
            "operating_threshold": self.threshold,
            "claim_count": len(claims_struct),
            "claims": [c["text"] for c in claims_struct],
            "explanation": explanation,
            "confidence_score": round(abs(prob_hybrid - 0.5) * 2.0, 4),
            "local_attribution": local_attribution_dict,
            "semantic_grounding": semantic_grounding,
            "candidate_comparison": candidate_comparison,
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
