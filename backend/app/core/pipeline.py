"""HalluciSense Unified Production Inference Pipeline.

Executes end-to-end real production inference pipeline:
Input text -> Claim Extraction -> Knowledge Retrieval -> Pillar 1 Grounding -> Pillar 2 Structure -> Hybrid Fusion -> Calibration -> Real Explanation.

Zero synthetic defaults. Zero retraining or modification of frozen research models.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

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
                p1_feats[0], p1_feats[1], p1_feats[2], p1_feats[3], p1_feats[4],  # P1 features (5)
                p2_feats[0], p2_feats[1], p2_feats[2], p2_feats[3], p2_feats[4],  # P2 features (5)
                prob_p1, prob_p2, l1, l2,                                          # Probabilities & Logits (4)
                disagg_abs, p_mean, p_max, p_min, p_ratio,                        # Meta signals (5)
            ]
            X_raw = np.array(hybrid_vector, dtype=np.float64).reshape(1, -1)

        # Task 7: Hybrid Fusion Inference
        X_scaled = self.scaler.transform(X_raw)
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

        return {
            "is_hallucinated": is_hallucinated,
            "hallucination_probability": round(prob_hybrid, 4),
            "operating_threshold": self.threshold,
            "claim_count": len(claims_struct),
            "claims": [c["text"] for c in claims_struct],
            "explanation": explanation,
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


# Singleton pipeline instance
pipeline = HalluciSensePipeline()
