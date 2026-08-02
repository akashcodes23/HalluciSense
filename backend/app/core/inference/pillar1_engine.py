"""Production Pillar 1 (Evidence Consistency) Feature Extraction & Base Model Inference Engine."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import structlog

from app.models.registry import registry
from app.modules.knowledge.retriever import HybridRetriever

logger = structlog.get_logger(__name__)

# The 5 locked Pillar-1 features (Phase 6K / 6M)
PILLAR1_LOCKED_FEATURES = [
    "mean_entailment",
    "max_entailment",
    "mean_contradiction",
    "min_support_margin",
    "num_claims",
]


class Pillar1Engine:
    """Production Engine for Pillar-1 Evidence Consistency Analysis."""

    def __init__(self):
        self.retriever = HybridRetriever()
        self.scaler, self.clf, self.meta = registry.load_pillar1_model()

    def extract_features_and_predict(
        self,
        claims: List[Dict[str, Any]],
    ) -> Tuple[List[float], float, List[Dict[str, Any]]]:
        """Extract Pillar-1 features, retrieve evidence, and predict base P1 probability.

        Args:
            claims: List of claim dicts containing claim_id and text.

        Returns:
            Tuple of:
                p1_features (5 floats)
                p1_prob (float)
                retrieved_evidence (list of structured evidence dicts)
        """
        n_claims = float(len(claims))
        if n_claims == 0:
            raw_features = [0.0, 0.0, 0.0, 0.0, 0.0]
            return raw_features, 0.5, []

        claim_entailments = []
        claim_contradictions = []
        claim_margins = []
        evidence_attribution = []

        for c in claims:
            c_text = c.get("text", "")
            passages = self.retriever.get_evidence(c_text)
            
            ent_scores = []
            con_scores = []

            for p in passages:
                if isinstance(p, dict):
                    score = float(p.get("score", 0.75))
                    text_snippet = p.get("snippet", p.get("text", str(p)))
                else:
                    score = float(getattr(p, "score", 0.75))
                    text_snippet = getattr(p, "snippet", getattr(p, "text", str(p)))
                ent_scores.append(score)
                con_scores.append(1.0 - score)

            max_ent = max(ent_scores) if ent_scores else 0.5
            mean_con = float(np.mean(con_scores)) if con_scores else 0.5
            margin = max_ent - mean_con

            claim_entailments.append(max_ent)
            claim_contradictions.append(mean_con)
            claim_margins.append(margin)

            evidence_attribution.append({
                "claim_id": c.get("claim_id", 0),
                "claim_text": c_text,
                "evidence_passages": [p.get("snippet", str(p)) if isinstance(p, dict) else str(p) for p in passages],
                "top_entailment": round(max_ent, 4),
            })

        f_mean_entailment = float(np.mean(claim_entailments))
        f_max_entailment = float(max(claim_entailments))
        f_mean_contradiction = float(np.mean(claim_contradictions))
        f_min_support_margin = float(min(claim_margins))
        f_num_claims = n_claims

        raw_features = [
            f_mean_entailment,
            f_max_entailment,
            f_mean_contradiction,
            f_min_support_margin,
            f_num_claims,
        ]

        X_raw = np.array(raw_features, dtype=np.float64).reshape(1, -1)
        X_scaled = self.scaler.transform(X_raw)
        prob_p1 = float(self.clf.predict_proba(X_scaled)[0, 1])

        return raw_features, prob_p1, evidence_attribution
