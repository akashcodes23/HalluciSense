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


def _relevance_to_nli(relevance: float) -> Tuple[float, float, float]:
    """Convert CrossEncoder relevance score to NLI-compatible (entailment, contradiction, neutral).

    Training data features were produced by a 3-class NLI model where:
        - entailment: P(text entails claim) — typically 0.0-0.3, median ~0.002
        - contradiction: P(text contradicts claim) — typically 0.0-0.8
        - neutral: P(neither) — often dominant class (~0.5-0.9)

    CrossEncoder relevance scores are semantically different (how relevant
    a passage is to a query, range 0.0-1.0). High relevance does NOT equal
    NLI entailment — even perfectly relevant passages show NLI entailment
    of only ~0.1-0.3 because NLI requires strict logical implication.

    Calibrated against Phase 6I development set (N=58,002):
        mean_entailment: median=0.0024, mean=0.1167, max=0.999
        mean_contradiction: median=0.0373, mean=0.3269
        min_support_margin: median=-0.0195

    Mapping:
        relevance 1.00 → ent ~0.30, con ~0.00 (strong evidence)
        relevance 0.75 → ent ~0.17, con ~0.15 (moderate evidence)
        relevance 0.50 → ent ~0.08, con ~0.35 (weak evidence)
        relevance 0.10 → ent ~0.00, con ~0.71 (irrelevant/contradictory)
    """
    relevance = max(0.0, min(1.0, float(relevance)))

    # Entailment: quadratic scaling capped at 0.30 to match training range
    # 0.999→0.30, 0.75→0.17, 0.50→0.08, 0.10→0.003, 0.0→0.0
    entailment = 0.3 * (relevance ** 2)

    # Contradiction: superlinear decay from irrelevant to relevant
    # 0.999→0.00, 0.75→0.15, 0.50→0.35, 0.10→0.71, 0.0→0.80
    contradiction = 0.8 * ((1.0 - relevance) ** 1.2)

    # Neutral absorbs remainder (dominant class, matching training data)
    neutral = max(0.0, 1.0 - entailment - contradiction)

    # Normalize to valid probability distribution
    total = entailment + contradiction + neutral
    if total > 0:
        entailment /= total
        contradiction /= total
        neutral /= total

    return entailment, contradiction, neutral


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
                    # Read the correct key from CrossEncoder evidence dicts
                    relevance = float(p.get("similarity_score", p.get("score", 0.0)))
                    text_snippet = p.get("snippet", p.get("text", str(p)))
                else:
                    relevance = float(getattr(p, "similarity_score", getattr(p, "score", 0.0)))
                    text_snippet = getattr(p, "snippet", getattr(p, "text", str(p)))

                # Convert CrossEncoder relevance to NLI-compatible features
                ent, con, neu = _relevance_to_nli(relevance)
                ent_scores.append(ent)
                con_scores.append(con)

            max_ent = max(ent_scores) if ent_scores else 0.0
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

