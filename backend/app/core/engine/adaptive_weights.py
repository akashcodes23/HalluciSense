"""Part 3 — Query-Dependent Adaptive Weight Learning Engine.

Computes input-dependent dynamic coefficients:
alpha(q), beta(q), gamma(q), delta(q)
where alpha(q) + beta(q) + gamma(q) + delta(q) = 1.0

Features considered:
- Query Complexity C(q)
- Claim Density D(c)
- Retrieval Quality Q(r)
- Model Uncertainty U(m)
"""

from __future__ import annotations

import re
from typing import Dict, Any, List, Tuple
import numpy as np


class AdaptiveWeightEstimator:
    """Estimates dynamic coefficients alpha(q), beta(q), gamma(q), delta(q)."""

    def estimate_weights(
        self,
        query: str,
        response_text: str,
        retrieval_similarity: float = 0.85,
        model_entropy: float = 0.25,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Estimate dynamic coefficients for a given query and response context."""
        # 1. Query Complexity C(q)
        q_words = re.findall(r"\w+", query.lower())
        q_len = len(q_words)
        has_reasoning = any(w in query.lower() for w in ["why", "how", "explain", "compare", "analyze"])
        query_complexity = min(1.0, (q_len / 30.0) * (1.5 if has_reasoning else 1.0))

        # 2. Claim Density D(c)
        r_words = re.findall(r"\w+", response_text.lower())
        r_len = max(1, len(r_words))
        sentences = [s.strip() for s in re.split(r"[.!?]+", response_text) if s.strip()]
        claim_density = min(1.0, len(sentences) / (r_len / 15.0))

        # 3. Features vector: [query_complexity, claim_density, retrieval_similarity, model_entropy]
        feat = np.array([query_complexity, claim_density, retrieval_similarity, model_entropy])

        # Projection matrices
        w_proj = np.array([
            [0.50, 0.20, 0.20, 0.10],  # alpha(q) Evidence Grounding
            [0.15, 0.45, 0.20, 0.20],  # beta(q)  Confidence Estimation
            [0.20, 0.20, 0.50, 0.10],  # gamma(q) Consistency Reasoning
            [0.15, 0.15, 0.10, 0.60],  # delta(q) Uncertainty Component
        ])

        raw_weights = np.dot(w_proj, feat)
        exp_w = np.exp(raw_weights - np.max(raw_weights))
        norm_w = exp_w / np.sum(exp_w)

        weights = {
            "alpha_q": round(float(norm_w[0]), 4),
            "beta_q": round(float(norm_w[1]), 4),
            "gamma_q": round(float(norm_w[2]), 4),
            "delta_q": round(float(norm_w[3]), 4),
        }

        diagnostics = {
            "query_complexity": round(query_complexity, 4),
            "claim_density": round(claim_density, 4),
            "retrieval_quality": round(retrieval_similarity, 4),
            "model_uncertainty": round(model_entropy, 4),
            "feature_importance": {
                "evidence_grounding_importance": round(weights["alpha_q"] * retrieval_similarity, 4),
                "confidence_gap_importance": round(weights["beta_q"] * model_entropy, 4),
                "consistency_importance": round(weights["gamma_q"] * query_complexity, 4),
            },
        }

        return weights, diagnostics
