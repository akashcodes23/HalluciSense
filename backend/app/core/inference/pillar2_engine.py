"""Production Pillar 2 (Structural Consistency) Feature Extraction & Base Model Inference Engine."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import structlog

from app.models.registry import registry
from evaluation.phase6l.claim_pairs import generate_unordered_claim_pairs
from evaluation.phase6l.pairwise_nli import evaluate_bidirectional_nli_and_similarity
from evaluation.phase6l.entity_extractor import extract_entity_consistency_features
from evaluation.phase6l.numeric_extractor import extract_numeric_consistency_features
from evaluation.phase6l.temporal_extractor import extract_temporal_consistency_features
from evaluation.phase6l.graph_builder import extract_graph_topological_features
from evaluation.phase6l.feature_extractor import extract_structural_features_for_response

logger = structlog.get_logger(__name__)

# The 5 locked Pillar-2 features (Phase 6L / 6M)
PILLAR2_LOCKED_FEATURES = [
    "max_pairwise_contradiction",
    "mean_pairwise_contradiction",
    "max_pairwise_similarity",
    "fraction_contradictory_pairs",
    "num_claims",
]


class Pillar2Engine:
    """Production Engine for Pillar-2 Structural Consistency Analysis."""

    def __init__(self):
        self.scaler, self.clf, self.meta = registry.load_pillar2_model()

    def extract_features_and_predict(
        self,
        claims: List[Dict[str, Any]],
    ) -> Tuple[List[float], float, Dict[str, Any]]:
        """Extract Pillar-2 features, build contradiction graph, and predict base P2 probability.

        Args:
            claims: List of claim dicts containing claim_id and text.

        Returns:
            Tuple of:
                p2_features (5 floats)
                p2_prob (float)
                structural_diagnostics (dict containing entity/numeric/temporal conflicts, graph stats)
        """
        claim_texts = [c.get("text", "").strip() for c in claims if c.get("text", "").strip()]
        n_claims = len(claim_texts)

        if n_claims < 2:
            raw_features = [0.0, 0.0, 0.0, 0.0, float(n_claims)]
            X_raw = np.array(raw_features, dtype=np.float64).reshape(1, -1)
            X_scaled = self.scaler.transform(X_raw)
            prob_p2 = float(self.clf.predict_proba(X_scaled)[0, 1])

            diagnostics = {
                "entity_conflicts": [],
                "numeric_conflicts": [],
                "temporal_conflicts": [],
                "graph_stats": {"contradiction_pair_count": 0, "graph_density": 0.0},
            }
            return raw_features, prob_p2, diagnostics

        # 1. Construct response record and generate claim pairs
        response_rec = {"example_id": "inference_response", "claim_details": [{"claim": c} for c in claim_texts]}
        # Cap max claims to 15 to prevent quadratic NLI pair explosion on 40+ sentence prompts
        claim_details = response_rec.get("claim_details", [])
        if len(claim_details) > 15:
            response_rec = {
                "example_id": response_rec.get("example_id", "sample"),
                "claim_details": claim_details[:15],
            }

        pairs = generate_unordered_claim_pairs(response_rec)

        # 2. Pairwise NLI and similarity
        nli_res = evaluate_bidirectional_nli_and_similarity(pairs)
        if isinstance(nli_res, dict) and "evaluated_pairs" in nli_res:
            evaluated_pairs = nli_res["evaluated_pairs"]
        elif isinstance(nli_res, list):
            evaluated_pairs = nli_res
        else:
            evaluated_pairs = []

        # 3. Structural feature extraction (24 features)
        feat_dict = extract_structural_features_for_response(response_rec, evaluated_pairs)
        full_24_features = feat_dict.get("features", {})

        # Extract 5 locked Pillar-2 features
        raw_features = [float(full_24_features.get(fn, 0.0)) for fn in PILLAR2_LOCKED_FEATURES]

        # 4. Detailed diagnostic extractions
        entity_feats = extract_entity_consistency_features(claim_texts)
        numeric_feats = extract_numeric_consistency_features(claim_texts)
        temporal_feats = extract_temporal_consistency_features(claim_texts)
        graph_feats = extract_graph_topological_features(n_claims, evaluated_pairs)

        # Predict P2 probability using Pillar 2 frozen model
        X_raw = np.array(raw_features, dtype=np.float64).reshape(1, -1)
        X_scaled = self.scaler.transform(X_raw)
        prob_p2 = float(self.clf.predict_proba(X_scaled)[0, 1])

        diagnostics = {
            "entity_features": entity_feats,
            "numeric_features": numeric_feats,
            "temporal_features": temporal_feats,
            "graph_stats": {
                "contradiction_pair_count": graph_feats.get("contradiction_pair_count", 0),
                "graph_density": graph_feats.get("contradiction_graph_density", 0.0),
            },
            "evaluated_pairs": evaluated_pairs,
        }

        return raw_features, prob_p2, diagnostics
