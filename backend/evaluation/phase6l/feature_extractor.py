"""Phase 6L.1B — Structural Feature Extractor & Pipeline Orchestrator.

Orchestrates full 24-feature extraction across 8 structural signal families (A-H),
handles degenerate responses (n=0, 1), enforces exact feature schema contracts,
and manages persistent content-addressable caching.

Strict Data Firewall Rule:
    * Label-free: No rule, feature, or threshold depends on ground truth target y.
    * Accesses DEV partition ONLY. Validation partition (N=12,483) is strictly sealed.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

import structlog

from evaluation.phase6j.utils import _serializable
from evaluation.phase6l.config import (
    STRUCTURAL_FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    TAU_CONTRADICTION,
    TAU_SUPPORT,
    TAU_SIMILARITY_DUPLICATE,
    PHASE6L_CACHE_DIR,
)
from evaluation.phase6l.claim_pairs import generate_unordered_claim_pairs
from evaluation.phase6l.pairwise_nli import evaluate_bidirectional_nli_and_similarity
from evaluation.phase6l.entity_extractor import extract_entity_consistency_features
from evaluation.phase6l.numeric_extractor import extract_numeric_consistency_features
from evaluation.phase6l.temporal_extractor import extract_temporal_consistency_features
from evaluation.phase6l.graph_builder import extract_graph_topological_features

logger = structlog.get_logger(__name__)


def extract_structural_features_for_response(
    response_record: Dict[str, Any],
    evaluated_pairs_for_response: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Extract exact 24-feature structural vector for a single response record.

    Args:
        response_record: Dict containing 'example_id', 'claim_details', 'ground_truth'.
        evaluated_pairs_for_response: Evaluated claim pair dicts for this specific response.

    Returns:
        Dict containing response_id, pair_count, features (24 floats), and diagnostics.
    """
    example_id = response_record.get("example_id", "")
    claim_details = response_record.get("claim_details", [])
    claims_text = [str(cd.get("claim", "")).strip() for cd in claim_details if cd.get("claim")]
    n_claims = len(claims_text)

    # Family H: Response Controls
    # 23. num_claims
    f_num_claims = float(n_claims)

    # 24. claim_length_variance (whitespace-token count variance across claims)
    if n_claims > 1:
        token_lengths = [len(c.split()) for c in claims_text]
        f_claim_length_var = float(np.var(token_lengths, ddof=0))  # Population variance
    else:
        f_claim_length_var = 0.0

    # Family D: Entity Consistency
    entity_feat = extract_entity_consistency_features(claims_text)

    # Family E: Numerical Consistency
    numeric_feat = extract_numeric_consistency_features(claims_text)

    # Family F: Temporal Consistency
    temporal_feat = extract_temporal_consistency_features(claims_text)

    # Family G: Claim Graph Topology
    graph_feat = extract_graph_topological_features(n_claims, evaluated_pairs_for_response, TAU_CONTRADICTION)

    # Handle pairwise features (Families A, B, C) for n < 2
    if n_claims < 2 or not evaluated_pairs_for_response:
        pairwise_feat = {
            "mean_pairwise_contradiction": 0.0,
            "max_pairwise_contradiction": 0.0,
            "p95_pairwise_contradiction": 0.0,
            "fraction_contradictory_pairs": 0.0,
            "contradiction_pair_count": 0.0,
            "mean_pairwise_entailment": 0.0,
            "max_pairwise_entailment": 0.0,
            "fraction_mutually_supportive_pairs": 0.0,
            "mean_pairwise_similarity": 0.0,
            "max_pairwise_similarity": 0.0,
            "near_duplicate_claim_fraction": 0.0,
        }
        pair_count = 0
    else:
        pair_count = len(evaluated_pairs_for_response)
        c_max_arr = np.array([p["c_max"] for p in evaluated_pairs_for_response], dtype=np.float64)

        # Symmetric entailment: E_mean = (E_ij + E_ji) / 2
        e_mean_arr = np.array([(p["e_ij"] + p["e_ji"]) / 2.0 for p in evaluated_pairs_for_response], dtype=np.float64)
        e_max_arr = np.array([max(p["e_ij"], p["e_ji"]) for p in evaluated_pairs_for_response], dtype=np.float64)

        sim_arr = np.array([p["embedding_cosine_similarity"] for p in evaluated_pairs_for_response], dtype=np.float64)

        # Family A Features
        f_mean_c = float(np.mean(c_max_arr))
        f_max_c = float(np.max(c_max_arr))
        f_p95_c = float(np.percentile(c_max_arr, 95))
        f_cnt_c = float(np.sum(c_max_arr >= TAU_CONTRADICTION))
        f_frac_c = float(f_cnt_c / pair_count)

        # Family B Features
        f_mean_e = float(np.mean(e_mean_arr))
        f_max_e = float(np.max(e_max_arr))
        # Mutual support: BOTH E_ij >= tau_E AND E_ji >= tau_E
        mut_supp_count = sum(1 for p in evaluated_pairs_for_response if p["e_ij"] >= TAU_SUPPORT and p["e_ji"] >= TAU_SUPPORT)
        f_frac_supp = float(mut_supp_count / pair_count)

        # Family C Features
        f_mean_sim = float(np.mean(sim_arr))
        f_max_sim = float(np.max(sim_arr))
        f_dup_frac = float(np.sum(sim_arr >= TAU_SIMILARITY_DUPLICATE) / pair_count)

        pairwise_feat = {
            "mean_pairwise_contradiction": f_mean_c,
            "max_pairwise_contradiction": f_max_c,
            "p95_pairwise_contradiction": f_p95_c,
            "fraction_contradictory_pairs": f_frac_c,
            "contradiction_pair_count": f_cnt_c,
            "mean_pairwise_entailment": f_mean_e,
            "max_pairwise_entailment": f_max_e,
            "fraction_mutually_supportive_pairs": f_frac_supp,
            "mean_pairwise_similarity": f_mean_sim,
            "max_pairwise_similarity": f_max_sim,
            "near_duplicate_claim_fraction": f_dup_frac,
        }

    # Combine all feature dicts
    all_features_dict = {}
    all_features_dict.update(pairwise_feat)
    all_features_dict.update({
        "entity_conflict_count": entity_feat["entity_conflict_count"],
        "entity_conflict_ratio": entity_feat["entity_conflict_ratio"],
        "entity_attribute_disagreement_score": entity_feat["entity_attribute_disagreement_score"],
        "numeric_conflict_count": numeric_feat["numeric_conflict_count"],
        "numeric_conflict_ratio": numeric_feat["numeric_conflict_ratio"],
        "max_numeric_disagreement": numeric_feat["max_numeric_disagreement"],
        "temporal_conflict_count": temporal_feat["temporal_conflict_count"],
        "timeline_order_violation_score": temporal_feat["timeline_order_violation_score"],
        "contradiction_graph_density": graph_feat["contradiction_graph_density"],
        "max_contradiction_degree": graph_feat["max_contradiction_degree"],
        "largest_contradictory_component_ratio": graph_feat["largest_contradictory_component_ratio"],
        "num_claims": f_num_claims,
        "claim_length_variance": f_claim_length_var,
    })

    # Validate exact 24-feature schema contract & ordering
    features_ordered = {}
    for col_name in STRUCTURAL_FEATURE_COLUMNS:
        if col_name not in all_features_dict:
            raise KeyError(f"Feature schema error: Missing column '{col_name}'")
        val = all_features_dict[col_name]
        if not np.isfinite(val):
            raise ValueError(f"Feature contract violation: Non-finite value '{val}' for feature '{col_name}'")
        features_ordered[col_name] = float(val)

    diagnostics = {
        "entity_explainability": entity_feat.get("explainability_records", []),
        "numeric_explainability": numeric_feat.get("explainability_records", []),
        "temporal_explainability": temporal_feat.get("explainability_records", []),
        "total_graph_edges": graph_feat.get("total_graph_edges", 0),
    }

    return {
        "example_id": example_id,
        "dataset_partition": "development",
        "num_claims": n_claims,
        "pair_count": pair_count,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": features_ordered,
        "diagnostics": diagnostics,
    }


def extract_structural_features_for_subset(
    subset_records: List[Dict[str, Any]],
    cache_dir: Path = PHASE6L_CACHE_DIR,
) -> Dict[str, Any]:
    """Extract structural feature vectors for 1,000-response subset with persistent joblib caching.

    Args:
        subset_records: List of 1,000 DEV response records.
        cache_dir: Cache output directory.

    Returns:
        Dict containing extracted records, feature matrix array, and metadata.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Subset fingerprint
    sub_hash_src = "".join(r.get("example_id", "") for r in subset_records[:100])
    sub_key = hashlib.sha256(f"{sub_hash_src}:{len(subset_records)}:{FEATURE_SCHEMA_VERSION}".encode("utf-8")).hexdigest()[:16]
    cache_path = cache_dir / f"structural_features_1b_{sub_key}.joblib"

    if cache_path.exists():
        logger.info("loading_structural_features_from_cache", path=str(cache_path))
        cached_payload = joblib.load(cache_path)
        cached_payload["cache_hit"] = True
        return cached_payload

    t0 = time.time()
    logger.info("extracting_structural_features_subset_start", n_responses=len(subset_records))

    # 1. Generate claim pairs for all subset responses
    all_pairs: List[Dict[str, Any]] = []
    resp_pair_map: Dict[str, List[Dict[str, Any]]] = {}

    for rec in subset_records:
        ex_id = rec.get("example_id", "")
        pairs = generate_unordered_claim_pairs(rec)
        resp_pair_map[ex_id] = pairs
        all_pairs.extend(pairs)

    # 2. Evaluate NLI & Similarity for all subset pairs (cached)
    nli_eval_payload = evaluate_bidirectional_nli_and_similarity(all_pairs, cache_dir=cache_dir)
    evaluated_pairs_list = nli_eval_payload["evaluated_pairs"]

    # Map evaluated pairs back to example_id
    eval_by_example: Dict[str, List[Dict[str, Any]]] = {}
    for ep in evaluated_pairs_list:
        ex_id = ep["example_id"]
        eval_by_example.setdefault(ex_id, []).append(ep)

    # 3. Extract 24-feature vectors for each response
    extracted_responses = []
    matrix_rows = []

    for rec in subset_records:
        ex_id = rec.get("example_id", "")
        e_pairs = eval_by_example.get(ex_id, [])
        feat_res = extract_structural_features_for_response(rec, e_pairs)
        extracted_responses.append(feat_res)

        row_vals = [feat_res["features"][col] for col in STRUCTURAL_FEATURE_COLUMNS]
        matrix_rows.append(row_vals)

    X_matrix = np.array(matrix_rows, dtype=np.float64)
    elapsed = time.time() - t0

    payload = {
        "cache_hit": False,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "n_responses": len(extracted_responses),
        "n_features": len(STRUCTURAL_FEATURE_COLUMNS),
        "feature_names": STRUCTURAL_FEATURE_COLUMNS,
        "elapsed_seconds": float(elapsed),
        "X_matrix": X_matrix,
        "extracted_responses": extracted_responses,
    }

    # Save to atomic persistent cache
    tmp_path = cache_dir / f"structural_features_1b_{sub_key}.tmp"
    joblib.dump(payload, tmp_path)
    os.replace(tmp_path, cache_path)

    logger.info("extracting_structural_features_subset_complete", elapsed_s=round(elapsed, 2), path=str(cache_path))
    return payload
