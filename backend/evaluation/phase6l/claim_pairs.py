"""Phase 6L.1A — Claim Pair Generation and Exact Pair Complexity Audit.

Audits exact claim and pair complexity across the full Development partition (N=58,002),
constructs deterministic DEV research subsets, and generates unordered claim pairs.

Strict Data Firewall Rule:
    * Accesses DEV partition ONLY. Validation partition (N=12,483) is strictly sealed.
"""

from __future__ import annotations

import hashlib
import json
import math

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import structlog

from evaluation.phase6j.utils import _serializable
from evaluation.phase6l.config import DEV_FEATURES_JSONL, PHASE6L_DIR, RANDOM_STATE

logger = structlog.get_logger(__name__)


def audit_dev_pair_complexity(dev_jsonl_path: Path = DEV_FEATURES_JSONL) -> Dict[str, Any]:
    """Inspect full DEV JSONL file and calculate exact claim and pair complexity statistics.

    Args:
        dev_jsonl_path: Path to claim_evidence_features_development.jsonl.

    Returns:
        Dict containing exact claim counts, quantiles, and pair counts.
    """
    logger.info("auditing_dev_pair_complexity_start", path=str(dev_jsonl_path))

    if not dev_jsonl_path.exists():
        raise FileNotFoundError(f"DEV feature JSONL file not found at: {dev_jsonl_path}")

    response_counts = 0
    total_claims = 0
    claims_per_response: List[int] = []
    unordered_pairs_per_response: List[int] = []

    m_unordered_total = 0

    with open(dev_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            response_counts += 1

            claim_details = rec.get("claim_details", [])
            n_r = len(claim_details) if claim_details else rec.get("num_claims", 0)

            claims_per_response.append(n_r)
            total_claims += n_r

            m_r = (n_r * (n_r - 1)) // 2 if n_r >= 2 else 0
            unordered_pairs_per_response.append(m_r)
            m_unordered_total += m_r

    arr_claims = np.array(claims_per_response, dtype=np.int64)
    arr_pairs = np.array(unordered_pairs_per_response, dtype=np.int64)

    m_directional_total = 2 * m_unordered_total

    # Pair distribution breakdown (responses with 0, 1, 2, 3+ pairs)
    pair_freq_counts = {
        "0_pairs_n_less_than_2": int(np.sum(arr_pairs == 0)),
        "1_pair_n_2": int(np.sum(arr_pairs == 1)),
        "3_pairs_n_3": int(np.sum(arr_pairs == 3)),
        "6_pairs_n_4": int(np.sum(arr_pairs == 6)),
        "10_pairs_n_5": int(np.sum(arr_pairs == 10)),
        "more_than_10_pairs_n_gt_5": int(np.sum(arr_pairs > 10)),
    }

    audit_results = {
        "dataset_partition": "development",
        "total_responses": response_counts,
        "total_claims": total_claims,
        "claims_per_response_stats": {
            "mean": float(np.mean(arr_claims)),
            "std": float(np.std(arr_claims)),
            "median": float(np.median(arr_claims)),
            "min": int(np.min(arr_claims)),
            "max": int(np.max(arr_claims)),
            "p75": float(np.percentile(arr_claims, 75)),
            "p90": float(np.percentile(arr_claims, 90)),
            "p95": float(np.percentile(arr_claims, 95)),
            "p99": float(np.percentile(arr_claims, 99)),
        },
        "exact_pair_counts": {
            "m_unordered_total": int(m_unordered_total),
            "m_directional_total": int(m_directional_total),
            "mean_unordered_pairs_per_response": float(np.mean(arr_pairs)),
            "max_unordered_pairs_per_response": int(np.max(arr_pairs)),
        },
        "response_pair_distribution": pair_freq_counts,
    }

    logger.info(
        "auditing_dev_pair_complexity_complete",
        responses=response_counts,
        total_claims=total_claims,
        m_unordered=m_unordered_total,
        m_directional=m_directional_total,
    )

    out_dir = PHASE6L_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pair_complexity_audit.json", "w", encoding="utf-8") as f:
        json.dump(_serializable(audit_results), f, indent=2)

    return audit_results


def extract_deterministic_dev_subset(
    dev_jsonl_path: Path = DEV_FEATURES_JSONL,
    subset_size: int = 1000,
    seed: int = RANDOM_STATE,
) -> List[Dict[str, Any]]:
    """Extract a deterministic stratified subset of DEV responses for Phase 6L.1A.

    Args:
        dev_jsonl_path: Path to claim_evidence_features_development.jsonl.
        subset_size: Number of responses to sample (default 1,000).
        seed: Random seed for sampling (default 42).

    Returns:
        List of response records.
    """
    records: List[Dict[str, Any]] = []
    with open(dev_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if len(records) < subset_size:
        raise ValueError(f"DEV dataset has only {len(records)} records; requested subset {subset_size}")

    labels = np.array([r.get("ground_truth", 0) for r in records], dtype=int)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    rng = np.random.RandomState(seed)
    pos_ratio = len(pos_idx) / len(labels)
    n_pos_target = int(round(subset_size * pos_ratio))
    n_neg_target = subset_size - n_pos_target

    sampled_pos = rng.choice(pos_idx, size=n_pos_target, replace=False)
    sampled_neg = rng.choice(neg_idx, size=n_neg_target, replace=False)

    sampled_indices = np.concatenate([sampled_pos, sampled_neg])
    sampled_indices.sort()

    subset_records = [records[i] for i in sampled_indices]

    logger.info("dev_subset_extracted", count=len(subset_records), seed=seed)
    return subset_records


def generate_unordered_claim_pairs(response_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate deterministic unordered claim pairs (i < j) for a single response.

    Args:
        response_record: Dict containing 'example_id', 'claim_details', 'ground_truth'.

    Returns:
        List of claim pair dicts with exact claim texts and indices.
    """
    example_id = response_record.get("example_id", "")
    claim_details = response_record.get("claim_details", [])
    ground_truth = response_record.get("ground_truth", 0)

    pairs: List[Dict[str, Any]] = []
    n = len(claim_details)
    if n < 2:
        return pairs

    for i in range(n):
        c_i_text = str(claim_details[i].get("claim", "")).strip()
        for j in range(i + 1, n):
            c_j_text = str(claim_details[j].get("claim", "")).strip()
            pair_obj = {
                "example_id": example_id,
                "claim_i_index": i,
                "claim_j_index": j,
                "claim_i_text": c_i_text,
                "claim_j_text": c_j_text,
                "ground_truth": int(ground_truth),
                "pair_hash": hashlib.sha256(f"{example_id}:{i}:{j}:{c_i_text}:{c_j_text}".encode("utf-8")).hexdigest()[:16],
            }
            pairs.append(pair_obj)

    return pairs
