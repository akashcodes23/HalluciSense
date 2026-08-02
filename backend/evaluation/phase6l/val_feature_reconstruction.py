"""Phase 6L.3 — Stage 0: VAL Partition Structural Feature Reconstruction.

Extracts the identical 24-feature structural matrix for the held-out Validation
partition (N = 12,483) using the frozen Phase 6L.1C measurement pipeline.

This is PURE MEASUREMENT EXTRACTION — no labels, no model fitting, no feature selection.
The exact same feature extraction code (claim_pairs, pairwise_nli, feature_extractor)
used for DEV is applied to VAL.

Produces: evaluation_results/phase6l/structural_features_full_val.jsonl
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import structlog

from evaluation.phase6l.config import (
    VAL_FEATURES_JSONL,
    STRUCTURAL_FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    PHASE6L_DIR,
    TAU_CONTRADICTION,
    TAU_SUPPORT,
    TAU_SIMILARITY_DUPLICATE,
)
from evaluation.phase6l.claim_pairs import generate_unordered_claim_pairs
from evaluation.phase6l.pairwise_nli import evaluate_bidirectional_nli_and_similarity
from evaluation.phase6l.feature_extractor import extract_structural_features_for_response

logger = structlog.get_logger(__name__)

SHARD_SIZE = 1000


def verify_val_dataset_integrity(val_path: Path = VAL_FEATURES_JSONL) -> Dict[str, Any]:
    """Verify VAL dataset record count, unique IDs, total claims, total pairs."""
    if not val_path.exists():
        raise FileNotFoundError(f"VAL dataset file missing: {val_path}")

    sha256_hash = hashlib.sha256()
    total_records = 0
    unique_ids = set()
    total_claims = 0
    claims_per_resp = []
    total_unordered_pairs = 0

    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            sha256_hash.update(line.encode("utf-8"))
            record = json.loads(line)
            ex_id = record.get("example_id", "")

            if ex_id in unique_ids:
                raise ValueError(f"Duplicate example_id in VAL: {ex_id}")
            unique_ids.add(ex_id)

            claims = [c.get("claim", "") for c in record.get("claim_details", []) if c.get("claim")]
            n_c = len(claims)
            total_records += 1
            total_claims += n_c
            claims_per_resp.append(n_c)
            total_unordered_pairs += (n_c * (n_c - 1)) // 2

    fingerprint = sha256_hash.hexdigest()

    if total_records != 12483:
        raise ValueError(f"VAL record count error: Expected 12,483, got {total_records}")

    return {
        "dataset_path": str(val_path),
        "dataset_sha256": fingerprint,
        "total_responses": total_records,
        "unique_response_ids": len(unique_ids),
        "total_claims": total_claims,
        "mean_claims_per_response": float(np.mean(claims_per_resp)) if claims_per_resp else 0.0,
        "median_claims_per_response": float(np.median(claims_per_resp)) if claims_per_resp else 0.0,
        "max_claims_per_response": int(np.max(claims_per_resp)) if claims_per_resp else 0,
        "total_unordered_pairs": total_unordered_pairs,
        "total_directional_inferences": total_unordered_pairs * 2,
    }


def reconstruct_val_structural_features(
    val_path: Path = VAL_FEATURES_JSONL,
    out_dir: Path = PHASE6L_DIR,
    shard_size: int = SHARD_SIZE,
) -> Dict[str, Any]:
    """Reconstruct 24-feature structural matrix for full VAL partition.

    Returns:
        Dict with reconstruction stats and output file path.
    """
    t_start = time.time()
    logger.info("val_feature_reconstruction_start")

    val_integrity = verify_val_dataset_integrity(val_path)
    print(f"  VAL Integrity: {val_integrity['total_responses']:,} responses, "
          f"{val_integrity['total_claims']:,} claims, "
          f"{val_integrity['total_unordered_pairs']:,} pairs")

    output_path = out_dir / "structural_features_full_val.jsonl"
    shards_dir = out_dir / "val_shards"
    cache_dir = out_dir / "cache"

    shards_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load all VAL records
    val_records: List[Dict[str, Any]] = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            val_records.append(json.loads(line))

    n_total = len(val_records)
    n_shards = (n_total + shard_size - 1) // shard_size

    all_feature_records: List[Dict[str, Any]] = []
    total_pairs = 0
    total_inferences = 0

    for s_idx in range(n_shards):
        start_i = s_idx * shard_size
        end_i = min(start_i + shard_size, n_total)
        shard_records = val_records[start_i:end_i]

        shard_filename = f"val_structural_{start_i:05d}_{end_i - 1:05d}.jsonl"
        shard_path = shards_dir / shard_filename

        # Check if shard is already completed
        if shard_path.exists():
            with open(shard_path, "r", encoding="utf-8") as sf:
                shard_lines = [json.loads(l) for l in sf if l.strip()]
            if len(shard_lines) == (end_i - start_i):
                all_feature_records.extend(shard_lines)
                print(f"  Shard {s_idx + 1}/{n_shards} [{start_i}:{end_i}] — resumed from cache")
                continue

        t_s = time.time()

        # 1. Generate claim pairs
        shard_pairs: List[Dict[str, Any]] = []
        resp_pair_map: Dict[str, List[Dict[str, Any]]] = {}
        for rec in shard_records:
            ex_id = rec.get("example_id", "")
            pairs = generate_unordered_claim_pairs(rec)
            resp_pair_map[ex_id] = pairs
            shard_pairs.extend(pairs)

        # 2. Bidirectional NLI & Similarity
        nli_payload = evaluate_bidirectional_nli_and_similarity(shard_pairs, cache_dir=cache_dir)
        eval_pairs_list = nli_payload["evaluated_pairs"]

        eval_by_example: Dict[str, List[Dict[str, Any]]] = {}
        for ep in eval_pairs_list:
            ex_id = ep["example_id"]
            eval_by_example.setdefault(ex_id, []).append(ep)

        # 3. Extract structural features per response
        shard_feature_records: List[Dict[str, Any]] = []
        for rec in shard_records:
            ex_id = rec.get("example_id", "")
            claims = [c.get("claim", "") for c in rec.get("claim_details", []) if c.get("claim")]
            evaluated_pairs = eval_by_example.get(ex_id, [])

            features = extract_structural_features_for_response(
                response_record=rec,
                evaluated_pairs_for_response=evaluated_pairs,
            )

            feature_record = {
                "example_id": ex_id,
                "dataset_partition": "validation",
                "num_claims": len(claims),
                "pair_count": len(resp_pair_map.get(ex_id, [])),
                "features": features,
            }
            shard_feature_records.append(feature_record)

        # Write shard
        with open(shard_path, "w", encoding="utf-8") as sf:
            for fr in shard_feature_records:
                sf.write(json.dumps(fr) + "\n")

        shard_pairs_count = sum(len(resp_pair_map.get(r.get("example_id", ""), [])) for r in shard_records)
        total_pairs += shard_pairs_count
        total_inferences += shard_pairs_count * 2

        all_feature_records.extend(shard_feature_records)
        elapsed_s = time.time() - t_s
        print(f"  Shard {s_idx + 1}/{n_shards} [{start_i}:{end_i}] — "
              f"{len(shard_feature_records)} responses, {shard_pairs_count} pairs, {elapsed_s:.1f}s")

    # Merge all shards into final output
    with open(output_path, "w", encoding="utf-8") as out_f:
        for fr in all_feature_records:
            out_f.write(json.dumps(fr) + "\n")

    # Validate
    n_written = len(all_feature_records)
    assert n_written == 12483, f"Expected 12,483 VAL records, got {n_written}"

    elapsed_total = time.time() - t_start
    stats = {
        "output_path": str(output_path),
        "total_responses": n_written,
        "total_pairs": total_pairs,
        "total_inferences": total_inferences,
        "elapsed_s": round(elapsed_total, 2),
        "val_integrity": val_integrity,
    }

    with open(out_dir / "val_reconstruction_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("val_feature_reconstruction_complete", n_responses=n_written, elapsed_s=round(elapsed_total, 2))
    return stats


def load_val_structural_features(
    val_structural_path: Path = PHASE6L_DIR / "structural_features_full_val.jsonl",
    val_labels_path: Path = VAL_FEATURES_JSONL,
    feature_columns: List[str] = STRUCTURAL_FEATURE_COLUMNS,
) -> Dict[str, Any]:
    """Load VAL structural feature matrix and join ground-truth labels.

    Returns:
        Dict with X_val, y_val, example_ids, feature_names.
    """
    if not val_structural_path.exists():
        raise FileNotFoundError(f"VAL structural features missing: {val_structural_path}")

    # Load labels
    labels_by_id: Dict[str, int] = {}
    with open(val_labels_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                ex_id = rec.get("example_id")
                gt = rec.get("ground_truth")
                if ex_id is not None and gt is not None:
                    labels_by_id[ex_id] = int(gt)

    # Load features
    example_ids: List[str] = []
    X_rows: List[List[float]] = []
    y_vals: List[int] = []

    with open(val_structural_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ex_id = rec.get("example_id", "")
            feats = rec.get("features", {})
            row = [float(feats.get(col, 0.0)) for col in feature_columns]

            if ex_id not in labels_by_id:
                raise ValueError(f"VAL example '{ex_id}' missing from ground-truth labels")

            example_ids.append(ex_id)
            X_rows.append(row)
            y_vals.append(labels_by_id[ex_id])

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_vals, dtype=np.int64)

    assert X.shape[0] == 12483, f"Expected 12,483 VAL rows, got {X.shape[0]}"
    assert X.shape[1] == len(feature_columns)

    logger.info("val_structural_features_loaded", shape=X.shape, n_pos=int((y == 1).sum()), n_neg=int((y == 0).sum()))
    return {
        "X": X,
        "y": y,
        "example_ids": example_ids,
        "feature_names": feature_columns,
    }
