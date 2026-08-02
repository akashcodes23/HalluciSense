"""Phase 6M — Hybrid Dataset Builder and Probability Joiner.

Joins frozen Pillar-1 features (Phase 6I/6K), frozen Pillar-2 features (Phase 6L),
frozen model predictions P1 and P2, agreement/disagreement signals, and meta features
into a unified, aligned hybrid feature matrix across DEV (N=58,002) and VAL (N=12,483).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import structlog

from evaluation.phase6m.config import (
    DEV_PHASE6I_PATH,
    VAL_PHASE6I_PATH,
    DEV_PHASE6L_PATH,
    VAL_PHASE6L_PATH,
    PILLAR1_SCALER_PATH,
    PILLAR1_CLASSIFIER_PATH,
    PILLAR2_SCALER_PATH,
    PILLAR2_CLASSIFIER_PATH,
    PILLAR1_LOCKED_FEATURES,
    PILLAR2_LOCKED_FEATURES,
    HYBRID_FEATURE_SCHEMA,
    EPSILON,
    PHASE6M_DIR,
)

logger = structlog.get_logger(__name__)


def compute_logit(p: float, eps: float = EPSILON) -> float:
    """Compute log-odds (logit) of probability value with epsilon clipping."""
    p_clipped = max(eps, min(1.0 - eps, float(p)))
    return float(math.log(p_clipped / (1.0 - p_clipped)))


def load_and_assemble_hybrid_matrix(
    partition: str = "development",
    out_dir: Path = PHASE6M_DIR,
) -> Dict[str, Any]:
    """Assemble complete 19-feature hybrid matrix for DEV or VAL partition.

    Args:
        partition: 'development' (N=58,002) or 'validation' (N=12,483).
        out_dir: Output directory path.

    Returns:
        Dict containing:
            X: numpy feature matrix (N, 19)
            y: numpy target vector (N,)
            example_ids: List[str]
            feature_names: List[str]
            p1_probs: numpy array (N,)
            p2_probs: numpy array (N,)
            record_payloads: List[Dict[str, Any]]
    """
    logger.info("load_and_assemble_hybrid_matrix_start", partition=partition)

    if partition == "development":
        p1_path = DEV_PHASE6I_PATH
        p2_path = DEV_PHASE6L_PATH
        expected_count = 58002
    elif partition == "validation":
        p1_path = VAL_PHASE6I_PATH
        p2_path = VAL_PHASE6L_PATH
        expected_count = 12483
    else:
        raise ValueError(f"Invalid partition: '{partition}'. Must be 'development' or 'validation'.")

    if not p1_path.exists():
        raise FileNotFoundError(f"Pillar-1 feature file missing: {p1_path}")
    if not p2_path.exists():
        raise FileNotFoundError(f"Pillar-2 feature file missing: {p2_path}")

    # 1. Load Pillar 1 records (features + ground truth labels)
    p1_records_by_id: Dict[str, Dict[str, Any]] = {}
    p1_order: List[str] = []
    with open(p1_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ex_id = rec.get("example_id", "")
            if ex_id in p1_records_by_id:
                raise ValueError(f"Duplicate example_id in Pillar-1 {partition}: {ex_id}")
            p1_records_by_id[ex_id] = rec
            p1_order.append(ex_id)

    # 2. Load Pillar 2 records
    p2_records_by_id: Dict[str, Dict[str, Any]] = {}
    with open(p2_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ex_id = rec.get("example_id", "")
            if ex_id in p2_records_by_id:
                raise ValueError(f"Duplicate example_id in Pillar-2 {partition}: {ex_id}")
            p2_records_by_id[ex_id] = rec

    # Integrity verification of ID alignment
    if len(p1_order) != expected_count:
        raise ValueError(f"Pillar-1 {partition} row count error: Expected {expected_count}, got {len(p1_order)}")
    if len(p2_records_by_id) != expected_count:
        raise ValueError(f"Pillar-2 {partition} row count error: Expected {expected_count}, got {len(p2_records_by_id)}")

    for ex_id in p1_order:
        if ex_id not in p2_records_by_id:
            raise ValueError(f"Example ID '{ex_id}' in Pillar-1 missing from Pillar-2 {partition} dataset!")

    # 3. Extract raw feature arrays for Pillar 1 and Pillar 2 base models
    X_p1_raw_rows: List[List[float]] = []
    X_p2_raw_rows: List[List[float]] = []
    y_list: List[int] = []

    for ex_id in p1_order:
        r1 = p1_records_by_id[ex_id]
        r2 = p2_records_by_id[ex_id]

        y_gt = int(r1.get("ground_truth", 0))
        y_list.append(y_gt)

        # Pillar 1 5 locked features
        p1_feats = [float(r1.get(fn, 0.0)) for fn in PILLAR1_LOCKED_FEATURES]
        X_p1_raw_rows.append(p1_feats)

        # Pillar 2 5 locked features
        p2_feat_dict = r2.get("features", {})
        p2_feats = [float(p2_feat_dict.get(fn, 0.0)) for fn in PILLAR2_LOCKED_FEATURES]
        X_p2_raw_rows.append(p2_feats)

    X_p1_raw = np.array(X_p1_raw_rows, dtype=np.float64)
    X_p2_raw = np.array(X_p2_raw_rows, dtype=np.float64)
    y_arr = np.array(y_list, dtype=np.int64)

    # 4. Predict P1 probabilities using frozen Pillar-1 model
    p1_scaler = joblib.load(PILLAR1_SCALER_PATH)
    p1_clf = joblib.load(PILLAR1_CLASSIFIER_PATH)
    X_p1_scaled = p1_scaler.transform(X_p1_raw)
    p1_probs = p1_clf.predict_proba(X_p1_scaled)[:, 1]

    # 5. Predict P2 probabilities using frozen Pillar-2 model
    p2_scaler = joblib.load(PILLAR2_SCALER_PATH)
    p2_clf = joblib.load(PILLAR2_CLASSIFIER_PATH)
    X_p2_scaled = p2_scaler.transform(X_p2_raw)
    p2_probs = p2_clf.predict_proba(X_p2_scaled)[:, 1]

    # 6. Construct 19-feature hybrid matrix
    X_hybrid_rows: List[List[float]] = []
    record_payloads: List[Dict[str, Any]] = []

    for idx, ex_id in enumerate(p1_order):
        p1_f = X_p1_raw[idx]
        p2_f = X_p2_raw[idx]
        prob1 = float(p1_probs[idx])
        prob2 = float(p2_probs[idx])

        # Probability features
        l1 = compute_logit(prob1)
        l2 = compute_logit(prob2)

        # Agreement features
        disagg_abs = float(abs(prob1 - prob2))
        p_mean = float((prob1 + prob2) / 2.0)
        p_max = float(max(prob1, prob2))
        p_min = float(min(prob1, prob2))

        p_ratio = float((prob1 + EPSILON) / (prob2 + EPSILON))
        p_ratio = max(1e-3, min(1e3, p_ratio))  # Numerical safety clipping

        # Assemble 19 features exactly matching HYBRID_FEATURE_SCHEMA ordering
        row = [
            p1_f[0], p1_f[1], p1_f[2], p1_f[3], p1_f[4],  # P1 features (5)
            p2_f[0], p2_f[1], p2_f[2], p2_f[3], p2_f[4],  # P2 features (5)
            prob1, prob2, l1, l2,                         # Probability features (4)
            disagg_abs, p_mean, p_max, p_min, p_ratio,     # Agreement features (5)
        ]

        X_hybrid_rows.append(row)

        record_obj = {
            "example_id": ex_id,
            "dataset_partition": partition,
            "ground_truth": int(y_arr[idx]),
            "features": dict(zip(HYBRID_FEATURE_SCHEMA, row)),
        }
        record_payloads.append(record_obj)

    X_hybrid = np.array(X_hybrid_rows, dtype=np.float64)

    assert X_hybrid.shape[0] == expected_count, f"Expected {expected_count} rows, got {X_hybrid.shape[0]}"
    assert X_hybrid.shape[1] == len(HYBRID_FEATURE_SCHEMA), f"Expected {len(HYBRID_FEATURE_SCHEMA)} cols, got {X_hybrid.shape[1]}"

    logger.info("load_and_assemble_hybrid_matrix_complete", shape=X_hybrid.shape, partition=partition)

    return {
        "X": X_hybrid,
        "y": y_arr,
        "example_ids": p1_order,
        "feature_names": HYBRID_FEATURE_SCHEMA,
        "p1_probs": p1_probs,
        "p2_probs": p2_probs,
        "record_payloads": record_payloads,
    }
