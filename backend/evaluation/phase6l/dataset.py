"""Phase 6L.2 — Stage 1: Feature Matrix Validation Engine.

Loads structural_features_full_dev.jsonl, joins ground-truth labels from Phase 6I DEV features,
validates schema, completeness, finiteness, and label balance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import structlog

from evaluation.phase6l.config import PHASE6L_DIR, STRUCTURAL_FEATURE_COLUMNS

logger = structlog.get_logger(__name__)

PHASE6I_DIR = Path(__file__).resolve().parents[2] / "evaluation_results" / "phase6i"
DEV_STRUCTURAL_FEATURES_PATH = PHASE6L_DIR / "structural_features_full_dev.jsonl"
DEV_LABELS_PATH = PHASE6I_DIR / "claim_evidence_features_development.jsonl"


def load_and_validate_full_dev_matrix(
    structural_path: Path = DEV_STRUCTURAL_FEATURES_PATH,
    labels_path: Path = DEV_LABELS_PATH,
    feature_columns: List[str] = STRUCTURAL_FEATURE_COLUMNS,
    out_dir: Path = PHASE6L_DIR,
) -> Dict[str, Any]:
    """Load full DEV structural feature matrix, join ground-truth labels, and validate.

    Returns:
        Dict containing:
            X: numpy feature matrix (58002, 24)
            y: numpy label vector (58002,)
            example_ids: List[str]
            feature_names: List[str]
            validation_payload: Dict[str, Any]
    """
    logger.info("stage1_feature_matrix_validation_start", structural_path=str(structural_path))

    if not structural_path.exists():
        raise FileNotFoundError(f"Structural feature matrix missing: {structural_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Phase 6I ground-truth labels missing: {labels_path}")

    # 1. Load Ground Truth Labels from Phase 6I DEV
    labels_by_id: Dict[str, int] = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                ex_id = rec.get("example_id")
                gt = rec.get("ground_truth")
                if ex_id is not None and gt is not None:
                    labels_by_id[ex_id] = int(gt)

    # 2. Load Structural Feature Matrix
    example_ids: List[str] = []
    X_rows: List[List[float]] = []
    y_vals: List[int] = []

    duplicate_ids_count = 0
    seen_ids = set()

    with open(structural_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            ex_id = rec.get("example_id", "")

            if ex_id in seen_ids:
                duplicate_ids_count += 1
            else:
                seen_ids.add(ex_id)

            feats = rec.get("features", {})
            row = [float(feats.get(col, 0.0)) for col in feature_columns]

            if ex_id not in labels_by_id:
                raise ValueError(f"Example ID '{ex_id}' at line {line_idx} missing from ground-truth labels!")

            y_val = labels_by_id[ex_id]

            example_ids.append(ex_id)
            X_rows.append(row)
            y_vals.append(y_val)

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_vals, dtype=np.int64)

    n_samples, n_feats = X.shape

    # 3. Validation Checks
    nan_count = int(np.isnan(X).sum())
    inf_count = int(np.isinf(X).sum())
    missing_value_count = nan_count + inf_count

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    # Row duplicates check
    row_hashes = set()
    duplicate_rows_count = 0
    for r in X_rows:
        h = hash(tuple(r))
        if h in row_hashes:
            duplicate_rows_count += 1
        else:
            row_hashes.add(h)

    validation_payload = {
        "record_count": n_samples,
        "feature_count": n_feats,
        "feature_schema_version": "6L.1B.0",
        "feature_names": feature_columns,
        "duplicate_ids_count": duplicate_ids_count,
        "duplicate_rows_count": duplicate_rows_count,
        "missing_values_count": missing_value_count,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "dtype": str(X.dtype),
        "target_balance": {
            "n_positive": n_pos,
            "n_negative": n_neg,
            "pos_ratio": float(n_pos / max(1, n_samples)),
        },
        "status": "PASS" if (n_samples == 58002 and missing_value_count == 0 and duplicate_ids_count == 0) else "FAIL",
    }

    out_file = out_dir / "feature_matrix_validation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(validation_payload, f, indent=2)

    logger.info(
        "stage1_feature_matrix_validation_complete",
        n_samples=n_samples,
        n_features=n_feats,
        missing=missing_value_count,
        status=validation_payload["status"],
    )

    return {
        "X": X,
        "y": y,
        "example_ids": example_ids,
        "feature_names": feature_columns,
        "validation_payload": validation_payload,
    }
