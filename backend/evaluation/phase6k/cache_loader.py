"""Phase 6K — Cache Discovery, Loading, and Schema Validation.

Loads cached feature matrices from Phase 6I (development and validation),
verifies schema integrity, target alignment, and class distributions.

Strictly read-only: never modifies cached JSONL files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import structlog

from evaluation.phase6k.config import (
    PHASE6I_DIR,
    DEV_CACHE_FILENAME,
    VAL_CACHE_FILENAME,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)

logger = structlog.get_logger(__name__)


# =========================================================
# DATACLASSES
# =========================================================

@dataclass
class DatasetPartition:
    """Loaded dataset partition containing features X and binary target y."""

    name: str
    X: np.ndarray
    y: np.ndarray
    n_samples: int
    n_features: int
    n_positive: int
    n_negative: int
    positive_ratio: float
    feature_names: List[str]


@dataclass
class LoadedCache:
    """Complete loaded Phase 6I cache container with DEV and VAL partitions."""

    dev: DatasetPartition
    val: DatasetPartition
    dev_path: Path
    val_path: Path
    schema_valid: bool = True
    validation_messages: List[str] = field(default_factory=list)


# =========================================================
# CACHE LOADING & VALIDATION
# =========================================================

def _load_partition_jsonl(
    file_path: Path,
    partition_name: str,
    feature_columns: List[str],
    target_column: str,
) -> DatasetPartition:
    """Read a Phase 6I JSONL feature file into a DatasetPartition object.

    Args:
        file_path: Path to the cached JSONL file.
        partition_name: Name of partition ('development' or 'validation').
        feature_columns: Ordered list of required feature names.
        target_column: Ground truth target column name.

    Returns:
        DatasetPartition container.

    Raises:
        FileNotFoundError: If the cache file does not exist.
        ValueError: If file is empty or schema validation fails.
    """
    if not file_path.exists():
        msg = (
            f"Phase 6I cached feature file missing: {file_path}\n"
            f"Action Required: Run Phase 6I feature extraction first using:\n"
            f"  python -m evaluation.run_phase6i_retrieval_reconstruction"
        )
        logger.error("phase6k_cache_file_missing", path=str(file_path))
        raise FileNotFoundError(msg)

    records: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as err:
                    raise ValueError(f"Corrupted JSON at line {line_num} in {file_path}: {err}")

    if not records:
        raise ValueError(f"Phase 6I cache file is empty: {file_path}")

    # Verify schema
    first_rec = records[0]
    missing_cols = [col for col in feature_columns if col not in first_rec]
    if missing_cols:
        raise ValueError(f"Missing required feature columns in {file_path}: {missing_cols}")

    if target_column not in first_rec:
        raise ValueError(f"Missing target column '{target_column}' in {file_path}")

    # Extract feature matrix X and target y
    X = np.array(
        [[r.get(col, 0.0) for col in feature_columns] for r in records],
        dtype=float,
    )
    y = np.array([r.get(target_column, 0) for r in records], dtype=int)

    # Validate target values are binary 0/1
    unique_targets = set(np.unique(y))
    if not unique_targets.issubset({0, 1}):
        raise ValueError(f"Target contains non-binary values {unique_targets} in {file_path}")

    n_samples = int(X.shape[0])
    n_features = int(X.shape[1])
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    pos_ratio = float(n_pos / max(1, n_samples))

    logger.info(
        "phase6k_partition_loaded",
        partition=partition_name,
        n_samples=n_samples,
        n_features=n_features,
        n_positive=n_pos,
        n_negative=n_neg,
        positive_ratio=round(pos_ratio, 4),
    )

    return DatasetPartition(
        name=partition_name,
        X=X,
        y=y,
        n_samples=n_samples,
        n_features=n_features,
        n_positive=n_pos,
        n_negative=n_neg,
        positive_ratio=pos_ratio,
        feature_names=list(feature_columns),
    )


def load_phase6i_cache(
    cache_dir: Path = PHASE6I_DIR,
    feature_columns: List[str] = FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
) -> LoadedCache:
    """Discover, load, and validate Phase 6I cached feature matrices for DEV and VAL.

    This is the single public entry point for cache loading in Phase 6K.

    Args:
        cache_dir: Directory containing Phase 6I JSONL cache files.
        feature_columns: Required feature names.
        target_column: Ground truth column name.

    Returns:
        LoadedCache object with DEV and VAL partitions and integrity metadata.
    """
    logger.info("phase6k_load_cache_start", cache_dir=str(cache_dir))

    dev_path = cache_dir / DEV_CACHE_FILENAME
    val_path = cache_dir / VAL_CACHE_FILENAME

    dev_partition = _load_partition_jsonl(dev_path, "development", feature_columns, target_column)
    val_partition = _load_partition_jsonl(val_path, "validation", feature_columns, target_column)

    messages: List[str] = [
        f"DEV: {dev_partition.n_samples} samples x {dev_partition.n_features} features (pos={dev_partition.n_positive}, neg={dev_partition.n_negative})",
        f"VAL: {val_partition.n_samples} samples x {val_partition.n_features} features (pos={val_partition.n_positive}, neg={val_partition.n_negative})",
    ]

    cache = LoadedCache(
        dev=dev_partition,
        val=val_partition,
        dev_path=dev_path,
        val_path=val_path,
        schema_valid=True,
        validation_messages=messages,
    )

    logger.info("phase6k_load_cache_complete", dev_samples=dev_partition.n_samples, val_samples=val_partition.n_samples)
    return cache
