"""Phase 6J — Orchestrator for Numerical Stability & Feature Validation.

Loads cached Phase 6I feature matrices, runs all Phase 6J analysis modules
in sequence, and saves outputs under evaluation_results/phase6j/.

Read-only: never modifies Phase 6I outputs.

Usage:
    python -m evaluation.phase6j.run_phase6j
    python -m evaluation.phase6j.run_phase6j --split development
    python -m evaluation.phase6j.run_phase6j --split validation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import structlog

from evaluation.phase6j.statistics import compute_statistics
from evaluation.phase6j.distributions import compute_distributions
from evaluation.phase6j.scaling import compute_scaling
from evaluation.phase6j.separation import compute_separation
from evaluation.phase6j.stability import compute_stability
from evaluation.phase6j.report import generate_report

logger = structlog.get_logger(__name__)

# =========================================================
# PATHS
# =========================================================

PHASE6I_DIR = Path("evaluation_results/phase6i")
PHASE6J_DIR = Path("evaluation_results/phase6j")

# Phase 6I feature columns — must match exactly.
FEATURE_COLUMNS: List[str] = [
    "mean_entailment",
    "max_entailment",
    "mean_contradiction",
    "max_contradiction",
    "mean_support_margin",
    "min_support_margin",
    "fraction_supported",
    "fraction_contradicted",
    "fraction_unsupported",
    "num_claims",
]


# =========================================================
# DATA LOADING
# =========================================================

def load_feature_matrix(
    jsonl_path: Path,
    feature_columns: List[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Load a cached Phase 6I feature JSONL file into numpy arrays.

    Args:
        jsonl_path: Path to the Phase 6I claim_evidence_features JSONL file.
        feature_columns: Ordered list of feature column names to extract.

    Returns:
        Tuple of (X, y) where X is shape (n_samples, n_features) and y is (n_samples,).

    Raises:
        FileNotFoundError: If the JSONL file does not exist.
        ValueError: If the file is empty or contains no valid records.
    """
    if not jsonl_path.exists():
        msg = (
            f"Required Phase 6I cached feature file missing: {jsonl_path}\n"
            f"Phase 6J analysis requires pre-computed Phase 6I feature matrices.\n"
            f"Action Required: Run Phase 6I cache generation first using:\n"
            f"  python -m evaluation.run_phase6i_retrieval_reconstruction"
        )
        logger.error("phase6j_missing_cache_file", path=str(jsonl_path))
        raise FileNotFoundError(msg)

    records: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"No records found in: {jsonl_path}")

    X = np.array(
        [[r.get(col, 0.0) for col in feature_columns] for r in records],
        dtype=float,
    )
    y = np.array([r.get("ground_truth", 0) for r in records], dtype=int)

    logger.info(
        "phase6j_data_loaded",
        path=str(jsonl_path),
        n_samples=X.shape[0],
        n_features=X.shape[1],
        n_positive=int((y == 1).sum()),
        n_negative=int((y == 0).sum()),
    )
    return X, y


# =========================================================
# ORCHESTRATOR
# =========================================================

def run_phase6j(split: str = "development") -> None:
    """Execute the full Phase 6J analysis pipeline.

    Args:
        split: Primary Phase 6I split to analyze ('development' or 'validation').
              Both splits are always loaded for statistics; the primary split
              is used for remaining modules.
    """
    logger.info("phase6j_pipeline_start", split=split)
    t0 = time.time()

    PHASE6J_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load both cached Phase 6I feature matrices ---
    dev_path = PHASE6I_DIR / "claim_evidence_features_development.jsonl"
    val_path = PHASE6I_DIR / "claim_evidence_features_validation.jsonl"

    print(f"\n{'=' * 60}")
    print("HalluciSense Phase 6J — Numerical Stability & Feature Validation")
    print(f"{'=' * 60}")
    print(f"  Primary Split    : {split}")
    print(f"  DEV Source       : {dev_path}")
    print(f"  VAL Source       : {val_path}")
    print(f"  Output Directory : {PHASE6J_DIR}")
    print(f"{'=' * 60}\n")

    X_dev, y_dev = load_feature_matrix(dev_path, FEATURE_COLUMNS)
    X_val, y_val = load_feature_matrix(val_path, FEATURE_COLUMNS)

    print(f"  DEV: {X_dev.shape[0]} samples × {X_dev.shape[1]} features"
          f"  (pos={int((y_dev == 1).sum())}, neg={int((y_dev == 0).sum())})")
    print(f"  VAL: {X_val.shape[0]} samples × {X_val.shape[1]} features"
          f"  (pos={int((y_val == 1).sum())}, neg={int((y_val == 0).sum())})\n")

    # Select primary split for modules that take a single matrix
    if split == "validation":
        X_primary, y_primary = X_val, y_val
    else:
        X_primary, y_primary = X_dev, y_dev

    # --- Execute analysis modules in sequence ---

    print("=== Module 1: Descriptive Statistics ===")
    stats_report = compute_statistics(X_dev, y_dev, X_val, y_val, FEATURE_COLUMNS, PHASE6J_DIR)

    print("=== Module 2: Distribution Analysis ===")
    dist_report = compute_distributions(X_primary, y_primary, FEATURE_COLUMNS, PHASE6J_DIR)

    print("=== Module 3: Scaling Diagnostics ===")
    scaling_report = compute_scaling(X_primary, y_primary, FEATURE_COLUMNS, PHASE6J_DIR)

    print("=== Module 4: Class Separation ===")
    sep_report = compute_separation(X_primary, y_primary, FEATURE_COLUMNS, PHASE6J_DIR)

    print("=== Module 5: Numerical Stability ===")
    stab_report = compute_stability(X_dev, y_dev, FEATURE_COLUMNS, PHASE6J_DIR, X_val=X_val, y_val=y_val)

    # --- Generate consolidated report ---

    print("=== Module 6: Report Generation ===")
    phase6j_report = generate_report(
        statistics=stats_report,
        distributions=dist_report,
        scaling=scaling_report,
        separation=sep_report,
        stability=stab_report,
        out_dir=PHASE6J_DIR,
    )

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Phase 6J Complete — {elapsed:.1f}s elapsed")
    print(f"Verdict: {phase6j_report.verdict}")
    print(f"{'=' * 60}\n")

    logger.info("phase6j_pipeline_complete", elapsed_s=round(elapsed, 2), verdict=phase6j_report.verdict)


# =========================================================
# CLI ENTRY POINT
# =========================================================

def main() -> None:
    """CLI entry point for Phase 6J."""
    parser = argparse.ArgumentParser(
        description="HalluciSense Phase 6J — Numerical Stability & Feature Validation",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="development",
        choices=["development", "validation"],
        help="Which Phase 6I split to analyze (default: development).",
    )
    args = parser.parse_args()

    run_phase6j(split=args.split)


if __name__ == "__main__":
    main()
