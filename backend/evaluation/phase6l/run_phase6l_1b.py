"""Phase 6L.1B — Orchestrator for Structural Feature Extractor Implementation & Measurement Validation.

Usage:
    python -m evaluation.phase6l.run_phase6l_1b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import structlog

from evaluation.phase6j.utils import _serializable
from evaluation.phase6l.config import DEV_FEATURES_JSONL, PHASE6L_DIR, SUBSET_SIZE, RANDOM_STATE
from evaluation.phase6l.claim_pairs import extract_deterministic_dev_subset
from evaluation.phase6l.feature_extractor import extract_structural_features_for_subset
from evaluation.phase6l.feature_validation import (
    audit_feature_distributions,
    audit_feature_correlations,
    verify_structural_invariants,
    generate_phase6l_1b_sanity_artifacts,
    generate_phase6l_1b_report,
)

logger = structlog.get_logger(__name__)


def run_phase6l_1b() -> Dict[str, Any]:
    """Orchestrate Phase 6L.1B structural feature extraction & measurement validation.

    Returns:
        Dict containing validation audit summary and decision answers.
    """
    logger.info("phase6l_1b_orchestrator_start")
    t0 = time.time()

    print(f"\n{'=' * 85}")
    print("HalluciSense Phase 6L.1B — Structural Feature Extractor Validation Gate")
    print(f"{'=' * 85}")

    # 1. Reuse Exact 1,000-Example DEV Research Subset from Phase 6L.1A
    print("=== Task 1: Loading Deterministic 1,000-Example DEV Subset ===")
    subset_records = extract_deterministic_dev_subset(DEV_FEATURES_JSONL, subset_size=SUBSET_SIZE, seed=RANDOM_STATE)
    print(f"  Subset Responses Loaded: {len(subset_records):,} (Seed = {RANDOM_STATE})\n")

    # 2. Extract 24-Feature Structural Vectors with Persistent Caching
    print("=== Task 2: Extracting 24 Structural Features (Families A-H) ===")
    feat_payload = extract_structural_features_for_subset(subset_records, cache_dir=PHASE6L_DIR / "cache")
    X_matrix = feat_payload["X_matrix"]
    extracted_responses = feat_payload["extracted_responses"]

    print(f"  Responses Extracted  : {len(extracted_responses):,}")
    print(f"  Feature Matrix Shape : {X_matrix.shape}")
    print(f"  Elapsed Time         : {feat_payload['elapsed_seconds']:.2f}s")
    print(f"  Cache Hit            : {feat_payload['cache_hit']}\n")

    # Export subset_1000 jsonl file
    subset_jsonl_path = PHASE6L_DIR / "structural_features_subset_1000.jsonl"
    with open(subset_jsonl_path, "w", encoding="utf-8") as f:
        for resp_dict in extracted_responses:
            f.write(json.dumps(_serializable(resp_dict)) + "\n")
    print(f"  Exported: {subset_jsonl_path}\n")

    # 3. Audits & Sanity Checks
    print("=== Task 3: Running Audits & Invariant Checks ===")
    dist_audit = audit_feature_distributions(X_matrix, out_dir=PHASE6L_DIR)
    corr_audit = audit_feature_correlations(X_matrix, out_dir=PHASE6L_DIR)
    invariants_audit = verify_structural_invariants(extracted_responses)
    sanity_payload = generate_phase6l_1b_sanity_artifacts(extracted_responses, out_dir=PHASE6L_DIR)

    print(f"  All Features Finite  : {'Yes' if dist_audit['all_finite'] else 'No'}")
    print(f"  Constant Features    : {len(dist_audit['constant_features'])}")
    print(f"  Redundant Pairs (|r|>=0.90): {corr_audit['redundant_pearson_count']}")
    print(f"  Invariants Violation : {invariants_audit['violation_count']}\n")

    # 4. Generate Report
    print("=== Task 4: Generating Validation Report ===")
    report_path = generate_phase6l_1b_report(
        dist_audit=dist_audit,
        corr_audit=corr_audit,
        invariants_audit=invariants_audit,
        perf_payload=feat_payload,
        out_dir=PHASE6L_DIR,
    )

    elapsed = time.time() - t0
    print(f"\n{'=' * 85}")
    print(f"Phase 6L.1B Execution Completed — {elapsed:.2f}s elapsed")
    print(f"Report Generated: {report_path}")
    print(f"{'=' * 85}\n")

    logger.info("phase6l_1b_orchestrator_complete", elapsed_s=round(elapsed, 2))
    return {
        "dist_audit": dist_audit,
        "corr_audit": corr_audit,
        "invariants_audit": invariants_audit,
        "feature_matrix_shape": X_matrix.shape,
    }


def main() -> None:
    """CLI entry point for Phase 6L.1B."""
    parser = argparse.ArgumentParser(
        description="HalluciSense Phase 6L.1B — Structural Feature Extractor Validation Gate",
    )
    _ = parser.parse_args()
    _ = run_phase6l_1b()


if __name__ == "__main__":
    main()
