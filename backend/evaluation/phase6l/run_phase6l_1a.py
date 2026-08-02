"""Phase 6L.1A — Orchestrator for Pairwise NLI Feasibility, Directionality & Measurement Validation.

Usage:
    python -m evaluation.phase6l.run_phase6l_1a
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import structlog

from evaluation.phase6l.config import DEV_FEATURES_JSONL, PHASE6L_DIR, SUBSET_SIZE, RANDOM_STATE
from evaluation.phase6l.claim_pairs import (
    audit_dev_pair_complexity,
    extract_deterministic_dev_subset,
    generate_unordered_claim_pairs,
)
from evaluation.phase6l.pairwise_nli import evaluate_bidirectional_nli_and_similarity
from evaluation.phase6l.nli_feasibility import (
    audit_nli_model_metadata,
    audit_directional_asymmetry,
    audit_symmetric_aggregation,
    audit_embedding_screening,
    generate_manual_sanity_samples,
    generate_phase6l_1a_report,
)

logger = structlog.get_logger(__name__)


def run_phase6l_1a() -> Dict[str, Any]:
    """Orchestrate Phase 6L.1A measurement validation gate.

    Returns:
        Dict containing gate audit summary and decision answers.
    """
    logger.info("phase6l_1a_orchestrator_start")
    t0 = time.time()

    print(f"\n{'=' * 85}")
    print("HalluciSense Phase 6L.1A — Pairwise NLI Feasibility & Measurement Validation Gate")
    print(f"{'=' * 85}")

    # 1. Exact Pair Complexity Audit on Full DEV (N=58,002)
    print("=== Task 1: Full DEV Claim & Pair Complexity Audit (N=58,002) ===")
    complexity_audit = audit_dev_pair_complexity(DEV_FEATURES_JSONL)

    print(f"  Total DEV Responses  : {complexity_audit['total_responses']:,}")
    print(f"  Total Atomic Claims  : {complexity_audit['total_claims']:,}")
    print(f"  Mean Claims/Response : {complexity_audit['claims_per_response_stats']['mean']:.2f}")
    print(f"  Exact Unordered Pairs: {complexity_audit['exact_pair_counts']['m_unordered_total']:,}")
    print(f"  Exact Directional NLI: {complexity_audit['exact_pair_counts']['m_directional_total']:,}\n")

    # 2. Extract Deterministic 1,000-Example DEV Research Subset
    print("=== Task 2: Extracting Deterministic 1,000-Example DEV Research Subset ===")
    subset_records = extract_deterministic_dev_subset(DEV_FEATURES_JSONL, subset_size=SUBSET_SIZE, seed=RANDOM_STATE)

    subset_pairs: List[Dict[str, Any]] = []
    for rec in subset_records:
        subset_pairs.extend(generate_unordered_claim_pairs(rec))

    print(f"  Subset Responses     : {len(subset_records):,}")
    print(f"  Subset Unordered Pairs: {len(subset_pairs):,}")
    print(f"  Subset Inferences     : {len(subset_pairs) * 2:,}\n")

    # 3. Execute Bidirectional NLI & Similarity on Research Subset
    print("=== Task 3: Executing Bidirectional NLI & Embedding Similarity ===")
    nli_results = evaluate_bidirectional_nli_and_similarity(subset_pairs, cache_dir=PHASE6L_DIR / "cache")

    print(f"  Pairs Evaluated      : {nli_results['total_pairs_evaluated']:,}")
    print(f"  Inferences Executed  : {nli_results['total_directional_inferences']:,}")
    print(f"  Elapsed Time         : {nli_results['elapsed_seconds']:.2f}s")
    print(f"  Inferences / Second  : {nli_results['inferences_per_second']:.1f}")
    print(f"  Warnings Captured    : {nli_results['total_warnings']}\n")

    # 4. Audits & Markdown Reports
    print("=== Task 4: Running Audits & Generating Reports ===")
    nli_audit = audit_nli_model_metadata(PHASE6L_DIR)
    asymmetry_audit = audit_directional_asymmetry(nli_results["evaluated_pairs"], PHASE6L_DIR)
    aggregation_audit = audit_symmetric_aggregation(nli_results["evaluated_pairs"], PHASE6L_DIR)
    screening_audit = audit_embedding_screening(nli_results["evaluated_pairs"], PHASE6L_DIR)
    sanity_audit = generate_manual_sanity_samples(nli_results["evaluated_pairs"], PHASE6L_DIR)

    report_path = generate_phase6l_1a_report(
        complexity_audit=complexity_audit,
        nli_audit=nli_audit,
        asymmetry_audit=asymmetry_audit,
        aggregation_audit=aggregation_audit,
        screening_audit=screening_audit,
        perf_audit=nli_results,
        out_dir=PHASE6L_DIR,
    )

    elapsed = time.time() - t0
    print(f"\n{'=' * 85}")
    print(f"Phase 6L.1A Execution Completed — {elapsed:.2f}s elapsed")
    print(f"Report Generated: {report_path}")
    print(f"{'=' * 85}\n")

    logger.info("phase6l_1a_orchestrator_complete", elapsed_s=round(elapsed, 2))
    return {
        "complexity_audit": complexity_audit,
        "nli_results": nli_results,
        "asymmetry_audit": asymmetry_audit,
        "aggregation_audit": aggregation_audit,
        "screening_audit": screening_audit,
    }


def main() -> None:
    """CLI entry point for Phase 6L.1A."""
    parser = argparse.ArgumentParser(
        description="HalluciSense Phase 6L.1A — Pairwise NLI Feasibility & Measurement Validation Gate",
    )
    _ = parser.parse_args()
    _ = run_phase6l_1a()


if __name__ == "__main__":
    main()
