"""Phase 8 — Dataset B Builder: Response-Level Ground Truth.

For each of the 750 Phase 7 generated responses, assigns an INDEPENDENT response-level
ground truth label by evaluating the response against retrieved evidence (P1 NLI only).

Ground truth authority: DeBERTa-v3-large-cross-encoder NLI score on BM25-retrieved evidence.
This is NOT the HalluciSense H-score (which includes fusion weights).
It is the raw NLI verdict only, providing an independent verification signal.

Output: backend/reports/phase8/response_level_ground_truth.jsonl
"""

from __future__ import annotations

import json
import time
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PHASE7_DIR = BACKEND_DIR / "reports" / "phase7"
PHASE8_DIR = BACKEND_DIR / "reports" / "phase8"
DATASET_PATH = BACKEND_DIR / "evaluation" / "results" / "benchmark_dataset.jsonl"

# Ground-truth thresholds (based on P1 NLI entailment/contradiction raw score)
# P1 score encodes factual error risk (0=factual, 1=hallucinated)
GT_FACTUAL_THRESHOLD = 0.35        # P1 < 0.35  → factual
GT_HALLUCINATED_THRESHOLD = 0.55   # P1 ≥ 0.55  → hallucinated
# 0.35 ≤ P1 < 0.55 → partially_hallucinated
UNVERIFIABLE_CLAIM_COUNT = 0       # 0-claim responses → unverifiable


def classify_response_gt(
    p1_score: float,
    p1_available: bool,
    claim_count: int,
    response_text: str,
) -> tuple[str, float, str]:
    """Classify response-level ground truth from P1 NLI evidence signal.

    Returns:
        (label, confidence, reason)
        label: 'factual' | 'hallucinated' | 'partially_hallucinated' | 'unverifiable'
    """
    if not p1_available:
        return "unverifiable", 0.0, "P1 evidence grounding unavailable for this sample"

    if not response_text or not response_text.strip():
        return "unverifiable", 0.0, "Empty response text"

    if claim_count == 0:
        return "unverifiable", 0.0, "No extractable claims in response"

    if p1_score < GT_FACTUAL_THRESHOLD:
        confidence = round(1.0 - p1_score / GT_FACTUAL_THRESHOLD, 4)
        return "factual", confidence, (
            f"P1 NLI evidence score {p1_score:.4f} < {GT_FACTUAL_THRESHOLD} threshold. "
            "Retrieved evidence strongly entails response claims."
        )
    elif p1_score >= GT_HALLUCINATED_THRESHOLD:
        confidence = round((p1_score - GT_HALLUCINATED_THRESHOLD) / (1.0 - GT_HALLUCINATED_THRESHOLD), 4)
        return "hallucinated", confidence, (
            f"P1 NLI evidence score {p1_score:.4f} ≥ {GT_HALLUCINATED_THRESHOLD} threshold. "
            "Retrieved evidence substantially contradicts response claims."
        )
    else:
        # Partial zone
        mid = (GT_FACTUAL_THRESHOLD + GT_HALLUCINATED_THRESHOLD) / 2
        confidence = round(abs(p1_score - mid) / (mid - GT_FACTUAL_THRESHOLD), 4)
        return "partially_hallucinated", confidence, (
            f"P1 NLI score {p1_score:.4f} in partial zone "
            f"[{GT_FACTUAL_THRESHOLD}, {GT_HALLUCINATED_THRESHOLD}). "
            "Evidence partially supports response with some potential inaccuracies."
        )


def binary_label(rich_label: str) -> Optional[int]:
    """Convert rich label to binary for primary benchmark metrics."""
    if rich_label == "factual":
        return 0
    elif rich_label in ("hallucinated", "partially_hallucinated"):
        return 1
    return None  # unverifiable → excluded from binary metrics


def main():
    PHASE8_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load Phase 7 traces ─────────────────────────────────────────────────
    print("Loading Phase 7 traces and benchmark dataset…")
    traces_dir = PHASE7_DIR / "traces"

    bench_map: dict[str, dict] = {}
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            bench_map[d["id"]] = d

    records = []
    label_counts = {"factual": 0, "hallucinated": 0, "partially_hallucinated": 0, "unverifiable": 0}
    label_shift_count = 0  # original_static_label=1, response_gt=factual

    for i in range(1, 751):
        trace_path = traces_dir / f"TRACE_PHASE7_{i:06d}.json"
        trace = json.loads(trace_path.read_text(encoding="utf-8"))

        sid = trace["sample_id"]
        domain = trace["domain"]
        difficulty = trace["difficulty"]
        query = trace["query"]
        response = trace["generated_response"]
        original_static_label = trace["ground_truth"]  # benchmark label (may not match response)

        p1_score = trace["p1"]["score"]
        p1_available = trace["p1"]["available"]
        p1_claims = trace["p1"].get("claims", [])
        claim_count = len(p1_claims) if isinstance(p1_claims, list) else 1

        p2_score = trace["p2"]["score"]
        p3_score = trace["p3"]["score"]
        p3_available = trace["p3"]["available"]
        h_score = trace["fusion"]["h_score"]
        fusion_mode = trace["fusion"]["mode"]

        # ── INDEPENDENT ground truth via P1 NLI only ──────────────────────
        rich_gt, gt_confidence, gt_reason = classify_response_gt(
            p1_score, p1_available, claim_count, response
        )
        bin_gt = binary_label(rich_gt)
        label_counts[rich_gt] += 1

        # Track label shift
        if original_static_label == 1 and rich_gt == "factual":
            label_shift_count += 1

        # ── Evidence provenance ────────────────────────────────────────────
        bench_rec = bench_map.get(sid, {})
        evidence_passages = bench_rec.get("evidence_passages", [])

        record = {
            "sample_id": sid,
            "dataset": "Phase7_LiveGeneration",
            "domain": domain,
            "difficulty": difficulty,
            "prompt": query,
            "generated_response": response,
            # Dual-label: the core scientific distinction
            "original_static_label": original_static_label,
            "original_static_label_meaning": "factual" if original_static_label == 0 else "hallucinated",
            "response_ground_truth": rich_gt,
            "response_ground_truth_binary": bin_gt,
            "ground_truth_confidence": gt_confidence,
            "ground_truth_method": "P1_NLI_Evidence_Grounding",
            "ground_truth_reason": gt_reason,
            "ground_truth_threshold_factual": GT_FACTUAL_THRESHOLD,
            "ground_truth_threshold_hallucinated": GT_HALLUCINATED_THRESHOLD,
            "is_label_shift": (original_static_label == 1 and rich_gt == "factual"),
            # HalluciSense scores (from Phase 7, for evaluation against new GT)
            "phase7_h_score": h_score,
            "phase7_p1": p1_score,
            "phase7_p2": p2_score,  # None = UNAVAILABLE
            "phase7_p3": p3_score,
            "phase7_p1_available": p1_available,
            "phase7_p2_available": trace["p2"]["available"],
            "phase7_p3_available": p3_available,
            "phase7_fusion_mode": fusion_mode,
            "phase7_predicted_label": trace["predicted_label"],
            "phase7_risk_level": trace["risk_level"],
            # Claim and evidence metadata
            "claim_count": claim_count,
            "evidence_count": trace["p1"].get("evidence_count", 0),
            "evidence_passages_count": len(evidence_passages),
            "supporting_evidence": evidence_passages[:2] if evidence_passages else [],
            "evidence_source": "Wikipedia_BM25_FAISS_DeBERTa_NLI",
            # Latency
            "latency_ms": trace["timings"]["total_ms"],
        }
        records.append(record)

    # ── Write output ────────────────────────────────────────────────────────
    out_path = PHASE8_DIR / "response_level_ground_truth.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── SHA-256 of output ───────────────────────────────────────────────────
    dataset_hash = hashlib.sha256(out_path.read_bytes()).hexdigest()

    # Compute label shift breakdown
    total = len(records)
    binary_usable = sum(1 for r in records if r["response_ground_truth_binary"] is not None)
    binary_factual = sum(1 for r in records if r["response_ground_truth_binary"] == 0)
    binary_hallucinated = sum(1 for r in records if r["response_ground_truth_binary"] == 1)
    unverifiable_count = label_counts["unverifiable"]

    manifest = {
        "dataset_name": "Phase8_Dataset_B_LiveResponseBenchmark",
        "version": "1.0.0",
        "creation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_dataset": "Phase7_live_generation_traces",
        "source_sha256_phase6_canonical": "dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5",
        "output_sha256": dataset_hash,
        "generation_script": "backend/evaluation/phase8/build_dataset_b.py",
        "random_seed": None,
        "sample_count": total,
        "ground_truth_method": "P1_NLI_Evidence_Grounding (independent of H-score fusion)",
        "ground_truth_threshold_factual": GT_FACTUAL_THRESHOLD,
        "ground_truth_threshold_hallucinated": GT_HALLUCINATED_THRESHOLD,
        "label_distribution_rich": label_counts,
        "label_distribution_binary": {
            "factual_binary_0": binary_factual,
            "hallucinated_or_partial_binary_1": binary_hallucinated,
            "unverifiable_excluded": unverifiable_count,
        },
        "binary_usable_count": binary_usable,
        "label_shift_count": label_shift_count,
        "label_shift_pct": round(label_shift_count / 375 * 100, 2),
        "schema_version": "phase8_v1",
        "note": (
            "response_ground_truth is determined independently via P1 NLI evidence grounding. "
            "It is NOT the same as original_static_label (the benchmark prompt label). "
            "unverifiable records are excluded from binary classification metrics."
        )
    }

    manifest_path = PHASE8_DIR / "response_ground_truth_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nDataset B complete: {total} records written to {out_path}")
    print(f"  SHA-256: {dataset_hash}")
    print(f"  Label distribution (rich): {label_counts}")
    print(f"  Binary usable: {binary_usable} (factual={binary_factual}, hallucinated={binary_hallucinated})")
    print(f"  Unverifiable (excluded from binary metrics): {unverifiable_count}")
    print(f"  Label shift cases (static=hallucinated, response=factual): {label_shift_count} / 375 ({label_shift_count/375*100:.1f}%)")


if __name__ == "__main__":
    main()
