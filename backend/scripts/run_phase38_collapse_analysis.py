"""Phase 38.3 & 38.4 — Feature Collapse Analysis and Minimal-Pair Metrics Script.

Executes all 162 adversarial cases, dumps backend/reports/phase38/feature_vectors.json,
calculates L1, L2, cosine similarities, per-feature differences, and compiles
quantitative discrimination metrics across all minimal-pair categories.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.pipeline import get_hallucisense_pipeline
from tests.test_phase38_adversarial_matrix import ADVERSARIAL_CASES


def compute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 1.0 if np.array_equal(v1, v2) else 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def main():
    pipe = get_hallucisense_pipeline()
    output_dir = BACKEND_DIR / "reports" / "phase38"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Evaluating 162 adversarial test cases...")
    records = []
    
    for cat_name, items in ADVERSARIAL_CASES.items():
        print(f"  Running category: {cat_name} ({len(items)} items)")
        for idx, item in enumerate(items):
            res = pipe.predict(response_text=item["text"])
            attr = res.get("local_attribution", {})
            vec = [float(f["value"]) for f in attr.get("features", [])]
            
            rec = {
                "id": item["id"],
                "category": cat_name,
                "text": item["text"],
                "expected": item["expected"],
                "claim_count": int(res["claim_count"]),
                "claims": res["claims"],
                "probability": float(res["hallucination_probability"]),
                "verdict": bool(res["is_hallucinated"]),
                "confidence": float(res["confidence_score"]),
                "vector": vec,
                "top_hallucination_drivers": [f["feature_name"] for f in attr.get("top_hallucination_drivers", [])],
                "top_protective_drivers": [f["feature_name"] for f in attr.get("top_protective_drivers", [])],
                "interaction_gap": float(attr.get("interaction_gap", 0.0)),
                "baseline_probability": float(attr.get("baseline_probability", 0.0)),
            }
            records.append(rec)
            
    # Save feature_vectors.json
    json_path = output_dir / "feature_vectors.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} feature records to {json_path}")
    
    # ─── Feature Collapse Analysis ─────────────────────────────────────────
    # Define "near-identical" threshold:
    # A vector difference is considered "near-identical" if L2 distance <= 0.01 (1% across 19 features)
    NEAR_IDENTICAL_L2_THRESHOLD = 0.01
    
    print("\nAnalyzing minimal pairs for representation collapse...")
    
    # Minimal pair categories
    pair_categories = [
        "category_a_minimal_pairs",
        "category_b_entity_swaps",
        "category_c_numerical_mutations",
        "category_d_negations",
        "category_e_temporal_mutations",
        "category_f_multiclaim_pairs",
    ]
    
    pair_results = []
    
    for cat in pair_categories:
        cat_records = [r for r in records if r["category"] == cat]
        n_pairs = len(cat_records) // 2
        for p_idx in range(n_pairs):
            r1 = cat_records[2 * p_idx]
            r2 = cat_records[2 * p_idx + 1]
            
            v1 = np.array(r1["vector"])
            v2 = np.array(r2["vector"])
            
            l1_dist = float(np.sum(np.abs(v1 - v2)))
            l2_dist = float(np.linalg.norm(v1 - v2))
            cos_sim = compute_cosine_similarity(v1, v2)
            prob_diff = abs(r1["probability"] - r2["probability"])
            verdict_diff = (r1["verdict"] != r2["verdict"])
            
            is_identical = (l2_dist == 0.0)
            is_near_identical = (l2_dist <= NEAR_IDENTICAL_L2_THRESHOLD)
            
            pair_results.append({
                "pair_id": f"{r1['id']} vs {r2['id']}",
                "category": cat,
                "item1": r1["text"],
                "item2": r2["text"],
                "p1": r1["probability"],
                "p2": r2["probability"],
                "prob_diff": round(prob_diff, 4),
                "v1_verdict": r1["verdict"],
                "v2_verdict": r2["verdict"],
                "verdict_separated": verdict_diff,
                "l1_distance": round(l1_dist, 6),
                "l2_distance": round(l2_dist, 6),
                "cosine_similarity": round(cos_sim, 6),
                "is_identical": is_identical,
                "is_near_identical": is_near_identical,
            })
            
    # Calculate Summary Metrics
    total_pairs = len(pair_results)
    identical_pairs = sum(1 for p in pair_results if p["is_identical"])
    near_identical_pairs = sum(1 for p in pair_results if p["is_near_identical"])
    representation_discriminated = total_pairs - near_identical_pairs
    rep_discrimination_rate = (representation_discriminated / total_pairs) * 100.0
    verdict_separated_pairs = sum(1 for p in pair_results if p["verdict_separated"])
    verdict_separation_rate = (verdict_separated_pairs / total_pairs) * 100.0
    
    print("\n=== SUMMARY METRICS ===")
    print(f"Total Minimal Pairs Evaluated: {total_pairs}")
    print(f"Identical Representation Pairs (L2 == 0): {identical_pairs} ({identical_pairs/total_pairs*100:.1f}%)")
    print(f"Near-Identical Pairs (L2 <= {NEAR_IDENTICAL_L2_THRESHOLD}): {near_identical_pairs} ({near_identical_pairs/total_pairs*100:.1f}%)")
    print(f"Representation Discrimination Rate: {rep_discrimination_rate:.1f}%")
    print(f"Verdict Separation Rate: {verdict_separation_rate:.1f}%")
    
    # Save Pair Analysis JSON
    with open(output_dir / "pair_analysis.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_pairs": total_pairs,
            "identical_pairs": identical_pairs,
            "near_identical_pairs": near_identical_pairs,
            "rep_discrimination_rate": rep_discrimination_rate,
            "verdict_separation_rate": verdict_separation_rate,
            "pairs": pair_results,
        }, f, indent=2)
    print(f"Saved pair analysis to {output_dir / 'pair_analysis.json'}")


if __name__ == "__main__":
    main()
