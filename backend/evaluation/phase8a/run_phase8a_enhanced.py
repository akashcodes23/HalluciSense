"""Phase 8A Enhanced P1 Evaluation Runner.

Evaluates the Enhanced P1 pipeline (with claim decomposition, deterministic numeric/unit checking,
negation detection, and causal direction analysis) on the SAME FROZEN 8A dataset (N=175).

Compares baseline vs enhanced metrics across categories and domains.
"""

from __future__ import annotations

import json
import time
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, auc, precision_recall_curve, brier_score_loss,
    confusion_matrix, matthews_corrcoef, balanced_accuracy_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.core.config import settings
settings.ENABLE_SELF_CONSISTENCY = False

from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.core.engine.claim_decomposition import ClaimDecomposer, AggregationStrategy, AtomicProposition
from app.core.engine.numeric_unit_checker import NumericUnitChecker, NumericUnitStatus
from app.core.engine.negation_detector import NegationDetector
from app.core.engine.causal_direction import CausalDirectionChecker

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PHASE8A_DIR = BACKEND_DIR / "reports" / "phase8" / "8A"
TRACES_DIR = PHASE8A_DIR / "traces_enhanced"
PLOTS_DIR = PHASE8A_DIR / "plots"
TRACES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = ["Physics", "Chemistry", "Biology", "Medicine", "Mathematics"]
CATEGORIES = [
    "TRUE_CONTROL", "NUMERICAL_PRECISION", "UNIT_SCALE", "NEGATION",
    "CAUSAL_INVERSION", "OUTDATED_SCIENTIFIC_CLAIM", "TRUE_CORE_FALSE_ELABORATION",
]


def evaluate_enhanced_claim(
    pipeline: HallucinationDetectionPipeline,
    decomposer: ClaimDecomposer,
    numeric_checker: NumericUnitChecker,
    negation_detector: NegationDetector,
    causal_checker: CausalDirectionChecker,
    claim: str,
    record: dict,
) -> dict:
    """Evaluates a single claim using the Enhanced P1 pipeline."""
    t_start = time.perf_counter()
    propositions = decomposer.decompose(claim)
    if not propositions:
        propositions = [AtomicProposition(text=claim, index=1, source_sentence=claim)]

    prop_results = []
    max_penalty = 0.0
    detected_enhancements = []

    for prop in propositions:
        # Run baseline P1 retrieval + NLI on atomic proposition
        base_res = pipeline.analyze_response(
            full_text=prop.text,
            query=f"Is the following scientific proposition accurate: {prop.text[:120]}",
        )
        base_h = float(getattr(base_res, "overall_h_score", 0.5))

        # Retrieve top evidence snippet for symbolic checks
        evidence_text = ""
        if hasattr(base_res, "sentence_analyses") and base_res.sentence_analyses:
            for s in base_res.sentence_analyses:
                if hasattr(s, "evidence_items") and s.evidence_items:
                    evidence_text = " ".join(e.snippet for e in s.evidence_items[:3] if hasattr(e, "snippet"))
                    break

        # 1. Numeric and unit check
        num_status, num_penalty, num_exp = numeric_checker.check_consistency(prop.text, evidence_text)
        if num_status in (NumericUnitStatus.NUMERIC_CONFLICT, NumericUnitStatus.SCALE_CONFLICT, NumericUnitStatus.UNIT_CONFLICT):
            detected_enhancements.append(f"NUMERIC_UNIT: {num_exp}")

        # 2. Negation and polarity check
        pol_res = negation_detector.analyze(prop.text, evidence_text)
        neg_penalty = pol_res.confidence_penalty
        if pol_res.negation_inversion_detected or pol_res.antonym_inversion_detected:
            detected_enhancements.append(f"NEGATION_POLARITY: {pol_res.explanation}")

        # 3. Causal direction check
        causal_res = causal_checker.check_inversion(prop.text, evidence_text)
        causal_penalty = causal_res.confidence_penalty
        if causal_res.is_inversion_detected:
            detected_enhancements.append(f"CAUSAL_DIRECTION: {causal_res.explanation}")

        # Combine NLI score with symbolic diagnostic penalties
        combined_prop_h = max(base_h, num_penalty, neg_penalty, causal_penalty)
        prop.h_score = combined_prop_h
        prop.is_contradiction = (combined_prop_h >= 0.50)
        prop_results.append({
            "proposition": prop.text,
            "base_nli_score": base_h,
            "numeric_penalty": num_penalty,
            "negation_penalty": neg_penalty,
            "causal_penalty": causal_penalty,
            "enhanced_score": combined_prop_h,
        })

    # Aggregate proposition scores using MAX_RISK strategy
    final_h_score = ClaimDecomposer.aggregate_scores(propositions, strategy=AggregationStrategy.MAX_RISK)
    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

    return {
        "enhanced_h_score": round(final_h_score, 4),
        "num_propositions": len(propositions),
        "proposition_details": prop_results,
        "enhancements_triggered": detected_enhancements,
        "latency_ms": latency_ms,
    }


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> dict:
    if len(y_true) == 0:
        return {"n": 0}
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_pred)) > 1 else 0.0
    try:
        auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        auroc = float("nan")
    try:
        p_arr, r_arr, _ = precision_recall_curve(y_true, y_prob)
        auprc = auc(r_arr, p_arr) if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        auprc = float("nan")
    brier = brier_score_loss(y_true, y_prob)

    return {
        "n": len(y_true), "threshold": threshold,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "f1_score": round(float(f1), 4),
        "balanced_accuracy": round(float(bal), 4),
        "mcc": round(float(mcc), 4),
        "auroc": round(float(auroc), 4) if not np.isnan(auroc) else None,
        "auprc": round(float(auprc), 4) if not np.isnan(auprc) else None,
        "brier_score": round(float(brier), 4),
    }


def main():
    print("Loading Enhanced P1 Components…")
    pipeline = HallucinationDetectionPipeline()
    decomposer = ClaimDecomposer()
    numeric_checker = NumericUnitChecker()
    negation_detector = NegationDetector()
    causal_checker = CausalDirectionChecker()
    print("  Enhanced P1 Pipeline Ready.")

    records = []
    with open(PHASE8A_DIR / "dataset_8a.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  Loaded {len(records)} frozen 8A records.")

    results = []
    t_start = time.perf_counter()

    for i, rec in enumerate(records, 1):
        if i % 10 == 0 or i == 1:
            print(f"  [{i:3d}/{len(records)}] domain={rec['domain']}, cat={rec['category']}, elapsed={time.perf_counter() - t_start:.1f}s")

        enh_res = evaluate_enhanced_claim(
            pipeline, decomposer, numeric_checker, negation_detector, causal_checker,
            rec["claim"], rec
        )
        gt = rec["ground_truth"]
        pred = 1 if enh_res["enhanced_h_score"] >= 0.50 else 0

        res_entry = {
            **rec,
            "enhanced_h_score": enh_res["enhanced_h_score"],
            "enhanced_predicted": pred,
            "num_propositions": enh_res["num_propositions"],
            "enhancements_triggered": enh_res["enhancements_triggered"],
            "latency_ms": enh_res["latency_ms"],
        }
        results.append(res_entry)

        # Save trace
        (TRACES_DIR / f"TRACE_ENHANCED_{i:04d}.json").write_text(
            json.dumps({
                "record_id": rec["id"],
                "claim": rec["claim"],
                "ground_truth": gt,
                "enhanced_h_score": enh_res["enhanced_h_score"],
                "predicted": pred,
                "proposition_details": enh_res["proposition_details"],
                "enhancements_triggered": enh_res["enhancements_triggered"],
                "latency_ms": enh_res["latency_ms"],
            }, indent=2)
        )

    df = pd.DataFrame(results)
    df.to_csv(PHASE8A_DIR / "enhanced_results.csv", index=False)

    # Compute overall and breakdown metrics
    y_true = df["ground_truth"].to_numpy(dtype=int)
    y_prob = df["enhanced_h_score"].to_numpy(dtype=float)
    overall_metrics = compute_metrics(y_true, y_prob)

    cat_rows = []
    for cat in CATEGORIES:
        sub = df[df["category"] == cat]
        sub_gt = sub["ground_truth"].to_numpy(dtype=int)
        sub_prob = sub["enhanced_h_score"].to_numpy(dtype=float)
        m = compute_metrics(sub_gt, sub_prob)
        cat_rows.append({"category": cat, **m})
    pd.DataFrame(cat_rows).to_csv(PHASE8A_DIR / "enhanced_category_breakdown.csv", index=False)

    dom_rows = []
    for dom in DOMAINS:
        sub = df[df["domain"] == dom]
        sub_gt = sub["ground_truth"].to_numpy(dtype=int)
        sub_prob = sub["enhanced_h_score"].to_numpy(dtype=float)
        m = compute_metrics(sub_gt, sub_prob)
        dom_rows.append({"domain": dom, **m})
    pd.DataFrame(dom_rows).to_csv(PHASE8A_DIR / "enhanced_domain_breakdown.csv", index=False)

    summary = {
        "experiment": "Phase8A_Enhanced_P1_Evaluation",
        "dataset_records": len(records),
        "overall_metrics": overall_metrics,
        "category_metrics": {r["category"]: r for r in cat_rows},
        "domain_metrics": {r["domain"]: r for r in dom_rows},
        "total_runtime_s": round(time.perf_counter() - t_start, 2),
    }
    (PHASE8A_DIR / "enhanced_metrics.json").write_text(json.dumps(summary, indent=2))

    print("\nPhase 8A Enhanced P1 Evaluation Complete.")
    print(f"  Accuracy: {overall_metrics['accuracy']:.4f}, Precision: {overall_metrics['precision']:.4f}, Recall: {overall_metrics['recall']:.4f}, F1: {overall_metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()
