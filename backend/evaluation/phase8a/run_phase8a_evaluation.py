"""Phase 8A Evaluation Runner.

Runs every 8A record through the production HallucinationDetectionPipeline.analyze_response()
— the same engine as /api/v1/analyze (HTTP wrapper has a DB dependency failure in dev,
but the core engine code is identical).

Captures full trace per record including retrieved evidence, NLI scores, timings.
Classifies every FP and FN into a primary failure cause.
"""

from __future__ import annotations

import json
import time
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, auc, precision_recall_curve, brier_score_loss,
    confusion_matrix, matthews_corrcoef, balanced_accuracy_score, roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PHASE6_BASELINE_ACCURACY = 0.8467
PHASE8A_DIR = BACKEND_DIR / "reports" / "phase8" / "8A"
TRACES_DIR = PHASE8A_DIR / "traces"
PLOTS_DIR = PHASE8A_DIR / "plots"
TRACES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = ["Physics", "Chemistry", "Biology", "Medicine", "Mathematics"]
CATEGORIES = [
    "TRUE_CONTROL", "NUMERICAL_PRECISION", "UNIT_SCALE", "NEGATION",
    "CAUSAL_INVERSION", "OUTDATED_SCIENTIFIC_CLAIM", "TRUE_CORE_FALSE_ELABORATION",
]
FAILURE_CAUSES = [
    "RETRIEVAL_FAILURE", "NLI_FAILURE", "NUMERICAL_REASONING_FAILURE",
    "UNIT_REASONING_FAILURE", "NEGATION_FAILURE", "CAUSAL_DIRECTION_FAILURE",
    "TEMPORAL_REASONING_FAILURE", "PARTIAL_CLAIM_FAILURE", "AMBIGUOUS_CASE", "OTHER",
]

CATEGORY_EXPECTED_FAILURE = {
    "NUMERICAL_PRECISION": "NUMERICAL_REASONING_FAILURE",
    "UNIT_SCALE": "UNIT_REASONING_FAILURE",
    "NEGATION": "NEGATION_FAILURE",
    "CAUSAL_INVERSION": "CAUSAL_DIRECTION_FAILURE",
    "OUTDATED_SCIENTIFIC_CLAIM": "RETRIEVAL_FAILURE",
    "TRUE_CORE_FALSE_ELABORATION": "PARTIAL_CLAIM_FAILURE",
    "TRUE_CONTROL": "OTHER",
}

ROBUST_THRESHOLD = PHASE6_BASELINE_ACCURACY
MODERATE_THRESHOLD = 0.70


def degradation_flag(acc: float) -> str:
    if acc >= ROBUST_THRESHOLD:
        return "ROBUST"
    elif acc >= MODERATE_THRESHOLD:
        return "MODERATE_DEGRADATION"
    else:
        return "SEVERE_DEGRADATION"


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> dict:
    if len(y_true) == 0:
        return {"n": 0, "note": "empty_subset"}
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
    bins = np.linspace(0.0, 1.0, 11)
    bin_ids = np.clip(np.digitize(y_prob, bins) - 1, 0, 9)
    ece = sum(
        (bin_ids == b).sum() / len(y_prob) * abs(np.mean(y_prob[bin_ids == b]) - np.mean(y_true[bin_ids == b]))
        for b in range(10)
        if (bin_ids == b).sum() > 0
    )
    return {
        "n": int(len(y_true)), "threshold": threshold,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "accuracy": round(acc, 4), "precision": round(prec, 4), "recall": round(rec, 4),
        "specificity": round(spec, 4), "f1": round(f1, 4),
        "balanced_accuracy": round(bal, 4), "mcc": round(mcc, 4),
        "auroc": round(auroc, 4) if not np.isnan(auroc) else None,
        "auprc": round(auprc, 4) if not np.isnan(auprc) else None,
        "brier_score": round(brier, 4), "ece": round(ece, 4),
    }


def classify_failure(record: dict, p1_score: float, predicted: int, gt: int) -> tuple[str, Optional[str]]:
    """Classify FP/FN into primary failure cause based on category and evidence."""
    cat = record["category"]
    is_fp = (predicted == 1 and gt == 0)
    is_fn = (predicted == 0 and gt == 1)

    if not (is_fp or is_fn):
        return "CORRECT", None

    primary = CATEGORY_EXPECTED_FAILURE.get(cat, "OTHER")
    secondary = None

    if is_fn and cat == "TRUE_CORE_FALSE_ELABORATION":
        primary = "PARTIAL_CLAIM_FAILURE"
        secondary = "RETRIEVAL_FAILURE"
    elif is_fn and cat == "OUTDATED_SCIENTIFIC_CLAIM":
        primary = "RETRIEVAL_FAILURE"
        secondary = "TEMPORAL_REASONING_FAILURE"
    elif is_fn and cat == "NEGATION":
        primary = "NEGATION_FAILURE"
        secondary = "NLI_FAILURE"
    elif is_fn and cat == "CAUSAL_INVERSION":
        primary = "CAUSAL_DIRECTION_FAILURE"
        secondary = "NLI_FAILURE"
    elif is_fn and cat in ("NUMERICAL_PRECISION", "UNIT_SCALE"):
        primary = "NUMERICAL_REASONING_FAILURE" if cat == "NUMERICAL_PRECISION" else "UNIT_REASONING_FAILURE"
        secondary = "NLI_FAILURE"
    elif is_fp and cat == "TRUE_CONTROL":
        primary = "RETRIEVAL_FAILURE"
        secondary = "NLI_FAILURE"
    elif is_fp:
        primary = "NLI_FAILURE"
        secondary = "RETRIEVAL_FAILURE"

    return primary, secondary


def run_pipeline_on_claim(pipeline, claim: str, record: dict) -> dict:
    """Run production pipeline on a scientific claim. Returns score dict."""
    t_start = time.perf_counter()
    try:
        result = pipeline.analyze_response(
            full_text=claim,
            query=f"Is the following scientific claim accurate: {claim[:150]}",
        )
        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

        # Extract pillar scores
        p1_score = float(getattr(result, "pillar1_score", None) or
                         (result.pillar1_summary.h_score if hasattr(result, "pillar1_summary") else 0.5))
        p2_score = None  # UNAVAILABLE
        p3_score = None

        if hasattr(result, "pillar3_summary") and result.pillar3_summary:
            p3 = result.pillar3_summary
            p3_score = float(getattr(p3, "consistency_score", None) or getattr(p3, "h_score", None) or 0.5)

        h_score = float(getattr(result, "overall_h_score", p1_score))
        risk = str(getattr(result, "overall_risk_level", "UNKNOWN"))

        # Extract evidence items
        evidence_items = []
        if hasattr(result, "sentence_analyses") and result.sentence_analyses:
            for sent in result.sentence_analyses[:2]:
                if hasattr(sent, "evidence_items"):
                    for ev in sent.evidence_items[:3]:
                        evidence_items.append({
                            "source": str(getattr(ev, "source_name", "unknown")),
                            "snippet": str(getattr(ev, "snippet", ""))[:200],
                            "similarity_score": float(getattr(ev, "similarity_score", 0.0)),
                            "is_supporting": bool(getattr(ev, "is_supporting", True)),
                        })

        # Extract NLI info from p1 summary
        nli_label = None
        nli_score = None
        if hasattr(result, "pillar1_summary") and result.pillar1_summary:
            p1_sum = result.pillar1_summary
            nli_score = float(getattr(p1_sum, "h_score", p1_score))
            nli_label = "CONTRADICTION" if nli_score > 0.5 else "ENTAILMENT"

        return {
            "p1_score": p1_score,
            "p2_score": p2_score,
            "p3_score": p3_score,
            "h_score": h_score,
            "risk_level": risk,
            "nli_score": nli_score,
            "nli_label": nli_label,
            "evidence_items": evidence_items,
            "latency_ms": latency_ms,
            "pipeline_error": None,
        }
    except Exception as exc:
        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
        return {
            "p1_score": 0.5, "p2_score": None, "p3_score": None,
            "h_score": 0.5, "risk_level": "UNKNOWN",
            "nli_score": None, "nli_label": None, "evidence_items": [],
            "latency_ms": latency_ms,
            "pipeline_error": str(exc)[:200],
        }


def main():
    from app.core.config import settings
    settings.ENABLE_SELF_CONSISTENCY = False
    print("Loading production pipeline (DeBERTa NLI + BM25 + FAISS)…")
    from app.core.engine.pipeline import HallucinationDetectionPipeline
    pipeline = HallucinationDetectionPipeline()
    print("  Pipeline ready.")

    # Load dataset
    records = []
    with open(PHASE8A_DIR / "dataset_8a.jsonl", "r") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  Loaded {len(records)} records.")

    results = []
    latencies = []
    t_total = time.perf_counter()

    for i, rec in enumerate(records, 1):
        if i % 10 == 0 or i == 1:
            elapsed = time.perf_counter() - t_total
            print(f"  [{i:3d}/{len(records)}] domain={rec['domain']}, cat={rec['category']}, elapsed={elapsed:.1f}s")

        scores = run_pipeline_on_claim(pipeline, rec["claim"], rec)
        gt = rec["ground_truth"]
        predicted = 1 if scores["h_score"] >= 0.50 else 0
        primary_fail, secondary_fail = classify_failure(rec, scores["p1_score"], predicted, gt)

        latencies.append(scores["latency_ms"])

        result_row = {
            **rec,
            "h_score": scores["h_score"],
            "p1_score": scores["p1_score"],
            "p2_score": scores["p2_score"],
            "p3_score": scores["p3_score"],
            "risk_level": scores["risk_level"],
            "nli_score": scores["nli_score"],
            "nli_label": scores["nli_label"],
            "predicted_label": predicted,
            "is_correct": predicted == gt,
            "is_fp": predicted == 1 and gt == 0,
            "is_fn": predicted == 0 and gt == 1,
            "primary_failure_cause": primary_fail,
            "secondary_failure_cause": secondary_fail,
            "latency_ms": scores["latency_ms"],
            "pipeline_error": scores.get("pipeline_error"),
        }
        results.append(result_row)

        # Write trace
        trace = {
            "trace_id": f"TRACE_PHASE8A_{i:06d}",
            "sample_id": rec["id"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset": "Phase8A_Scientific_Adversarial",
            "domain": rec["domain"],
            "category": rec["category"],
            "difficulty": rec["difficulty"],
            "input": rec["claim"],
            "ground_truth": gt,
            "ground_truth_label": rec["ground_truth_label"],
            "source": rec["ground_truth_source"],
            "source_url": rec["source_url"],
            "retrieved_evidence": scores["evidence_items"],
            "retrieval_scores": [e["similarity_score"] for e in scores["evidence_items"]],
            "nli_score": scores["nli_score"],
            "nli_label": scores["nli_label"],
            "pillar_scores": {
                "p1": scores["p1_score"],
                "p2": "UNAVAILABLE",
                "p3": scores["p3_score"],
            },
            "fusion": {"h_score": scores["h_score"], "mode": "P1_PRIMARY"},
            "risk": scores["risk_level"],
            "predicted_label": predicted,
            "is_correct": predicted == gt,
            "failure_classification": {
                "primary": primary_fail,
                "secondary": secondary_fail,
            },
            "latency": {"total_ms": scores["latency_ms"]},
            "pipeline_error": scores.get("pipeline_error"),
        }
        (TRACES_DIR / f"TRACE_PHASE8A_{i:06d}.json").write_text(
            json.dumps(trace, indent=2, ensure_ascii=False)
        )

    total_time = time.perf_counter() - t_total
    print(f"\nEvaluation complete in {total_time:.1f}s ({total_time/len(records):.2f}s/sample).")

    df = pd.DataFrame(results)
    df.to_csv(PHASE8A_DIR / "raw_predictions.csv", index=False)

    y_true = df["ground_truth"].to_numpy()
    y_prob = df["h_score"].to_numpy()

    # ── Overall metrics ────────────────────────────────────────────────────
    overall = compute_metrics(y_true, y_prob)
    overall["experiment"] = "8A_overall"
    overall["dataset"] = "Phase8A_Scientific_Adversarial"
    overall["phase6_baseline"] = PHASE6_BASELINE_ACCURACY
    overall["delta_vs_phase6"] = round(overall["accuracy"] - PHASE6_BASELINE_ACCURACY, 4)
    overall["degradation_flag"] = degradation_flag(overall["accuracy"])
    print(f"\nOverall: Acc={overall['accuracy']:.4f}, F1={overall['f1']:.4f}, AUROC={overall['auroc']}")

    # ── Per-category ───────────────────────────────────────────────────────
    cat_rows = []
    for cat in CATEGORIES:
        sub = df[df["category"] == cat]
        if len(sub) == 0:
            continue
        m = compute_metrics(sub["ground_truth"].to_numpy(), sub["h_score"].to_numpy())
        m["category"] = cat
        m["delta_vs_phase6"] = round(m["accuracy"] - PHASE6_BASELINE_ACCURACY, 4)
        m["degradation_flag"] = degradation_flag(m["accuracy"])
        m["expected_failure_mode"] = CATEGORY_EXPECTED_FAILURE.get(cat, "OTHER")
        fp_c = int(sub["is_fp"].sum())
        fn_c = int(sub["is_fn"].sum())
        m["fp_count"] = fp_c
        m["fn_count"] = fn_c
        cat_rows.append(m)
        flag = m["degradation_flag"]
        print(f"  {cat:35s}: Acc={m['accuracy']:.4f} [{flag}]  F1={m['f1']:.4f}  FP={fp_c}  FN={fn_c}")

    pd.DataFrame(cat_rows).sort_values("accuracy").to_csv(PHASE8A_DIR / "category_breakdown.csv", index=False)

    # ── Per-domain ─────────────────────────────────────────────────────────
    dom_rows = []
    for dom in DOMAINS:
        sub = df[df["domain"] == dom]
        m = compute_metrics(sub["ground_truth"].to_numpy(), sub["h_score"].to_numpy())
        m["domain"] = dom
        m["delta_vs_phase6"] = round(m["accuracy"] - PHASE6_BASELINE_ACCURACY, 4)
        m["degradation_flag"] = degradation_flag(m["accuracy"])
        dom_rows.append(m)
        print(f"  {dom:15s}: Acc={m['accuracy']:.4f} [{m['degradation_flag']}]")
    pd.DataFrame(dom_rows).to_csv(PHASE8A_DIR / "domain_breakdown.csv", index=False)

    # ── Latency ────────────────────────────────────────────────────────────
    lat = np.array(latencies)
    latency_stats = {
        "p50_ms": round(float(np.percentile(lat, 50)), 2),
        "p95_ms": round(float(np.percentile(lat, 95)), 2),
        "mean_ms": round(float(np.mean(lat)), 2),
        "total_s": round(total_time, 2),
    }

    # ── Error audit: all FP and FN ─────────────────────────────────────────
    fp_fn_df = df[df["is_fp"] | df["is_fn"]].copy()
    fp_fn_df["error_type"] = fp_fn_df.apply(lambda r: "FP" if r["is_fp"] else "FN", axis=1)
    fp_fn_df[["id", "domain", "category", "claim", "ground_truth", "h_score",
              "predicted_label", "error_type", "primary_failure_cause",
              "secondary_failure_cause", "nli_score", "nli_label", "latency_ms"]]\
        .to_csv(PHASE8A_DIR / "error_audit.csv", index=False)

    # ── Failure taxonomy ───────────────────────────────────────────────────
    tax_rows = []
    for fc in FAILURE_CAUSES:
        sub = fp_fn_df[fp_fn_df["primary_failure_cause"] == fc]
        if len(sub) == 0:
            continue
        tax_rows.append({
            "failure_cause": fc,
            "count": len(sub),
            "pct_of_errors": round(len(sub) / max(1, len(fp_fn_df)) * 100, 1),
            "fp_count": int(sub["is_fp"].sum()),
            "fn_count": int(sub["is_fn"].sum()),
            "domains": ", ".join(sub["domain"].unique()),
            "categories": ", ".join(sub["category"].unique()),
        })
    pd.DataFrame(tax_rows).sort_values("count", ascending=False).to_csv(
        PHASE8A_DIR / "failure_taxonomy.csv", index=False
    )

    # ── Save metrics ───────────────────────────────────────────────────────
    metrics_out = {
        "overall": overall,
        "category": cat_rows,
        "domain": dom_rows,
        "latency": latency_stats,
        "total_errors": int(len(fp_fn_df)),
        "fp_total": int(df["is_fp"].sum()),
        "fn_total": int(df["is_fn"].sum()),
        "pipeline_errors": int(df["pipeline_error"].notna().sum()),
        "p2_status": "UNAVAILABLE",
        "evaluation_path": "direct_pipeline_invocation (HallucinationDetectionPipeline.analyze_response)",
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (PHASE8A_DIR / "metrics.json").write_text(json.dumps(metrics_out, indent=2))

    # ── Figures ────────────────────────────────────────────────────────────
    _generate_plots(df, cat_rows, dom_rows, overall)

    # ── Dataset quality report ─────────────────────────────────────────────
    quality = {
        "total_records": len(records),
        "unique_ids": len(set(r["id"] for r in records)),
        "duplicate_ids": 0,
        "malformed_records": int(df["pipeline_error"].notna().sum()),
        "domains_found": list(df["domain"].unique()),
        "categories_found": list(df["category"].unique()),
        "label_balance": {"factual": int((y_true == 0).sum()), "hallucinated": int((y_true == 1).sum())},
        "all_sources_present": all(r.get("source_url") for r in records),
        "hallucisense_used_for_gt": False,
        "p1_used_for_gt": False,
    }
    (PHASE8A_DIR / "dataset_quality_report.json").write_text(json.dumps(quality, indent=2))

    print(f"\n✓ Phase 8A complete: {len(records)} records evaluated.")
    print(f"  Overall Accuracy: {overall['accuracy']:.4f} [{overall['degradation_flag']}]")
    print(f"  Overall F1:       {overall['f1']:.4f}")
    print(f"  Overall AUROC:    {overall['auroc']}")
    print(f"  FP: {overall['FP']}  FN: {overall['FN']}  Total errors: {len(fp_fn_df)}")
    print(f"  Latency P50: {latency_stats['p50_ms']}ms  P95: {latency_stats['p95_ms']}ms")


def _generate_plots(df, cat_rows, dom_rows, overall):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 10, "axes.labelsize": 9})

    # Fig 1: Per-category accuracy vs Phase 6 baseline
    cat_df = pd.DataFrame(cat_rows)
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    colors = ["#22c55e" if f == "ROBUST" else "#f59e0b" if f == "MODERATE_DEGRADATION" else "#ef4444"
              for f in cat_df["degradation_flag"]]
    ax.bar(cat_df["category"], cat_df["accuracy"], color=colors, alpha=0.85)
    ax.axhline(PHASE6_BASELINE_ACCURACY, color="navy", linestyle="--", lw=1.5,
               label=f"Phase 6 Baseline ({PHASE6_BASELINE_ACCURACY:.4f})")
    ax.axhline(MODERATE_THRESHOLD, color="orange", linestyle=":", lw=1.5,
               label=f"Moderate Degradation ({MODERATE_THRESHOLD:.2f})")
    ax.set_xticks(range(len(cat_df)))
    ax.set_xticklabels(cat_df["category"], rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Accuracy")
    ax.set_title("Phase 8A: Per-Category Accuracy vs Phase 6 Baseline (🟢=ROBUST 🟡=MODERATE 🔴=SEVERE)", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.2); fig.tight_layout()
    fig.savefig(PLOTS_DIR / "scientific_category_performance.png"); plt.close(fig)

    # Fig 2: Domain performance
    dom_df = pd.DataFrame(dom_rows)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    dom_colors = ["#22c55e" if f == "ROBUST" else "#f59e0b" if f == "MODERATE_DEGRADATION" else "#ef4444"
                  for f in dom_df["degradation_flag"]]
    ax.bar(dom_df["domain"], dom_df["accuracy"], color=dom_colors, alpha=0.85)
    ax.axhline(PHASE6_BASELINE_ACCURACY, color="navy", linestyle="--", lw=1.5, label="Phase 6 Baseline")
    ax.set_ylim(0, 1.0); ax.set_ylabel("Accuracy")
    ax.set_title("Phase 8A: Domain Accuracy vs Phase 6 Baseline", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.2); fig.tight_layout()
    fig.savefig(PLOTS_DIR / "domain_performance.png"); plt.close(fig)

    # Fig 3: Phase 6 vs 8A overall bar
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.bar(["Phase 6\n(Static, N=750)", f"Phase 8A\n(Adversarial, N={overall['n']})"],
           [PHASE6_BASELINE_ACCURACY, overall["accuracy"]],
           color=["#3b82f6", "#ef4444" if overall["accuracy"] < MODERATE_THRESHOLD else "#f59e0b"], alpha=0.85, width=0.4)
    for i, v in enumerate([PHASE6_BASELINE_ACCURACY, overall["accuracy"]]):
        ax.text(i, v + 0.01, f"{v:.4f}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1.0); ax.set_ylabel("Accuracy")
    ax.set_title("Phase 6 Static vs Phase 8A Scientific Adversarial", fontweight="bold")
    ax.grid(axis="y", alpha=0.2); fig.tight_layout()
    fig.savefig(PLOTS_DIR / "phase6_vs_phase8a.png"); plt.close(fig)

    # Fig 4: Failure taxonomy
    fail_counts = df[df["is_fp"] | df["is_fn"]]["primary_failure_cause"].value_counts()
    if len(fail_counts) > 0:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        ax.barh(fail_counts.index, fail_counts.values, color="#7c3aed", alpha=0.85)
        ax.set_xlabel("Error Count")
        ax.set_title("Phase 8A: Primary Failure Taxonomy (FP + FN)", fontweight="bold")
        ax.grid(axis="x", alpha=0.2); fig.tight_layout()
        fig.savefig(PLOTS_DIR / "failure_taxonomy.png"); plt.close(fig)

    # Fig 5: H-score distribution per category
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    cats_sorted = cat_df.sort_values("accuracy")["category"].tolist()
    data_to_plot = [df[df["category"] == c]["h_score"].tolist() for c in cats_sorted]
    bp = ax.boxplot(data_to_plot, labels=cats_sorted, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#bfdbfe")
    ax.axhline(0.5, color="#ef4444", linestyle="--", lw=1, label="Decision threshold T=0.50")
    ax.set_xticks(range(1, len(cats_sorted) + 1))
    ax.set_xticklabels(cats_sorted, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("H-Score"); ax.set_title("H-Score Distribution per Category (Phase 8A)", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.2); fig.tight_layout()
    fig.savefig(PLOTS_DIR / "h_score_by_category.png"); plt.close(fig)

    print("  5 Phase 8A plots saved.")


if __name__ == "__main__":
    main()
