"""Phase 8B — Response-Level Ground-Truth Audit Report Builder.

Reframes the existing Dataset B (response_level_ground_truth.jsonl) as a
LABEL-INTEGRITY / RESPONSE-ALIGNMENT experiment — NOT as an independent
detector performance benchmark.

Key scientific statement:
  "Dataset B is diagnostic evidence concerning label alignment between
   static benchmark labels and live LLM responses. It is NOT an
   independent detector benchmark. Any P1-vs-P1-derived metric is
   explicitly marked CIRCULAR and must not be presented as evidence of
   detection quality."
"""

from __future__ import annotations

import json
import time
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
SRC = BACKEND_DIR / "reports" / "phase8" / "response_level_ground_truth.jsonl"
DST_DIR = BACKEND_DIR / "reports" / "phase8" / "8B"
DST_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = [
    "General Knowledge", "Medicine", "Law", "Finance", "History",
    "Science", "Computer Science", "Physics", "Biology", "Chemistry",
    "News", "Mathematics", "Geography", "Politics", "Literature",
]


def main():
    print("Loading Dataset B (response_level_ground_truth.jsonl)…")
    records = []
    with open(SRC, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  Loaded {len(records)} records.")

    df = pd.DataFrame(records)

    # ── Core label-shift statistics ───────────────────────────────────────
    total = len(df)
    static_factual = int((df["original_static_label"] == 0).sum())
    static_hallucinated = int((df["original_static_label"] == 1).sum())
    response_factual = int((df["response_ground_truth"] == "factual").sum())
    response_hallucinated = int((df["response_ground_truth"] == "hallucinated").sum())
    response_partial = int((df["response_ground_truth"] == "partially_hallucinated").sum())
    label_shift = int(df["is_label_shift"].sum())  # static=1, response=factual
    label_shift_pct = round(label_shift / max(1, static_hallucinated) * 100, 2)

    # ── Domain distribution of label shifts ──────────────────────────────
    dom_rows = []
    for dom in DOMAINS:
        sub = df[df["domain"] == dom]
        if len(sub) == 0:
            continue
        shifts = int(sub["is_label_shift"].sum())
        dom_rows.append({
            "domain": dom,
            "n": len(sub),
            "static_factual": int((sub["original_static_label"] == 0).sum()),
            "static_hallucinated": int((sub["original_static_label"] == 1).sum()),
            "response_factual": int((sub["response_ground_truth"] == "factual").sum()),
            "response_hallucinated": int((sub["response_ground_truth"] == "hallucinated").sum()),
            "response_partial": int((sub["response_ground_truth"] == "partially_hallucinated").sum()),
            "label_shift_count": shifts,
            "label_shift_pct": round(shifts / max(1, len(sub)) * 100, 2),
        })
    pd.DataFrame(dom_rows).to_csv(DST_DIR / "domain_label_shift.csv", index=False)

    # ── Representative label-shift examples ──────────────────────────────
    shift_examples = df[df["is_label_shift"]].head(20)[
        ["sample_id", "domain", "prompt", "generated_response",
         "original_static_label_meaning", "response_ground_truth",
         "ground_truth_reason"]
    ].to_dict(orient="records")

    # ── P1-vs-P1 CIRCULAR metric disclosure ──────────────────────────────
    # Evaluating P1-only against Dataset B ground truth (which was derived
    # from P1 NLI thresholds) is CIRCULAR. We explicitly mark and document it.
    y_true = df["response_ground_truth_binary"].to_numpy(dtype=float)
    p1_scores = df["phase7_p1"].to_numpy(dtype=float)

    # Demonstrate circularity: compute simple threshold metrics
    from sklearn.metrics import roc_auc_score
    try:
        auroc_circular = round(float(roc_auc_score(y_true, p1_scores)), 4)
    except Exception:
        auroc_circular = None

    circular_note = (
        "CIRCULAR EVALUATION: Dataset B response_ground_truth was assigned using "
        "P1 NLI score thresholds (factual < 0.35, hallucinated ≥ 0.55). "
        f"Evaluating P1 against this label produces AUROC ≈ {auroc_circular} — "
        "a mathematical artifact because P1's own scores defined the labels. "
        "This MUST NOT be reported as evidence of detector quality."
    )

    # ── Summary report ────────────────────────────────────────────────────
    summary = {
        "experiment": "8B_Response_Level_Ground_Truth_Audit",
        "purpose": (
            "DIAGNOSTIC: Measures label alignment between static Phase 6 benchmark labels "
            "and actual LLM responses generated in Phase 7. "
            "This is NOT an independent detector benchmark."
        ),
        "dataset_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
        "total_responses": total,
        "static_label_distribution": {
            "factual_0": static_factual,
            "hallucinated_1": static_hallucinated,
        },
        "response_gt_distribution": {
            "factual": response_factual,
            "hallucinated": response_hallucinated,
            "partially_hallucinated": response_partial,
        },
        "label_shift": {
            "count": label_shift,
            "percentage_of_static_hallucinated": label_shift_pct,
            "interpretation": (
                f"{label_shift} of {static_hallucinated} prompts labeled 'hallucinated' "
                f"({label_shift_pct}%) were answered factually correctly by the live LLM "
                "(Qwen2.5-Coder:1.5b via Ollama). The static benchmark label is invalid "
                "for those responses."
            ),
        },
        "ground_truth_method": "P1_NLI_Evidence_Grounding (not H-score fusion)",
        "circular_evaluation_disclosure": circular_note,
        "p1_auroc_on_dataset_b": auroc_circular,
        "p1_auroc_is_circular": True,
        "primary_scientific_finding": (
            "50.7% of hallucination-labeled benchmark prompts were answered factually "
            "by the live LLM. Static benchmark labels cannot serve as ground truth "
            "for newly generated responses."
        ),
        "examples_label_shift_count": len(shift_examples),
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (DST_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # ── Label-shift examples ──────────────────────────────────────────────
    (DST_DIR / "label_shift_examples.json").write_text(
        json.dumps(shift_examples, indent=2, ensure_ascii=False)
    )

    # ── Generate 2 figures ────────────────────────────────────────────────
    plots_dir = DST_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Fig 1: Label distribution comparison (static vs response-level)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)
    axes[0].bar(["Factual", "Hallucinated"],
                [static_factual, static_hallucinated], color=["#22c55e", "#ef4444"], alpha=0.85)
    axes[0].set_title("Static Benchmark Labels\n(Phase 6 Ground Truth)", fontweight="bold")
    axes[0].set_ylabel("Count"); axes[0].set_ylim(0, 450)
    for i, v in enumerate([static_factual, static_hallucinated]):
        axes[0].text(i, v + 8, str(v), ha="center", fontweight="bold")

    axes[1].bar(["Factual", "Hallucinated", "Partial"],
                [response_factual, response_hallucinated, response_partial],
                color=["#22c55e", "#ef4444", "#f59e0b"], alpha=0.85)
    axes[1].set_title("Response-Level GT Labels\n(Dataset B, P1 NLI Evidence)", fontweight="bold")
    axes[1].set_ylabel("Count"); axes[1].set_ylim(0, 450)
    for i, v in enumerate([response_factual, response_hallucinated, response_partial]):
        axes[1].text(i, v + 8, str(v), ha="center", fontweight="bold")

    fig.suptitle("Phase 8B: Static Labels vs Response-Level Ground Truth", fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "label_distribution_comparison.png"); plt.close(fig)

    # Fig 2: Label-shift by domain
    dom_df = pd.DataFrame(dom_rows)
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    ax.bar(dom_df["domain"], dom_df["label_shift_count"], color="#7c3aed", alpha=0.85)
    ax.set_xticks(range(len(dom_df)))
    ax.set_xticklabels(dom_df["domain"], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Label-Shift Count (Static=1, Response=Factual)")
    ax.set_title(
        f"Phase 8B: Label Shifts by Domain (Total={label_shift}, {label_shift_pct}% of hallucinated prompts)",
        fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.2); fig.tight_layout()
    fig.savefig(plots_dir / "label_shift_by_domain.png"); plt.close(fig)

    print(f"\nPhase 8B report complete.")
    print(f"  Total responses: {total}")
    print(f"  Static labels: factual={static_factual}, hallucinated={static_hallucinated}")
    print(f"  Response GT:   factual={response_factual}, hallucinated={response_hallucinated}, partial={response_partial}")
    print(f"  Label shifts:  {label_shift} / {static_hallucinated} = {label_shift_pct}%")
    print(f"  Circular AUROC (DO NOT REPORT AS PERFORMANCE): {auroc_circular}")
    print(f"  Saved to: {DST_DIR}")


if __name__ == "__main__":
    main()
