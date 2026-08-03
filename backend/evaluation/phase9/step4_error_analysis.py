"""
HalluciSense Phase 9 — Step 4: Error Analysis
==============================================
Systematic analysis of TP/TN/FP/FN prediction errors.
Failure pattern clustering and weakness identification.

FROZEN FIREWALL: No models, scalers, or thresholds are modified.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler as SS

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
PHASE6K = ROOT / "evaluation_results" / "phase6k"
FINAL_MODEL_DIR = PHASE6K / "final_model"
PHASE6I = ROOT / "evaluation_results" / "phase6i"
PUB = PHASE6K / "publication"
PUB.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = [
    "mean_entailment",
    "max_entailment",
    "mean_contradiction",
    "min_support_margin",
    "num_claims",
]
FEATURE_LABELS = [
    "Mean Entail.",
    "Max Entail.",
    "Mean Contra.",
    "Min Sup. Margin",
    "Num Claims",
]
OPERATING_THRESHOLD = 0.56
DPI = 300
QUADRANT_COLORS = {"TP": "#2c7bb6", "TN": "#1a9641", "FP": "#d7191c", "FN": "#fdae61"}


def load_val():
    rows, labels = [], []
    with open(PHASE6I / "claim_evidence_features_validation.jsonl") as f:
        for line in f:
            obj = json.loads(line.strip()) if line.strip() else {}
            rows.append([obj.get(fn, float("nan")) for fn in FEATURE_NAMES])
            labels.append(int(obj.get("ground_truth", 0)))
    return np.array(rows, dtype=np.float64), np.array(labels, dtype=np.int32)


def feature_stats(X: np.ndarray) -> dict:
    stats = {}
    for i, fn in enumerate(FEATURE_NAMES):
        col = X[:, i]
        stats[fn] = {
            "mean": float(col.mean()),
            "std": float(col.std()),
            "min": float(col.min()),
            "p25": float(np.percentile(col, 25)),
            "median": float(np.median(col)),
            "p75": float(np.percentile(col, 75)),
            "max": float(col.max()),
        }
    return stats


def run() -> None:
    print("=" * 70)
    print("HalluciSense Phase 9 — Step 4: Error Analysis")
    print("=" * 70)
    t0 = time.time()

    model = joblib.load(FINAL_MODEL_DIR / "pillar1_logistic_model.joblib")
    scaler = joblib.load(FINAL_MODEL_DIR / "robust_scaler.joblib")

    X_val, y_val = load_val()
    X_val_scaled = scaler.transform(X_val)
    probs = model.predict_proba(X_val_scaled)[:, 1]
    preds = (probs >= OPERATING_THRESHOLD).astype(int)

    # Quadrant masks
    tp_mask = (preds == 1) & (y_val == 1)
    tn_mask = (preds == 0) & (y_val == 0)
    fp_mask = (preds == 1) & (y_val == 0)
    fn_mask = (preds == 0) & (y_val == 1)

    masks = {"TP": tp_mask, "TN": tn_mask, "FP": fp_mask, "FN": fn_mask}

    print(f"\n  Quadrant sizes: " + " | ".join(
        f"{q}={m.sum()}" for q, m in masks.items()
    ))

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "9_step4_error_analysis",
        "quadrant_counts": {q: int(m.sum()) for q, m in masks.items()},
        "quadrant_feature_stats": {},
        "quadrant_confidence_stats": {},
        "failure_clusters": {},
        "systematic_weaknesses": [],
        "recommendations": [],
    }

    # ── 1. Per-quadrant feature distribution ─────────────────────────────────
    print("\n[1/4] Per-quadrant feature distribution analysis...")
    for q, mask in masks.items():
        if mask.sum() == 0:
            continue
        Xq = X_val[mask]
        pq = probs[mask]
        report["quadrant_feature_stats"][q] = feature_stats(Xq)
        report["quadrant_confidence_stats"][q] = {
            "prob_mean": float(pq.mean()),
            "prob_std": float(pq.std()),
            "prob_min": float(pq.min()),
            "prob_max": float(pq.max()),
            "prob_p25": float(np.percentile(pq, 25)),
            "prob_median": float(np.median(pq)),
            "prob_p75": float(np.percentile(pq, 75)),
            "count": int(mask.sum()),
        }

    # ── 2. K-means clustering of FP and FN ───────────────────────────────────
    print("\n[2/4] K-means clustering of failure patterns (k=3)...")
    N_CLUSTERS = 3
    for err_q, err_label in [("FP", "False Positives"), ("FN", "False Negatives")]:
        mask = masks[err_q]
        if mask.sum() < N_CLUSTERS:
            report["failure_clusters"][err_q] = {"error": "Insufficient samples"}
            continue
        Xerr = X_val[mask]
        Perr = probs[mask]

        # Normalize for clustering
        ss_err = SS()
        Xerr_norm = ss_err.fit_transform(Xerr)

        km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
        km.fit(Xerr_norm)
        cluster_labels = km.labels_

        clusters = []
        for cid in range(N_CLUSTERS):
            cmask = cluster_labels == cid
            Xc = Xerr[cmask]
            Pc = Perr[cmask]
            # Characterize cluster
            center_raw = ss_err.inverse_transform(km.cluster_centers_[cid:cid+1])[0]
            dominant_feature = FEATURE_NAMES[
                np.argmax(np.abs(km.cluster_centers_[cid]))
            ]
            clusters.append({
                "cluster_id": cid,
                "size": int(cmask.sum()),
                "size_pct": round(float(cmask.sum()) / mask.sum() * 100, 1),
                "dominant_feature": dominant_feature,
                "center_raw": dict(zip(FEATURE_NAMES, center_raw.tolist())),
                "mean_probability": float(Pc.mean()),
                "std_probability": float(Pc.std()),
                "feature_means": {fn: float(Xc[:, i].mean())
                                  for i, fn in enumerate(FEATURE_NAMES)},
            })
        clusters.sort(key=lambda c: -c["size"])
        report["failure_clusters"][err_q] = {
            "n_clusters": N_CLUSTERS,
            "total_errors": int(mask.sum()),
            "clusters": clusters,
        }
        print(f"  {err_label}: {mask.sum()} samples → {N_CLUSTERS} clusters")

    # ── 3. Systematic weakness identification ─────────────────────────────────
    print("\n[3/4] Identifying systematic weaknesses...")
    weaknesses = []

    # Compare FP vs TN feature means
    if masks["FP"].sum() > 0 and masks["TN"].sum() > 0:
        for i, fn in enumerate(FEATURE_NAMES):
            fp_mean = float(X_val[fp_mask, i].mean())
            tn_mean = float(X_val[tn_mask, i].mean())
            diff = abs(fp_mean - tn_mean)
            if diff > 0.1:
                weaknesses.append({
                    "quadrant_pair": "FP vs TN",
                    "feature": fn,
                    "fp_mean": round(fp_mean, 4),
                    "tn_mean": round(tn_mean, 4),
                    "mean_diff": round(diff, 4),
                    "interpretation": (
                        f"FP samples have {fn} = {fp_mean:.3f} vs TN = {tn_mean:.3f}. "
                        f"Model over-predicts hallucination when this feature differs."
                    ),
                })

    # Compare FN vs TP feature means
    if masks["FN"].sum() > 0 and masks["TP"].sum() > 0:
        for i, fn in enumerate(FEATURE_NAMES):
            fn_mean = float(X_val[fn_mask, i].mean())
            tp_mean = float(X_val[tp_mask, i].mean())
            diff = abs(fn_mean - tp_mean)
            if diff > 0.1:
                weaknesses.append({
                    "quadrant_pair": "FN vs TP",
                    "feature": fn,
                    "fn_mean": round(fn_mean, 4),
                    "tp_mean": round(tp_mean, 4),
                    "mean_diff": round(diff, 4),
                    "interpretation": (
                        f"FN samples have {fn} = {fn_mean:.3f} vs TP = {tp_mean:.3f}. "
                        f"Model misses hallucinations when this feature differs."
                    ),
                })

    report["systematic_weaknesses"] = weaknesses

    # ── 4. Recommendations ────────────────────────────────────────────────────
    report["recommendations"] = [
        {
            "id": "R1",
            "type": "Feature Engineering",
            "recommendation": (
                "min_support_margin is the strongest predictor. "
                "Pillar 2/3 should engineer richer support margin signals "
                "(e.g., per-document, per-sentence margins) to reduce FN rate."
            ),
        },
        {
            "id": "R2",
            "type": "Threshold Calibration",
            "recommendation": (
                "The 0.56 threshold was optimized for balanced F1/MCC. "
                "For applications requiring higher recall (catching more hallucinations), "
                "lower threshold to 0.50 at the cost of precision."
            ),
        },
        {
            "id": "R3",
            "type": "Feature Expansion",
            "recommendation": (
                "mean_entailment and max_entailment show weak discriminative power. "
                "Consider claim-level aggregation variants or attention-weighted NLI "
                "scores to capture subtle entailment patterns in longer responses."
            ),
        },
        {
            "id": "R4",
            "type": "Ensemble Strategy",
            "recommendation": (
                "Pillar-1 misses hallucinations with low contradiction scores. "
                "A Pillar-3 semantic similarity Pillar could complement Pillar-1 "
                "for hallucinations expressed without direct contradiction."
            ),
        },
        {
            "id": "R5",
            "type": "Dataset Expansion",
            "recommendation": (
                "The 0.47:0.53 class ratio in VAL differs from DEV (0.46:0.54). "
                "Future work should evaluate on more diverse benchmarks beyond "
                "HaluBench/HaluEval/RAGTruth to assess generalization."
            ),
        },
    ]

    # ── Generate figures ──────────────────────────────────────────────────────
    fig_dir = PUB / "figures"
    fig_dir.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # Figure 1: Feature distributions by quadrant (box plots)
    fig, axes = plt.subplots(1, len(FEATURE_NAMES), figsize=(16, 5))
    for i, (fn, fl, ax) in enumerate(zip(FEATURE_NAMES, FEATURE_LABELS, axes)):
        data_by_q = [X_val[masks[q], i] for q in ["TP", "TN", "FP", "FN"]]
        bp = ax.boxplot(data_by_q, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, q in zip(bp["boxes"], ["TP", "TN", "FP", "FN"]):
            patch.set_facecolor(QUADRANT_COLORS[q])
            patch.set_alpha(0.7)
        ax.set_xticklabels(["TP", "TN", "FP", "FN"], fontsize=8)
        ax.set_title(fl, fontsize=9)
        ax.set_ylabel("Value" if i == 0 else "", fontsize=8)
    fig.suptitle("Feature Distributions by Prediction Quadrant (VAL set)", fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_dir / "step4_quadrant_feature_distributions.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    # Figure 2: Probability distribution by quadrant
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (q, mask) in zip(axes.flat, masks.items()):
        pq = probs[mask]
        ax.hist(pq, bins=30, color=QUADRANT_COLORS[q], alpha=0.75, edgecolor="black",
                linewidth=0.5)
        ax.axvline(OPERATING_THRESHOLD, color="red", linestyle="--", linewidth=1.2,
                   label=f"Threshold={OPERATING_THRESHOLD}")
        ax.set_title(f"{q} (n={mask.sum()})", fontsize=11)
        ax.set_xlabel("Predicted Probability", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("Predicted Probability Distributions by Quadrant", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "step4_probability_by_quadrant.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    # Figure 3: Mean feature values heatmap by quadrant
    fig, ax = plt.subplots(figsize=(9, 4))
    heatmap_data = np.array([
        [report["quadrant_feature_stats"][q][fn]["mean"] for fn in FEATURE_NAMES]
        for q in ["TP", "TN", "FP", "FN"]
    ])
    im = ax.imshow(heatmap_data, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_LABELS, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(4))
    ax.set_yticklabels(["TP", "TN", "FP", "FN"], fontsize=10)
    for i in range(4):
        for j in range(len(FEATURE_NAMES)):
            ax.text(j, i, f"{heatmap_data[i, j]:.3f}", ha="center", va="center",
                    fontsize=7, color="black")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Mean Feature Values by Prediction Quadrant", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "step4_feature_heatmap.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    # Figure 4: FP cluster scatter (top 2 features)
    if fp_mask.sum() >= N_CLUSTERS:
        fig, ax = plt.subplots(figsize=(7, 5))
        Xfp = X_val[fp_mask]
        ss_fp = SS()
        Xfp_n = ss_fp.fit_transform(Xfp)
        km_fp = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
        km_fp.fit(Xfp_n)
        cluster_pal = ["#e41a1c", "#377eb8", "#4daf4a"]
        for cid in range(N_CLUSTERS):
            cmask = km_fp.labels_ == cid
            ax.scatter(Xfp[cmask, 3], Xfp[cmask, 2],
                       c=cluster_pal[cid], alpha=0.6, s=20,
                       label=f"Cluster {cid+1} (n={cmask.sum()})")
        ax.set_xlabel("min_support_margin", fontsize=11)
        ax.set_ylabel("mean_contradiction", fontsize=11)
        ax.set_title("False Positive Failure Clusters", fontsize=12)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(fig_dir / "step4_fp_clusters.png", dpi=DPI, bbox_inches="tight")
        plt.close()

    # Figure 5: FN cluster scatter
    if fn_mask.sum() >= N_CLUSTERS:
        fig, ax = plt.subplots(figsize=(7, 5))
        Xfn = X_val[fn_mask]
        ss_fn = SS()
        Xfn_n = ss_fn.fit_transform(Xfn)
        km_fn = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
        km_fn.fit(Xfn_n)
        cluster_pal = ["#e41a1c", "#377eb8", "#4daf4a"]
        for cid in range(N_CLUSTERS):
            cmask = km_fn.labels_ == cid
            ax.scatter(Xfn[cmask, 3], Xfn[cmask, 2],
                       c=cluster_pal[cid], alpha=0.6, s=20,
                       label=f"Cluster {cid+1} (n={cmask.sum()})")
        ax.set_xlabel("min_support_margin", fontsize=11)
        ax.set_ylabel("mean_contradiction", fontsize=11)
        ax.set_title("False Negative Failure Clusters", fontsize=12)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(fig_dir / "step4_fn_clusters.png", dpi=DPI, bbox_inches="tight")
        plt.close()

    elapsed = time.time() - t0
    report["elapsed_seconds"] = round(elapsed, 2)

    # Write JSON
    json_out = PUB / "step4_error_analysis.json"
    with open(json_out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  JSON → {json_out}")

    # Write Markdown
    qc = report["quadrant_counts"]
    total = sum(qc.values())
    md_lines = [
        "# Phase 9 — Step 4: Error Analysis",
        "",
        f"**Generated**: {report['generated_at_utc']}",
        "",
        "## 1. Quadrant Summary",
        "",
        "| Quadrant | Count | % | Description |",
        "| --- | --- | --- | --- |",
        f"| TP | {qc['TP']} | {qc['TP']/total*100:.1f}% | Correctly predicted hallucination |",
        f"| TN | {qc['TN']} | {qc['TN']/total*100:.1f}% | Correctly predicted grounded |",
        f"| FP | {qc['FP']} | {qc['FP']/total*100:.1f}% | Incorrectly predicted hallucination (false alarm) |",
        f"| FN | {qc['FN']} | {qc['FN']/total*100:.1f}% | Missed hallucination (miss) |",
        "",
        "## 2. Feature Statistics by Quadrant",
        "",
        "| Feature | TP Mean | TN Mean | FP Mean | FN Mean |",
        "| --- | --- | --- | --- | --- |",
    ]
    for fn in FEATURE_NAMES:
        row = []
        for q in ["TP", "TN", "FP", "FN"]:
            s = report["quadrant_feature_stats"].get(q, {}).get(fn, {})
            row.append(f"{s.get('mean', float('nan')):.4f}")
        md_lines.append(f"| `{fn}` | {' | '.join(row)} |")

    md_lines += [
        "",
        "## 3. Failure Cluster Analysis",
        "",
    ]
    for q in ["FP", "FN"]:
        fc = report["failure_clusters"].get(q, {})
        md_lines.append(f"### {q} ({fc.get('total_errors', 0)} samples → {fc.get('n_clusters', 0)} clusters)")
        for c in fc.get("clusters", []):
            md_lines.append(
                f"- **Cluster {c['cluster_id']+1}** ({c['size']} samples, {c['size_pct']}%): "
                f"dominant=`{c['dominant_feature']}`, mean_prob={c['mean_probability']:.3f}"
            )
        md_lines.append("")

    md_lines += [
        "## 4. Systematic Weaknesses",
        "",
    ]
    for w in report["systematic_weaknesses"][:5]:
        md_lines.append(f"- **{w['quadrant_pair']} | {w['feature']}**: {w['interpretation']}")
    md_lines.append("")

    md_lines += [
        "## 5. Recommendations",
        "",
    ]
    for r in report["recommendations"]:
        md_lines.append(f"- **[{r['id']}] {r['type']}**: {r['recommendation']}")

    md_lines += [
        "",
        "## 6. Figures",
        "",
        "- `step4_quadrant_feature_distributions.png` — Feature box plots by quadrant",
        "- `step4_probability_by_quadrant.png` — Predicted probability histograms",
        "- `step4_feature_heatmap.png` — Mean feature value heatmap",
        "- `step4_fp_clusters.png` — FP cluster scatter (min_support_margin vs mean_contradiction)",
        "- `step4_fn_clusters.png` — FN cluster scatter",
    ]

    md_out = PUB / "step4_error_analysis.md"
    with open(md_out, "w") as f:
        f.write("\n".join(md_lines))
    print(f"  MD  → {md_out}")

    print(f"\n✅ Step 4 complete in {elapsed:.1f}s")


if __name__ == "__main__":
    run()
