"""
HalluciSense Phase 9 — Step 3: Feature Importance Analysis
===========================================================
Coefficient ranking, standardized importance, odds ratios with bootstrap CIs,
permutation importance, partial dependence plots.

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
from scipy.stats import norm

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
    "Mean Entailment",
    "Max Entailment",
    "Mean Contradiction",
    "Min Support Margin",
    "Num Claims",
]
OPERATING_THRESHOLD = 0.56
DPI = 300
PALETTE = ["#2c7bb6", "#d7191c", "#fdae61", "#1a9641", "#a6d96a"]
NEG_COLOR = "#d7191c"
POS_COLOR = "#2c7bb6"


def roc_auc_score(y_true, probs):
    from sklearn.metrics import roc_auc_score as _roc_auc
    return float(_roc_auc(y_true, probs))


def load_val():
    rows, labels = [], []
    with open(PHASE6I / "claim_evidence_features_validation.jsonl") as f:
        for line in f:
            obj = json.loads(line.strip()) if line.strip() else {}
            rows.append([obj.get(fn, float("nan")) for fn in FEATURE_NAMES])
            labels.append(int(obj.get("ground_truth", 0)))
    return np.array(rows, dtype=np.float64), np.array(labels, dtype=np.int32)


def run() -> None:
    print("=" * 70)
    print("HalluciSense Phase 9 — Step 3: Feature Importance Analysis")
    print("=" * 70)
    t0 = time.time()

    model = joblib.load(FINAL_MODEL_DIR / "pillar1_logistic_model.joblib")
    scaler = joblib.load(FINAL_MODEL_DIR / "robust_scaler.joblib")

    X_val, y_val = load_val()
    X_val_scaled = scaler.transform(X_val)
    probs = model.predict_proba(X_val_scaled)[:, 1]
    base_auc = roc_auc_score(y_val, probs)

    coef = model.coef_[0]
    print(f"\n[1/5] Coefficients: {dict(zip(FEATURE_NAMES, coef.round(4)))}")

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "9_step3_feature_importance",
        "base_val_roc_auc": base_auc,
    }

    # ── 1. Coefficient ranking ────────────────────────────────────────────────
    print("\n[1/5] Coefficient ranking...")
    # Standardize by RobustScaler IQR (scale_)
    iqr_scales = scaler.scale_  # IQR for each feature
    std_coef = coef * iqr_scales  # multiply coef by IQR to get natural-unit importance

    coef_ranking = []
    for i, fn in enumerate(FEATURE_NAMES):
        coef_ranking.append({
            "feature": fn,
            "coefficient": float(coef[i]),
            "standardized_coefficient": float(std_coef[i]),
            "abs_standardized_coefficient": float(abs(std_coef[i])),
            "direction": "positive" if coef[i] > 0 else "negative",
            "odds_ratio": float(np.exp(coef[i])),
        })
    coef_ranking.sort(key=lambda x: x["abs_standardized_coefficient"], reverse=True)
    report["coefficient_ranking"] = coef_ranking

    # ── 2. Odds ratios with bootstrap CIs ────────────────────────────────────
    print("\n[2/5] Computing odds ratios with bootstrap CIs (n=2000)...")
    N_BOOTSTRAP = 2000
    rng = np.random.default_rng(42)
    n_val = len(X_val)

    boot_coefs = np.zeros((N_BOOTSTRAP, len(FEATURE_NAMES)))
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_val, size=n_val)
        Xb = X_val_scaled[idx]
        yb = y_val[idx]
        if len(np.unique(yb)) < 2:
            boot_coefs[b] = coef
            continue
        from sklearn.linear_model import LogisticRegression
        clf_b = LogisticRegression(
            solver="liblinear", penalty="l2", C=model.C,
            max_iter=1000, random_state=42
        )
        try:
            clf_b.fit(Xb, yb)
            boot_coefs[b] = clf_b.coef_[0]
        except Exception:
            boot_coefs[b] = coef

    odds_ratios = []
    for i, fn in enumerate(FEATURE_NAMES):
        bc = boot_coefs[:, i]
        or_vals = np.exp(bc)
        or_lo = float(np.percentile(or_vals, 2.5))
        or_hi = float(np.percentile(or_vals, 97.5))
        odds_ratios.append({
            "feature": fn,
            "coefficient": float(coef[i]),
            "odds_ratio": float(np.exp(coef[i])),
            "ci_lower_95": or_lo,
            "ci_upper_95": or_hi,
            "ci_width": round(or_hi - or_lo, 4),
            "significant": not (or_lo <= 1.0 <= or_hi),
        })
    report["odds_ratios"] = odds_ratios
    print(f"  Bootstrap done. CIs computed for {len(odds_ratios)} features.")

    # ── 3. Permutation importance ─────────────────────────────────────────────
    print("\n[3/5] Permutation importance (n=100 repeats)...")
    N_PERM = 100
    perm_importance = []
    for fi, fn in enumerate(FEATURE_NAMES):
        drops = []
        for _ in range(N_PERM):
            Xp = X_val_scaled.copy()
            rng.shuffle(Xp[:, fi])
            perm_probs = model.predict_proba(Xp)[:, 1]
            drop = base_auc - roc_auc_score(y_val, perm_probs)
            drops.append(drop)
        drops_arr = np.array(drops)
        perm_importance.append({
            "feature": fn,
            "mean_auc_drop": float(drops_arr.mean()),
            "std_auc_drop": float(drops_arr.std()),
            "min_auc_drop": float(drops_arr.min()),
            "max_auc_drop": float(drops_arr.max()),
            "importance_rank": 0,  # filled below
        })
        print(f"  {fn}: mean_drop={drops_arr.mean():.4f} ± {drops_arr.std():.4f}")

    perm_importance.sort(key=lambda x: x["mean_auc_drop"], reverse=True)
    for rank, pi in enumerate(perm_importance):
        pi["importance_rank"] = rank + 1
    report["permutation_importance"] = perm_importance

    # ── 4. Partial dependence (key features) ─────────────────────────────────
    print("\n[4/5] Partial dependence for key features...")
    pd_features = ["min_support_margin", "mean_contradiction"]
    pdp_results = {}
    for fi_name in pd_features:
        fi = FEATURE_NAMES.index(fi_name)
        grid_raw = np.linspace(X_val[:, fi].min(), X_val[:, fi].max(), 100)
        X_pdp = X_val_scaled.copy()
        pdp_probs = []
        for gv in grid_raw:
            Xg = X_val_scaled.copy()
            # Scale the grid value
            dummy = X_val.copy()
            dummy[:, fi] = gv
            Xg = scaler.transform(dummy)
            mean_prob = float(model.predict_proba(Xg)[:, 1].mean())
            pdp_probs.append(mean_prob)
        pdp_results[fi_name] = {
            "grid_raw": grid_raw.tolist(),
            "mean_predicted_probability": pdp_probs,
        }

    report["partial_dependence"] = pdp_results

    # ── 5. Generate figures ───────────────────────────────────────────────────
    print("\n[5/5] Generating publication figures...")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig_dir = PUB / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Figure 1: Coefficient bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [c["feature"].replace("_", "\n") for c in coef_ranking]
    vals = [c["standardized_coefficient"] for c in coef_ranking]
    colors = [POS_COLOR if v > 0 else NEG_COLOR for v in vals]
    bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1], edgecolor="black",
                   linewidth=0.5, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardized Coefficient (IQR-scaled)", fontsize=11)
    ax.set_title("Pillar-1 Feature Coefficients\n(Positive = Hallucination Evidence)", fontsize=12)
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "step3_coefficient_ranking.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    # Figure 2: Odds ratio forest plot
    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = list(range(len(odds_ratios)))
    for i, od in enumerate(sorted(odds_ratios, key=lambda x: x["odds_ratio"])):
        color = POS_COLOR if od["odds_ratio"] > 1 else NEG_COLOR
        ax.barh(i, od["odds_ratio"] - 1, left=1, color=color, alpha=0.6, height=0.5)
        ax.plot(
            [od["ci_lower_95"], od["ci_upper_95"]], [i, i],
            color="black", linewidth=1.5
        )
        ax.plot(od["odds_ratio"], i, "D", color=color, markersize=7)
    ax.axvline(1.0, color="black", linewidth=1.2, linestyle="--")
    ax.set_yticks(range(len(odds_ratios)))
    ax.set_yticklabels([od["feature"].replace("_", "\n") for od in
                        sorted(odds_ratios, key=lambda x: x["odds_ratio"])], fontsize=9)
    ax.set_xlabel("Odds Ratio (95% Bootstrap CI)", fontsize=11)
    ax.set_title("Pillar-1 Odds Ratios with 95% Bootstrap CIs", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "step3_odds_ratios.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    # Figure 3: Permutation importance
    fig, ax = plt.subplots(figsize=(8, 4))
    pi_sorted = sorted(perm_importance, key=lambda x: x["mean_auc_drop"])
    names_pi = [p["feature"].replace("_", "\n") for p in pi_sorted]
    means_pi = [p["mean_auc_drop"] for p in pi_sorted]
    stds_pi = [p["std_auc_drop"] for p in pi_sorted]
    colors_pi = [POS_COLOR if m > 0 else "#888888" for m in means_pi]
    ax.barh(names_pi, means_pi, xerr=stds_pi, color=colors_pi, alpha=0.85,
            edgecolor="black", linewidth=0.5, capsize=4)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean ROC-AUC Drop (n=100 permutations)", fontsize=11)
    ax.set_title("Permutation Importance (Pillar-1, VAL set)", fontsize=12)
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "step3_permutation_importance.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    # Figure 4: Partial dependence — min_support_margin
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, fn in zip(axes, pd_features):
        pdp = pdp_results[fn]
        ax.plot(pdp["grid_raw"], pdp["mean_predicted_probability"],
                color=POS_COLOR, linewidth=2)
        ax.axhline(OPERATING_THRESHOLD, color=NEG_COLOR, linewidth=1.2,
                   linestyle="--", label=f"Threshold={OPERATING_THRESHOLD}")
        ax.set_xlabel(fn.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Mean Predicted Probability", fontsize=10)
        ax.set_title(f"Partial Dependence: {fn}", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(fig_dir / "step3_partial_dependence.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    # Figure 5: Coefficient magnitude comparison (abs)
    fig, ax = plt.subplots(figsize=(8, 4))
    names_all = [c["feature"].replace("_", "\n") for c in coef_ranking]
    abs_std = [abs(c["standardized_coefficient"]) for c in coef_ranking]
    abs_raw = [abs(c["coefficient"]) for c in coef_ranking]
    x = np.arange(len(coef_ranking))
    width = 0.35
    ax.bar(x - width/2, abs_std, width, label="|Std Coef| (IQR-scaled)", color=POS_COLOR,
           alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, abs_raw, width, label="|Raw Coef|", color="#fdae61",
           alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names_all, fontsize=9)
    ax.set_ylabel("Absolute Coefficient Magnitude", fontsize=11)
    ax.set_title("Feature Importance: Raw vs Standardized Coefficients", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "step3_coefficient_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"  5 figures saved to {fig_dir}")

    # ── Write report ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    report["elapsed_seconds"] = round(elapsed, 2)

    json_out = PUB / "step3_feature_importance.json"
    with open(json_out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  JSON → {json_out}")

    # Markdown
    md_lines = [
        "# Phase 9 — Step 3: Feature Importance Analysis",
        "",
        f"**Generated**: {report['generated_at_utc']}",
        f"**Base VAL ROC-AUC**: {base_auc:.4f}",
        "",
        "## 1. Coefficient Ranking (Standardized by IQR)",
        "",
        "| Rank | Feature | Raw Coef | Std Coef | Odds Ratio | Direction |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for rank, c in enumerate(coef_ranking, 1):
        md_lines.append(
            f"| {rank} | `{c['feature']}` | {c['coefficient']:.4f} | "
            f"{c['standardized_coefficient']:.4f} | {c['odds_ratio']:.4f} | {c['direction']} |"
        )

    md_lines += [
        "",
        "## 2. Odds Ratios with 95% Bootstrap CIs",
        "",
        "| Feature | OR | 95% CI Lower | 95% CI Upper | Significant |",
        "| --- | --- | --- | --- | --- |",
    ]
    for od in odds_ratios:
        sig = "✅" if od["significant"] else "❌"
        md_lines.append(
            f"| `{od['feature']}` | {od['odds_ratio']:.4f} | "
            f"{od['ci_lower_95']:.4f} | {od['ci_upper_95']:.4f} | {sig} |"
        )

    md_lines += [
        "",
        "## 3. Permutation Importance",
        "",
        "| Rank | Feature | Mean AUC Drop | Std |",
        "| --- | --- | --- | --- |",
    ]
    for pi in perm_importance:
        md_lines.append(
            f"| {pi['importance_rank']} | `{pi['feature']}` | "
            f"{pi['mean_auc_drop']:.4f} | {pi['std_auc_drop']:.4f} |"
        )

    md_lines += [
        "",
        "## 4. Figures Generated",
        "",
        "- `step3_coefficient_ranking.png` — Standardized coefficient bar chart",
        "- `step3_odds_ratios.png` — Forest plot with bootstrap CIs",
        "- `step3_permutation_importance.png` — ROC-AUC drop per feature",
        "- `step3_partial_dependence.png` — PDP for min_support_margin + mean_contradiction",
        "- `step3_coefficient_comparison.png` — Raw vs standardized coefficient comparison",
    ]

    md_out = PUB / "step3_feature_importance.md"
    with open(md_out, "w") as f:
        f.write("\n".join(md_lines))
    print(f"  MD  → {md_out}")

    print(f"\n✅ Step 3 complete in {elapsed:.1f}s")


if __name__ == "__main__":
    run()
