"""
HalluciSense Phase 9 — Step 2: Prediction Explainability
=========================================================
Implements coefficient-based explainability for every inference.
No black-box explainers. No SHAP. Uses frozen logistic coefficients.

FROZEN FIREWALL: No models, scalers, or thresholds are modified.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

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
FEATURE_DESCRIPTIONS = {
    "mean_entailment": "Mean NLI entailment score across all claim-evidence pairs",
    "max_entailment": "Maximum NLI entailment score (strongest supporting evidence)",
    "mean_contradiction": "Mean NLI contradiction score (evidence contradiction signal)",
    "min_support_margin": "Minimum support margin (weakest evidence-claim alignment)",
    "num_claims": "Number of atomic claims extracted from response",
}
OPERATING_THRESHOLD = 0.56


# ── Explainer ─────────────────────────────────────────────────────────────────
class PillarOneExplainer:
    """
    Coefficient-based explainability for Pillar-1 LogisticRegression.
    Computes feature contributions as: contribution_i = coef_i * scaled_feature_i
    """

    CONFIDENCE_BINS = [
        (0.90, 1.01, "Very High"),
        (0.75, 0.90, "High"),
        (0.60, 0.75, "Moderate"),
        (0.50, 0.60, "Low"),
        (0.40, 0.50, "Low"),
        (0.25, 0.40, "Moderate"),
        (0.10, 0.25, "High"),
        (0.00, 0.10, "Very High"),
    ]

    def __init__(self, model, scaler, feature_names: list[str], threshold: float):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.threshold = threshold
        self.coef = model.coef_[0]
        self.intercept = float(model.intercept_[0])

    def _confidence_category(self, prob: float) -> str:
        dist = abs(prob - self.threshold)
        if dist >= 0.25:
            return "Very High"
        elif dist >= 0.15:
            return "High"
        elif dist >= 0.08:
            return "Moderate"
        else:
            return "Low"

    def _reasoning_summary(
        self,
        pred_label: int,
        prob: float,
        top_pos: list[dict],
        top_neg: list[dict],
    ) -> str:
        direction = "HALLUCINATED" if pred_label == 1 else "GROUNDED"
        conf = self._confidence_category(prob)
        margin = abs(prob - self.threshold)

        pos_str = ", ".join(
            f"{e['feature']} (+{e['contribution']:.3f})" for e in top_pos[:2]
        )
        neg_str = ", ".join(
            f"{e['feature']} ({e['contribution']:.3f})" for e in top_neg[:2]
        )

        return (
            f"Prediction: {direction} (p={prob:.3f}, threshold={self.threshold}). "
            f"{conf} confidence with margin {margin:.3f}. "
            f"Key supporting signals: [{pos_str or 'none'}]. "
            f"Key opposing signals: [{neg_str or 'none'}]."
        )

    def explain(self, x_raw: np.ndarray) -> dict[str, Any]:
        """
        Generate full explanation for a single raw feature vector.

        Parameters
        ----------
        x_raw : np.ndarray, shape (n_features,)
            Unscaled feature vector.

        Returns
        -------
        dict with full explanation structure.
        """
        # Scale
        x_scaled = self.scaler.transform(x_raw.reshape(1, -1))[0]

        # Logit and probability
        logit = float(np.dot(self.coef, x_scaled) + self.intercept)
        prob = float(1.0 / (1.0 + np.exp(-logit)))
        pred_label = int(prob >= self.threshold)

        # Feature contributions
        contributions = []
        for i, fname in enumerate(self.feature_names):
            contrib = float(self.coef[i] * x_scaled[i])
            contributions.append({
                "feature": fname,
                "description": FEATURE_DESCRIPTIONS.get(fname, fname),
                "raw_value": float(x_raw[i]),
                "scaled_value": float(x_scaled[i]),
                "coefficient": float(self.coef[i]),
                "contribution": contrib,
            })

        contributions.sort(key=lambda c: c["contribution"], reverse=True)

        top_pos = [c for c in contributions if c["contribution"] > 0]
        top_neg = [c for c in contributions if c["contribution"] < 0]
        top_neg_sorted = sorted(top_neg, key=lambda c: c["contribution"])

        margin = prob - self.threshold
        confidence_cat = self._confidence_category(prob)
        reasoning = self._reasoning_summary(pred_label, prob, top_pos, top_neg_sorted)

        return {
            "prediction": {
                "label": pred_label,
                "label_str": "HALLUCINATED" if pred_label == 1 else "GROUNDED",
                "probability": round(prob, 6),
                "logit": round(logit, 6),
                "decision_threshold": self.threshold,
                "margin_to_threshold": round(margin, 6),
                "confidence_category": confidence_cat,
            },
            "feature_contributions": contributions,
            "top_positive_evidence": top_pos[:3],
            "top_negative_evidence": top_neg_sorted[:3],
            "reasoning_summary": reasoning,
            "intercept_contribution": round(self.intercept, 6),
        }


# ── Main ──────────────────────────────────────────────────────────────────────
def run() -> None:
    print("=" * 70)
    print("HalluciSense Phase 9 — Step 2: Prediction Explainability")
    print("=" * 70)
    t0 = time.time()

    model = joblib.load(FINAL_MODEL_DIR / "pillar1_logistic_model.joblib")
    scaler = joblib.load(FINAL_MODEL_DIR / "robust_scaler.joblib")
    explainer = PillarOneExplainer(model, scaler, FEATURE_NAMES, OPERATING_THRESHOLD)

    # Load VAL features and labels
    print("\n[1/4] Loading VAL features...")
    val_rows = []
    with open(PHASE6I / "claim_evidence_features_validation.jsonl") as f:
        for line in f:
            obj = json.loads(line.strip()) if line.strip() else {}
            row = [obj.get(fn, float("nan")) for fn in FEATURE_NAMES]
            val_rows.append({"features": row, "label": int(obj.get("ground_truth", 0))})
    X_val = np.array([r["features"] for r in val_rows], dtype=np.float64)
    y_val = np.array([r["label"] for r in val_rows], dtype=np.int32)
    print(f"  VAL samples: {len(X_val)}")

    # Run explainability on full VAL set
    print("\n[2/4] Running explainability on all 3,500 VAL samples...")
    X_val_scaled = scaler.transform(X_val)
    probs = model.predict_proba(X_val_scaled)[:, 1]
    preds = (probs >= OPERATING_THRESHOLD).astype(int)

    all_explanations = []
    for i, (x_raw, y_true, pred, prob) in enumerate(
        zip(X_val, y_val, preds, probs)
    ):
        expl = explainer.explain(x_raw)
        expl["sample_index"] = i
        expl["true_label"] = int(y_true)
        expl["true_label_str"] = "HALLUCINATED" if y_true == 1 else "GROUNDED"
        quadrant = (
            "TP" if pred == 1 and y_true == 1
            else "TN" if pred == 0 and y_true == 0
            else "FP" if pred == 1 and y_true == 0
            else "FN"
        )
        expl["quadrant"] = quadrant
        all_explanations.append(expl)

    # Quadrant counts
    quadrant_counts = {q: sum(1 for e in all_explanations if e["quadrant"] == q)
                       for q in ["TP", "TN", "FP", "FN"]}
    print(f"  Quadrant counts: {quadrant_counts}")

    # Spot-check: 5 examples from each quadrant
    print("\n[3/4] Selecting spot-check examples (5 per quadrant)...")
    spot_checks = {}
    for q in ["TP", "TN", "FP", "FN"]:
        pool = [e for e in all_explanations if e["quadrant"] == q]
        # Select by highest confidence (most extreme margin)
        pool.sort(key=lambda e: abs(e["prediction"]["margin_to_threshold"]), reverse=True)
        spot_checks[q] = pool[:5]

    # Compute aggregate statistics
    print("\n[4/4] Computing aggregate explainability statistics...")
    contrib_by_feature: dict[str, list[float]] = {fn: [] for fn in FEATURE_NAMES}
    for expl in all_explanations:
        for fc in expl["feature_contributions"]:
            contrib_by_feature[fc["feature"]].append(fc["contribution"])

    agg_stats = {}
    for fn in FEATURE_NAMES:
        contribs = np.array(contrib_by_feature[fn])
        agg_stats[fn] = {
            "mean_contribution": float(contribs.mean()),
            "std_contribution": float(contribs.std()),
            "abs_mean_contribution": float(np.abs(contribs).mean()),
            "positive_count": int((contribs > 0).sum()),
            "negative_count": int((contribs < 0).sum()),
            "zero_count": int((contribs == 0).sum()),
        }

    # Confidence category distribution
    conf_dist = {}
    for expl in all_explanations:
        cc = expl["prediction"]["confidence_category"]
        conf_dist[cc] = conf_dist.get(cc, 0) + 1

    # Build report
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "9_step2_explainability",
        "val_samples": len(X_val),
        "operating_threshold": OPERATING_THRESHOLD,
        "quadrant_counts": quadrant_counts,
        "confidence_category_distribution": conf_dist,
        "aggregate_feature_contribution_stats": agg_stats,
        "spot_check_examples": spot_checks,
        "method": (
            "Linear coefficient-based explainability: "
            "contribution_i = coef_i × RobustScaled(feature_i). "
            "No SHAP, no black-box approximations."
        ),
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    # Write JSON
    json_out = PUB / "step2_explainability_examples.json"
    with open(json_out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  JSON → {json_out}")

    # Write Markdown report
    md_lines = [
        "# Phase 9 — Step 2: Prediction Explainability",
        "",
        f"**Generated**: {report['generated_at_utc']}",
        "",
        "## Explainability Method",
        "",
        "Each prediction is explained using the frozen logistic regression coefficients:",
        "",
        "```",
        "contribution_i = coef_i × RobustScaled(feature_i)",
        "logit = Σ contribution_i + intercept",
        "probability = sigmoid(logit)",
        "prediction = (probability ≥ 0.56)",
        "```",
        "",
        "No SHAP, no gradient-based attribution, no black-box approximations.",
        "",
        "## Quadrant Distribution (VAL 3,500 samples)",
        "",
        "| Quadrant | Count | % |",
        "| --- | --- | --- |",
    ]
    total = len(X_val)
    for q, cnt in quadrant_counts.items():
        md_lines.append(f"| {q} | {cnt} | {cnt/total*100:.1f}% |")

    md_lines += [
        "",
        "## Confidence Category Distribution",
        "",
        "| Category | Count | % |",
        "| --- | --- | --- |",
    ]
    for cc, cnt in sorted(conf_dist.items(), key=lambda x: -x[1]):
        md_lines.append(f"| {cc} | {cnt} | {cnt/total*100:.1f}% |")

    md_lines += [
        "",
        "## Aggregate Feature Contribution Statistics",
        "",
        "| Feature | Mean Contrib | Std | |Contrib| Mean | Positive | Negative |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for fn in FEATURE_NAMES:
        s = agg_stats[fn]
        md_lines.append(
            f"| `{fn}` | {s['mean_contribution']:.4f} | {s['std_contribution']:.4f} "
            f"| {s['abs_mean_contribution']:.4f} | {s['positive_count']} | {s['negative_count']} |"
        )

    md_lines += [
        "",
        "## Spot-Check Examples",
        "",
    ]
    for q in ["TP", "TN", "FP", "FN"]:
        md_lines.append(f"### {q} Examples (Top 3 by confidence margin)")
        md_lines.append("")
        for e in spot_checks[q][:3]:
            p = e["prediction"]
            md_lines.append(
                f"- **Sample #{e['sample_index']}** | True: `{e['true_label_str']}` | "
                f"Predicted: `{p['label_str']}` | prob={p['probability']:.4f} | "
                f"margin={p['margin_to_threshold']:.4f} | conf={p['confidence_category']}"
            )
            md_lines.append(f"  > {e['reasoning_summary']}")
            md_lines.append("")

    md_out = PUB / "step2_explainability_report.md"
    with open(md_out, "w") as f:
        f.write("\n".join(md_lines))
    print(f"  MD  → {md_out}")

    elapsed = time.time() - t0
    print(f"\n✅ Step 2 complete in {elapsed:.1f}s")
    print(f"   Explained {len(X_val)} predictions | Quadrants: {quadrant_counts}")


if __name__ == "__main__":
    run()
