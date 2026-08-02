"""Phase 6L.2 — Stage 4: Feature Discrimination Audit Engine.

Evaluates univariate predictive strength for all 24 structural features against target label y:
- Mutual Information (MI)
- Univariate ROC-AUC
- Cohen's d Effect Size
- Point-biserial correlation (r_pb)
- Feature Variance (sigma^2)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import scipy.stats as scipy_stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
import structlog

from evaluation.phase6l.config import PHASE6L_DIR, STRUCTURAL_FEATURE_COLUMNS

logger = structlog.get_logger(__name__)


def run_feature_discrimination_audit(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str] = STRUCTURAL_FEATURE_COLUMNS,
    out_dir: Path = PHASE6L_DIR,
) -> Dict[str, Any]:
    """Audit univariate feature discrimination across all 24 structural features.

    Returns:
        Dict containing univariate metrics and composite discrimination ranking.
    """
    logger.info("stage4_feature_discrimination_start", n_samples=X.shape[0], n_features=len(feature_names))

    n_samples, n_feats = X.shape

    # 1. Compute Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)

    pos_mask = (y == 1)
    neg_mask = (y == 0)

    n1 = pos_mask.sum()
    n0 = neg_mask.sum()

    records: List[Dict[str, Any]] = []

    for i in range(n_feats):
        feat_name = feature_names[i]
        vals = X[:, i]

        vals_1 = vals[pos_mask]
        vals_0 = vals[neg_mask]

        # Univariate ROC-AUC
        # Handle zero-variance features gracefully
        if np.std(vals) < 1e-12:
            auc = 0.50
        else:
            auc = float(roc_auc_score(y, vals))
            # If AUC < 0.50, report directional AUC as max(auc, 1-auc)
            auc = max(auc, 1.0 - auc)

        # Cohen's d Effect Size
        m1 = float(np.mean(vals_1)) if n1 > 0 else 0.0
        m0 = float(np.mean(vals_0)) if n0 > 0 else 0.0
        v1 = float(np.var(vals_1, ddof=1)) if n1 > 1 else 0.0
        v0 = float(np.var(vals_0, ddof=1)) if n0 > 1 else 0.0

        pooled_std = np.sqrt(((n1 - 1) * v1 + (n0 - 1) * v0) / max(1, n1 + n0 - 2))
        if pooled_std < 1e-12:
            cohens_d = 0.0
        else:
            cohens_d = float(abs(m1 - m0) / pooled_std)

        # Point-biserial correlation
        if np.std(vals) < 1e-12 or np.std(y) < 1e-12:
            r_pb = 0.0
        else:
            r_val, _ = scipy_stats.pointbiserialr(y, vals)
            r_pb = 0.0 if np.isnan(r_val) else float(abs(r_val))

        variance = float(np.var(vals))

        records.append({
            "feature": feat_name,
            "mutual_information": float(round(mi_scores[i], 6)),
            "roc_auc": float(round(auc, 6)),
            "cohens_d": float(round(cohens_d, 6)),
            "point_biserial_r": float(round(r_pb, 6)),
            "variance": float(round(variance, 6)),
        })

    # Composite stability score calculation
    # Rank features by normalized average of (MI_norm + AUC_norm + Cohen_norm)
    mi_arr = np.array([r["mutual_information"] for r in records])
    auc_arr = np.array([r["roc_auc"] for r in records])
    d_arr = np.array([r["cohens_d"] for r in records])

    mi_norm = (mi_arr - mi_arr.min()) / max(1e-9, mi_arr.max() - mi_arr.min())
    auc_norm = (auc_arr - auc_arr.min()) / max(1e-9, auc_arr.max() - auc_arr.min())
    d_norm = (d_arr - d_arr.min()) / max(1e-9, d_arr.max() - d_arr.min())

    composite_scores = (mi_norm + auc_norm + d_norm) / 3.0

    for idx, rec in enumerate(records):
        rec["composite_score"] = float(round(composite_scores[idx], 6))

    # Sort records by composite_score descending
    sorted_records = sorted(records, key=lambda x: x["composite_score"], reverse=True)
    for rank, rec in enumerate(sorted_records, start=1):
        rec["rank"] = rank

    payload = {
        "n_samples": n_samples,
        "n_features": n_feats,
        "feature_rankings": sorted_records,
        "top_5_features": [r["feature"] for r in sorted_records[:5]],
    }

    with open(out_dir / "feature_selection_report.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(
        "stage4_feature_discrimination_complete",
        top_feature=sorted_records[0]["feature"],
        top_auc=sorted_records[0]["roc_auc"],
        top_mi=sorted_records[0]["mutual_information"],
    )

    return payload
