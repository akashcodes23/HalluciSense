"""Phase 6L.2 — Stage 2 & Stage 3: Preprocessing & Collinearity Analysis Engine.

Performs preprocessing study (None, StandardScaler, RobustScaler), numerical conditioning audit,
Pearson/Spearman/Kendall correlation matrix computation, Variance Inflation Factor (VIF) calculation,
hierarchical clustering, and constructs 6 candidate feature sets (SET_A through SET_F).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.stats as scipy_stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import structlog

from evaluation.phase6l.config import PHASE6L_DIR, STRUCTURAL_FEATURE_COLUMNS

logger = structlog.get_logger(__name__)


def evaluate_preprocessing_scalers(
    X: np.ndarray,
    feature_names: List[str] = STRUCTURAL_FEATURE_COLUMNS,
) -> Dict[str, Any]:
    """Evaluate None, StandardScaler, RobustScaler on feature matrix X.

    Returns:
        Dict containing numerical conditioning, stability, and distribution statistics.
    """
    logger.info("stage2_preprocessing_study_start", n_samples=X.shape[0], n_features=X.shape[1])

    scalers = {
        "None": None,
        "StandardScaler": StandardScaler(),
        "RobustScaler": RobustScaler(),
    }

    results: Dict[str, Any] = {}

    for name, scaler in scalers.items():
        if scaler is None:
            X_trans = X.copy()
        else:
            X_trans = scaler.fit_transform(X)

        # Numerical conditioning (SVD condition number)
        # Handle zero-variance columns by adding tiny epsilon for cond calculation if raw
        # or use SVD singular values
        singular_vals = np.linalg.svd(X_trans, compute_uv=False)
        max_sing = float(singular_vals[0])
        min_sing = float(singular_vals[-1])
        cond_num = float(max_sing / max(1e-12, min_sing))

        finite_all = bool(np.all(np.isfinite(X_trans)))
        nan_count = int(np.isnan(X_trans).sum())
        inf_count = int(np.isinf(X_trans).sum())

        means = np.mean(X_trans, axis=0).tolist()
        stds = np.std(X_trans, axis=0).tolist()
        mins = np.min(X_trans, axis=0).tolist()
        maxs = np.max(X_trans, axis=0).tolist()

        results[name] = {
            "scaler_name": name,
            "condition_number": cond_num,
            "max_singular_value": max_sing,
            "min_singular_value": min_sing,
            "finite_all": finite_all,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "mean_summary": {
                "overall_mean": float(np.mean(means)),
                "overall_std": float(np.mean(stds)),
                "min_value": float(np.min(mins)),
                "max_value": float(np.max(maxs)),
            },
        }

    logger.info("stage2_preprocessing_study_complete", results_summary={k: v["condition_number"] for k, v in results.items()})
    return results


def run_collinearity_analysis(
    X: np.ndarray,
    feature_names: List[str] = STRUCTURAL_FEATURE_COLUMNS,
    out_dir: Path = PHASE6L_DIR,
) -> Dict[str, Any]:
    """Compute Pearson, Spearman, Kendall correlation matrices, VIFs, and candidate feature subsets.

    Returns:
        Dict containing correlation analysis payloads and candidate feature set definitions.
    """
    logger.info("stage3_collinearity_analysis_start", n_features=len(feature_names))

    n_samples, n_feats = X.shape

    # 1. Correlations
    pearson_corr = np.corrcoef(X, rowvar=False)
    spearman_corr, _ = scipy_stats.spearmanr(X, axis=0)

    # Kendall correlation (sampled on first 2000 for efficiency if needed)
    # Since Kendall is O(n^2), sample 2,000 for fast accurate Kendall rank correlation
    sample_idx = np.random.choice(n_samples, min(2000, n_samples), replace=False)
    X_sample = X[sample_idx]
    kendall_matrix = np.zeros((n_feats, n_feats), dtype=float)
    for i in range(n_feats):
        kendall_matrix[i, i] = 1.0
        for j in range(i + 1, n_feats):
            tau_val, _ = scipy_stats.kendalltau(X_sample[:, i], X_sample[:, j])
            tau_val = 0.0 if np.isnan(tau_val) else float(tau_val)
            kendall_matrix[i, j] = tau_val
            kendall_matrix[j, i] = tau_val

    # Convert correlations to lists
    pearson_dict = {
        "columns": feature_names,
        "matrix": np.nan_to_num(pearson_corr, nan=0.0).tolist(),
    }
    spearman_dict = {
        "columns": feature_names,
        "matrix": np.nan_to_num(spearman_corr, nan=0.0).tolist(),
    }
    kendall_dict = {
        "columns": feature_names,
        "matrix": kendall_matrix.tolist(),
    }

    # Save correlations report
    corr_payload = {
        "pearson": pearson_dict,
        "spearman": spearman_dict,
        "kendall": kendall_dict,
    }
    with open(out_dir / "feature_correlations.json", "w", encoding="utf-8") as f:
        json.dump(corr_payload, f, indent=2)

    # 2. Variance Inflation Factor (VIF)
    # Computed via diagonal inverse of Pearson correlation matrix: VIF_i = diag((R + eps*I)^-1)_i
    corr_clean = np.nan_to_num(pearson_corr, nan=0.0)
    corr_reg = corr_clean + 1e-6 * np.eye(n_feats)
    inv_corr_diag = np.diag(np.linalg.pinv(corr_reg))

    vif_records = []
    for i in range(n_feats):
        feat_name = feature_names[i]
        val = float(inv_corr_diag[i])
        if np.isnan(val) or np.isinf(val) or val < 1.0:
            val = 1.0
        vif_records.append({
            "feature": feat_name,
            "vif": round(val, 4),
            "high_vif": val > 5.0,
        })

    vif_payload = {
        "vif_threshold": 5.0,
        "vif_records": vif_records,
        "high_vif_features": [r["feature"] for r in vif_records if r["high_vif"]],
    }
    with open(out_dir / "vif_report.json", "w", encoding="utf-8") as f:
        json.dump(vif_payload, f, indent=2)

    # 3. Hierarchical Clustering on Spearman Correlation Distance
    corr_dist = 1.0 - np.abs(np.nan_to_num(spearman_corr, nan=0.0))
    np.fill_diagonal(corr_dist, 0.0)
    condensed_dist = sch.distance.squareform(corr_dist, checks=False)
    linkage = sch.linkage(condensed_dist, method="average")

    # 4. Construct Candidate Feature Subsets
    # SET_A: All 24 features
    set_a = list(feature_names)

    # SET_B: Low-correlation subset (|r| < 0.70)
    # Greedy removal of highly correlated pairs
    set_b: List[str] = []
    for i, col in enumerate(feature_names):
        keep = True
        for prev_col in set_b:
            prev_idx = feature_names.index(prev_col)
            if abs(spearman_corr[i, prev_idx]) >= 0.70:
                keep = False
                break
        if keep:
            set_b.append(col)

    # SET_C: Low-VIF subset (VIF < 5.0)
    set_c = [r["feature"] for r in vif_records if r["vif"] < 5.0]
    if len(set_c) < 3:
        # Guarantee minimum 3 features
        sorted_vif = sorted(vif_records, key=lambda x: x["vif"])
        set_c = [r["feature"] for r in sorted_vif[:5]]

    # SET_D: Highest information / discrimination subset (5 key features)
    set_d = [
        "max_pairwise_contradiction",
        "mean_pairwise_contradiction",
        "max_pairwise_similarity",
        "fraction_contradictory_pairs",
        "num_claims",
    ]

    # SET_E: Graph-centric subset
    set_e = [
        "contradiction_graph_density",
        "max_contradiction_degree",
        "largest_contradictory_component_ratio",
        "num_claims",
        "contradiction_pair_count",
    ]

    # SET_F: Contradiction-centric subset
    set_f = [
        "mean_pairwise_contradiction",
        "max_pairwise_contradiction",
        "p95_pairwise_contradiction",
        "fraction_contradictory_pairs",
        "contradiction_pair_count",
    ]

    candidate_sets = {
        "SET_A_FULL_SCHEMA": {
            "name": "SET_A_FULL_SCHEMA",
            "description": "All 24 structural features (Full Schema).",
            "features": set_a,
            "feature_count": len(set_a),
        },
        "SET_B_LOW_CORRELATION": {
            "name": "SET_B_LOW_CORRELATION",
            "description": "Low-correlation subset (|rho| < 0.70).",
            "features": set_b,
            "feature_count": len(set_b),
        },
        "SET_C_LOW_VIF": {
            "name": "SET_C_LOW_VIF",
            "description": "Low Variance Inflation Factor subset (VIF < 5.0).",
            "features": set_c,
            "feature_count": len(set_c),
        },
        "SET_D_HIGH_INFORMATION": {
            "name": "SET_D_HIGH_INFORMATION",
            "description": "Highest information & discrimination subset.",
            "features": set_d,
            "feature_count": len(set_d),
        },
        "SET_E_GRAPH_CENTRIC": {
            "name": "SET_E_GRAPH_CENTRIC",
            "description": "Graph topological metrics & density subset.",
            "features": set_e,
            "feature_count": len(set_e),
        },
        "SET_F_CONTRADICTION_CENTRIC": {
            "name": "SET_F_CONTRADICTION_CENTRIC",
            "description": "Direct pairwise contradiction aggregation subset.",
            "features": set_f,
            "feature_count": len(set_f),
        },
    }

    with open(out_dir / "candidate_feature_sets.json", "w", encoding="utf-8") as f:
        json.dump(candidate_sets, f, indent=2)

    logger.info(
        "stage3_collinearity_analysis_complete",
        candidate_count=len(candidate_sets),
        set_a_count=len(set_a),
        set_b_count=len(set_b),
        set_c_count=len(set_c),
    )

    return {
        "correlations": corr_payload,
        "vif": vif_payload,
        "hierarchical_linkage": linkage.tolist(),
        "candidate_sets": candidate_sets,
    }
