"""Phase 6K — Controlled Feature Subset Selection.

Constructs four scientifically motivated candidate feature sets using ONLY the
Development partition (X_dev, y_dev):

    1. SET A — ALL (10 features): Full original Pillar-1 feature matrix.
    2. SET B — DECOLLINEARIZED (5 features): Features retained after collinearity audit.
    3. SET C — TOP_DISCRIMINATIVE (5 features): Top features ranked by composite discriminative power (MI, ROC-AUC, Cohen's d, KS stat).
    4. SET D — DECOLLINEARIZED_DISCRIMINATIVE (3 features): Minimalist non-redundant top discriminative features.

STRICT DATA ISOLATION:
Validation labels (y_val) and validation features (X_val) are NEVER accessed
or referenced by feature selection algorithms.

Outputs:
    * Exported artifact: ``evaluation_results/phase6k/feature_sets.json``

This module is analysis-only and read-only.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
import structlog

from evaluation.phase6j.utils import _serializable
from evaluation.phase6k.config import PHASE6K_DIR, FEATURE_COLUMNS

logger = structlog.get_logger(__name__)


# =========================================================
# DATACLASSES
# =========================================================

@dataclass
class FeatureSetMetadata:
    """Detailed audit metadata for a single candidate feature set."""

    set_name: str
    feature_names: List[str]
    feature_count: int
    indices: List[int]
    matrix_rank: int
    condition_number_unscaled: float
    condition_number_robust_scaled: float
    mean_pairwise_abs_correlation: float
    max_pairwise_abs_correlation: float
    description: str


@dataclass
class CandidateFeatureSetsReport:
    """Aggregated report container for all four candidate feature sets."""

    n_dev_samples: int
    n_master_features: int
    candidate_sets: Dict[str, FeatureSetMetadata] = field(default_factory=dict)


# =========================================================
# COMPUTATION HELPERS
# =========================================================

def _compute_pairwise_abs_correlations(X_sub: np.ndarray) -> Tuple[float, float]:
    """Compute mean and max pairwise absolute Pearson correlations for a feature matrix subset.

    Args:
        X_sub: Matrix subset of shape (n_samples, n_sub_features).

    Returns:
        Tuple of (mean_pairwise_abs_corr, max_pairwise_abs_corr).
    """
    n_cols = X_sub.shape[1]
    if n_cols <= 1:
        return 0.0, 0.0

    X_clean = np.nan_to_num(X_sub, nan=0.0, posinf=0.0, neginf=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr_mat = np.abs(np.corrcoef(X_clean, rowvar=False))

    upper_tri_indices = np.triu_indices(n_cols, k=1)
    upper_vals = corr_mat[upper_tri_indices]

    if len(upper_vals) == 0:
        return 0.0, 0.0

    mean_corr = float(np.mean(upper_vals))
    max_corr = float(np.max(upper_vals))
    return mean_corr, max_corr


def compute_feature_set_metadata(
    set_name: str,
    feature_names_subset: List[str],
    master_feature_names: List[str],
    X_dev: np.ndarray,
    description: str,
) -> FeatureSetMetadata:
    """Compute numerical conditioning and correlation metrics for a feature subset on DEV.

    Args:
        set_name: Unique identifier for feature set (e.g. 'SET_A_ALL').
        feature_names_subset: List of feature names included in this set.
        master_feature_names: Master list of all 10 feature names.
        X_dev: Development feature matrix (n_dev, 10).
        description: Scientific description of the set.

    Returns:
        FeatureSetMetadata container.
    """
    indices = [master_feature_names.index(f) for f in feature_names_subset]
    X_sub = X_dev[:, indices]

    # Matrix rank & Unscaled condition number
    X_clean = np.nan_to_num(X_sub, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        rank_val = int(np.linalg.matrix_rank(X_clean))
    except Exception:
        rank_val = 0

    try:
        cond_unscaled = float(np.linalg.cond(X_clean))
        if not math.isfinite(cond_unscaled):
            cond_unscaled = 1e12
    except Exception:
        cond_unscaled = 1e12

    # Robust scaled condition number (fitted on DEV subset)
    try:
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_clean)
        cond_scaled = float(np.linalg.cond(X_scaled))
        if not math.isfinite(cond_scaled):
            cond_scaled = 1e12
    except Exception:
        cond_scaled = 1e12

    mean_abs_r, max_abs_r = _compute_pairwise_abs_correlations(X_clean)

    return FeatureSetMetadata(
        set_name=set_name,
        feature_names=list(feature_names_subset),
        feature_count=len(feature_names_subset),
        indices=indices,
        matrix_rank=rank_val,
        condition_number_unscaled=cond_unscaled,
        condition_number_robust_scaled=cond_scaled,
        mean_pairwise_abs_correlation=mean_abs_r,
        max_pairwise_abs_correlation=max_abs_r,
        description=description,
    )


def compute_composite_discriminative_ranks(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    feature_names: List[str],
) -> List[Tuple[str, float]]:
    """Compute multi-metric composite discriminative ranking of features on DEV.

    Combines:
        1. Mutual Information (`mutual_info_classif`)
        2. Absolute ROC distance |ROC_AUC - 0.5|
        3. Absolute Cohen's d effect size
        4. Kolmogorov-Smirnov statistic D

    Args:
        X_dev: Development feature matrix (n_dev, n_features).
        y_dev: Development labels (n_dev,).
        feature_names: Feature column names.

    Returns:
        List of (feature_name, composite_score) sorted descending.
    """
    n_features = len(feature_names)
    X_clean = np.nan_to_num(X_dev, nan=0.0, posinf=0.0, neginf=0.0)

    # 1. MI scores
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mi = mutual_info_classif(X_clean, y_dev, random_state=42)
        except Exception:
            mi = np.zeros(n_features)

    # 2. ROC-AUC distance
    roc_dist = np.zeros(n_features)
    for i in range(n_features):
        try:
            auc_val = float(roc_auc_score(y_dev, X_clean[:, i]))
            roc_dist[i] = abs(auc_val - 0.5)
        except Exception:
            roc_dist[i] = 0.0

    # 3. Cohen's d magnitude
    cohen_d = np.zeros(n_features)
    pos_mask = y_dev == 1
    neg_mask = y_dev == 0
    for i in range(n_features):
        pos_vals = X_clean[pos_mask, i]
        neg_vals = X_clean[neg_mask, i]
        if len(pos_vals) > 1 and len(neg_vals) > 1:
            v_p = np.var(pos_vals, ddof=1)
            v_n = np.var(neg_vals, ddof=1)
            p_std = math.sqrt(((len(pos_vals) - 1) * v_p + (len(neg_vals) - 1) * v_n) / (len(y_dev) - 2))
            if p_std > 1e-12:
                cohen_d[i] = abs((np.mean(pos_vals) - np.mean(neg_vals)) / p_std)

    # 4. KS Statistic
    ks_stat = np.zeros(n_features)
    for i in range(n_features):
        pos_vals = X_clean[pos_mask, i]
        neg_vals = X_clean[neg_mask, i]
        if len(pos_vals) > 0 and len(neg_vals) > 0:
            ks_res = scipy_stats.ks_2samp(pos_vals, neg_vals)
            ks_stat[i] = float(ks_res.statistic)

    # Normalize metrics to [0, 1] range to calculate composite score
    def _norm(arr: np.ndarray) -> np.ndarray:
        rng = np.max(arr) - np.min(arr)
        return (arr - np.min(arr)) / rng if rng > 1e-12 else np.zeros_like(arr)

    composite = (
        0.30 * _norm(mi) +
        0.30 * _norm(roc_dist) +
        0.20 * _norm(cohen_d) +
        0.20 * _norm(ks_stat)
    )

    ranked_tuples = [
        (feature_names[i], float(composite[i]))
        for i in range(n_features)
    ]
    ranked_tuples.sort(key=lambda t: t[1], reverse=True)
    return ranked_tuples


# =========================================================
# PUBLIC API
# =========================================================

def construct_candidate_feature_sets(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    feature_names: List[str] = FEATURE_COLUMNS,
    delinearized_retained_features: Optional[List[str]] = None,
    out_dir: Path = PHASE6K_DIR,
) -> CandidateFeatureSetsReport:
    """Construct four scientifically motivated candidate feature sets using ONLY DEV data.

    STRICT DATA ISOLATION:
        Validation data (X_val, y_val) is NEVER passed to or accessed by this function.

    Exports:
        * ``evaluation_results/phase6k/feature_sets.json``

    Args:
        X_dev: Development feature matrix (n_dev, n_features).
        y_dev: Development binary target array.
        feature_names: Master list of 10 feature names.
        delinearized_retained_features: Pre-computed retained features from collinearity audit.
        out_dir: Output directory path.

    Returns:
        CandidateFeatureSetsReport object.
    """
    logger.info("phase6k_construct_feature_sets_start", n_dev=X_dev.shape[0])

    if delinearized_retained_features is None:
        delinearized_retained_features = [
            "mean_entailment",
            "max_entailment",
            "mean_contradiction",
            "min_support_margin",
            "num_claims",
        ]

    # --- 1. SET A — ALL (10 features) ---
    set_a_cols = list(feature_names)
    meta_a = compute_feature_set_metadata(
        set_name="SET_A_ALL",
        feature_names_subset=set_a_cols,
        master_feature_names=feature_names,
        X_dev=X_dev,
        description="Full original 10-feature Pillar 1 set",
    )

    # --- 2. SET B — DECOLLINEARIZED (5 features) ---
    set_b_cols = list(delinearized_retained_features)
    meta_b = compute_feature_set_metadata(
        set_name="SET_B_DECOLLINEARIZED",
        feature_names_subset=set_b_cols,
        master_feature_names=feature_names,
        X_dev=X_dev,
        description="Deduplicated feature set with pairwise |r| < 0.90",
    )

    # --- 3. SET C — TOP_DISCRIMINATIVE (5 features) ---
    composite_ranks = compute_composite_discriminative_ranks(X_dev, y_dev, feature_names)
    top_5_disc = [t[0] for t in composite_ranks[:5]]
    meta_c = compute_feature_set_metadata(
        set_name="SET_C_TOP_DISCRIMINATIVE",
        feature_names_subset=top_5_disc,
        master_feature_names=feature_names,
        X_dev=X_dev,
        description="Top 5 features ranked by composite discriminative power (MI, ROC-AUC, Cohen's d, KS)",
    )

    # --- 4. SET D — DECOLLINEARIZED_DISCRIMINATIVE (3 features) ---
    # Pick top 3 features from the composite discriminative ranking that belong to the de-collinearized set B
    set_d_cols = [f for f, _ in composite_ranks if f in delinearized_retained_features][:3]
    meta_d = compute_feature_set_metadata(
        set_name="SET_D_DECOLLINEARIZED_DISCRIMINATIVE",
        feature_names_subset=set_d_cols,
        master_feature_names=feature_names,
        X_dev=X_dev,
        description="Minimalist 3-feature set combining strict decorrelation and top discriminative power",
    )

    sets_dict: Dict[str, FeatureSetMetadata] = {
        "SET_A_ALL": meta_a,
        "SET_B_DECOLLINEARIZED": meta_b,
        "SET_C_TOP_DISCRIMINATIVE": meta_c,
        "SET_D_DECOLLINEARIZED_DISCRIMINATIVE": meta_d,
    }

    report = CandidateFeatureSetsReport(
        n_dev_samples=int(X_dev.shape[0]),
        n_master_features=len(feature_names),
        candidate_sets=sets_dict,
    )

    # Export JSON report
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "feature_sets.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_serializable(asdict(report)), f, indent=2)

    logger.info("phase6k_construct_feature_sets_complete", output=str(out_path), count=len(sets_dict))
    return report
