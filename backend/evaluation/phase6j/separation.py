"""Phase 6J — Class-conditional separation metrics and feature ranking.

Measures how effectively each numerical feature discriminates between
hallucinated (positive) and factual (negative) outputs.

Computes for every feature:
    * ROC-AUC
    * Mutual Information
    * Cohen's d
    * Point Biserial Correlation
    * Mann-Whitney U test (statistic & p-value)
    * Welch's t-test (statistic & p-value)
    * Kolmogorov-Smirnov test (statistic & p-value)

Outputs:
    * Exported artifact: ``feature_separation.json``
    * ROC curves figure for top features: ``figures/top10_roc_curves.png``

Detection:
    * Weak features (|ROC-AUC - 0.5| < 0.05, |Cohen's d| < 0.2)
    * Highly discriminative features (ROC-AUC > 0.7 or < 0.3, |Cohen's d| > 0.5)
    * Redundant feature pairs (absolute correlation > 0.9)

This module is analysis-only. It never modifies feature values.
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
from sklearn.metrics import roc_auc_score, roc_curve
import structlog
from evaluation.phase6j.utils import _serializable

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

logger = structlog.get_logger(__name__)


# =========================================================
# DATA CLASSES
# =========================================================

@dataclass
class TestResult:
    """Statistical test result containing test statistic and p-value."""

    statistic: float = 0.0
    p_value: float = 1.0


@dataclass
class FeatureSeparation:
    """Comprehensive class-conditional separation metrics for a single feature."""

    name: str
    rank: int = 0

    # Primary metrics
    roc_auc: float = 0.5
    roc_auc_distance: float = 0.0  # |roc_auc - 0.5|
    mutual_information: float = 0.0
    cohens_d: float = 0.0
    point_biserial_r: float = 0.0

    # Hypothesis tests
    mann_whitney_u: TestResult = field(default_factory=TestResult)
    welch_t_test: TestResult = field(default_factory=TestResult)
    kolmogorov_smirnov: TestResult = field(default_factory=TestResult)

    # Class means & stds
    mean_positive: float = 0.0
    mean_negative: float = 0.0
    std_positive: float = 0.0
    std_negative: float = 0.0

    # Classification
    discrimination_category: str = "weak"  # 'highly_discriminative', 'moderate', 'weak'
    anomalies: List[str] = field(default_factory=list)


@dataclass
class RedundantPair:
    """Pair of features with high mutual correlation indicating redundancy."""

    feature1: str
    feature2: str
    correlation: float


@dataclass
class SeparationReport:
    """Aggregated class separation report for all features."""

    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0
    feature_count: int = 0

    # Ranked feature list
    ranked_features: List[FeatureSeparation] = field(default_factory=list)
    features: Dict[str, FeatureSeparation] = field(default_factory=dict)

    # Category summaries
    highly_discriminative_features: List[str] = field(default_factory=list)
    weak_features: List[str] = field(default_factory=list)
    redundant_pairs: List[RedundantPair] = field(default_factory=list)

    # Figure path
    roc_curve_figure_path: str = ""


# =========================================================
# PURE COMPUTATION FUNCTIONS
# =========================================================

def _compute_cohens_d(pos: np.ndarray, neg: np.ndarray) -> float:
    """Compute Cohen's d effect size between positive and negative samples.

    Formula: (mean(pos) - mean(neg)) / pooled_std
    """
    n_pos, n_neg = len(pos), len(neg)
    if n_pos < 2 or n_neg < 2:
        return 0.0

    var_pos = np.var(pos, ddof=1)
    var_neg = np.var(neg, ddof=1)
    pooled_std = math.sqrt(((n_pos - 1) * var_pos + (n_neg - 1) * var_neg) / (n_pos + n_neg - 2))

    if pooled_std < 1e-12:
        return 0.0

    return float((np.mean(pos) - np.mean(neg)) / pooled_std)


def compute_single_feature_separation(
    col: np.ndarray,
    y: np.ndarray,
    feature_name: str,
    mi_score: float = 0.0,
) -> FeatureSeparation:
    """Compute class-conditional separation metrics for a single feature column.

    Pure function — no side effects.

    Args:
        col: 1-D numpy array for the feature (finite values).
        y: 1-D binary label array aligned with col.
        feature_name: Name of the feature column.
        mi_score: Pre-computed mutual information score.

    Returns:
        FeatureSeparation object with all 7 separation metrics and test results.
    """
    fs = FeatureSeparation(name=feature_name, mutual_information=mi_score)

    pos = col[y == 1]
    neg = col[y == 0]

    fs.mean_positive = float(np.mean(pos)) if len(pos) > 0 else 0.0
    fs.mean_negative = float(np.mean(neg)) if len(neg) > 0 else 0.0
    fs.std_positive = float(np.std(pos, ddof=0)) if len(pos) > 0 else 0.0
    fs.std_negative = float(np.std(neg, ddof=0)) if len(neg) > 0 else 0.0

    if len(pos) == 0 or len(neg) == 0 or np.std(col) < 1e-12:
        fs.discrimination_category = "weak"
        fs.anomalies.append("zero variance or empty class")
        return fs

    # 1. ROC-AUC
    try:
        auc = float(roc_auc_score(y, col))
        fs.roc_auc = auc
        fs.roc_auc_distance = float(abs(auc - 0.5))
    except Exception:
        fs.roc_auc = 0.5
        fs.roc_auc_distance = 0.0

    # 2. Cohen's d
    fs.cohens_d = _compute_cohens_d(pos, neg)

    # 3. Point Biserial Correlation
    try:
        pb_r, _ = scipy_stats.pointbiserialr(y, col)
        fs.point_biserial_r = float(pb_r) if math.isfinite(pb_r) else 0.0
    except Exception:
        fs.point_biserial_r = 0.0

    # 4. Mann-Whitney U Test
    try:
        mwu = scipy_stats.mannwhitneyu(pos, neg, alternative="two-sided")
        fs.mann_whitney_u = TestResult(statistic=float(mwu.statistic), p_value=float(mwu.pvalue))
    except Exception:
        fs.mann_whitney_u = TestResult()

    # 5. Welch's t-test
    try:
        tt = scipy_stats.ttest_ind(pos, neg, equal_var=False)
        fs.welch_t_test = TestResult(
            statistic=float(tt.statistic) if math.isfinite(tt.statistic) else 0.0,
            p_value=float(tt.pvalue) if math.isfinite(tt.pvalue) else 1.0,
        )
    except Exception:
        fs.welch_t_test = TestResult()

    # 6. Kolmogorov-Smirnov Test
    try:
        ks = scipy_stats.ks_2samp(pos, neg)
        fs.kolmogorov_smirnov = TestResult(statistic=float(ks.statistic), p_value=float(ks.pvalue))
    except Exception:
        fs.kolmogorov_smirnov = TestResult()

    # Classification into discrimination categories
    if fs.roc_auc_distance > 0.2 or abs(fs.cohens_d) > 0.5 or fs.mutual_information > 0.1:
        fs.discrimination_category = "highly_discriminative"
    elif fs.roc_auc_distance < 0.05 and abs(fs.cohens_d) < 0.2 and fs.mutual_information < 0.02:
        fs.discrimination_category = "weak"
    else:
        fs.discrimination_category = "moderate"

    return fs


def find_redundant_pairs(
    X: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.90,
) -> List[RedundantPair]:
    """Find pairs of features with high linear correlation.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        feature_names: Names of feature columns.
        threshold: Absolute correlation threshold (default 0.90).

    Returns:
        List of RedundantPair objects.
    """
    redundant: List[RedundantPair] = []
    n_features = len(feature_names)
    if n_features < 2:
        return redundant

    # Clean array for correlation calculation
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr_matrix = np.corrcoef(X_clean, rowvar=False)

    for i in range(n_features):
        for j in range(i + 1, n_features):
            val = corr_matrix[i, j]
            if math.isfinite(val) and abs(val) >= threshold:
                redundant.append(
                    RedundantPair(
                        feature1=feature_names[i],
                        feature2=feature_names[j],
                        correlation=float(val),
                    )
                )

    return redundant


def plot_top_roc_curves(
    X: np.ndarray,
    y: np.ndarray,
    ranked_features: List[FeatureSeparation],
    out_fig_path: Path,
    top_n: int = 10,
) -> None:
    """Generate and save ROC curve plot for top N discriminating features.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Binary target labels.
        ranked_features: Features ordered by separation rank.
        out_fig_path: Output PNG file path.
        top_n: Number of top features to display (default 10).
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    top_features = ranked_features[:top_n]
    feature_name_to_idx = {name: idx for idx, name in enumerate([f.name for f in ranked_features])}

    cmap = plt.get_cmap("tab10")

    for i, feat in enumerate(top_features):
        col = X[:, feature_name_to_idx[feat.name]]
        # Align direction if ROC < 0.5 (invert for display)
        score_col = col if feat.roc_auc >= 0.5 else -col
        try:
            fpr, tpr, _ = roc_curve(y, score_col)
            auc_val = roc_auc_score(y, score_col)
            color = cmap(i % 10)
            ax.plot(
                fpr,
                tpr,
                lw=1.5,
                color=color,
                label=f"{feat.name} (AUC={auc_val:.3f})",
            )
        except Exception as e:
            logger.warning("phase6j_roc_curve_error", feature=feat.name, error=str(e))

    # Diagonal chance line
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1.0, label="Random Chance")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(f"ROC Curves for Top {len(top_features)} Features", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    out_fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_fig_path), dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("phase6j_top10_roc_saved", path=str(out_fig_path))


# =========================================================
# PUBLIC API
# =========================================================

def compute_separation(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    out_dir: Path,
) -> SeparationReport:
    """Compute class-conditional separation metrics and feature ranking.

    This is the single public entry point for the separation module.

    Computes for every feature:
        ROC-AUC, Mutual Information, Cohen's d, Point Biserial r,
        Mann-Whitney U, Welch's t-test, Kolmogorov-Smirnov test.

    Generates:
        * ``feature_separation.json``
        * ``figures/top10_roc_curves.png``

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Binary label array of shape (n_samples,).
        feature_names: Ordered list of feature column names.
        out_dir: Directory to write output artifacts.

    Returns:
        SeparationReport containing ranked feature objects and redundancy audit.
    """
    logger.info("phase6j_separation_start", n_samples=X.shape[0], n_features=len(feature_names))

    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Pre-compute Mutual Information
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mi_scores = mutual_info_classif(X_clean, y, random_state=42)
        except Exception as e:
            logger.warning("phase6j_mi_computation_failed", error=str(e))
            mi_scores = np.zeros(len(feature_names))

    # Compute separation metrics per feature
    features_dict: Dict[str, FeatureSeparation] = {}
    feature_list: List[FeatureSeparation] = []

    for idx, name in enumerate(feature_names):
        col = X_clean[:, idx]
        mi_val = float(mi_scores[idx]) if idx < len(mi_scores) else 0.0
        fs = compute_single_feature_separation(col, y, name, mi_score=mi_val)
        features_dict[name] = fs
        feature_list.append(fs)

    # Rank features primarily by ROC-AUC distance from 0.5, secondarily by MI
    feature_list.sort(key=lambda f: (f.roc_auc_distance, f.mutual_information, abs(f.cohens_d)), reverse=True)

    # Assign rank 1..N
    for rank_idx, feat in enumerate(feature_list, start=1):
        feat.rank = rank_idx
        features_dict[feat.name].rank = rank_idx

    # Redundancy check
    redundant_pairs = find_redundant_pairs(X_clean, feature_names, threshold=0.90)

    # Categorize features
    highly_disc = [f.name for f in feature_list if f.discrimination_category == "highly_discriminative"]
    weak_feats = [f.name for f in feature_list if f.discrimination_category == "weak"]

    # Generate ROC curve plot
    fig_dir = out_dir / "figures"
    fig_path = fig_dir / "top10_roc_curves.png"
    plot_top_roc_curves(X_clean, y, feature_list, fig_path, top_n=min(10, len(feature_list)))

    report = SeparationReport(
        n_samples=int(X.shape[0]),
        n_positive=int((y == 1).sum()),
        n_negative=int((y == 0).sum()),
        feature_count=len(feature_names),
        ranked_features=feature_list,
        features=features_dict,
        highly_discriminative_features=highly_disc,
        weak_features=weak_feats,
        redundant_pairs=redundant_pairs,
        roc_curve_figure_path=str(fig_path),
    )

    # Export JSON
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "feature_separation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_serializable(asdict(report)), f, indent=2)

    logger.info(
        "phase6j_separation_complete",
        output=str(out_path),
        highly_discriminative=len(highly_disc),
        weak=len(weak_feats),
        redundant_pairs=len(redundant_pairs),
    )

    return report
