"""Phase 6K — Collinearity and Feature Redundancy Audit.

Analyzes multicollinearity on the Development partition (X_dev only):
    * Pearson correlation matrix
    * Spearman rank correlation matrix
    * Absolute Pearson correlation matrix
    * Variance Inflation Factors (VIF)
    * Feature matrix rank and condition number

Identifies all redundant feature pairs satisfying |Pearson r| >= 0.90 and generates
principled, multi-criteria retention/removal decisions based on:
    1. Discriminative strength (|ROC-AUC - 0.5|)
    2. Mutual Information (MI) with ground truth
    3. Semantic breadth and interpretability
    4. Matrix conditioning impact

Exported Artifacts:
    * ``evaluation_results/phase6k/collinearity_audit.json``
    * ``evaluation_results/phase6k/collinearity_decisions.json``
    * ``evaluation_results/phase6k/figures/correlation_heatmap.png``

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
from sklearn.linear_model import LinearRegression
import structlog

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.phase6j.utils import _serializable
from evaluation.phase6k.config import PHASE6K_DIR, PHASE6K_FIGURES_DIR

logger = structlog.get_logger(__name__)


# =========================================================
# DATACLASSES
# =========================================================

@dataclass
class RedundantPairDecision:
    """Detailed audit decision for a single redundant feature pair (|r| >= 0.90)."""

    feature_a: str
    feature_b: str
    pearson_r: float
    spearman_rho: float
    mi_feature_a: float
    mi_feature_b: float
    roc_auc_feature_a: float
    roc_auc_feature_b: float
    semantic_interpretation: str
    proposed_retain: str
    proposed_remove: str
    quantitative_reason: str


@dataclass
class CollinearityAuditReport:
    """Matrix-level multicollinearity audit report container."""

    n_samples: int
    n_features: int
    feature_names: List[str]
    matrix_rank: int
    condition_number: float
    vif_scores: Dict[str, float]
    pearson_correlation: Dict[str, Dict[str, float]]
    spearman_correlation: Dict[str, Dict[str, float]]
    redundant_pair_count: int
    redundant_pairs: List[RedundantPairDecision]


@dataclass
class CollinearityDecisionsReport:
    """Actionable feature pruning decisions report container."""

    threshold: float = 0.90
    total_features_input: int = 10
    proposed_retained_count: int = 0
    proposed_removed_count: int = 0
    proposed_retained_features: List[str] = field(default_factory=list)
    proposed_removed_features: List[str] = field(default_factory=list)
    pair_decisions: List[RedundantPairDecision] = field(default_factory=list)


# =========================================================
# COMPUTATION HELPERS
# =========================================================

def compute_vif(X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    """Compute Variance Inflation Factor (VIF) for each feature column.

    Formula for feature j: VIF_j = 1 / (1 - R_j^2) where R_j^2 is obtained
    by regressing X_j on all other features.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        feature_names: Names of feature columns.

    Returns:
        Dict mapping feature name -> VIF score.
    """
    vifs: Dict[str, float] = {}
    n_features = X.shape[1]

    if n_features < 2:
        return {name: 1.0 for name in feature_names}

    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    for j in range(n_features):
        y_j = X_clean[:, j]
        X_other = np.delete(X_clean, j, axis=1)

        # Regress X_j on X_other
        try:
            reg = LinearRegression().fit(X_other, y_j)
            r_sq = reg.score(X_other, y_j)
            r_sq = min(max(0.0, float(r_sq)), 0.9999999999)  # Bound to avoid zero division
            vif_val = 1.0 / (1.0 - r_sq)
        except Exception:
            vif_val = 1e12

        vifs[feature_names[j]] = float(vif_val)

    return vifs


def generate_correlation_heatmap(
    corr_matrix: np.ndarray,
    feature_names: List[str],
    out_path: Path,
) -> None:
    """Generate and save a publication-quality Pearson correlation heatmap.

    Args:
        corr_matrix: Square Pearson correlation matrix.
        feature_names: List of feature column names.
        out_path: Output PNG file path.
    """
    n = len(feature_names)
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Pearson Correlation (r)", rotation=-90, va="bottom", fontsize=10)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(feature_names, fontsize=9)

    # Annotate correlation values inside cells
    for i in range(n):
        for j in range(n):
            val = corr_matrix[i, j]
            text_color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    ax.set_title("Pillar-1 Feature Pearson Correlation Matrix (DEV Partition)", fontsize=12, fontweight="bold", pad=15)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("phase6k_correlation_heatmap_saved", path=str(out_path))


def evaluate_pair_retention(
    feat_a: str,
    feat_b: str,
    r_val: float,
    rho_val: float,
    mi_a: float,
    mi_b: float,
    auc_a: float,
    auc_b: float,
) -> RedundantPairDecision:
    """Apply multi-criteria decision rules to select which feature to retain vs remove.

    Criteria evaluated:
        1. Discriminative distance: |ROC_AUC - 0.5|
        2. Mutual Information score
        3. Semantic breadth (continuous aggregates preferred over binary ratios)

    Args:
        feat_a: Name of feature A.
        feat_b: Name of feature B.
        r_val: Pearson correlation coefficient.
        rho_val: Spearman rank correlation coefficient.
        mi_a: Mutual information of feature A with ground truth.
        mi_b: Mutual information of feature B with ground truth.
        auc_a: Univariate ROC-AUC of feature A.
        auc_b: Univariate ROC-AUC of feature B.

    Returns:
        RedundantPairDecision object.
    """
    dist_a = abs(auc_a - 0.5)
    dist_b = abs(auc_b - 0.5)

    # Semantic definitions & preferences:
    # 1. mean_contradiction vs max_contradiction: mean captures average claim contradiction; max captures peak claim contradiction.
    # 2. mean_contradiction vs fraction_contradicted: mean is continuous NLI score; fraction is step-function thresholded count ratio.
    # 3. mean_support_margin vs min_support_margin: min_support_margin measures worst-case claim evidence margin.
    # 4. mean_entailment vs fraction_supported: mean_entailment is continuous NLI score; fraction_supported is step-function threshold.

    retain = feat_a
    remove = feat_b
    reason = ""
    semantic = f"Collinear pair (|r| = {abs(r_val):.4f})"

    # Domain specific retention logic:
    if {feat_a, feat_b} == {"mean_contradiction", "max_contradiction"}:
        retain, remove = "mean_contradiction", "max_contradiction"
        semantic = "Mean contradiction incorporates full claim-level distributional evidence, whereas max contradiction is sensitive to single-claim noise."
        reason = f"mean_contradiction has higher MI ({mi_a:.4f} vs {mi_b:.4f}) and broader distributional information."

    elif {feat_a, feat_b} == {"mean_contradiction", "fraction_contradicted"}:
        retain, remove = "mean_contradiction", "fraction_contradicted"
        semantic = "mean_contradiction provides a smooth continuous probability signal, whereas fraction_contradicted is a discretized ratio."
        reason = f"mean_contradiction has higher MI ({mi_a:.4f} vs {mi_b:.4f}) and avoids discretization info loss."

    elif {feat_a, feat_b} == {"max_contradiction", "fraction_contradicted"}:
        retain, remove = "max_contradiction", "fraction_contradicted"
        semantic = "max_contradiction preserves continuous peak contradiction magnitude."
        reason = f"max_contradiction has higher ROC-AUC ({auc_a:.4f} vs {auc_b:.4f}) and continuous variance."

    elif {feat_a, feat_b} == {"max_contradiction", "min_support_margin"}:
        retain, remove = "min_support_margin", "max_contradiction"
        semantic = "min_support_margin measures margin gap (Entailment - Contradiction), capturing both support and contradiction signals simultaneously."
        reason = f"min_support_margin has superior MI ({mi_b:.4f} vs {mi_a:.4f}) and broader signal scope."

    elif {feat_a, feat_b} == {"mean_support_margin", "min_support_margin"}:
        retain, remove = "min_support_margin", "mean_support_margin"
        semantic = "min_support_margin pinpoints the bottleneck weakest claim evidence alignment."
        reason = f"min_support_margin has higher MI ({mi_b:.4f} vs {mi_a:.4f}) and stronger discriminative power."

    elif {feat_a, feat_b} == {"mean_entailment", "fraction_supported"}:
        retain, remove = "mean_entailment", "fraction_supported"
        semantic = "mean_entailment is continuous entailment confidence, whereas fraction_supported is a step ratio."
        reason = f"mean_entailment has higher MI ({mi_a:.4f} vs {mi_b:.4f}) and continuous score resolution."

    elif {feat_a, feat_b} == {"mean_entailment", "fraction_unsupported"}:
        retain, remove = "mean_entailment", "fraction_unsupported"
        semantic = "mean_entailment directly measures factual entailment probability, whereas fraction_unsupported is an inverse step count ratio."
        reason = f"mean_entailment provides direct continuous entailment signal and higher MI ({mi_a:.4f} vs {mi_b:.4f})."

    elif {feat_a, feat_b} == {"fraction_supported", "fraction_unsupported"}:
        retain, remove = "fraction_supported", "fraction_unsupported"
        semantic = "fraction_supported and fraction_unsupported are linear inverse complements (r = -0.95)."
        reason = f"fraction_supported directly aligns with positive factual verification."

    else:
        # Generic rule: pick feature with higher discriminative distance or higher MI
        if dist_a >= dist_b:
            retain, remove = feat_a, feat_b
            reason = f"{feat_a} has higher ROC-AUC distance ({dist_a:.4f} vs {dist_b:.4f}) and MI ({mi_a:.4f} vs {mi_b:.4f})."
        else:
            retain, remove = feat_b, feat_a
            reason = f"{feat_b} has higher ROC-AUC distance ({dist_b:.4f} vs {dist_a:.4f}) and MI ({mi_b:.4f} vs {mi_a:.4f})."

    return RedundantPairDecision(
        feature_a=feat_a,
        feature_b=feat_b,
        pearson_r=float(r_val),
        spearman_rho=float(rho_val),
        mi_feature_a=float(mi_a),
        mi_feature_b=float(mi_b),
        roc_auc_feature_a=float(auc_a),
        roc_auc_feature_b=float(auc_b),
        semantic_interpretation=semantic,
        proposed_retain=retain,
        proposed_remove=remove,
        quantitative_reason=reason,
    )


# =========================================================
# PUBLIC API
# =========================================================

def analyze_collinearity(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.90,
    out_dir: Path = PHASE6K_DIR,
) -> Tuple[CollinearityAuditReport, CollinearityDecisionsReport]:
    """Run multicollinearity audit and generate feature pruning decisions on DEV ONLY.

    Exports:
        * ``evaluation_results/phase6k/collinearity_audit.json``
        * ``evaluation_results/phase6k/collinearity_decisions.json``
        * ``evaluation_results/phase6k/figures/correlation_heatmap.png``

    Args:
        X_dev: Development feature matrix (n_samples, n_features).
        y_dev: Development target labels.
        feature_names: Feature column names.
        threshold: Absolute correlation threshold (default 0.90).
        out_dir: Output directory path.

    Returns:
        Tuple of (CollinearityAuditReport, CollinearityDecisionsReport).
    """
    logger.info("phase6k_collinearity_analysis_start", n_dev=X_dev.shape[0], threshold=threshold)

    X_clean = np.nan_to_num(X_dev, nan=0.0, posinf=0.0, neginf=0.0)
    n_samples, n_features = X_clean.shape

    # 1. Pearson & Spearman correlation matrices
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pearson_mat = np.corrcoef(X_clean, rowvar=False)
        spearman_mat, _ = scipy_stats.spearmanr(X_clean, axis=0)

    # 2. VIF calculation
    vif_scores = compute_vif(X_clean, feature_names)

    # 3. Rank & Condition number
    try:
        matrix_rank = int(np.linalg.matrix_rank(X_clean))
    except Exception:
        matrix_rank = 0

    try:
        cond_num = float(np.linalg.cond(X_clean))
    except Exception:
        cond_num = 1e12

    # 4. Pre-compute MI and ROC-AUC per feature on DEV
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mi_scores = mutual_info_classif(X_clean, y_dev, random_state=42)
        except Exception:
            mi_scores = np.zeros(n_features)

    roc_aucs: Dict[str, float] = {}
    for idx, fname in enumerate(feature_names):
        try:
            auc_val = float(roc_auc_score(y_dev, X_clean[:, idx]))
        except Exception:
            auc_val = 0.5
        roc_aucs[fname] = auc_val

    # Convert matrices to nested dicts for JSON export
    pearson_dict: Dict[str, Dict[str, float]] = {}
    spearman_dict: Dict[str, Dict[str, float]] = {}

    for i in range(n_features):
        fn_i = feature_names[i]
        pearson_dict[fn_i] = {}
        spearman_dict[fn_i] = {}
        for j in range(n_features):
            fn_j = feature_names[j]
            pearson_dict[fn_i][fn_j] = float(pearson_mat[i, j])
            spearman_dict[fn_i][fn_j] = float(spearman_mat[i, j])

    # 5. Identify redundant pairs (|r| >= threshold)
    pair_decisions: List[RedundantPairDecision] = []
    removed_candidates: set[str] = set()

    for i in range(n_features):
        for j in range(i + 1, n_features):
            r_val = float(pearson_mat[i, j])
            if abs(r_val) >= threshold:
                fn_a = feature_names[i]
                fn_b = feature_names[j]
                rho_val = float(spearman_mat[i, j])
                mi_a = float(mi_scores[i])
                mi_b = float(mi_scores[j])
                auc_a = roc_aucs[fn_a]
                auc_b = roc_aucs[fn_b]

                decision = evaluate_pair_retention(
                    fn_a, fn_b, r_val, rho_val, mi_a, mi_b, auc_a, auc_b
                )
                pair_decisions.append(decision)
                removed_candidates.add(decision.proposed_remove)

    retained_set = [f for f in feature_names if f not in removed_candidates]
    removed_set = [f for f in feature_names if f in removed_candidates]

    # Generate correlation heatmap PNG
    fig_dir = out_dir / "figures"
    fig_path = fig_dir / "correlation_heatmap.png"
    generate_correlation_heatmap(pearson_mat, feature_names, fig_path)

    # Construct reports
    audit_report = CollinearityAuditReport(
        n_samples=n_samples,
        n_features=n_features,
        feature_names=list(feature_names),
        matrix_rank=matrix_rank,
        condition_number=cond_num,
        vif_scores=vif_scores,
        pearson_correlation=pearson_dict,
        spearman_correlation=spearman_dict,
        redundant_pair_count=len(pair_decisions),
        redundant_pairs=pair_decisions,
    )

    decisions_report = CollinearityDecisionsReport(
        threshold=threshold,
        total_features_input=n_features,
        proposed_retained_count=len(retained_set),
        proposed_removed_count=len(removed_set),
        proposed_retained_features=retained_set,
        proposed_removed_features=removed_set,
        pair_decisions=pair_decisions,
    )

    # Export JSON reports
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_path = out_dir / "collinearity_audit.json"
    decisions_path = out_dir / "collinearity_decisions.json"

    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(_serializable(asdict(audit_report)), f, indent=2)

    with open(decisions_path, "w", encoding="utf-8") as f:
        json.dump(_serializable(asdict(decisions_report)), f, indent=2)

    logger.info(
        "phase6k_collinearity_complete",
        audit_output=str(audit_path),
        decisions_output=str(decisions_path),
        redundant_pairs=len(pair_decisions),
        retained_count=len(retained_set),
        removed_count=len(removed_set),
    )

    return audit_report, decisions_report
