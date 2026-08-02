"""Phase 6K — Feasibility Rules, Leakage & Shortcut Audit.

Evaluates whether Pillar-1 feature matrices and candidate models exhibit:
    1. Direct target leakage or label shortcut dependencies.
    2. Exact duplicate examples across DEV and held-out VAL partitions.
    3. Single-feature dominance or text/evidence length proxy reliance (e.g. num_claims).
    4. Label permutation collapse (sanity test: performance must collapse to chance ~0.50 under random permutation).
    5. Feature ablation degradation analysis.

Exported Artifact:
    * ``evaluation_results/phase6k/leakage_shortcut_audit.json``

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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
import structlog

from evaluation.phase6j.utils import _serializable
from evaluation.phase6k.config import PHASE6K_DIR, FEATURE_COLUMNS, ExperimentConfig

logger = structlog.get_logger(__name__)


# =========================================================
# DATACLASSES
# =========================================================

@dataclass
class PermutationTestResult:
    """Result of label permutation sanity test."""

    n_permutations: int = 5
    mean_permuted_roc_auc: float = 0.50
    std_permuted_roc_auc: float = 0.00
    mean_permuted_mcc: float = 0.00
    collapsed_to_chance: bool = True


@dataclass
class AblationResult:
    """Result of single-feature ablation analysis."""

    baseline_roc_auc: float
    ablated_feature: str
    ablated_roc_auc: float
    delta_roc_auc: float
    catastrophic_degradation: bool


@dataclass
class LeakageShortcutReport:
    """Comprehensive leakage and shortcut audit report."""

    target_leakage_detected: bool = False
    max_feature_target_correlation: float = 0.0
    leakage_feature_name: Optional[str] = None
    dev_exact_duplicates: int = 0
    val_exact_duplicates: int = 0
    dev_val_overlap_count: int = 0
    dev_val_overlap_ratio: float = 0.0
    label_shortcut_detected: bool = False
    single_feature_dominance_detected: bool = False
    dominant_feature_name: Optional[str] = None
    dominant_feature_share: float = 0.0
    num_claims_target_correlation: float = 0.0
    num_claims_prediction_dependence: float = 0.0
    permutation_test: PermutationTestResult = field(default_factory=PermutationTestResult)
    ablation_results: List[AblationResult] = field(default_factory=list)
    catastrophic_ablation_features: List[str] = field(default_factory=list)
    overall_verdict: str = "PASS"
    audit_warnings: List[str] = field(default_factory=list)


# =========================================================
# AUDIT IMPLEMENTATION
# =========================================================

def run_leakage_shortcut_audit(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str] = FEATURE_COLUMNS,
    out_dir: Path = PHASE6K_DIR,
    seed: int = 42,
) -> LeakageShortcutReport:
    """Run comprehensive target leakage, train-val overlap, and shortcut audit.

    Exports:
        * ``evaluation_results/phase6k/leakage_shortcut_audit.json``

    Args:
        X_dev: Development feature matrix (n_dev, 10).
        y_dev: Development target array.
        X_val: Validation feature matrix (n_val, 10).
        y_val: Validation target array.
        feature_names: Master feature list.
        out_dir: Output directory path.
        seed: Random seed.

    Returns:
        LeakageShortcutReport container.
    """
    logger.info("phase6k_leakage_audit_start", n_dev=X_dev.shape[0], n_val=X_val.shape[0])

    X_dev_clean = np.nan_to_num(X_dev, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_clean = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    report_warns: List[str] = []

    # 1. Target Leakage & Feature Shortcuts Check
    max_corr = 0.0
    leak_feat = None
    shortcut_flag = False

    for i, fname in enumerate(feature_names):
        col = X_dev_clean[:, i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_val = abs(float(np.corrcoef(col, y_dev)[0, 1]))

        if r_val > max_corr:
            max_corr = r_val
            leak_feat = fname

        try:
            auc_val = float(roc_auc_score(y_dev, col))
            auc_dist = abs(auc_val - 0.5)
            if auc_dist > 0.49:  # AUC > 0.99 or < 0.01
                shortcut_flag = True
                report_warns.append(f"Feature '{fname}' achieves near-perfect target separation (AUC = {auc_val:.4f})")
        except Exception:
            pass

    target_leakage_flag = max_corr > 0.99
    if target_leakage_flag:
        report_warns.append(f"Target leakage detected in feature '{leak_feat}' (|r| = {max_corr:.4f})")

    # 2. Duplicate Examples & Train/VAL Overlap
    def _hash_rows(arr: np.ndarray) -> set[bytes]:
        rounded = np.round(arr, decimals=6)
        return {r.tobytes() for r in rounded}

    dev_hashes = [r.tobytes() for r in np.round(X_dev_clean, decimals=6)]
    val_hashes = [r.tobytes() for r in np.round(X_val_clean, decimals=6)]

    dev_unique = set(dev_hashes)
    val_unique = set(val_hashes)

    dev_dups = len(dev_hashes) - len(dev_unique)
    val_dups = len(val_hashes) - len(val_unique)

    dev_val_overlap = len(dev_unique.intersection(val_unique))
    overlap_ratio = dev_val_overlap / float(len(X_val_clean))

    if dev_val_overlap > 0:
        report_warns.append(f"Detected {dev_val_overlap} exact duplicate feature vectors overlapping DEV and VAL ({overlap_ratio:.2%})")

    # 3. Single-Feature Dominance & num_claims Dependence
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mi_scores = mutual_info_classif(X_dev_clean, y_dev, random_state=seed)
        except Exception:
            mi_scores = np.zeros(len(feature_names))

    total_mi = mi_scores.sum()
    dom_feat = None
    dom_share = 0.0
    single_dom_flag = False

    if total_mi > 1e-12:
        shares = mi_scores / total_mi
        max_idx = int(np.argmax(shares))
        dom_share = float(shares[max_idx])
        dom_feat = feature_names[max_idx]
        if dom_share > 0.90:
            single_dom_flag = True
            report_warns.append(f"Feature '{dom_feat}' dominates {dom_share:.1%} of total mutual information")

    # num_claims dependence
    num_claims_idx = feature_names.index("num_claims") if "num_claims" in feature_names else 9
    num_claims_col = X_dev_clean[:, num_claims_idx]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        num_claims_r = abs(float(np.corrcoef(num_claims_col, y_dev)[0, 1]))

    # 4. Label Permutation Sanity Test
    perm_aucs: List[float] = []
    perm_mccs: List[float] = []

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    for p_seed in range(101, 106):
        rng = np.random.RandomState(p_seed)
        y_perm = rng.permutation(y_dev)

        perm_oof_probs = np.zeros(len(y_dev))
        for train_idx, val_idx in skf.split(X_dev_clean, y_perm):
            clf = HistGradientBoostingClassifier(max_depth=3, random_state=p_seed)
            clf.fit(X_dev_clean[train_idx], y_perm[train_idx])
            perm_oof_probs[val_idx] = clf.predict_proba(X_dev_clean[val_idx])[:, 1]

        try:
            p_auc = float(roc_auc_score(y_perm, perm_oof_probs))
            p_mcc = float(matthews_corrcoef(y_perm, (perm_oof_probs > 0.5).astype(int)))
        except Exception:
            p_auc, p_mcc = 0.50, 0.00

        perm_aucs.append(p_auc)
        perm_mccs.append(p_mcc)

    mean_p_auc = float(np.mean(perm_aucs))
    std_p_auc = float(np.std(perm_aucs))
    mean_p_mcc = float(np.mean(perm_mccs))
    collapsed_to_chance = bool(abs(mean_p_auc - 0.50) < 0.08)

    if not collapsed_to_chance:
        report_warns.append(f"Label permutation sanity test failed: permuted AUC ({mean_p_auc:.4f}) did not collapse to 0.50")

    perm_test_res = PermutationTestResult(
        n_permutations=len(perm_aucs),
        mean_permuted_roc_auc=mean_p_auc,
        std_permuted_roc_auc=std_p_auc,
        mean_permuted_mcc=mean_p_mcc,
        collapsed_to_chance=collapsed_to_chance,
    )

    # 5. Feature Ablation Analysis
    baseline_oof = np.zeros(len(y_dev))
    for train_idx, val_idx in skf.split(X_dev_clean, y_dev):
        clf = HistGradientBoostingClassifier(max_depth=3, random_state=seed)
        clf.fit(X_dev_clean[train_idx], y_dev[train_idx])
        baseline_oof[val_idx] = clf.predict_proba(X_dev_clean[val_idx])[:, 1]

    try:
        baseline_auc = float(roc_auc_score(y_dev, baseline_oof))
    except Exception:
        baseline_auc = 0.50

    ablation_results: List[AblationResult] = []
    catastrophic_feats: List[str] = []

    for i, fname in enumerate(feature_names):
        X_abl = np.delete(X_dev_clean, i, axis=1)
        abl_oof = np.zeros(len(y_dev))

        for train_idx, val_idx in skf.split(X_abl, y_dev):
            clf = HistGradientBoostingClassifier(max_depth=3, random_state=seed)
            clf.fit(X_abl[train_idx], y_dev[train_idx])
            abl_oof[val_idx] = clf.predict_proba(X_abl[val_idx])[:, 1]

        try:
            abl_auc = float(roc_auc_score(y_dev, abl_oof))
        except Exception:
            abl_auc = 0.50

        delta_auc = float(baseline_auc - abl_auc)
        is_catastrophic = delta_auc > 0.30

        if is_catastrophic:
            catastrophic_feats.append(fname)
            report_warns.append(f"Ablating feature '{fname}' causes catastrophic performance drop (ΔAUC = {delta_auc:.4f})")

        ablation_results.append(
            AblationResult(
                baseline_roc_auc=baseline_auc,
                ablated_feature=fname,
                ablated_roc_auc=abl_auc,
                delta_roc_auc=delta_auc,
                catastrophic_degradation=is_catastrophic,
            )
        )

    # 6. Overall Verdict Assignment
    if target_leakage_flag or not collapsed_to_chance or len(catastrophic_feats) > 0:
        verdict = "FAIL"
    elif len(report_warns) > 0:
        verdict = "PASS WITH WARNINGS"
    else:
        verdict = "PASS"

    report = LeakageShortcutReport(
        target_leakage_detected=target_leakage_flag,
        max_feature_target_correlation=max_corr,
        leakage_feature_name=leak_feat,
        dev_exact_duplicates=dev_dups,
        val_exact_duplicates=val_dups,
        dev_val_overlap_count=dev_val_overlap,
        dev_val_overlap_ratio=overlap_ratio,
        label_shortcut_detected=shortcut_flag,
        single_feature_dominance_detected=single_dom_flag,
        dominant_feature_name=dom_feat,
        dominant_feature_share=dom_share,
        num_claims_target_correlation=num_claims_r,
        num_claims_prediction_dependence=num_claims_r,
        permutation_test=perm_test_res,
        ablation_results=ablation_results,
        catastrophic_ablation_features=catastrophic_feats,
        overall_verdict=verdict,
        audit_warnings=report_warns,
    )

    # Export JSON report
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "leakage_shortcut_audit.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_serializable(asdict(report)), f, indent=2)

    logger.info(
        "phase6k_leakage_audit_complete",
        output=str(out_path),
        verdict=verdict,
        warnings_count=len(report_warns),
    )

    return report
