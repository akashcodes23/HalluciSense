"""Phase 6K — Numerical Stability Gate (1,000-Example Benchmark).

Evaluates LogisticRegression optimization stability across 16 configurations:
    4 Feature Sets x 4 Preprocessing Scalers

Data: Deterministic 1,000-example stratified subset from DEV (seed=42).
VAL is NEVER used for this stability gate.

Primary Objective: Verify that optimization completes without numerical instability
(zero overflow, divide-by-zero, or invalid matmul warnings and clean convergence).

Exported Artifact:
    * ``evaluation_results/phase6k/stability_gate_1000.json``

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import structlog

from evaluation.phase6j.utils import _serializable
from evaluation.phase6k.config import PHASE6K_DIR, FEATURE_COLUMNS
from evaluation.phase6k.preprocessing import fit_transform_strategy
from evaluation.phase6k.feature_selection import construct_candidate_feature_sets

logger = structlog.get_logger(__name__)

SCALERS_TO_TEST: List[str] = [
    "Original",
    "StandardScaler",
    "RobustScaler",
    "QuantileTransformer",
]


# =========================================================
# DATACLASSES
# =========================================================

@dataclass
class CandidateEvaluationResult:
    """Evaluation result for one candidate combination of (Scaler, Subset, Classifier)."""

    candidate_id: str
    scaler_name: str
    subset_name: str
    model_name: str
    feature_count: int
    feature_names: List[str]
    dev_cv_mcc_mean: float = 0.0
    val_accuracy: float = 0.0
    val_roc_auc: float = 0.0
    val_mcc: float = 0.0
    val_recall: float = 0.0
    val_specificity: float = 0.0
    condition_number: float = 0.0
    warnings_captured: List[str] = field(default_factory=list)
    converged: bool = True
    is_feasible: bool = False
    rejection_reasons: List[str] = field(default_factory=list)


@dataclass
class StabilityConfigResult:
    """Diagnostic stability results for one (Feature Set, Scaler) configuration."""

    config_id: str
    feature_set_name: str
    scaler_name: str
    feature_count: int
    condition_number: float = 0.0
    runtime_warning_count: int = 0
    convergence_warning_count: int = 0
    overflow_warning_count: int = 0
    divide_by_zero_warning_count: int = 0
    invalid_value_warning_count: int = 0
    converged: bool = False
    n_iter: int = 0
    max_abs_coef: float = 0.0
    coef_l2_norm: float = 0.0
    training_accuracy: float = 0.0
    pass_status: bool = False
    failure_reasons: List[str] = field(default_factory=list)
    warnings_captured: List[str] = field(default_factory=list)


@dataclass
class StabilityGate1000Report:
    """Aggregated report container for 1,000-example numerical stability gate."""

    n_subset_samples: int = 1000
    n_positive: int = 0
    n_negative: int = 0
    total_configs_tested: int = 0
    passing_configs_count: int = 0
    failing_configs_count: int = 0
    overall_verdict: str = "STABILITY GATE: UNKNOWN"
    ranked_config_ids: List[str] = field(default_factory=list)
    configs: Dict[str, StabilityConfigResult] = field(default_factory=dict)


@dataclass
class BenchmarkSuiteResult:
    """Aggregated benchmark suite container for forward compatibility."""

    total_candidates_evaluated: int = 0
    feasible_candidates_count: int = 0
    best_candidate: Optional[Any] = None

def get_stratified_1000_subset(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a deterministic, stratified 1,000-example subset from DEV.

    Args:
        X_dev: Development feature matrix (n_dev, n_features).
        y_dev: Development target array (n_dev,).
        seed: Deterministic random seed (default 42).

    Returns:
        Tuple of (X_sub, y_sub) with shape (1000, n_features) and (1000,).
    """
    if X_dev.shape[0] <= 1000:
        return X_dev.copy(), y_dev.copy()

    X_sub, _, y_sub, _ = train_test_split(
        X_dev,
        y_dev,
        train_size=1000,
        stratify=y_dev,
        random_state=seed,
    )
    return X_sub, y_sub


def evaluate_single_config_stability(
    feature_set_name: str,
    feature_names_subset: List[str],
    scaler_name: str,
    X_sub_full: np.ndarray,
    y_sub: np.ndarray,
    master_feature_names: List[str] = FEATURE_COLUMNS,
    seed: int = 42,
) -> StabilityConfigResult:
    """Train LogisticRegression on a 1,000-sample config and audit numerical stability.

    Captures all warnings, checks convergence, non-finite coefficients/predictions,
    and assigns PASS / FAIL status based strictly on numerical health.

    Args:
        feature_set_name: Key name of feature set (e.g. 'SET_B_DECOLLINEARIZED').
        feature_names_subset: List of feature names in this set.
        scaler_name: Preprocessing scaler to apply.
        X_sub_full: Full 10-feature subset matrix (1000, 10).
        y_sub: Target labels (1000,).
        master_feature_names: Master feature list.
        seed: Random seed.

    Returns:
        StabilityConfigResult object.
    """
    config_id = f"{feature_set_name}__{scaler_name}"
    indices = [master_feature_names.index(f) for f in feature_names_subset]
    X_sub = X_sub_full[:, indices]

    res = StabilityConfigResult(
        config_id=config_id,
        feature_set_name=feature_set_name,
        scaler_name=scaler_name,
        feature_count=len(feature_names_subset),
    )

    reasons: List[str] = []
    captured_warns: List[str] = []

    # 1. Scaler Transformation under warning recorder
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        try:
            X_scaled, _, _ = fit_transform_strategy(scaler_name, X_sub, X_val=None, seed=seed)
        except Exception as e:
            reasons.append(f"Scaler transformation failed: {e}")
            X_scaled = X_sub.copy()

        for w in recorded:
            captured_warns.append(f"[{w.category.__name__}] {w.message}")

    # Condition number
    try:
        cond = float(np.linalg.cond(X_scaled))
        res.condition_number = cond if math.isfinite(cond) else 1e12
    except Exception:
        res.condition_number = 1e12

    # 2. LogisticRegression Fitting under warning recorder (explicit solver lbfgs)
    model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=seed)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        try:
            model.fit(X_scaled, y_sub)
            res.converged = bool(model.n_iter_[0] < model.max_iter)
            res.n_iter = int(model.n_iter_[0])
        except Exception as e:
            reasons.append(f"Model fitting exception: {e}")
            res.converged = False
            res.n_iter = 1000

        for w in recorded:
            captured_warns.append(f"[{w.category.__name__}] {w.message}")

    # Categorize warning types
    for w_str in captured_warns:
        w_lower = w_str.lower()
        if "runtimewarning" in w_lower:
            res.runtime_warning_count += 1
        if "convergencewarning" in w_lower:
            res.convergence_warning_count += 1
        if "overflow" in w_lower:
            res.overflow_warning_count += 1
        if "divide by zero" in w_lower:
            res.divide_by_zero_warning_count += 1
        if "invalid" in w_lower:
            res.invalid_value_warning_count += 1

    # Coefficients and predictions check
    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        if not np.all(np.isfinite(coefs)):
            reasons.append("Non-finite coefficients detected")
        else:
            res.max_abs_coef = float(np.max(np.abs(coefs)))
            res.coef_l2_norm = float(np.linalg.norm(coefs))

        try:
            preds = model.predict(X_scaled)
            probs = model.predict_proba(X_scaled)
            if not np.all(np.isfinite(probs)):
                reasons.append("Non-finite predictions detected")
            else:
                res.training_accuracy = float(accuracy_score(y_sub, preds))
        except Exception as e:
            reasons.append(f"Prediction calculation failed: {e}")

    # Evaluate FAIL criteria
    if res.overflow_warning_count > 0:
        reasons.append(f"Emitted {res.overflow_warning_count} overflow warnings")
    if res.divide_by_zero_warning_count > 0:
        reasons.append(f"Emitted {res.divide_by_zero_warning_count} divide-by-zero warnings")
    if res.invalid_value_warning_count > 0:
        reasons.append(f"Emitted {res.invalid_value_warning_count} invalid matmul/value warnings")
    if not res.converged:
        reasons.append(f"Solver failed to converge in {res.n_iter} iterations")

    res.warnings_captured = captured_warns
    res.failure_reasons = reasons
    res.pass_status = len(reasons) == 0

    return res


# =========================================================
# PUBLIC API — STABILITY GATE
# =========================================================

def run_stability_gate_1000(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    feature_names: List[str] = FEATURE_COLUMNS,
    out_dir: Path = PHASE6K_DIR,
    seed: int = 42,
) -> StabilityGate1000Report:
    """Execute the 1,000-example numerical stability benchmark across 16 configurations.

    Tests 4 Feature Sets x 4 Scalers on a deterministic 1,000-sample DEV subset.
    VAL is NEVER accessed.

    Exports:
        * ``evaluation_results/phase6k/stability_gate_1000.json``

    Args:
        X_dev: Development feature matrix (n_dev, 10).
        y_dev: Development binary target array.
        feature_names: Master list of 10 feature names.
        out_dir: Output directory path.
        seed: Random seed (default 42).

    Returns:
        StabilityGate1000Report container.
    """
    logger.info("phase6k_stability_gate_start", n_dev=X_dev.shape[0])

    # 1. Extract 1,000-sample stratified DEV subset
    X_sub, y_sub = get_stratified_1000_subset(X_dev, y_dev, seed=seed)
    n_pos = int((y_sub == 1).sum())
    n_neg = int((y_sub == 0).sum())

    # 2. Get candidate feature sets
    sets_report = construct_candidate_feature_sets(X_dev, y_dev, feature_names, out_dir=out_dir)

    configs_dict: Dict[str, StabilityConfigResult] = {}
    passing_count = 0
    failing_count = 0

    # 3. Test all 16 combinations (4 Feature Sets x 4 Scalers)
    for set_key, set_meta in sets_report.candidate_sets.items():
        for sname in SCALERS_TO_TEST:
            res = evaluate_single_config_stability(
                feature_set_name=set_key,
                feature_names_subset=set_meta.feature_names,
                scaler_name=sname,
                X_sub_full=X_sub,
                y_sub=y_sub,
                master_feature_names=feature_names,
                seed=seed,
            )
            configs_dict[res.config_id] = res

            if res.pass_status:
                passing_count += 1
            else:
                failing_count += 1

    # 4. Rank configurations: passing configs first (sorted by condition number, then accuracy)
    config_ids = list(configs_dict.keys())
    config_ids.sort(
        key=lambda cid: (
            0 if configs_dict[cid].pass_status else 1,
            configs_dict[cid].condition_number,
            -configs_dict[cid].training_accuracy,
        )
    )

    # 5. Determine overall verdict
    verdict = "STABILITY GATE: PASS" if passing_count > 0 else "STABILITY GATE: FAIL"

    report = StabilityGate1000Report(
        n_subset_samples=int(X_sub.shape[0]),
        n_positive=n_pos,
        n_negative=n_neg,
        total_configs_tested=len(configs_dict),
        passing_configs_count=passing_count,
        failing_configs_count=failing_count,
        overall_verdict=verdict,
        ranked_config_ids=config_ids,
        configs=configs_dict,
    )

    # Export JSON
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stability_gate_1000.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_serializable(asdict(report)), f, indent=2)

    logger.info(
        "phase6k_stability_gate_complete",
        output=str(out_path),
        verdict=verdict,
        passing=passing_count,
        failing=failing_count,
    )

    return report
