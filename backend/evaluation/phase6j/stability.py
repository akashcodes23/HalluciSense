"""Phase 6J — Numerical stability diagnostics for Logistic Regression.

Investigates numerical hazards (divide by zero, overflow, invalid matmul)
encountered during Phase 6I training by evaluating Logistic Regression
under six distinct feature preprocessing strategies:

    1. Original (unscaled raw features)
    2. StandardScaler
    3. RobustScaler
    4. QuantileTransformer
    5. PowerTransformer (Yeo-Johnson)
    6. Winsorized Features (clipped to 1st-99th percentiles)

Captures for each strategy:
    * Warnings emitted during fitting & evaluation
    * Convergence status & iteration count
    * Coefficient magnitudes & intercept
    * Training accuracy, validation accuracy, ROC-AUC, MCC
    * Feature matrix condition number & feature L2 norms
    * Maximum absolute coefficient
    * Hessian approximation condition number

Never suppresses warnings — uses ``warnings.catch_warnings(record=True)``.
Identifies which preprocessing pipeline eliminates numerical instability.

Exported artifact: ``stability_report.json`` (and ``stability.json``)

This module is purely diagnostic and never modifies training logic or data.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, log_loss
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
)
import structlog
from evaluation.phase6j.utils import _serializable

logger = structlog.get_logger(__name__)

# =========================================================
# CONSTANTS
# =========================================================

STRATEGIES: List[str] = [
    "Original",
    "StandardScaler",
    "RobustScaler",
    "QuantileTransformer",
    "PowerTransformer",
    "Winsorized",
]


# =========================================================
# DATA CLASSES
# =========================================================

@dataclass
class PreprocessingStabilityResult:
    """Diagnostic stability results for a single preprocessing strategy."""

    strategy_name: str
    warnings_captured: List[str] = field(default_factory=list)
    warning_count: int = 0
    converged: bool = False
    iterations: int = 0
    coefficients: Dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    loss: float = 0.0
    training_accuracy: float = 0.0
    validation_accuracy: float = 0.0
    roc_auc: float = 0.5
    mcc: float = 0.0
    condition_number: float = 0.0
    feature_norms: Dict[str, float] = field(default_factory=dict)
    max_abs_coefficient: float = 0.0
    hessian_condition_number: float = 0.0
    eliminates_instability: bool = False


@dataclass
class StabilityReport:
    """Aggregated numerical stability diagnostic report."""

    n_samples_train: int = 0
    n_samples_val: int = 0
    n_features: int = 0
    recommended_pipeline: str = ""
    strategies: Dict[str, PreprocessingStabilityResult] = field(default_factory=dict)
    stable_strategies: List[str] = field(default_factory=list)
    unstable_strategies: List[str] = field(default_factory=list)


# =========================================================
# PREPROCESSING HELPERS
# =========================================================

def _apply_winsorization(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Clip features to lower and upper percentile bounds computed on training set.

    Args:
        X_train: Training feature matrix (n_train, n_features).
        X_val: Validation feature matrix (n_val, n_features), optional.
        lower_pct: Lower percentile threshold (default 1.0).
        upper_pct: Upper percentile threshold (default 99.0).

    Returns:
        Tuple of (X_train_clipped, X_val_clipped).
    """
    X_tr_clip = X_train.copy()
    X_val_clip = X_val.copy() if X_val is not None else None

    for col_idx in range(X_train.shape[1]):
        col_tr = X_train[:, col_idx]
        finite = col_tr[np.isfinite(col_tr)]
        if len(finite) > 0:
            low_b = float(np.percentile(finite, lower_pct))
            high_b = float(np.percentile(finite, upper_pct))
            X_tr_clip[:, col_idx] = np.clip(X_tr_clip[:, col_idx], low_b, high_b)
            if X_val_clip is not None:
                X_val_clip[:, col_idx] = np.clip(X_val_clip[:, col_idx], low_b, high_b)

    return X_tr_clip, X_val_clip


def preprocess_features(
    strategy: str,
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Preprocess training and validation matrices using a named strategy.

    Inputs are copied; originals are never modified.

    Args:
        strategy: Name of strategy from ``STRATEGIES``.
        X_train: Training feature matrix.
        X_val: Validation feature matrix (optional).

    Returns:
        Tuple of (X_train_processed, X_val_processed).
    """
    X_tr = np.nan_to_num(X_train.copy(), nan=0.0, posinf=0.0, neginf=0.0)
    X_v = np.nan_to_num(X_val.copy(), nan=0.0, posinf=0.0, neginf=0.0) if X_val is not None else None

    if strategy == "Original":
        return X_tr, X_v

    if strategy == "StandardScaler":
        scaler = StandardScaler()
        X_tr_proc = scaler.fit_transform(X_tr)
        X_v_proc = scaler.transform(X_v) if X_v is not None else None
        return X_tr_proc, X_v_proc

    if strategy == "RobustScaler":
        scaler = RobustScaler()
        X_tr_proc = scaler.fit_transform(X_tr)
        X_v_proc = scaler.transform(X_v) if X_v is not None else None
        return X_tr_proc, X_v_proc

    if strategy == "QuantileTransformer":
        n_quantiles = min(1000, X_tr.shape[0])
        scaler = QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal", random_state=42)
        X_tr_proc = scaler.fit_transform(X_tr)
        X_v_proc = scaler.transform(X_v) if X_v is not None else None
        return X_tr_proc, X_v_proc

    if strategy == "PowerTransformer":
        scaler = PowerTransformer(method="yeo-johnson")
        X_tr_proc = scaler.fit_transform(X_tr)
        X_v_proc = scaler.transform(X_v) if X_v is not None else None
        return X_tr_proc, X_v_proc

    if strategy == "Winsorized":
        return _apply_winsorization(X_tr, X_v, lower_pct=1.0, upper_pct=99.0)

    raise ValueError(f"Unknown strategy: {strategy!r}. Expected one of {STRATEGIES}")


# =========================================================
# PURE COMPUTATION HELPERS
# =========================================================

def _compute_hessian_cond(X: np.ndarray, probs: np.ndarray) -> float:
    """Compute condition number of Hessian approximation H = X^T W X.

    W is a diagonal matrix of p_i * (1 - p_i).
    """
    try:
        w = probs * (1.0 - probs)
        w = np.maximum(w, 1e-12)  # Avoid exact zero
        X_w = X * np.sqrt(w)[:, np.newaxis]
        H = X_w.T @ X_w
        cond = float(np.linalg.cond(H))
        return cond if math.isfinite(cond) else 1e12
    except Exception:
        return 1e12


def evaluate_strategy_stability(
    strategy_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    feature_names: List[str],
) -> PreprocessingStabilityResult:
    """Train Logistic Regression under a specific preprocessing strategy and record diagnostics.

    Captures all Python and NumPy warnings emitted during execution.

    Args:
        strategy_name: Name of preprocessing strategy.
        X_train: Raw training feature matrix.
        y_train: Training labels.
        X_val: Raw validation feature matrix.
        y_val: Validation labels.
        feature_names: Names of feature columns.

    Returns:
        PreprocessingStabilityResult with complete stability diagnostics.
    """
    res = PreprocessingStabilityResult(strategy_name=strategy_name)
    captured_warns: List[str] = []

    # 1. Preprocess matrices under warning recorder
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        try:
            X_tr_proc, X_v_proc = preprocess_features(strategy_name, X_train, X_val)
        except Exception as e:
            logger.warning("phase6j_preprocessing_error", strategy=strategy_name, error=str(e))
            X_tr_proc = X_train
            X_v_proc = X_val

        for w in recorded_warnings:
            captured_warns.append(f"[{w.category.__name__}] {w.message}")

    # Condition number & Feature norms
    try:
        res.condition_number = float(np.linalg.cond(X_tr_proc))
        if not math.isfinite(res.condition_number):
            res.condition_number = 1e12
    except Exception:
        res.condition_number = 1e12

    res.feature_norms = {
        name: float(np.linalg.norm(X_tr_proc[:, idx]))
        for idx, name in enumerate(feature_names)
    }

    # 2. Fit Logistic Regression under warning recorder
    model = LogisticRegression(max_iter=1000, random_state=42)

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        try:
            model.fit(X_tr_proc, y_train)
            res.converged = bool(model.n_iter_[0] < model.max_iter)
            res.iterations = int(model.n_iter_[0])
        except Exception as e:
            logger.warning("phase6j_fit_error", strategy=strategy_name, error=str(e))
            res.converged = False
            res.iterations = 1000

        for w in recorded_warnings:
            captured_warns.append(f"[{w.category.__name__}] {w.message}")

    # Coefficients & Intercept
    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        res.coefficients = {name: float(coefs[idx]) for idx, name in enumerate(feature_names)}
        res.max_abs_coefficient = float(np.max(np.abs(coefs)))
        res.intercept = float(model.intercept_[0]) if hasattr(model, "intercept_") else 0.0

    # 3. Evaluate Metrics under warning recorder
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        try:
            train_preds = model.predict(X_tr_proc)
            train_probs = model.predict_proba(X_tr_proc)[:, 1]
            res.training_accuracy = float(accuracy_score(y_train, train_preds))
            res.loss = float(log_loss(y_train, train_probs))

            # Hessian condition number
            res.hessian_condition_number = _compute_hessian_cond(X_tr_proc, train_probs)

            # Validation metrics
            eval_X = X_v_proc if X_v_proc is not None else X_tr_proc
            eval_y = y_val if y_val is not None else y_train

            val_preds = model.predict(eval_X)
            val_probs = model.predict_proba(eval_X)[:, 1]

            res.validation_accuracy = float(accuracy_score(eval_y, val_preds))
            res.roc_auc = float(roc_auc_score(eval_y, val_probs))
            res.mcc = float(matthews_corrcoef(eval_y, val_preds))
        except Exception as e:
            logger.warning("phase6j_eval_error", strategy=strategy_name, error=str(e))

        for w in recorded_warnings:
            captured_warns.append(f"[{w.category.__name__}] {w.message}")

    # Clean warnings list
    res.warnings_captured = captured_warns
    res.warning_count = len(captured_warns)

    # Check if instability is eliminated (no warnings, converged, well-conditioned)
    res.eliminates_instability = bool(
        res.warning_count == 0
        and res.converged
        and res.condition_number < 1e5
        and res.max_abs_coefficient < 50.0
    )

    return res


# =========================================================
# PUBLIC API
# =========================================================

def compute_stability(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    out_dir: Path,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> StabilityReport:
    """Determine why Logistic Regression produces numerical warnings and identify safe pipelines.

    Evaluates Logistic Regression across 6 preprocessing strategies.
    Captures all warnings, condition numbers, convergence, and metrics.

    Exports:
        * ``stability_report.json``
        * ``stability.json`` (for backwards compatibility)

    Args:
        X: Training feature matrix (n_samples, n_features).
        y: Binary label array for training set.
        feature_names: Ordered feature column names.
        out_dir: Output directory path.
        X_val: Validation feature matrix (optional).
        y_val: Validation label array (optional).

    Returns:
        StabilityReport containing diagnostics for all strategies.
    """
    logger.info("phase6j_stability_start", n_samples=X.shape[0], n_features=len(feature_names))

    strategy_results: Dict[str, PreprocessingStabilityResult] = {}
    stable_strats: List[str] = []
    unstable_strats: List[str] = []

    for sname in STRATEGIES:
        res = evaluate_strategy_stability(sname, X, y, X_val, y_val, feature_names)
        strategy_results[sname] = res
        if res.eliminates_instability:
            stable_strats.append(sname)
        else:
            unstable_strats.append(sname)

    # Recommend safest pipeline (prefer StandardScaler / RobustScaler / QuantileTransformer if stable)
    recommended = ""
    preferred_order = ["StandardScaler", "RobustScaler", "QuantileTransformer", "PowerTransformer", "Winsorized"]
    for pref in preferred_order:
        if pref in stable_strats:
            recommended = pref
            break

    if not recommended:
        recommended = stable_strats[0] if stable_strats else "StandardScaler (with clipping)"

    report = StabilityReport(
        n_samples_train=int(X.shape[0]),
        n_samples_val=int(X_val.shape[0]) if X_val is not None else 0,
        n_features=len(feature_names),
        recommended_pipeline=recommended,
        strategies=strategy_results,
        stable_strategies=stable_strats,
        unstable_strategies=unstable_strats,
    )

    # Export JSON to both stability_report.json and stability.json
    out_dir.mkdir(parents=True, exist_ok=True)

    report_dict = _serializable(asdict(report))

    out_path_main = out_dir / "stability_report.json"
    out_path_compat = out_dir / "stability.json"

    with open(out_path_main, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    with open(out_path_compat, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    logger.info(
        "phase6j_stability_complete",
        output=str(out_path_main),
        recommended=recommended,
        stable_count=len(stable_strats),
        unstable_count=len(unstable_strats),
    )

    return report
