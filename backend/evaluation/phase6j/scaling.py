"""Phase 6J — Scaling diagnostics.

Investigates whether numerical instability originates from feature scaling
by applying five standard scikit-learn transformers to each feature and
comparing the resulting statistics.

Pipeline per feature::

    Original → StandardScaler → RobustScaler → MinMaxScaler
             → PowerTransformer → QuantileTransformer

For each method the module computes: min, max, dynamic range, mean, std,
and coefficient of variation.  It then flags exploding ranges, compressed
ranges, and unstable transformations.

Exported artifact: ``feature_scaling.json``

Design:
    * Each scaler is independently testable via ``apply_scaler()``.
    * Original feature matrices are **never** modified.
    * Transformed matrices are returned for optional downstream benchmarking.

This module is analysis-only.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

from evaluation.phase6j.utils import _serializable

logger = structlog.get_logger(__name__)

# =========================================================
# CONSTANTS
# =========================================================

SCALER_NAMES: List[str] = [
    "original",
    "standard",
    "robust",
    "minmax",
    "power",
    "quantile",
]

# Anomaly thresholds
_EXPLODING_RANGE = 1e4        # dynamic range > 10 000 → exploding
_COMPRESSED_RANGE = 1e-6      # dynamic range < 1e-6 → compressed
_UNSTABLE_CV = 100.0          # |CV| > 100 → unstable transformation


# =========================================================
# DATA CLASSES
# =========================================================

@dataclass
class ScalerStats:
    """Statistics for one feature under one scaling method.

    All fields default to safe values so partial computation never raises.
    """

    min: float = 0.0
    max: float = 0.0
    dynamic_range: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    coefficient_of_variation: Optional[float] = None


@dataclass
class FeatureScaling:
    """Scaling diagnostics for a single feature across all scalers."""

    name: str
    scalers: Dict[str, ScalerStats] = field(default_factory=dict)

    # Anomaly flags
    has_exploding_range: bool = False
    has_compressed_range: bool = False
    has_unstable_transformation: bool = False
    anomalies: List[str] = field(default_factory=list)


@dataclass
class ScalingReport:
    """Aggregated scaling diagnostics for all features."""

    n_samples: int = 0
    feature_count: int = 0
    scaler_names: List[str] = field(default_factory=list)
    features: Dict[str, FeatureScaling] = field(default_factory=dict)

    # Anomaly summary lists
    exploding_range_features: List[str] = field(default_factory=list)
    compressed_range_features: List[str] = field(default_factory=list)
    unstable_features: List[str] = field(default_factory=list)

    # Transformed matrices keyed by scaler name (not serialised to JSON)
    transformed_matrices: Dict[str, np.ndarray] = field(
        default_factory=dict, repr=False,
    )


# =========================================================
# PURE COMPUTATION — SINGLE SCALER APPLICATION
# =========================================================

def apply_scaler(
    X: np.ndarray,
    scaler_name: str,
) -> np.ndarray:
    """Apply a named scaler to a feature matrix and return the result.

    The input matrix ``X`` is **never** modified.  A fresh copy is
    transformed and returned.

    Each scaler is imported and instantiated independently so that this
    function is fully testable in isolation.

    Args:
        X: Feature matrix of shape (n_samples, n_features).  Must contain
           only finite values (NaN/Inf should be handled upstream).
        scaler_name: One of ``SCALER_NAMES`` (excluding ``'original'``).

    Returns:
        Transformed copy of X with the same shape.

    Raises:
        ValueError: If ``scaler_name`` is not recognised.
    """
    from sklearn.preprocessing import (
        StandardScaler,
        RobustScaler,
        MinMaxScaler,
        PowerTransformer,
        QuantileTransformer,
    )

    X_clean = np.nan_to_num(X.copy(), nan=0.0, posinf=0.0, neginf=0.0)

    if scaler_name == "original":
        return X_clean

    if scaler_name == "standard":
        return StandardScaler().fit_transform(X_clean)

    if scaler_name == "robust":
        return RobustScaler().fit_transform(X_clean)

    if scaler_name == "minmax":
        return MinMaxScaler().fit_transform(X_clean)

    if scaler_name == "power":
        # PowerTransformer requires strictly positive data for 'box-cox'.
        # Use 'yeo-johnson' which handles zero and negative values.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return PowerTransformer(method="yeo-johnson").fit_transform(X_clean)
            except Exception:
                logger.warning("phase6j_power_transform_fallback", reason="yeo-johnson failed, returning original")
                return X_clean

    if scaler_name == "quantile":
        n_samples = X_clean.shape[0]
        n_quantiles = min(1000, n_samples)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return QuantileTransformer(
                    n_quantiles=n_quantiles,
                    output_distribution="normal",
                    random_state=42,
                ).fit_transform(X_clean)
            except Exception:
                logger.warning("phase6j_quantile_transform_fallback", reason="quantile transform failed, returning original")
                return X_clean

    raise ValueError(f"Unknown scaler: {scaler_name!r}.  Expected one of {SCALER_NAMES}")


# =========================================================
# PURE COMPUTATION — COLUMN STATISTICS
# =========================================================

def compute_column_stats(col: np.ndarray) -> ScalerStats:
    """Compute summary statistics for a single transformed column.

    Pure function — no I/O or side effects.

    Args:
        col: 1-D numpy array of transformed feature values.

    Returns:
        ScalerStats with min, max, dynamic_range, mean, std, and CV.
    """
    finite = col[np.isfinite(col)]

    if len(finite) == 0:
        return ScalerStats()

    min_v = float(np.min(finite))
    max_v = float(np.max(finite))
    mean_v = float(np.mean(finite))
    std_v = float(np.std(finite, ddof=0))
    dyn_range = max_v - min_v

    cv: Optional[float] = None
    if abs(mean_v) > 1e-15:
        cv = std_v / abs(mean_v)

    return ScalerStats(
        min=min_v,
        max=max_v,
        dynamic_range=dyn_range,
        mean=mean_v,
        std=std_v,
        coefficient_of_variation=cv,
    )


# =========================================================
# PURE COMPUTATION — SINGLE FEATURE ANALYSIS
# =========================================================

def analyse_feature_scaling(
    feature_name: str,
    scaler_stats: Dict[str, ScalerStats],
) -> FeatureScaling:
    """Analyse scaling diagnostics for one feature across all scalers.

    Detects exploding ranges, compressed ranges, and unstable
    transformations.  Pure function.

    Args:
        feature_name: Human-readable feature name.
        scaler_stats: Dict mapping scaler name → ScalerStats.

    Returns:
        FeatureScaling with anomaly flags populated.
    """
    fs = FeatureScaling(name=feature_name, scalers=scaler_stats)

    for sname, ss in scaler_stats.items():
        # Exploding range
        if abs(ss.dynamic_range) > _EXPLODING_RANGE:
            fs.has_exploding_range = True
            fs.anomalies.append(
                f"exploding range under {sname} "
                f"(dynamic_range={ss.dynamic_range:.2f})"
            )

        # Compressed range
        if abs(ss.dynamic_range) < _COMPRESSED_RANGE and sname != "original":
            fs.has_compressed_range = True
            fs.anomalies.append(
                f"compressed range under {sname} "
                f"(dynamic_range={ss.dynamic_range:.2e})"
            )

        # Unstable CV
        if ss.coefficient_of_variation is not None:
            if abs(ss.coefficient_of_variation) > _UNSTABLE_CV:
                fs.has_unstable_transformation = True
                fs.anomalies.append(
                    f"unstable transformation under {sname} "
                    f"(CV={ss.coefficient_of_variation:.2f})"
                )

    return fs


# =========================================================
# PUBLIC API
# =========================================================

def compute_scaling(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    out_dir: Path,
) -> ScalingReport:
    """Evaluate scaling behaviour of every feature under multiple transformers.

    This is the single public entry point for the scaling module.

    Pipeline::

        For each scaler in [original, standard, robust, minmax, power, quantile]:
            1. Transform the full matrix (copy — never mutates input).
            2. Compute per-column statistics.
            3. Detect anomalies.

    The transformed matrices are attached to the returned report for
    optional downstream benchmarking but are **not** serialised to JSON.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Binary label array of shape (n_samples,).
        feature_names: Ordered list of feature column names.
        out_dir: Directory to write ``feature_scaling.json``.

    Returns:
        ScalingReport with per-feature, per-scaler diagnostics and
        transformed matrices.
    """
    logger.info(
        "phase6j_scaling_start",
        n_samples=X.shape[0],
        n_features=len(feature_names),
        scalers=SCALER_NAMES,
    )

    # --- Apply all scalers ---
    transformed: Dict[str, np.ndarray] = {}
    for sname in SCALER_NAMES:
        transformed[sname] = apply_scaler(X, sname)
        logger.info("phase6j_scaler_applied", scaler=sname, shape=transformed[sname].shape)

    # --- Per-feature analysis ---
    feature_reports: Dict[str, FeatureScaling] = {}

    for idx, fname in enumerate(feature_names):
        per_scaler: Dict[str, ScalerStats] = {}
        for sname in SCALER_NAMES:
            col = transformed[sname][:, idx]
            per_scaler[sname] = compute_column_stats(col)

        feature_reports[fname] = analyse_feature_scaling(fname, per_scaler)

    # --- Build report ---
    report = ScalingReport(
        n_samples=int(X.shape[0]),
        feature_count=len(feature_names),
        scaler_names=list(SCALER_NAMES),
        features=feature_reports,
        transformed_matrices=transformed,
    )

    for fname, fs in feature_reports.items():
        if fs.has_exploding_range:
            report.exploding_range_features.append(fname)
        if fs.has_compressed_range:
            report.compressed_range_features.append(fname)
        if fs.has_unstable_transformation:
            report.unstable_features.append(fname)

    # --- Export JSON (exclude transformed matrices) ---
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "feature_scaling.json"

    export = asdict(report)
    export.pop("transformed_matrices", None)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_serializable(export), f, indent=2)

    logger.info(
        "phase6j_scaling_complete",
        output=str(out_path),
        exploding=len(report.exploding_range_features),
        compressed=len(report.compressed_range_features),
        unstable=len(report.unstable_features),
    )

    return report
