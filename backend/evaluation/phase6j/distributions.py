"""Phase 6J — Distribution shape analysis and visualisation.

Generates per-feature distribution plots (histogram, KDE, box plot,
violin plot, ECDF) and computes numerical shape metrics (skewness,
kurtosis, entropy).  Highlights distributional anomalies: heavy tails,
long tails, bimodal distributions, and near-constant features.

Exported artifacts:
    ``feature_distributions.json``  — numerical report
    ``figures/``                    — PNG images (one per plot type per feature)

This module is analysis-only.  It never modifies feature values.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats as scipy_stats
import structlog
from evaluation.phase6j.utils import _serializable

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — never shows figures
import matplotlib.pyplot as plt

logger = structlog.get_logger(__name__)

# =========================================================
# CONSTANTS
# =========================================================

# Thresholds for distributional anomaly flags
_HEAVY_TAIL_KURTOSIS = 3.0       # excess kurtosis > 3 → heavy tails
_LONG_TAIL_SKEW = 2.0            # |skewness| > 2 → long tail
_NEAR_CONSTANT_UNIQUE = 3        # ≤ 3 unique values → near constant
_BIMODAL_DIP_THRESHOLD = 0.05    # Hartigan dip p-value < 0.05 → bimodal (simplified)

# Plot styling
_FIG_DPI = 120
_FIG_SIZE_SINGLE = (7, 5)
_FIG_SIZE_WIDE = (9, 5)
_COLOR_POS = "#e74c3c"
_COLOR_NEG = "#2980b9"
_COLOR_ALL = "#7f8c8d"
_HIST_BINS = 60


# =========================================================
# DATA CLASSES
# =========================================================

@dataclass
class PlotMetadata:
    """Metadata for a single generated plot file."""

    plot_type: str = ""
    feature_name: str = ""
    filename: str = ""
    path: str = ""
    width_px: int = 0
    height_px: int = 0


@dataclass
class FeatureDistribution:
    """Distribution shape metrics and plot metadata for a single feature."""

    name: str

    # Numerical shape metrics
    skewness: float = 0.0
    kurtosis: float = 0.0
    entropy: float = 0.0

    # Anomaly flags
    has_heavy_tails: bool = False
    has_long_tail: bool = False
    is_bimodal: bool = False
    is_near_constant: bool = False

    # Anomaly descriptions (human readable)
    anomalies: List[str] = field(default_factory=list)

    # Plot metadata
    plots: List[PlotMetadata] = field(default_factory=list)


@dataclass
class DistributionsReport:
    """Aggregated distribution report for all features."""

    n_samples: int = 0
    feature_count: int = 0
    total_plots_generated: int = 0
    features: Dict[str, FeatureDistribution] = field(default_factory=dict)
    heavy_tail_features: List[str] = field(default_factory=list)
    long_tail_features: List[str] = field(default_factory=list)
    bimodal_features: List[str] = field(default_factory=list)
    near_constant_features: List[str] = field(default_factory=list)


# =========================================================
# PURE COMPUTATION FUNCTIONS
# =========================================================

def _safe_finite(col: np.ndarray) -> np.ndarray:
    """Extract finite (non-NaN, non-Inf) elements of a 1-D array.

    Args:
        col: 1-D numpy array, possibly containing NaN and ±Inf.

    Returns:
        1-D numpy array with only finite values.  May be empty.
    """
    return col[np.isfinite(col)]


def _compute_entropy(col_finite: np.ndarray, n_bins: int = 50) -> float:
    """Estimate Shannon entropy via histogram binning.

    Args:
        col_finite: 1-D array of finite values.
        n_bins: Number of histogram bins.

    Returns:
        Shannon entropy in nats.  0.0 for degenerate inputs.
    """
    if len(col_finite) < 2:
        return 0.0
    counts, _ = np.histogram(col_finite, bins=n_bins)
    probs = counts[counts > 0] / counts.sum()
    if len(probs) <= 1:
        return 0.0
    return float(-np.sum(probs * np.log(probs)))


def _detect_bimodality(col_finite: np.ndarray) -> bool:
    """Simple bimodality detection via Hartigan's dip test approximation.

    Uses a histogram-based heuristic: if the histogram has two or more
    peaks with a valley between them that drops below 60% of the lower
    peak, the distribution is flagged as bimodal.

    Args:
        col_finite: 1-D array of finite values (≥ 20 elements recommended).

    Returns:
        True if bimodal pattern is detected.
    """
    if len(col_finite) < 20:
        return False

    counts, _ = np.histogram(col_finite, bins=min(50, len(col_finite) // 10))
    if len(counts) < 5:
        return False

    # Smooth with a 3-point moving average to reduce noise
    smoothed = np.convolve(counts, np.ones(3) / 3, mode="same")

    # Find peaks (local maxima)
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            peaks.append((i, smoothed[i]))

    if len(peaks) < 2:
        return False

    # Sort peaks by height, take top 2
    peaks.sort(key=lambda p: p[1], reverse=True)
    p1_idx, p1_h = peaks[0]
    p2_idx, p2_h = peaks[1]

    # Find valley between the two peaks
    lo, hi = min(p1_idx, p2_idx), max(p1_idx, p2_idx)
    if lo == hi:
        return False
    valley = float(np.min(smoothed[lo:hi + 1]))
    threshold = 0.6 * min(p1_h, p2_h)

    return valley < threshold


def compute_single_distribution(
    col: np.ndarray,
    feature_name: str,
) -> FeatureDistribution:
    """Compute distribution shape metrics for a single feature column.

    Pure function — no I/O side effects.

    Args:
        col: 1-D numpy array for the feature, may contain NaN/±Inf.
        feature_name: Human-readable name of the feature.

    Returns:
        FeatureDistribution with numerical metrics and anomaly flags.
    """
    fd = FeatureDistribution(name=feature_name)
    finite = _safe_finite(col)

    if len(finite) < 2:
        fd.is_near_constant = True
        fd.anomalies.append("insufficient finite values (<2)")
        return fd

    n_unique = len(np.unique(finite))
    std = float(np.std(finite, ddof=0))

    # --- Shape metrics ---
    if std > 1e-15 and len(finite) >= 3:
        fd.skewness = float(scipy_stats.skew(finite, bias=False))
    if std > 1e-15 and len(finite) >= 4:
        fd.kurtosis = float(scipy_stats.kurtosis(finite, bias=False, fisher=True))

    fd.entropy = _compute_entropy(finite)

    # --- Anomaly detection ---

    # Heavy tails
    if fd.kurtosis > _HEAVY_TAIL_KURTOSIS:
        fd.has_heavy_tails = True
        fd.anomalies.append(f"heavy tails (excess kurtosis={fd.kurtosis:.2f})")

    # Long tail
    if abs(fd.skewness) > _LONG_TAIL_SKEW:
        fd.has_long_tail = True
        direction = "right" if fd.skewness > 0 else "left"
        fd.anomalies.append(f"long {direction} tail (skewness={fd.skewness:.2f})")

    # Near-constant
    if n_unique <= _NEAR_CONSTANT_UNIQUE or std < 1e-12:
        fd.is_near_constant = True
        fd.anomalies.append(f"near-constant ({n_unique} unique values, std={std:.6f})")

    # Bimodality
    if not fd.is_near_constant and _detect_bimodality(finite):
        fd.is_bimodal = True
        fd.anomalies.append("bimodal distribution detected")

    return fd


# =========================================================
# PLOT GENERATION FUNCTIONS
# =========================================================

def _save_fig(fig: plt.Figure, path: Path) -> None:
    """Save a matplotlib figure and close it immediately.

    Args:
        fig: Matplotlib Figure object.
        path: Destination file path.
    """
    fig.savefig(str(path), dpi=_FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _make_plot_meta(
    plot_type: str,
    feature_name: str,
    path: Path,
) -> PlotMetadata:
    """Create PlotMetadata for a saved figure.

    Args:
        plot_type: Type identifier (histogram, kde, boxplot, violin, ecdf).
        feature_name: Name of the feature.
        path: Absolute or relative path to the saved PNG file.

    Returns:
        PlotMetadata dataclass.
    """
    return PlotMetadata(
        plot_type=plot_type,
        feature_name=feature_name,
        filename=path.name,
        path=str(path),
        width_px=int(_FIG_SIZE_SINGLE[0] * _FIG_DPI),
        height_px=int(_FIG_SIZE_SINGLE[1] * _FIG_DPI),
    )


def _plot_histogram(
    finite: np.ndarray,
    y_finite: np.ndarray,
    feature_name: str,
    fd: FeatureDistribution,
    fig_dir: Path,
) -> Optional[PlotMetadata]:
    """Generate a class-conditional histogram for a feature.

    Overlays histograms for positive and negative classes with an overall
    distribution.  Annotates skewness, kurtosis, and any anomaly flags.

    Args:
        finite: 1-D array of finite feature values.
        y_finite: Corresponding binary labels.
        feature_name: Human-readable feature name.
        fd: FeatureDistribution with computed metrics.
        fig_dir: Directory to save the PNG file.

    Returns:
        PlotMetadata for the saved figure, or None on failure.
    """
    if len(finite) < 2:
        return None

    fig, ax = plt.subplots(figsize=_FIG_SIZE_SINGLE)

    pos_vals = finite[y_finite == 1]
    neg_vals = finite[y_finite == 0]

    ax.hist(finite, bins=_HIST_BINS, alpha=0.3, color=_COLOR_ALL, label="All", density=True)
    if len(pos_vals) > 0:
        ax.hist(pos_vals, bins=_HIST_BINS, alpha=0.5, color=_COLOR_POS, label="Positive (1)", density=True)
    if len(neg_vals) > 0:
        ax.hist(neg_vals, bins=_HIST_BINS, alpha=0.5, color=_COLOR_NEG, label="Negative (0)", density=True)

    ax.set_title(f"Histogram: {feature_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

    # Annotation
    ann = f"skew={fd.skewness:.2f}  kurt={fd.kurtosis:.2f}"
    if fd.anomalies:
        ann += "\n⚠ " + ", ".join(fd.anomalies)
    ax.text(0.02, 0.95, ann, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

    path = fig_dir / f"{feature_name}_histogram.png"
    _save_fig(fig, path)
    return _make_plot_meta("histogram", feature_name, path)


def _plot_kde(
    finite: np.ndarray,
    y_finite: np.ndarray,
    feature_name: str,
    fd: FeatureDistribution,
    fig_dir: Path,
) -> Optional[PlotMetadata]:
    """Generate a class-conditional KDE plot for a feature.

    Args:
        finite: 1-D array of finite feature values.
        y_finite: Corresponding binary labels.
        feature_name: Human-readable feature name.
        fd: FeatureDistribution with computed metrics.
        fig_dir: Directory to save the PNG file.

    Returns:
        PlotMetadata for the saved figure, or None on failure.
    """
    if len(finite) < 10:
        return None

    fig, ax = plt.subplots(figsize=_FIG_SIZE_SINGLE)

    # Full KDE
    std = float(np.std(finite, ddof=0))
    if std < 1e-12:
        return None

    try:
        kde_all = scipy_stats.gaussian_kde(finite)
        x_grid = np.linspace(float(np.min(finite)), float(np.max(finite)), 300)
        ax.plot(x_grid, kde_all(x_grid), color=_COLOR_ALL, linewidth=2, label="All")

        pos_vals = finite[y_finite == 1]
        neg_vals = finite[y_finite == 0]

        if len(pos_vals) > 5 and float(np.std(pos_vals)) > 1e-12:
            kde_pos = scipy_stats.gaussian_kde(pos_vals)
            ax.plot(x_grid, kde_pos(x_grid), color=_COLOR_POS, linewidth=1.5, linestyle="--", label="Positive (1)")

        if len(neg_vals) > 5 and float(np.std(neg_vals)) > 1e-12:
            kde_neg = scipy_stats.gaussian_kde(neg_vals)
            ax.plot(x_grid, kde_neg(x_grid), color=_COLOR_NEG, linewidth=1.5, linestyle="--", label="Negative (0)")
    except Exception as e:
        logger.warning("phase6j_kde_failed", feature=feature_name, error=str(e))
        plt.close(fig)
        return None

    ax.set_title(f"KDE: {feature_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

    ann = f"entropy={fd.entropy:.3f}"
    if fd.is_bimodal:
        ann += "  ⚠ bimodal"
    ax.text(0.02, 0.95, ann, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

    path = fig_dir / f"{feature_name}_kde.png"
    _save_fig(fig, path)
    return _make_plot_meta("kde", feature_name, path)


def _plot_boxplot(
    finite: np.ndarray,
    y_finite: np.ndarray,
    feature_name: str,
    fig_dir: Path,
) -> Optional[PlotMetadata]:
    """Generate a class-conditional box plot for a feature.

    Args:
        finite: 1-D array of finite feature values.
        y_finite: Corresponding binary labels.
        feature_name: Human-readable feature name.
        fig_dir: Directory to save the PNG file.

    Returns:
        PlotMetadata for the saved figure, or None on failure.
    """
    if len(finite) < 2:
        return None

    pos_vals = finite[y_finite == 1]
    neg_vals = finite[y_finite == 0]

    fig, ax = plt.subplots(figsize=_FIG_SIZE_SINGLE)

    data_to_plot = []
    labels = []
    colors = []

    if len(neg_vals) > 0:
        data_to_plot.append(neg_vals)
        labels.append("Negative (0)")
        colors.append(_COLOR_NEG)
    if len(pos_vals) > 0:
        data_to_plot.append(pos_vals)
        labels.append("Positive (1)")
        colors.append(_COLOR_POS)
    data_to_plot.append(finite)
    labels.append("All")
    colors.append(_COLOR_ALL)

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, notch=True,
                    showmeans=True, meanprops=dict(marker="D", markerfacecolor="black", markersize=5))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_title(f"Box Plot: {feature_name}", fontsize=12, fontweight="bold")
    ax.set_ylabel(feature_name)

    path = fig_dir / f"{feature_name}_boxplot.png"
    _save_fig(fig, path)
    return _make_plot_meta("boxplot", feature_name, path)


def _plot_violin(
    finite: np.ndarray,
    y_finite: np.ndarray,
    feature_name: str,
    fig_dir: Path,
) -> Optional[PlotMetadata]:
    """Generate a class-conditional violin plot for a feature.

    Args:
        finite: 1-D array of finite feature values.
        y_finite: Corresponding binary labels.
        feature_name: Human-readable feature name.
        fig_dir: Directory to save the PNG file.

    Returns:
        PlotMetadata for the saved figure, or None on failure.
    """
    if len(finite) < 10:
        return None

    pos_vals = finite[y_finite == 1]
    neg_vals = finite[y_finite == 0]

    # Need enough variance to draw a violin
    data_to_plot = []
    labels = []
    if len(neg_vals) > 5 and float(np.std(neg_vals)) > 1e-12:
        data_to_plot.append(neg_vals)
        labels.append("Negative (0)")
    if len(pos_vals) > 5 and float(np.std(pos_vals)) > 1e-12:
        data_to_plot.append(pos_vals)
        labels.append("Positive (1)")
    if float(np.std(finite)) > 1e-12:
        data_to_plot.append(finite)
        labels.append("All")

    if len(data_to_plot) == 0:
        return None

    fig, ax = plt.subplots(figsize=_FIG_SIZE_SINGLE)

    parts = ax.violinplot(data_to_plot, showmeans=True, showmedians=True)

    # Color the violins
    color_map = [_COLOR_NEG, _COLOR_POS, _COLOR_ALL]
    for i, pc in enumerate(parts["bodies"]):
        c_idx = min(i, len(color_map) - 1)
        pc.set_facecolor(color_map[c_idx])
        pc.set_alpha(0.5)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_title(f"Violin Plot: {feature_name}", fontsize=12, fontweight="bold")
    ax.set_ylabel(feature_name)

    path = fig_dir / f"{feature_name}_violin.png"
    _save_fig(fig, path)
    return _make_plot_meta("violin", feature_name, path)


def _plot_ecdf(
    finite: np.ndarray,
    y_finite: np.ndarray,
    feature_name: str,
    fig_dir: Path,
) -> Optional[PlotMetadata]:
    """Generate a class-conditional ECDF plot for a feature.

    Args:
        finite: 1-D array of finite feature values.
        y_finite: Corresponding binary labels.
        feature_name: Human-readable feature name.
        fig_dir: Directory to save the PNG file.

    Returns:
        PlotMetadata for the saved figure, or None on failure.
    """
    if len(finite) < 2:
        return None

    fig, ax = plt.subplots(figsize=_FIG_SIZE_SINGLE)

    def _ecdf(vals: np.ndarray):
        """Return sorted values and their ECDF y-coordinates."""
        xs = np.sort(vals)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        return xs, ys

    xs, ys = _ecdf(finite)
    ax.step(xs, ys, color=_COLOR_ALL, linewidth=2, label="All", where="post")

    pos_vals = finite[y_finite == 1]
    neg_vals = finite[y_finite == 0]
    if len(pos_vals) > 0:
        xs_p, ys_p = _ecdf(pos_vals)
        ax.step(xs_p, ys_p, color=_COLOR_POS, linewidth=1.5, linestyle="--", label="Positive (1)", where="post")
    if len(neg_vals) > 0:
        xs_n, ys_n = _ecdf(neg_vals)
        ax.step(xs_n, ys_n, color=_COLOR_NEG, linewidth=1.5, linestyle="--", label="Negative (0)", where="post")

    ax.set_title(f"ECDF: {feature_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Cumulative Probability")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    path = fig_dir / f"{feature_name}_ecdf.png"
    _save_fig(fig, path)
    return _make_plot_meta("ecdf", feature_name, path)


# =========================================================
# FEATURE-LEVEL ORCHESTRATOR
# =========================================================

def _process_feature(
    col: np.ndarray,
    y: np.ndarray,
    feature_name: str,
    fig_dir: Path,
) -> FeatureDistribution:
    """Compute distribution metrics and generate all plots for one feature.

    Args:
        col: 1-D numpy array for the feature (raw, may contain NaN/±Inf).
        y: Binary label array aligned with col.
        feature_name: Human-readable feature name.
        fig_dir: Directory to save PNG figures.

    Returns:
        FeatureDistribution with metrics, anomaly flags, and plot metadata.
    """
    fd = compute_single_distribution(col, feature_name)

    finite = _safe_finite(col)
    if len(finite) < 2:
        logger.warning("phase6j_dist_skip_plots", feature=feature_name, reason="insufficient finite values")
        return fd

    # Align labels to finite mask
    finite_mask = np.isfinite(col)
    y_finite = y[finite_mask]

    # Generate all plot types
    plot_funcs = [
        ("histogram", _plot_histogram, dict(finite=finite, y_finite=y_finite, feature_name=feature_name, fd=fd, fig_dir=fig_dir)),
        ("kde", _plot_kde, dict(finite=finite, y_finite=y_finite, feature_name=feature_name, fd=fd, fig_dir=fig_dir)),
        ("boxplot", _plot_boxplot, dict(finite=finite, y_finite=y_finite, feature_name=feature_name, fig_dir=fig_dir)),
        ("violin", _plot_violin, dict(finite=finite, y_finite=y_finite, feature_name=feature_name, fig_dir=fig_dir)),
        ("ecdf", _plot_ecdf, dict(finite=finite, y_finite=y_finite, feature_name=feature_name, fig_dir=fig_dir)),
    ]

    for plot_type, func, kwargs in plot_funcs:
        try:
            meta = func(**kwargs)
            if meta is not None:
                fd.plots.append(meta)
        except Exception as e:
            logger.warning("phase6j_plot_failed", feature=feature_name, plot_type=plot_type, error=str(e))

    return fd


# =========================================================
# PUBLIC API
# =========================================================

def compute_distributions(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    out_dir: Path,
) -> DistributionsReport:
    """Compute distribution analysis and generate plots for every feature.

    This is the single public entry point for the distributions module.
    It computes numerical shape metrics, detects distributional anomalies,
    generates five plot types per feature, and exports a JSON report with
    plot metadata.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Binary label array of shape (n_samples,).
        feature_names: Ordered list of feature column names.
        out_dir: Root output directory.  Figures are saved under
                 ``out_dir / figures/``.

    Returns:
        DistributionsReport with per-feature metrics and plot metadata.
    """
    logger.info("phase6j_distributions_start", n_samples=X.shape[0], n_features=len(feature_names))

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    report = DistributionsReport(
        n_samples=int(X.shape[0]),
        feature_count=len(feature_names),
    )

    total_plots = 0

    for idx, name in enumerate(feature_names):
        col = X[:, idx].astype(float)
        fd = _process_feature(col, y, name, fig_dir)
        report.features[name] = fd
        total_plots += len(fd.plots)

        # Collect anomaly lists
        if fd.has_heavy_tails:
            report.heavy_tail_features.append(name)
        if fd.has_long_tail:
            report.long_tail_features.append(name)
        if fd.is_bimodal:
            report.bimodal_features.append(name)
        if fd.is_near_constant:
            report.near_constant_features.append(name)

    report.total_plots_generated = total_plots

    # --- Export JSON ---
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "feature_distributions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_serializable(asdict(report)), f, indent=2)

    logger.info(
        "phase6j_distributions_complete",
        output=str(out_path),
        total_plots=total_plots,
        heavy_tails=len(report.heavy_tail_features),
        long_tails=len(report.long_tail_features),
        bimodal=len(report.bimodal_features),
        near_constant=len(report.near_constant_features),
    )

    return report
