"""Unit tests for Phase 6L.4 Root Cause Analysis Pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from evaluation.phase6l.config import STRUCTURAL_FEATURE_COLUMNS
from evaluation.phase6l.root_cause_analysis import (
    LOCKED_FEATURE_NAMES,
    compute_jensenshannon_divergence,
    compute_shannon_entropy,
    decompose_feature_distribution_shift,
    analyze_pairwise_nli_score_drift,
    analyze_structural_complexity,
    analyze_detector_activations,
    analyze_feature_stability,
    analyze_probability_compression,
    analyze_error_clusters,
    synthesize_root_cause,
)


def test_analyze_pairwise_nli_score_drift(tmp_path: Path):
    """Verify pairwise NLI score drift calculation."""
    np.random.seed(42)
    X_dev = np.random.randn(100, 24)
    X_val = np.random.randn(50, 24) + 0.3

    res = analyze_pairwise_nli_score_drift(X_dev, X_val, out_dir=tmp_path)
    assert "contradiction_score_drift" in res
    assert "smd" in res["contradiction_score_drift"]


def test_compute_jensenshannon_divergence():
    """Verify Jensen-Shannon Divergence calculation."""
    np.random.seed(42)
    u = np.random.randn(100)
    v = np.random.randn(100) + 1.0

    jsd = compute_jensenshannon_divergence(u, v)
    assert jsd >= 0.0
    assert np.isfinite(jsd)


def test_compute_shannon_entropy():
    """Verify Shannon entropy calculation."""
    np.random.seed(42)
    probs = np.random.uniform(0, 1, 200)

    entropy = compute_shannon_entropy(probs)
    assert entropy > 0.0
    assert np.isfinite(entropy)


def test_decompose_feature_distribution_shift(tmp_path: Path):
    """Verify feature distribution shift decomposition."""
    np.random.seed(42)
    X_dev = np.random.randn(100, 24)
    X_val = np.random.randn(50, 24) + 0.5

    payload = decompose_feature_distribution_shift(X_dev, X_val, out_dir=tmp_path)

    assert payload["total_features_analyzed"] == 24
    assert len(payload["locked_features_shift_summary"]) == 5
    assert (tmp_path / "distribution_shift_decomposition.json").exists()


def test_analyze_structural_complexity():
    """Verify DEV vs VAL structural complexity comparison."""
    np.random.seed(42)
    X_dev = np.random.randn(100, 24)
    X_val = np.random.randn(50, 24)

    comp = analyze_structural_complexity(X_dev, X_val)

    assert "num_claims" in comp
    assert "graph_density" in comp
    assert "smd" in comp["num_claims"]


def test_analyze_detector_activations(tmp_path: Path):
    """Verify detector activation frequency audit."""
    np.random.seed(42)
    X_dev = np.random.exponential(scale=0.1, size=(100, 24))
    X_val = np.random.exponential(scale=0.05, size=(50, 24))

    act = analyze_detector_activations(X_dev, X_val, out_dir=tmp_path)

    assert "families" in act
    assert "contradiction_family" in act["families"]
    assert (tmp_path / "detector_activation_analysis.json").exists()


def test_analyze_feature_stability(tmp_path: Path):
    """Verify impurity and permutation importance stability audit."""
    np.random.seed(42)
    X_dev = np.random.randn(100, 24)
    y_dev = np.random.randint(0, 2, 100)
    X_val = np.random.randn(50, 24)
    y_val = np.random.randint(0, 2, 50)

    feat_indices = [STRUCTURAL_FEATURE_COLUMNS.index(f) for f in LOCKED_FEATURE_NAMES]
    scaler = StandardScaler().fit(X_dev[:, feat_indices])
    clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42).fit(scaler.transform(X_dev[:, feat_indices]), y_dev)

    stab = analyze_feature_stability(X_dev, y_dev, X_val, y_val, scaler, clf, out_dir=tmp_path)

    assert "features" in stab
    assert len(stab["features"]) == 5
    assert (tmp_path / "feature_stability_analysis.json").exists()


def test_analyze_probability_compression(tmp_path: Path):
    """Verify probability compression analysis."""
    np.random.seed(42)
    X_dev = np.random.randn(100, 24)
    y_dev = np.random.randint(0, 2, 100)
    X_val = np.random.randn(50, 24) - 1.0
    y_val = np.random.randint(0, 2, 50)

    feat_indices = [STRUCTURAL_FEATURE_COLUMNS.index(f) for f in LOCKED_FEATURE_NAMES]
    scaler = StandardScaler().fit(X_dev[:, feat_indices])
    clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42).fit(scaler.transform(X_dev[:, feat_indices]), y_dev)

    pcomp = analyze_probability_compression(X_dev, y_dev, X_val, y_val, scaler, clf, out_dir=tmp_path)

    assert "dev_probabilities" in pcomp
    assert "val_probabilities" in pcomp
    assert (tmp_path / "probability_compression_analysis.json").exists()


def test_analyze_error_clusters(tmp_path: Path):
    """Verify error clustering analysis."""
    np.random.seed(42)
    X_val = np.random.randn(50, 24)
    y_val = np.random.randint(0, 2, 50)

    feat_indices = [STRUCTURAL_FEATURE_COLUMNS.index(f) for f in LOCKED_FEATURE_NAMES]
    scaler = StandardScaler().fit(X_val[:, feat_indices])
    clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42).fit(scaler.transform(X_val[:, feat_indices]), y_val)

    clusters = analyze_error_clusters(X_val, y_val, scaler, clf, out_dir=tmp_path)

    assert "clusters" in clusters
    assert "TP" in clusters["clusters"]
    assert (tmp_path / "error_cluster_analysis.json").exists()


def test_synthesize_root_cause(tmp_path: Path):
    """Verify root cause synthesis hierarchy generation."""
    hier = synthesize_root_cause({}, {}, {}, out_dir=tmp_path)

    assert "primary_root_cause" in hier
    assert "secondary_root_causes" in hier
    assert "contributing_factors" in hier
    assert (tmp_path / "root_cause_analysis.json").exists()
