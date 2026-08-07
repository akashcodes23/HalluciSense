"""Pytest Test Suite for Phase 26 Datasets, Baselines, Metrics, and Figures."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from evaluation.datasets.public_benchmark_loaders import load_all_benchmark_datasets, StandardizedBenchmarkDataset
from evaluation.baselines.unified_baselines import get_all_sota_baselines, HalluciSenseSystem
from evaluation.metrics_engine import compute_all_metrics
from evaluation.statistical_validation_engine import bootstrap_ci, mcnemar_test, cohens_d, cliffs_delta
from evaluation.ablation_studies_engine import run_ablation_studies

BASE_DIR = Path(__file__).resolve().parent.parent


def test_public_dataset_loaders():
    """Verify loading of 11 public benchmark datasets."""
    datasets = load_all_benchmark_datasets(max_per_dataset=5)
    assert len(datasets) == 11
    
    for d_name in StandardizedBenchmarkDataset.DATASET_NAMES:
        assert d_name in datasets
        samples = datasets[d_name]
        assert len(samples) > 0
        s = samples[0]
        assert "question" in s
        assert "response" in s
        assert "label" in s
        assert "domain" in s


def test_sota_baselines_prediction():
    """Verify unified SOTA baselines return standardized predict structure."""
    baselines = get_all_sota_baselines()
    assert len(baselines) == 10
    
    for b_name, b_inst in baselines.items():
        res = b_inst.predict("What is the capital of France?", "The capital of France is Paris.")
        assert "score" in res
        assert "confidence" in res
        assert "runtime_ms" in res
        assert "metadata" in res
        assert 0.0 <= res["score"] <= 1.0


def test_metrics_computation():
    """Verify calculation of classification, AUROC, ECE, and latency metrics."""
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_prob = [0.02, 0.05, 0.10, 0.08, 0.12, 0.88, 0.92, 0.95, 0.89, 0.96]
    latencies = [10.0, 12.0, 15.0, 11.0, 13.0, 14.0, 12.5, 11.5, 10.5, 13.5]
    
    m = compute_all_metrics(y_true, y_prob, latencies, threshold=0.54)
    assert m["accuracy"] == 1.0
    assert m["auroc"] == 1.0
    assert m["f1_score"] == 1.0
    assert m["ece"] <= 0.08


def test_statistical_tests():
    """Verify bootstrap CI, McNemar, Cohen's d, and Cliff's Delta functions."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_prob = np.array([0.05, 0.1, 0.9, 0.85, 0.1, 0.95, 0.05, 0.88])
    
    mean_acc, lower, upper = bootstrap_ci(y_true, y_prob, n_bootstraps=100)
    assert 0.0 <= lower <= mean_acc <= upper <= 1.0
    
    stat, p = mcnemar_test(y_true, (y_prob >= 0.54).astype(int), (y_prob >= 0.54).astype(int))
    assert p == 1.0
    
    d = cohens_d(y_prob, y_prob)
    assert d == 0.0
    
    delta = cliffs_delta(y_prob, y_prob)
    assert delta == 0.0


def test_ablation_studies():
    """Verify evaluation of 13 ablation variants."""
    y_true = np.array([0, 0, 1, 1, 0])
    base_probs = np.array([0.1, 0.2, 0.9, 0.8, 0.15])
    
    df = run_ablation_studies(y_true, base_probs)
    assert len(df) == 13
    assert "Full HalluciSense (Proposed)" in df["variant"].values
