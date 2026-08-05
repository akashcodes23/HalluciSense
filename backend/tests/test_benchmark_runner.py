"""Tests for Phase 14 Evaluation Dataset and Benchmark Suite."""

import pytest
import numpy as np
from evaluation.phase14.dataset_loader import EvaluationDataset, DOMAINS
from evaluation.phase14.evaluator import MetricAggregator, BaselineModelSimulator


def test_dataset_generation():
    dataset = EvaluationDataset.generate_benchmark_dataset(n_per_domain=10, random_seed=42)
    assert len(dataset) == 15 * 10
    assert len(DOMAINS) == 15

    med_dataset = dataset.filter_by_domain("Medicine")
    assert len(med_dataset) == 10


def test_metric_aggregator():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.85])

    metrics = MetricAggregator.compute_all_metrics(y_true, y_prob, threshold=0.50)
    assert metrics["accuracy"] >= 0.75
    assert metrics["auroc"] >= 0.80
    assert "ece" in metrics
    assert "brier_score" in metrics


def test_baseline_simulator():
    rng = np.random.default_rng(42)
    ds = EvaluationDataset.generate_benchmark_dataset(n_per_domain=5, random_seed=42)
    probs = [BaselineModelSimulator.predict_baseline("SelfCheckGPT", s, rng) for s in ds.samples]
    assert len(probs) == 75
    assert all(0.0 <= p <= 1.0 for p in probs)
