"""
Unit tests for HalluciSense Phase 11 — Benchmarking, Scientific Validation & Research Package.
"""

from pathlib import Path
import pytest
from evaluation.phase11.module11_1_datasets import BenchmarkDatasetAdapter
from evaluation.phase11.module11_2_baselines import (
    ConfidenceOnlyBaseline,
    FActScoreBaseline,
    MajorityBaseline,
    RAGASBaseline,
    SelfCheckGPTBaseline,
)
from evaluation.phase11.module11_3_evaluation import HeadToHeadEvaluator
from evaluation.phase11.module11_4_statistics import StatisticalSignificanceEngine
from evaluation.phase11.module11_5_ablations import AblationStudySuite
from evaluation.phase11.module11_6_robustness import RobustnessAnalyzer
from evaluation.phase11.module11_7_generalization import CrossDomainGeneralizationEvaluator
from evaluation.phase11.module11_8_error_taxonomy import ErrorTaxonomyAnalyzer
from evaluation.phase11.module11_9_latency_benchmarks import LatencyResourceProfiler
from evaluation.phase11.module11_10_figures import PublicationFigureRenderer
from evaluation.phase11.module11_11_reproducibility import ReproducibilityPackageBuilder
from evaluation.phase11.module11_12_paper_generator import IEEEPaperGenerator
from evaluation.phase11.module11_13_leaderboard import ScientificLeaderboardRenderer


@pytest.fixture
def dataset_adapter():
    return BenchmarkDatasetAdapter()


def test_module11_1_datasets(dataset_adapter):
    meta = dataset_adapter.get_all_metadata()
    assert len(meta) == 8
    assert "HaluEval" in meta
    assert "TruthfulQA" in meta

    samples = dataset_adapter.load_dataset("HaluEval", split="test", num_samples=10)
    assert len(samples) == 10
    assert samples[0].dataset_name == "HaluEval"


def test_module11_2_baselines(dataset_adapter):
    sample = dataset_adapter.load_dataset("TruthfulQA", num_samples=1)[0]

    baselines = [
        SelfCheckGPTBaseline(),
        FActScoreBaseline(),
        RAGASBaseline(),
        ConfidenceOnlyBaseline(),
        MajorityBaseline(),
    ]

    for b in baselines:
        prob, pred = b.predict_sample(sample)
        assert 0.0 <= prob <= 1.0
        assert pred in [0, 1]


def test_module11_3_head_to_head_evaluation(dataset_adapter):
    samples = dataset_adapter.load_dataset("HaluEval", num_samples=20)
    evaluator = HeadToHeadEvaluator()
    detector = FActScoreBaseline()

    metrics, y_true, y_prob = evaluator.evaluate_system(detector, samples)
    assert metrics.system_name == "FActScore"
    assert 0.0 <= metrics.roc_auc <= 1.0
    assert 0.0 <= metrics.f1_score <= 1.0
    assert metrics.mean_latency_ms >= 0.0


def test_module11_4_statistics(dataset_adapter):
    samples = dataset_adapter.load_dataset("HaluEval", num_samples=20)
    evaluator = HeadToHeadEvaluator()
    _, y_true, y_prob_a = evaluator.evaluate_system(FActScoreBaseline(), samples)
    _, _, y_prob_b = evaluator.evaluate_system(MajorityBaseline(), samples)

    stats_engine = StatisticalSignificanceEngine()
    sig = stats_engine.compare_systems("FActScore", y_prob_a, "Majority", y_prob_b, y_true)
    assert sig.auc_diff != 0.0
    assert 0.0 <= sig.p_value_delong <= 1.0
    assert len(sig.bootstrap_ci_a) == 2


def test_module11_5_ablations(dataset_adapter):
    samples = dataset_adapter.load_dataset("HaluEval", num_samples=10)
    suite = AblationStudySuite()
    results = suite.evaluate_ablations(samples)
    assert len(results) == 8
    assert results[0].variant_name == "Full HalluciSense"
    assert results[0].auc_drop_from_full == 0.0


def test_module11_6_robustness(dataset_adapter):
    samples = dataset_adapter.load_dataset("HaluEval", num_samples=10)
    analyzer = RobustnessAnalyzer()
    results = analyzer.evaluate_robustness(samples)
    assert len(results) == 8
    assert all(r.performance_retention_pct > 80.0 for r in results)


def test_module11_7_generalization(dataset_adapter):
    samples = dataset_adapter.load_dataset("HaluEval", num_samples=12)
    gen_eval = CrossDomainGeneralizationEvaluator()
    results = gen_eval.evaluate_generalization(samples)
    assert len(results) == 6
    assert any(r.domain_name == "Medicine" for r in results)


def test_module11_8_error_taxonomy(dataset_adapter):
    samples = dataset_adapter.load_dataset("HaluEval", num_samples=10)
    analyzer = ErrorTaxonomyAnalyzer()
    probs = [0.85 if s.ground_truth_label == 1 else 0.15 for s in samples]
    probs[0] = 1.0 - probs[0]  # Force one error
    report = analyzer.analyze_errors(samples, probs)
    assert report.total_samples_evaluated == 10
    assert len(report.categories) == 8


def test_module11_9_latency_profiler():
    profiler = LatencyResourceProfiler()
    report = profiler.profile_system(n_iterations=20)
    assert report.p50_latency_ms > 0.0
    assert report.p95_latency_ms >= report.p50_latency_ms
    assert report.peak_memory_mb > 0.0


def test_module11_10_11_12_13_generators(tmp_path):
    # Figures
    fig_renderer = PublicationFigureRenderer()
    fig_paths = fig_renderer.render_all_figures(tmp_path / "figures")
    assert len(fig_paths) >= 15  # PNG, SVG, PDF for 5 figures

    # Reproducibility
    repro_builder = ReproducibilityPackageBuilder()
    repro = repro_builder.generate_package(tmp_path / "repro")
    assert (tmp_path / "repro" / "Dockerfile").exists()

    # Paper
    paper_gen = IEEEPaperGenerator()
    paper_files = paper_gen.generate_paper(tmp_path / "docs")
    assert (tmp_path / "docs" / "paper.tex").exists()

    # Leaderboard
    lb_renderer = ScientificLeaderboardRenderer()
    lb_files = lb_renderer.generate_leaderboard(tmp_path / "lb")
    assert (tmp_path / "lb" / "leaderboard.md").exists()
