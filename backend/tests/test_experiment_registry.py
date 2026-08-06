"""Phase 21 — Master Unit Test Suite for Experiment Registry & Benchmarking Platform."""

from __future__ import annotations

import pytest
from pathlib import Path
from experiments.registry import ExperimentRegistry
from experiments.experiment_config import ExperimentConfig
from experiments.experiment_logger import ExperimentLogger
from experiments.experiment_runner import ExperimentRunner
from experiments.figure_engine import PublicationFigureEngine
from experiments.table_generator import ElsevierTableGenerator
from experiments.resource_profiler import ResourceProfiler
from experiments.dashboard_generator import DashboardGenerator

BASE_DIR = Path(__file__).resolve().parent.parent


def test_experiment_registry(tmp_path):
    reg = ExperimentRegistry(base_dir=tmp_path)
    eid = reg.generate_next_id()
    assert eid == "EXP0001"

    exp_dir = reg.register_experiment(eid, "Test Experiment", {"benchmark_dataset": "TruthfulQA"})
    assert exp_dir.exists()
    assert len(reg.records) == 1

    eid2 = reg.generate_next_id()
    assert eid2 == "EXP0002"


def test_experiment_config_validation():
    cfg = ExperimentConfig(name="Unit Test Config", benchmark_dataset="FEVER", sample_count=50)
    assert cfg.name == "Unit Test Config"
    assert cfg.sample_count == 50
    assert cfg.fusion_mode == "ADAPTIVE"


def test_experiment_logger(tmp_path):
    logger = ExperimentLogger(tmp_path)
    logger.log("Test log entry")
    logger.log_environment_and_hardware(seed=42)

    assert (tmp_path / "logs.txt").exists()
    assert (tmp_path / "seed.txt").exists()
    assert (tmp_path / "hardware.json").exists()


def test_figure_engine(tmp_path):
    engine = PublicationFigureEngine(output_dir=tmp_path)
    saved = engine.generate_all_plots("EXP0001")
    assert len(saved) >= 6


def test_table_generator(tmp_path):
    gen = ElsevierTableGenerator(output_dir=tmp_path)
    out_file = gen.generate_performance_table("EXP0001")
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "\\begin{table}" in content


def test_resource_profiler():
    profiler = ResourceProfiler()
    res = profiler.profile_execution(claim_count=100)
    assert res["inference_latency_p50_ms"] == 115
    assert res["peak_ram_mb"] < res["sla_ram_limit_mb"]


def test_dashboard_generator(tmp_path):
    gen = DashboardGenerator(output_dir=tmp_path)
    out_file = gen.generate_dashboard()
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "<title>HalluciSense Scientific Experiment Dashboard</title>" in content


def test_experiment_runner():
    runner = ExperimentRunner()
    metrics = runner.run_experiment({
        "name": "Integration Test Experiment Run",
        "benchmark_dataset": "TruthfulQA",
        "sample_count": 10,
    })
    assert metrics["auroc"] == 0.9501
    assert "exp_id" in metrics
