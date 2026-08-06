"""Phase 23 — Master Unit Test Suite for Scientific Landmark Extensions."""

from __future__ import annotations

import pytest
from pathlib import Path
from theory.theoretical_analysis import TheoreticalAnalysisEngine
from theory.information_theory import InformationTheoryEngine
from theory.pgm_engine import ProbabilisticGraphicalModelEngine
from theory.causal_engine import StructuralCausalModelEngine
from visualization.scientific_plots import ScientificPlotEngine

BASE_DIR = Path(__file__).resolve().parent.parent


def test_theoretical_analysis_engine():
    engine = TheoreticalAnalysisEngine()
    res = engine.compute_complexity_bounds()
    assert res["lipschitz_constant"] == 0.455
    assert "pillar1_hybrid_retrieval" in res["time_complexity"]


def test_information_theory_engine():
    engine = InformationTheoryEngine()
    info = engine.compute_information_flow()
    assert info["shannon_entropy_nats"] > 0
    assert info["mutual_information_I_Q_Y"] >= 0


def test_pgm_engine():
    pgm = ProbabilisticGraphicalModelEngine()
    res = pgm.compute_joint_factor_distribution()
    assert res["posterior_probability_h0_factual"] > 0.5
    assert "factor_graph_structure" in res


def test_causal_engine():
    scm = StructuralCausalModelEngine()
    res = scm.compute_causal_treatment_effects()
    assert res["average_treatment_effect_ATE_do_FE"] < 0.0
    assert "counterfactual_explanation" in res


def test_scientific_plot_engine(tmp_path):
    plot_engine = ScientificPlotEngine(output_dir=tmp_path)
    saved = plot_engine.generate_all_scientific_plots()
    assert len(saved) >= 3


def test_latex_proofs_exist():
    foundation = BASE_DIR / "paper" / "mathematical_foundation.tex"
    proofs = BASE_DIR / "paper" / "proofs.tex"
    assert foundation.exists()
    assert proofs.exists()
    content = proofs.read_text(encoding="utf-8")
    assert "Lipschitz" in content


def test_irb_protocol_exists():
    irb = BASE_DIR / "human_study" / "irb_protocol.md"
    assert irb.exists()


def test_hallucisense_bench_leaderboard_exists():
    lb = BASE_DIR.parent / "hallucisense_bench" / "leaderboard.json"
    assert lb.exists()
