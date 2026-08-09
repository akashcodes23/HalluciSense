"""Tests for the research-only seven-way pillar ablation harness."""

import asyncio

from app.core.engine.fusion import FusionEngine
from scripts.benchmark_ablation import MODES, fuse_selected


def test_ablation_contains_exactly_seven_required_modes():
    assert MODES == [
        "P1",
        "P2",
        "P3",
        "P1_P2",
        "P1_P3",
        "P2_P3",
        "P1_P2_P3",
    ]


def test_single_pillar_modes_use_only_the_selected_score():
    fusion = FusionEngine(alpha=0.5, beta=0.25, gamma=0.25)

    h1, _, w1 = fuse_selected(fusion, {"P1": 0.2}, "P1")
    h2, _, w2 = fuse_selected(fusion, {"P2": 0.6}, "P2")
    h3, _, w3 = fuse_selected(fusion, {"P3": 0.8}, "P3")

    assert h1 == 0.2
    assert h2 == 0.6
    assert h3 == 0.8
    assert w1 == {"P1": 1.0, "P2": 0.0, "P3": 0.0}
    assert w2 == {"P1": 0.0, "P2": 1.0, "P3": 0.0}
    assert w3 == {"P1": 0.0, "P2": 0.0, "P3": 1.0}


def test_pairwise_modes_renormalize_configured_weights():
    fusion = FusionEngine(alpha=0.5, beta=0.3, gamma=0.2)

    h12, _, w12 = fuse_selected(fusion, {"P1": 0.2, "P2": 0.8}, "P1_P2")
    h13, _, w13 = fuse_selected(fusion, {"P1": 0.2, "P3": 0.8}, "P1_P3")
    h23, _, w23 = fuse_selected(fusion, {"P2": 0.3, "P3": 0.9}, "P2_P3")

    assert w12 == {"P1": 0.625, "P2": 0.375, "P3": 0.0}
    assert w13 == {"P1": 0.7143, "P2": 0.0, "P3": 0.2857}
    assert w23 == {"P1": 0.0, "P2": 0.6, "P3": 0.4}
    assert h12 == 0.425
    assert h13 == 0.3714
    assert h23 == 0.54


def test_ablation_does_not_modify_fusion_configuration():
    fusion = FusionEngine(alpha=0.5, beta=0.3, gamma=0.2)
    original = (fusion.alpha, fusion.beta, fusion.gamma)

    fuse_selected(fusion, {"P1": 0.1, "P2": 0.2, "P3": 0.3}, "P1_P2_P3")

    assert (fusion.alpha, fusion.beta, fusion.gamma) == original


def test_unavailable_pillar_is_not_fabricated():
    fusion = FusionEngine(alpha=0.5, beta=0.3, gamma=0.2)

    # An unavailable P3 is represented by None. The harness must not invent a
    # synthetic consistency score just to complete an ablation row.
    h, risk, weights = fuse_selected(
        fusion,
        {"P1": 0.2, "P2": 0.4, "P3": None},
        "P1_P2_P3",
    )

    assert h == 0.275
    assert weights == {"P1": 0.625, "P2": 0.375, "P3": 0.0}
    assert risk == "VERIFIED"


def test_unavailable_p3_is_null_not_zero():
    """Regression test: verify unavailable P3 is represented as None/null and not 0.0.

    Verifies:
    1. Single-pillar mode P3 with unavailable P3 yields H=None, risk=None.
    2. Pairwise mode P1_P3 with unavailable P3 yields H=P1 score (renormalized, NOT treated as 0.0).
    3. Pairwise mode P2_P3 with unavailable P3 yields H=P2 score (renormalized, NOT treated as 0.0).
    4. Three-pillar mode P1_P2_P3 with unavailable P3 yields dynamic fusion of P1 and P2 (P3 weight = 0.0).
    """
    fusion = FusionEngine(alpha=0.4, beta=0.3, gamma=0.3)
    p1_score = 0.40
    p2_score = 0.60
    p3_score = None  # Quota error / circuit breaker tripped / no samples available

    scores = {"P1": p1_score, "P2": p2_score, "P3": p3_score}

    # 1. P3 mode: P3 is unavailable -> H must be None, NOT 0.0
    h_p3, risk_p3, w_p3 = fuse_selected(fusion, scores, "P3")
    assert h_p3 is None, f"Expected H=None for unavailable P3 mode, got {h_p3}"
    assert risk_p3 is None, f"Expected risk=None for unavailable P3 mode, got {risk_p3}"

    # 2. P1_P3 mode: P3 unavailable -> H must equal P1 (0.40), NOT (0.4*0.4 + 0.3*0.0)/(0.7) = 0.2286!
    h_p1_p3, risk_p1_p3, w_p1_p3 = fuse_selected(fusion, scores, "P1_P3")
    assert h_p1_p3 == p1_score, f"Expected H={p1_score} for P1_P3 when P3 is None, got {h_p1_p3}"
    assert w_p1_p3 == {"P1": 1.0, "P2": 0.0, "P3": 0.0}

    # 3. P2_P3 mode: P3 unavailable -> H must equal P2 (0.60), NOT (0.3*0.6 + 0.3*0.0)/(0.6) = 0.30!
    h_p2_p3, risk_p2_p3, w_p2_p3 = fuse_selected(fusion, scores, "P2_P3")
    assert h_p2_p3 == p2_score, f"Expected H={p2_score} for P2_P3 when P3 is None, got {h_p2_p3}"
    assert w_p2_p3 == {"P1": 0.0, "P2": 1.0, "P3": 0.0}

    # 4. P1_P2_P3 mode: P3 unavailable -> H must equal dynamic fusion of P1 and P2 (P3 weight = 0.0)
    h_full, risk_full, w_full = fuse_selected(fusion, scores, "P1_P2_P3")
    expected_h = round((0.4 * 0.40 + 0.3 * 0.60) / 0.7, 4)  # 0.34 / 0.7 = 0.4857
    assert h_full == expected_h, f"Expected H={expected_h} for P1_P2_P3 when P3 is None, got {h_full}"
    assert w_full["P3"] == 0.0
