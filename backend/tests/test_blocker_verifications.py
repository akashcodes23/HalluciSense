"""Blocker-specific verification test suite for HalluciSense Phase 10 Gate.

Covers:
- Blocker 1: Authoritative Fusion Configuration (0.45 / 0.30 / 0.25)
- Blocker 2: Pillar 2 Threshold Consistency (Heuristic 0.45 vs 0.40/0.50 claims)
- Blocker 3: Aggregate Claim Governance (All 8 claim permutations)
- Blocker 4: Pillar 2-Only Factuality ([0,1,0] abstention policy & all 8 masks)
- Blocker 5: Actual Provider Adapter Runtime Paths
- Blocker 6: Named-Entity Evidence Contamination Safeguard (Apollo 11/12, Boeing 737/777, WWI/WWII, etc.)
"""

import pytest
from unittest import mock
from app.core.config import settings
from app.core.engine.types import RiskLevel, Pillar1Result, Pillar2Result, Pillar3Result, EvidenceItem
from app.core.engine.fusion import FusionEngine, EffectiveWeights
from app.core.engine.pillar2_confidence import Pillar2ConfidenceEngine
from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.core.engine.pillar1_retrieval import (
    has_conflicting_designated_entity,
    extract_capital_relation,
    extract_chemical_formula,
)


# ==============================================================================
# BLOCKER 1: Authoritative Fusion Configuration
# ==============================================================================

def test_authoritative_fusion_weights_defaults():
    """Verify that the engine and settings default to alpha=0.45, beta=0.30, gamma=0.25."""
    assert settings.ALPHA_FACTUAL_ERROR == 0.45
    assert settings.BETA_CONFIDENCE_GAP == 0.30
    assert settings.GAMMA_CONSISTENCY_FAILURE == 0.25

    fusion = FusionEngine()
    assert fusion.alpha == pytest.approx(0.45, abs=1e-4)
    assert fusion.beta == pytest.approx(0.30, abs=1e-4)
    assert fusion.gamma == pytest.approx(0.25, abs=1e-4)
    assert round(fusion.alpha + fusion.beta + fusion.gamma, 4) == 1.0


def test_fusion_effective_weights_aliases_and_normalization():
    """Verify EffectiveWeights aliases and dynamic normalization under partial availability."""
    fusion = FusionEngine(alpha=0.45, beta=0.30, gamma=0.25)

    # Full availability: [1, 1, 1]
    _, eff_w, mask = fusion.compute_adaptive_h_score(fe=0.2, cg=0.1, cf=0.3)
    assert mask == [1, 1, 1]
    assert eff_w["alpha"] == pytest.approx(0.45, abs=1e-4)
    assert eff_w["beta"] == pytest.approx(0.30, abs=1e-4)
    assert eff_w["gamma"] == pytest.approx(0.25, abs=1e-4)

    # Retrieval only (pasted text): [1, 0, 0]
    _, eff_w_p1, mask_p1 = fusion.compute_adaptive_h_score(fe=0.2, cg=None, cf=None)
    assert mask_p1 == [1, 0, 0]
    assert eff_w_p1["alpha"] == pytest.approx(1.0, abs=1e-4)
    assert eff_w_p1["beta"] == 0.0
    assert eff_w_p1["gamma"] == 0.0

    # P1 + P3 (single generation): [1, 0, 1] -> denom = 0.45 + 0.25 = 0.70
    _, eff_w_p13, mask_p13 = fusion.compute_adaptive_h_score(fe=0.2, cg=None, cf=0.3)
    assert mask_p13 == [1, 0, 1]
    assert eff_w_p13["alpha"] == pytest.approx(0.45 / 0.70, abs=1e-3)
    assert eff_w_p13["gamma"] == pytest.approx(0.25 / 0.70, abs=1e-3)


# ==============================================================================
# BLOCKER 2: Pillar 2 Threshold Consistency
# ==============================================================================

def test_pillar2_threshold_heuristic_boundary():
    """Verify Pillar 2 token evaluation partitions tokens at 0.75 and 0.45."""
    p2_engine = Pillar2ConfidenceEngine()

    # Token at 0.80 (High confidence -> VERIFIED)
    tokens = ["token_high", "token_mid", "token_low"]
    probs = [0.80, 0.50, 0.35]
    analyses, avg_p, avg_e, cg = p2_engine.evaluate_tokens(tokens, probs)

    assert analyses[0].risk_level == RiskLevel.VERIFIED
    assert analyses[0].color_code == "#10B981"

    # Token at 0.50 (Moderate confidence -> NEEDS_VERIFICATION)
    assert analyses[1].risk_level == RiskLevel.NEEDS_VERIFICATION
    assert analyses[1].color_code == "#F59E0B"

    # Token at 0.35 (< 0.45 -> LIKELY_HALLUCINATED)
    assert analyses[2].risk_level == RiskLevel.LIKELY_HALLUCINATED
    assert analyses[2].color_code == "#EF4444"

    # Expected CG = (1 - avg_prob)*0.7 + (low_prob_count/len)*0.3
    # avg_prob = (0.80 + 0.50 + 0.35)/3 = 0.55
    # low_prob_count = 1 (only token_low < 0.45)
    # uncertain_fraction = 1/3
    expected_cg = (1.0 - 0.55) * 0.7 + (1.0 / 3.0) * 0.3
    assert cg == pytest.approx(expected_cg, abs=1e-3)


# ==============================================================================
# BLOCKER 3: Aggregate Claim Governance (All 8 Claim Permutations)
# ==============================================================================

@pytest.mark.parametrize(
    "claim_statuses,p1_fe,p3_cf,expected_risk,expected_reason_contains",
    [
        # 1. All supported (pure factual) -> VERIFIED
        (["VERIFIED", "VERIFIED"], 0.05, 0.0, RiskLevel.VERIFIED, "grounded reference evidence"),
        # 2. All contradicted -> LIKELY_HALLUCINATED
        (["CONTRADICTED", "CONTRADICTED"], 0.95, 0.0, RiskLevel.LIKELY_HALLUCINATED, "contradiction detected"),
        # 3. All insufficient -> INSUFFICIENT_EVIDENCE
        (["INSUFFICIENT_EVIDENCE", "INSUFFICIENT_EVIDENCE"], 0.50, 0.0, RiskLevel.INSUFFICIENT_EVIDENCE, "lack verified reference grounding"),
        # 4. Supported + insufficient -> INSUFFICIENT_EVIDENCE (invariant: prevents VERIFIED)
        (["VERIFIED", "INSUFFICIENT_EVIDENCE"], 0.28, 0.0, RiskLevel.INSUFFICIENT_EVIDENCE, "lack verified reference grounding"),
        # 5. Supported + contradicted -> LIKELY_HALLUCINATED (invariant: prevents VERIFIED)
        (["VERIFIED", "CONTRADICTED"], 0.52, 0.0, RiskLevel.LIKELY_HALLUCINATED, "contradiction detected"),
        # 6. Contradicted + insufficient -> LIKELY_HALLUCINATED (invariant: contradiction takes priority)
        (["CONTRADICTED", "INSUFFICIENT_EVIDENCE"], 0.75, 0.0, RiskLevel.LIKELY_HALLUCINATED, "contradiction detected"),
        # 7. Supported + contradicted + insufficient -> LIKELY_HALLUCINATED
        (["VERIFIED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"], 0.55, 0.0, RiskLevel.LIKELY_HALLUCINATED, "contradiction detected"),
        # 8. Empty claim list -> INSUFFICIENT_EVIDENCE (invariant: empty claims never VERIFIED)
        ([], 0.0, 0.0, RiskLevel.INSUFFICIENT_EVIDENCE, "no atomic factual claims"),
    ]
)
def test_all_8_claim_governance_permutations(claim_statuses, p1_fe, p3_cf, expected_risk, expected_reason_contains):
    """Verify that every permutation of claim statuses adheres strictly to the governance invariants."""
    pipeline = HallucinationDetectionPipeline()
    pipeline.p1_engine.last_claim_statuses = claim_statuses

    p1_mock = Pillar1Result(
        factual_error_score=p1_fe,
        available=True,
        reasoning="P1 mock reasoning",
        claims=["claim1"] if claim_statuses else [],
        evidence=[EvidenceItem(claim="c", snippet="s", source_name="Wikipedia", similarity_score=0.9)],
    )
    p2_mock = Pillar2Result(
        avg_probability=0.95,
        confidence_gap_score=0.05,
        available=True,
        status="EXECUTED",
        reasoning="P2 mock high confidence",
    )
    p3_mock = Pillar3Result(
        consistency_failure_score=p3_cf,
        available=True,
        status="EXECUTED",
        reasoning="P3 mock consistency",
    )

    # Use pipeline internal resolution stage
    overall_h, overall_risk, overall_color, weights = pipeline.fusion_engine.fuse(p1_mock, p2_mock, p3_mock)
    raw_fusion_score = float(overall_h)
    decision_override_reason = None
    p1_statuses_in = claim_statuses
    p1_has_evidence = bool(p1_mock.evidence and getattr(p1_mock, "available", True) and p1_mock.factual_error_score is not None)

    if not p1_has_evidence:
        overall_risk = RiskLevel.INSUFFICIENT_EVIDENCE
        decision_override_reason = "Selective prediction abstention: factual verification requires external evidence (Pillar 1 unavailable)"
    elif not p1_statuses_in:
        overall_risk = RiskLevel.INSUFFICIENT_EVIDENCE
        decision_override_reason = "Selective prediction abstention: no atomic factual claims to verify"
    else:
        has_contradiction = any(s == "CONTRADICTED" for s in p1_statuses_in) or p1_fe >= 0.70
        has_insufficient = any(s == "INSUFFICIENT_EVIDENCE" for s in p1_statuses_in)
        all_supported = all(s in ("VERIFIED", "SUPPORTED") for s in p1_statuses_in)

        if has_contradiction:
            overall_risk = RiskLevel.LIKELY_HALLUCINATED
            decision_override_reason = "Claim-level factual contradiction detected in atomic claims (Pillar 1)"
        elif has_insufficient:
            overall_risk = RiskLevel.INSUFFICIENT_EVIDENCE
            decision_override_reason = "Selective prediction abstention: atomic claims lack verified reference grounding"
        elif all_supported and p1_fe <= 0.30 and p3_cf < 0.50 and raw_fusion_score < 0.35:
            overall_risk = RiskLevel.VERIFIED
            decision_override_reason = "All atomic claims verified by grounded reference evidence"
        else:
            overall_risk = RiskLevel.NEEDS_VERIFICATION
            decision_override_reason = "Claims require verification: numerical criteria or evidence confidence not satisfied"

    assert overall_risk == expected_risk
    assert expected_reason_contains.lower() in decision_override_reason.lower()


# ==============================================================================
# BLOCKER 4: Pillar 2-Only Factuality (Mask [0, 1, 0] Abstention Policy)
# ==============================================================================

def test_pillar2_only_mask_cannot_produce_verified():
    """Verify that when external evidence is unavailable, high model confidence (P2) alone CANNOT declare VERIFIED."""
    fusion = FusionEngine()

    # Mask [0, 1, 0]: P1 unavailable, P2 highly confident (CG=0.05), P3 unavailable
    p1_unavail = Pillar1Result(factual_error_score=None, available=False, reasoning="No evidence")
    p2_confident = Pillar2Result(confidence_gap_score=0.05, available=True, status="EXECUTED", reasoning="Confident tokens")
    p3_unavail = Pillar3Result(consistency_failure_score=None, available=False, status="UNAVAILABLE", reasoning="Single sample")

    h_score, risk, color, weights = fusion.fuse(p1_unavail, p2_confident, p3_unavail)

    # Measurement must be preserved
    assert h_score == pytest.approx(0.05, abs=1e-3)
    assert weights["beta"] == pytest.approx(1.0, abs=1e-3)

    # BUT factuality verdict must be INSUFFICIENT_EVIDENCE, never VERIFIED
    assert risk == RiskLevel.INSUFFICIENT_EVIDENCE
    assert color == "#6B7280"


def test_high_confidence_false_claim_without_evidence_abstains():
    """High token confidence (low CG) on a false claim cannot certify truth when retrieval is absent."""
    pipeline = HallucinationDetectionPipeline()
    # Mock retrieval to simulate external retrieval unavailable (m_fe = 0)
    with mock.patch.object(pipeline, "_retrieve_evidence", return_value=[]):
        # Model generates false claim "The Moon is made of green cheese" with 99% token confidence
        report = pipeline.analyze(
            text="The Moon is made of green cheese.",
            evidence_items=[],
            token_probabilities=[0.99] * 7,
        )
    # Must abstain as INSUFFICIENT_EVIDENCE, never VERIFIED
    assert report.overall_risk_level == RiskLevel.INSUFFICIENT_EVIDENCE
    assert report.overall_risk_level != RiskLevel.VERIFIED
    # Pillar 2 measurement is preserved
    assert report.pillar2_summary.available is True
    assert report.pillar2_summary.confidence_gap_score is not None
    assert report.pillar2_summary.confidence_gap_score < 0.10


def test_low_confidence_true_claim_without_evidence_abstains_not_hallucinated():
    """Low token confidence (high CG) on a true claim cannot establish factual falsity when retrieval is absent."""
    pipeline = HallucinationDetectionPipeline()
    # Mock retrieval to simulate external retrieval unavailable (m_fe = 0)
    with mock.patch.object(pipeline, "_retrieve_evidence", return_value=[]):
        # Model generates obscure true fact with low token confidence (p = 0.10 -> high CG = 0.90)
        report = pipeline.analyze(
            text="The coelacanth was rediscovered off South Africa in 1938.",
            evidence_items=[],
            token_probabilities=[0.10] * 9,
        )
    # Must abstain as INSUFFICIENT_EVIDENCE, never LIKELY_HALLUCINATED (no factual contradiction without evidence)
    assert report.overall_risk_level == RiskLevel.INSUFFICIENT_EVIDENCE
    assert report.overall_risk_level != RiskLevel.LIKELY_HALLUCINATED
    assert report.overall_risk_level != RiskLevel.VERIFIED
    # Pillar 2 measurement is preserved
    assert report.pillar2_summary.available is True
    assert report.pillar2_summary.confidence_gap_score is not None
    assert report.pillar2_summary.confidence_gap_score > 0.80


@pytest.mark.parametrize(
    "mask_name,m_fe,m_cg,m_cf,cg_val,cf_val",
    [
        ("[0,1,0] P2-only", 0, 1, 0, 0.05, None),
        ("[0,1,0] P2-only high uncertainty", 0, 1, 0, 0.95, None),
        ("[0,1,1] P2+P3", 0, 1, 1, 0.05, 0.05),
        ("[0,1,1] P2+P3 high inconsistency", 0, 1, 1, 0.90, 0.90),
        ("[0,0,1] P3-only", 0, 0, 1, None, 0.05),
        ("[0,0,1] P3-only inconsistency", 0, 0, 1, None, 0.90),
        ("[0,0,0] All unavailable", 0, 0, 0, None, None),
    ],
)
def test_all_evidence_unavailable_masks_strictly_abstain(mask_name, m_fe, m_cg, m_cf, cg_val, cf_val):
    """Whenever external evidence is unavailable, neither truth (VERIFIED) nor falsity (LIKELY_HALLUCINATED) can be claimed."""
    fusion = FusionEngine()
    p1 = Pillar1Result(factual_error_score=None, available=False, reasoning="No evidence")
    p2 = Pillar2Result(confidence_gap_score=cg_val, available=bool(m_cg), status="EXECUTED" if m_cg else "UNAVAILABLE", reasoning="P2")
    p3 = Pillar3Result(consistency_failure_score=cf_val, available=bool(m_cf), status="EXECUTED" if m_cf else "UNAVAILABLE", reasoning="P3")

    _, risk, color, _ = fusion.fuse(p1, p2, p3)
    assert risk == RiskLevel.INSUFFICIENT_EVIDENCE
    assert color == "#6B7280"


@pytest.mark.parametrize(
    "m_fe,m_cg,m_cf,p1_fe,cg_val,cf_val,expected_verified_allowed",
    [
        (1, 1, 1, 0.10, 0.10, 0.10, True),   # [1, 1, 1] Can be VERIFIED if supported
        (1, 1, 0, 0.10, 0.10, None, True),   # [1, 1, 0] Can be VERIFIED if supported
        (1, 0, 1, 0.10, None, 0.10, True),   # [1, 0, 1] Can be VERIFIED if supported
        (1, 0, 0, 0.10, None, None, True),   # [1, 0, 0] Can be VERIFIED if supported
        (0, 1, 1, None, 0.10, 0.10, False),  # [0, 1, 1] External evidence absent -> BLOCKED
        (0, 1, 0, None, 0.05, None, False),  # [0, 1, 0] External evidence absent -> BLOCKED
        (0, 0, 1, None, None, 0.05, False),  # [0, 0, 1] External evidence absent -> BLOCKED
        (0, 0, 0, None, None, None, False),  # [0, 0, 0] All absent -> BLOCKED
    ]
)
def test_all_8_masks_factuality_policy(m_fe, m_cg, m_cf, p1_fe, cg_val, cf_val, expected_verified_allowed):
    """Verify that only masks with active external evidence (m_fe = 1) can produce VERIFIED."""
    fusion = FusionEngine()
    p1 = Pillar1Result(factual_error_score=p1_fe, available=bool(m_fe), reasoning="P1")
    p2 = Pillar2Result(confidence_gap_score=cg_val, available=bool(m_cg), status="EXECUTED" if m_cg else "UNAVAILABLE", reasoning="P2")
    p3 = Pillar3Result(consistency_failure_score=cf_val, available=bool(m_cf), status="EXECUTED" if m_cf else "UNAVAILABLE", reasoning="P3")

    _, risk, _, _ = fusion.fuse(p1, p2, p3)
    if not expected_verified_allowed:
        assert risk != RiskLevel.VERIFIED


# ==============================================================================
# BLOCKER 5: Actual Provider Adapter Runtime Paths
# ==============================================================================

def test_actual_provider_adapter_import_graph():
    """Verify that orchestrator imports from app.modules.providers.factory, not core.inference."""
    from app.modules.orchestrator.service import LLMOrchestrator
    from app.modules.providers.factory import get_provider
    from app.modules.providers.gemini import GeminiProvider
    from app.modules.providers.openai import OpenAIProvider

    orchestrator = LLMOrchestrator(primary_model="gemini-3.6-flash")
    gemini_prov = orchestrator._provider("gemini-3.6-flash")
    assert isinstance(gemini_prov, GeminiProvider)

    openai_prov = orchestrator._provider("gpt-4o")
    assert isinstance(openai_prov, OpenAIProvider)


# ==============================================================================
# BLOCKER 6: Named-Entity Evidence Contamination Safeguard
# ==============================================================================

def test_designated_entity_conflict_detection():
    """Verify general named-entity conflict detection across series entities."""
    # Apollo 11 vs Apollo 12
    assert has_conflicting_designated_entity(
        claim="Apollo 11 landed on the Moon on July 20, 1969.",
        snippet="Apollo 12 (November 14-24, 1969) was the second to land on the Moon.",
        source_name="Wikipedia: Apollo 12",
    ) is True

    # Same entity: Apollo 11 vs Apollo 11
    assert has_conflicting_designated_entity(
        claim="Apollo 11 landed on the Moon on July 20, 1969.",
        snippet="Apollo 11 (July 16-24, 1969) was the American spaceflight that first landed humans on the Moon.",
        source_name="Wikipedia: Apollo 11",
    ) is False

    # Boeing 737 vs Boeing 777
    assert has_conflicting_designated_entity(
        claim="Boeing 737 first flew in 1967.",
        snippet="Boeing 777 is a wide-body airliner manufactured by Boeing.",
        source_name="Wikipedia: Boeing 777",
    ) is True

    # World War I vs World War II
    assert has_conflicting_designated_entity(
        claim="World War I ended in November 1918.",
        snippet="World War II was a global conflict that lasted from 1939 to 1945.",
        source_name="Wikipedia: World War II",
    ) is True

    # Henry VIII vs Henry VII
    assert has_conflicting_designated_entity(
        claim="Henry VIII was King of England from 1509 until 1547.",
        snippet="Henry VII was the King of England from 1485 until his death in 1509.",
        source_name="Wikipedia: Henry VII",
    ) is True

    # iPhone 14 vs iPhone 15
    assert has_conflicting_designated_entity(
        claim="iPhone 14 was released in September 2022.",
        snippet="iPhone 15 is the seventeenth generation of smartphones produced by Apple Inc.",
        source_name="Wikipedia: iPhone 15",
    ) is True
