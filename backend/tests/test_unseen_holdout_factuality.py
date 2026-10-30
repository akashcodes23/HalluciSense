"""Independent Holdout Verification Suite for HalluciSense RC1.

Contains >=20 unseen test cases covering:
1. Paraphrased scientific claims
2. Incorrect numerical values
3. Negative claims
4. Contradictory clauses
5. Entity substitution
6. Unsupported obscure claims
7. Irrelevant but topically similar evidence
8. Conflicting evidence handling
9. Multi-claim compound answers

Execution adheres strictly to the independent evaluation protocol:
- No hardcoded claim strings in production code
- Clear separation between neural NLI outputs and deterministic rules
- Expected verdicts are established independently of the engine implementation.
"""

import pytest
from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.core.engine.pillar1_retrieval import Pillar1RetrievalEngine
from app.core.engine.types import EvidenceItem, RiskLevel


@pytest.fixture(scope="module")
def pipeline():
    return HallucinationDetectionPipeline()


@pytest.fixture(scope="module")
def p1_engine():
    return Pillar1RetrievalEngine()


# ==============================================================================
# DIMENSION 1: Paraphrased Scientific Claims
# ==============================================================================

def test_holdout_01_paraphrased_speed_of_light(pipeline):
    """Paraphrased speed of light in vacuum must be verified."""
    claim = "In a vacuum, the propagation velocity of electromagnetic radiation is exactly 299,792,458 meters per second."
    ev = [
        EvidenceItem(
            claim=claim,
            snippet="The speed of light in vacuum, commonly denoted c, is a universal physical constant that is exactly 299,792,458 m/s.",
            source_name="Wikipedia: Speed of light",
            similarity_score=0.92,
            is_supporting=True,
        )
    ]
    res = pipeline.analyze(claim, provided_evidence=ev)
    assert res.overall_h_score <= 0.35, f"Expected low hallucination score, got {res.overall_h_score}"
    assert res.pillar1_summary.factual_error_score <= 0.35


def test_holdout_02_paraphrased_water_molar_mass(pipeline):
    """Paraphrased water molar mass must be verified."""
    claim = "One mole of pure water possesses a mass of approximately 18.015 grams."
    ev = [
        EvidenceItem(
            claim=claim,
            snippet="Water has a molar mass of approximately 18.01528 g/mol, which corresponds to 18.015 g per mole.",
            source_name="Wikipedia: Properties of water",
            similarity_score=0.90,
            is_supporting=True,
        )
    ]
    res = pipeline.analyze(claim, provided_evidence=ev)
    assert res.overall_h_score <= 0.35
    assert res.pillar1_summary.factual_error_score <= 0.35


# ==============================================================================
# DIMENSION 2: Incorrect Numerical Values
# ==============================================================================

def test_holdout_03_incorrect_speed_of_light(pipeline):
    """Incorrect speed of light must be contradicted."""
    claim = "Light travels through a vacuum at exactly 150,000 km/s."
    ev = [
        EvidenceItem(
            claim=claim,
            snippet="The speed of light in vacuum is defined to be exactly 299,792,458 m/s (approximately 300,000 km/s).",
            source_name="Wikipedia: Speed of light",
            similarity_score=0.92,
            is_supporting=False,
        )
    ]
    res = pipeline.analyze(claim, provided_evidence=ev)
    assert res.overall_h_score >= 0.70, f"Expected high hallucination score, got {res.overall_h_score}"
    assert res.overall_risk_level == RiskLevel.LIKELY_HALLUCINATED


def test_holdout_04_incorrect_water_molar_mass(pipeline):
    """Wildly incorrect water molar mass must be contradicted."""
    claim = "The molar mass of water is 45.2 grams per mole."
    ev = [
        EvidenceItem(
            claim=claim,
            snippet="The molar mass of water (H2O) is 18.015 g/mol.",
            source_name="Wikipedia: Properties of water",
            similarity_score=0.90,
            is_supporting=False,
        )
    ]
    res = pipeline.analyze(claim, provided_evidence=ev)
    assert res.overall_h_score >= 0.70
    assert res.overall_risk_level == RiskLevel.LIKELY_HALLUCINATED


def test_holdout_05_incorrect_water_freezing_point(pipeline):
    """Incorrect freezing point of water must be contradicted."""
    claim = "Water freezes at 25 °C at 1 atmosphere of pressure."
    ev = [
        EvidenceItem(
            claim=claim,
            snippet="At standard atmospheric pressure (1 atm), the freezing point of water is 0 °C (32 °F).",
            source_name="Wikipedia: Water",
            similarity_score=0.91,
            is_supporting=False,
        )
    ]
    res = pipeline.analyze(claim, provided_evidence=ev)
    assert res.overall_h_score >= 0.70
    assert res.overall_risk_level == RiskLevel.LIKELY_HALLUCINATED


# ==============================================================================
# DIMENSION 3: Negative Claims
# ==============================================================================

def test_holdout_06_negative_factually_false_mitochondria(pipeline):
    """False negative claim must be contradicted."""
    claim = "Mitochondria do not possess their own DNA."
    ev = [
        EvidenceItem(
            claim=claim,
            snippet="Mitochondria contain their own circular DNA, known as mtDNA, which encodes essential respiratory proteins.",
            source_name="Wikipedia: Mitochondrion",
            similarity_score=0.88,
            is_supporting=False,
        )
    ]
    res = pipeline.analyze(claim, provided_evidence=ev)
    assert res.overall_h_score >= 0.70, f"Expected contradiction on false negative claim, got {res.overall_h_score}"


def test_holdout_07_negative_factually_true_paris_germany(pipeline):
    """True negative claim must not be falsely inverted into contradiction."""
    claim = "Paris is not the capital of Germany."
    ev = [
        EvidenceItem(
            claim=claim,
            snippet="Berlin is the capital and largest city of Germany by both area and population.",
            source_name="Wikipedia: Germany",
            similarity_score=0.85,
            is_supporting=True,
        )
    ]
    res = pipeline.analyze(claim, provided_evidence=ev)
    # Must not be flagged as a high-confidence hallucination
    assert res.overall_risk_level != RiskLevel.LIKELY_HALLUCINATED
    assert res.overall_h_score < 0.60


def test_holdout_08_negative_factually_true_mars_oceans(pipeline):
    """True negative claim about Mars liquid surface oceans."""
    claim = "Mars does not possess oceans of liquid surface water today."
    ev = [
        EvidenceItem(
            claim=claim,
            snippet="Liquid water cannot exist on the surface of Mars due to low atmospheric pressure, which is less than 1% of Earth's.",
            source_name="Wikipedia: Water on Mars",
            similarity_score=0.89,
            is_supporting=True,
        )
    ]
    res = pipeline.analyze(claim, provided_evidence=ev)
    assert res.overall_risk_level != RiskLevel.LIKELY_HALLUCINATED


# ==============================================================================
# DIMENSION 4: Contradictory Clauses / Internal Semantic Conflict
# ==============================================================================

def test_holdout_09_contradictory_freezing_boiling_claim(pipeline):
    """Compound claim asserting impossible equality must be contradicted."""
    claim = "Water freezes at 0 °C and boils at 0 °C at standard atmospheric pressure."
    ev = [
        EvidenceItem(
            claim=claim,
            snippet="At standard atmospheric pressure, water freezes at 0 °C and boils at 100 °C.",
            source_name="Wikipedia: Water",
            similarity_score=0.92,
            is_supporting=False,
        )
    ]
    res = pipeline.analyze(claim, provided_evidence=ev)
    assert res.overall_h_score >= 0.70
    assert res.overall_risk_level == RiskLevel.LIKELY_HALLUCINATED


def test_holdout_10_contradictory_apollo_claim(pipeline):
    """Clause asserting Apollo landed while simultaneously denying human lunar landing."""
    claim = "Apollo 11 landed on the Moon, but humans have never landed on the lunar surface."
    ev = [
        EvidenceItem(
            claim=claim,
            snippet="Apollo 11 was the American spaceflight that first landed humans on the Moon on July 20, 1969.",
            source_name="Wikipedia: Apollo 11",
            similarity_score=0.90,
            is_supporting=False,
        )
    ]
    res = pipeline.analyze(claim, provided_evidence=ev)
    assert res.overall_h_score >= 0.45


# ==============================================================================
# DIMENSION 5: Entity Substitution
# ==============================================================================

def test_holdout_11_entity_sub_hubli_capital(pipeline):
    """Hubli asserted as capital of Karnataka must be contradicted."""
    claim = "Hubli is the capital city of Karnataka."
    ev = [
        EvidenceItem(
            claim=claim,
            snippet="Bengaluru, also known as Bangalore, is the capital and largest city of the Indian state of Karnataka.",
            source_name="Wikipedia: Bengaluru",
            similarity_score=0.91,
            is_supporting=False,
        )
    ]
    res = pipeline.analyze(claim, provided_evidence=ev)
    assert res.overall_h_score >= 0.70
    assert res.overall_risk_level == RiskLevel.LIKELY_HALLUCINATED


def test_holdout_12_entity_sub_madrid_germany(pipeline):
    """Madrid asserted as capital of Germany must be contradicted."""
    claim = "Madrid is the official capital of Germany."
    ev = [
        EvidenceItem(
            claim=claim,
            snippet="Berlin is the capital and largest city of Germany by both area and population.",
            source_name="Wikipedia: Berlin",
            similarity_score=0.90,
            is_supporting=False,
        )
    ]
    res = pipeline.analyze(claim, provided_evidence=ev)
    assert res.overall_h_score >= 0.70
    assert res.overall_risk_level == RiskLevel.LIKELY_HALLUCINATED


def test_holdout_13_entity_sub_boeing_737_four_engines(pipeline):
    """Boeing 737 asserted to have four engines must be contradicted."""
    claim = "The Boeing 737 is a four-engine wide-body passenger aircraft."
    ev = [
        EvidenceItem(
            claim=claim,
            snippet="The Boeing 737 is a narrow-body aircraft produced by Boeing Commercial Airplanes, powered by two turbofan engines.",
            source_name="Wikipedia: Boeing 737",
            similarity_score=0.92,
            is_supporting=False,
        )
    ]
    res = pipeline.analyze(claim, provided_evidence=ev)
    assert res.overall_h_score >= 0.70
    assert res.overall_risk_level == RiskLevel.LIKELY_HALLUCINATED


# ==============================================================================
# DIMENSION 6: Unsupported Obscure Claims (Abstention Testing)
# ==============================================================================

def test_holdout_14_unsupported_obscure_emperor(pipeline):
    """Completely fabricated historical figure with no evidence must return INSUFFICIENT_EVIDENCE."""
    claim = "The ancient emperor Zandor ruled the kingdom of Valoria from 412 BC to 380 BC."
    res = pipeline.analyze(claim, provided_evidence=[])
    assert res.overall_risk_level in (RiskLevel.INSUFFICIENT_EVIDENCE, RiskLevel.NEEDS_VERIFICATION)
    assert 0.35 <= res.overall_h_score <= 0.65


def test_holdout_15_unsupported_obscure_martian_city(pipeline):
    """Sci-fi claim with no evidence must abstain."""
    claim = "Deep beneath the Martian crust lies an underground city built by the Arkonians."
    res = pipeline.analyze(claim, provided_evidence=[])
    assert res.overall_risk_level in (RiskLevel.INSUFFICIENT_EVIDENCE, RiskLevel.NEEDS_VERIFICATION)
    assert 0.35 <= res.overall_h_score <= 0.65


# ==============================================================================
# DIMENSION 7: Irrelevant but Topically Similar Evidence Filtering
# ==============================================================================

def test_holdout_16_irrelevant_traffic_evidence_rejected(p1_engine):
    """Traffic and election articles must be filtered out for capital claim."""
    claim = "Bengaluru is the capital of Karnataka."
    ev = [
        EvidenceItem(
            claim=claim,
            snippet="The 2026 Greater Bengaluru Authority elections are the first elections to the five City corporation.",
            source_name="Wikipedia: Greater Bengaluru Elections",
            similarity_score=0.45,
            is_supporting=True,
        ),
        EvidenceItem(
            claim=claim,
            snippet="Bengaluru, also known as Bangalore, is the capital and largest city of the south Indian state of Karnataka.",
            source_name="Wikipedia: Bengaluru",
            similarity_score=0.95,
            is_supporting=True,
        ),
    ]
    fe, processed = p1_engine.evaluate_claims_against_evidence([claim], ev)
    assert fe <= 0.35, f"Expected verified factual error, got {fe}"
    # The election snippet must be deemed irrelevant or not override the genuine ground truth
    assert any(it.relevance_score is not None and it.relevance_score >= 0.80 for it in processed)


def test_holdout_17_irrelevant_sound_evidence_for_light(p1_engine):
    """Speed of sound evidence must be filtered as irrelevant for speed of light claim."""
    claim = "The speed of light in vacuum is 299,792,458 m/s."
    ev = [
        EvidenceItem(
            claim=claim,
            snippet="The speed of sound in dry air at 20 °C is 343 meters per second.",
            source_name="Wikipedia: Speed of sound",
            similarity_score=0.60,
            is_supporting=False,
        ),
        EvidenceItem(
            claim=claim,
            snippet="The speed of light in vacuum, commonly denoted c, is a universal physical constant exactly equal to 299,792,458 m/s.",
            source_name="Wikipedia: Speed of light",
            similarity_score=0.95,
            is_supporting=True,
        ),
    ]
    fe, processed = p1_engine.evaluate_claims_against_evidence([claim], ev)
    assert fe <= 0.35, f"Sound snippet caused false contradiction: {fe}"


# ==============================================================================
# DIMENSION 8: Conflicting Evidence Handling
# ==============================================================================

def test_holdout_18_conflicting_evidence_uncertainty(p1_engine):
    """Genuinely conflicting relevant evidence must produce an uncertainty state."""
    claim = "The ancient settlement was founded in 1200 BC."
    ev = [
        EvidenceItem(
            claim=claim,
            snippet="Archaeological excavations confirmed the ancient settlement was founded in 1200 BC during the late Bronze Age.",
            source_name="Source A",
            similarity_score=0.90,
            is_supporting=True,
        ),
        EvidenceItem(
            claim=claim,
            snippet="Recent radiocarbon dating conclusively demonstrated that the ancient settlement was founded in 850 BC, refuting earlier 1200 BC hypotheses.",
            source_name="Source B",
            similarity_score=0.90,
            is_supporting=False,
        ),
    ]
    fe, processed = p1_engine.evaluate_claims_against_evidence([claim], ev)
    # Conflicting evidence must not be a dogmatic 0.0 or 1.0, but an uncertainty state (around 0.50)
    assert 0.40 <= fe <= 0.60, f"Expected uncertainty score for conflicting evidence, got {fe}"


# ==============================================================================
# DIMENSION 9: Multi-Claim Compound Answers
# ==============================================================================

def test_holdout_19_multiclaim_all_true(pipeline):
    """Compound response with two verified scientific assertions."""
    text = "The speed of light in vacuum is 299,792,458 m/s. Water freezes at 0 °C at standard atmospheric pressure."
    ev = [
        EvidenceItem(
            claim="The speed of light in vacuum is 299,792,458 m/s.",
            snippet="The speed of light in vacuum is a universal physical constant that is exactly 299,792,458 m/s.",
            source_name="Wikipedia: Speed of light",
            similarity_score=0.95,
            is_supporting=True,
        ),
        EvidenceItem(
            claim="Water freezes at 0 °C at standard atmospheric pressure.",
            snippet="At standard atmospheric pressure, the freezing point of water is 0 °C.",
            source_name="Wikipedia: Water",
            similarity_score=0.95,
            is_supporting=True,
        ),
    ]
    res = pipeline.analyze(text, provided_evidence=ev)
    assert res.overall_h_score <= 0.35
    assert res.overall_risk_level in (RiskLevel.VERIFIED, RiskLevel.LOW_RISK)


def test_holdout_20_multiclaim_mixed_true_and_false(pipeline):
    """Compound response with one true assertion and one false assertion."""
    text = "Bengaluru is the capital of Karnataka. Hubli is the capital of Karnataka."
    ev = [
        EvidenceItem(
            claim="Bengaluru is the capital of Karnataka.",
            snippet="Bengaluru, also known as Bangalore, is the capital and largest city of the Indian state of Karnataka.",
            source_name="Wikipedia: Bengaluru",
            similarity_score=0.95,
            is_supporting=True,
        ),
        EvidenceItem(
            claim="Hubli is the capital of Karnataka.",
            snippet="Bengaluru is the capital of the state of Karnataka, while Hubli is a major commercial city in northern Karnataka.",
            source_name="Wikipedia: Karnataka",
            similarity_score=0.90,
            is_supporting=False,
        ),
    ]
    res = pipeline.analyze(text, provided_evidence=ev)
    # The presence of a blatant false statement must flag the response
    assert res.overall_h_score >= 0.45
    assert len(res.sentence_analyses) == 2
    # Second sentence must be flagged as likely hallucinated or high score
    assert res.sentence_analyses[1].risk_level == RiskLevel.LIKELY_HALLUCINATED or res.sentence_analyses[1].hallucination_score >= 0.50


def test_holdout_21_multiclaim_capital_isolation(pipeline):
    """Bengaluru/Karnataka and New Delhi/India must not cross-contaminate."""
    text = "Bengaluru is the capital of Karnataka. New Delhi is the capital of India."
    ev = [
        EvidenceItem(
            claim="Bengaluru is the capital of Karnataka.",
            snippet="Bengaluru, also known as Bangalore, is the capital and largest city of the south Indian state of Karnataka.",
            source_name="Wikipedia: Bengaluru",
            similarity_score=0.95,
            is_supporting=True,
        ),
        EvidenceItem(
            claim="New Delhi is the capital of India.",
            snippet="New Delhi is the capital of India and part of the National Capital Territory of Delhi.",
            source_name="Wikipedia: New Delhi",
            similarity_score=0.95,
            is_supporting=True,
        ),
    ]
    res = pipeline.analyze(text, provided_evidence=ev)
    assert res.overall_h_score <= 0.35
    for s in res.sentence_analyses:
        assert s.hallucination_score <= 0.35, f"Sentence '{s.text}' falsely flagged with score {s.hallucination_score}"


def test_holdout_22_provenance_transparency(p1_engine):
    """Every scored evidence item must transparently expose its evaluation source."""
    claim = "The speed of light in vacuum is 299,792,458 m/s."
    ev = [
        EvidenceItem(
            claim=claim,
            snippet="The speed of light in vacuum is exactly 299,792,458 m/s.",
            source_name="Wikipedia: Speed of light",
            similarity_score=0.95,
            is_supporting=True,
        )
    ]
    fe, processed = p1_engine.evaluate_claims_against_evidence([claim], ev)
    assert len(processed) > 0
    for item in processed:
        assert hasattr(item, "evaluation_source")
        assert item.evaluation_source in (
            "neural_deberta_nli",
            "deterministic_scientific_constant_rule",
            "deterministic_capital_relation_rule",
            "deterministic_chemical_formula_rule",
            "deterministic_quantity_spec_rule",
        )
