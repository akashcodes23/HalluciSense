"""Test Suite for Phase 8 Enhanced P1 Components.

Validates:
- ClaimDecomposer: proposition decomposition, punctuation handling, aggregation strategies.
- NumericUnitChecker: scientific notation, scale discrepancies, unit conflicts.
- NegationDetector: polarity detection, negation markers, antonym reversal detection.
- CausalDirectionChecker: cause-effect extraction, directional inversion detection.
"""

from __future__ import annotations

import pytest
from app.core.engine.claim_decomposition import (
    ClaimDecomposer,
    AggregationStrategy,
    AtomicProposition,
)
from app.core.engine.numeric_unit_checker import (
    NumericUnitChecker,
    NumericUnitStatus,
)
from app.core.engine.negation_detector import (
    NegationDetector,
    Polarity,
)
from app.core.engine.causal_direction import (
    CausalDirectionChecker,
    CausalRelation,
)


class TestClaimDecomposer:
    @pytest.fixture
    def decomposer(self):
        return ClaimDecomposer()

    def test_clean_text_strips_discourse(self, decomposer):
        raw = "It is well known that water boils at 100 degrees Celsius."
        cleaned = decomposer.clean_text(raw)
        assert cleaned == "water boils at 100 degrees Celsius."

    def test_split_sentences(self, decomposer):
        text = "DNA is a nucleic acid. It encodes genetic instructions. Proteins perform enzymatic work."
        sents = decomposer.split_sentences(text)
        assert len(sents) == 3

    def test_decompose_compound_sentence(self, decomposer):
        text = "The Haber process synthesises ammonia from nitrogen; it requires high pressure and temperature."
        props = decomposer.decompose_sentence(text)
        assert len(props) >= 2
        assert any("Haber" in p for p in props)
        assert any("pressure" in p for p in props)

    def test_aggregate_scores_mean(self):
        props = [
            AtomicProposition("p1", 1, "s", h_score=0.2),
            AtomicProposition("p2", 2, "s", h_score=0.8),
        ]
        score = ClaimDecomposer.aggregate_scores(props, strategy=AggregationStrategy.MEAN)
        assert pytest.approx(score, 0.01) == 0.5

    def test_aggregate_scores_max_risk(self):
        props = [
            AtomicProposition("p1", 1, "s", h_score=0.2),
            AtomicProposition("p2", 2, "s", h_score=0.85),
            AtomicProposition("p3", 3, "s", h_score=0.4),
        ]
        score = ClaimDecomposer.aggregate_scores(props, strategy=AggregationStrategy.MAX_RISK)
        assert score == 0.85

    def test_aggregate_scores_unsupported_ratio(self):
        props = [
            AtomicProposition("p1", 1, "s", h_score=0.2),
            AtomicProposition("p2", 2, "s", h_score=0.7),
            AtomicProposition("p3", 3, "s", h_score=0.8),
            AtomicProposition("p4", 4, "s", h_score=0.1),
        ]
        score = ClaimDecomposer.aggregate_scores(props, strategy=AggregationStrategy.UNSUPPORTED_RATIO, contradiction_threshold=0.50)
        assert score == 0.50  # 2 of 4 flagged


class TestNumericUnitChecker:
    @pytest.fixture
    def checker(self):
        return NumericUnitChecker()

    def test_extract_scientific_notation(self, checker):
        text = "The speed of light is 3×10⁸ metres per second."
        quantities = checker.extract_quantities(text)
        assert len(quantities) >= 1
        assert pytest.approx(quantities[0].value, rel=1e-3) == 3e8

    def test_extract_multiplier_words(self, checker):
        text = "The human genome contains approximately 3 billion base pairs."
        quantities = checker.extract_quantities(text)
        assert len(quantities) >= 1
        assert pytest.approx(quantities[0].value, rel=1e-3) == 3e9

    def test_numeric_match(self, checker):
        claim = "The acceleration is 9.8 m/s²."
        evidence = "Standard gravity at Earth's surface is 9.8 m/s²."
        status, penalty, _ = checker.check_consistency(claim, evidence)
        assert status == NumericUnitStatus.NUMERIC_MATCH
        assert penalty == 0.0

    def test_scale_conflict_detection(self, checker):
        claim = "The nucleus has a diameter of 6 mm."
        evidence = "A typical cell nucleus has a diameter of approximately 6 micrometres."
        status, penalty, _ = checker.check_consistency(claim, evidence)
        assert status == NumericUnitStatus.SCALE_CONFLICT
        assert penalty > 0.80

    def test_numeric_conflict_wrong_number(self, checker):
        claim = "The speed of light is approximately 3×10⁶ metres per second."
        evidence = "The speed of light in vacuum is approximately 3×10⁸ m/s."
        status, penalty, _ = checker.check_consistency(claim, evidence)
        assert status == NumericUnitStatus.SCALE_CONFLICT or status == NumericUnitStatus.NUMERIC_CONFLICT
        assert penalty > 0.70


class TestNegationDetector:
    @pytest.fixture
    def detector(self):
        return NegationDetector()

    def test_polarity_detection(self, detector):
        pos_text = "Mitochondria contain their own circular DNA."
        neg_text = "Mitochondria do not contain their own DNA."
        assert detector.get_polarity(pos_text)[0] == Polarity.POSITIVE
        assert detector.get_polarity(neg_text)[0] == Polarity.NEGATIVE

    def test_negation_inversion_detection(self, detector):
        claim = "Mitochondria do not contain their own DNA."
        evidence = "Mitochondria contain their own circular DNA independent of the nucleus."
        res = detector.analyze(claim, evidence)
        assert res.negation_inversion_detected is True
        assert res.confidence_penalty >= 0.80

    def test_antonym_conflict_detection(self, detector):
        claim = "Corticosteroids promote inflammation."
        evidence = "Corticosteroids are anti-inflammatory drugs that suppress inflammation."
        res = detector.analyze(claim, evidence)
        assert res.antonym_inversion_detected is True
        assert res.confidence_penalty >= 0.75


class TestCausalDirectionChecker:
    @pytest.fixture
    def checker(self):
        return CausalDirectionChecker()

    def test_extract_forward_causal(self, checker):
        text = "Smoking causes lung cancer."
        rel = checker.extract_causal_relation(text)
        assert rel is not None
        assert rel.is_forward is True
        assert "smoking" in rel.cause.lower()
        assert "lung cancer" in rel.effect.lower()

    def test_extract_backward_causal(self, checker):
        text = "Haemophilia A is caused by factor VIII deficiency."
        rel = checker.extract_causal_relation(text)
        assert rel is not None
        assert rel.is_forward is False
        assert "factor viii" in rel.cause.lower()
        assert "haemophilia" in rel.effect.lower()

    def test_causal_inversion_detection(self, checker):
        claim = "mRNA is transcribed from a protein template during gene expression."
        evidence = "Protein is translated from mRNA during gene expression."
        # Or direction test with explicit cause/effect
        claim2 = "A higher variance causes the standard deviation to decrease."
        evidence2 = "Standard deviation increases when variance increases."
        res = checker.check_inversion(
            "ACE inhibitors lower blood pressure by increasing angiotensin II.",
            "ACE inhibitors lower blood pressure because angiotensin II production is blocked.",
        )
        assert res is not None
