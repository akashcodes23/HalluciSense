"""Phase 16 — Evidence Reliability, NLI Aggregation, Precision Policy, and Multi-Claim Regression Suite.

Verifies:
1. Phase 16B & 16D NLI Evidence Aggregation Conditions:
   A. One relevant supporting passage and one irrelevant contradictory passage -> VERIFIED.
   B. One relevant contradictory passage and one irrelevant supporting passage -> CONTRADICTED.
   C. Multiple relevant supporting passages -> VERIFIED.
   D. Multiple relevant contradictory passages -> CONTRADICTED.
   E. Relevant but neutral evidence -> INSUFFICIENT_EVIDENCE (never contradiction).
   F. No relevant evidence -> INSUFFICIENT_EVIDENCE (abstention).
   G. Conflicting relevant evidence -> Uncertainty state.
2. Phase 16E Numerical Precision Policy:
   - Exact unit conversion
   - Defined physical constant (c = 299,792,458 m/s vs 150,000 km/s)
   - Rounded representation with approximation markers
   - Measurement uncertainty (molar mass 18.015 g/mol)
   - Absolute-scale temperature comparison (0°C vs 1.4°C rejected)
   - Discrete count (4 engines vs 2 engines rejected)
3. Phase 16F Multi-Claim Verdict Aggregation:
   - Three supported claims
   - Two supported and one contradicted
   - Two supported and one insufficient
   - One contradicted and one insufficient
   - All insufficient
   - Empty claim list
   - Conversational framing with factual claims
   - Repeated claims
   - Duplicate claims with conflicting values
4. Phase 16G Provider and Pillar Status:
   - P2 unavailable for pasted text without logprobs
   - P3 unavailable for single generation
   - No fabricated probabilities
"""

import pytest
from app.core.engine.types import EvidenceItem, RiskLevel
from app.core.engine.pillar1_retrieval import Pillar1RetrievalEngine
from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.core.engine.scientific_claim_reasoner import ScientificClaimReasoner, ClaimSRO
from app.core.verification.precision_policy import (
    NumericalPrecisionPolicy,
    PrecisionDomain,
    has_approximation_framing,
)


# =====================================================================
# Phase 16B & 16D: NLI Evidence Aggregation Conditions A-G
# =====================================================================

class TestNLIEvidenceAggregation:
    @pytest.fixture
    def engine(self):
        return Pillar1RetrievalEngine()

    def test_condition_a_relevant_support_irrelevant_contradiction(self, engine):
        """Condition A: One relevant supporting passage and one irrelevant contradictory passage."""
        claim = "The speed of light in vacuum is 299,792,458 m/s."
        evidence = [
            EvidenceItem(
                claim="",
                snippet="The speed of light in vacuum is a universal physical constant exactly equal to 299,792,458 m/s.",
                source_name="Wikipedia: Speed of light",
                source_url="https://en.wikipedia.org/wiki/Speed_of_light",
                similarity_score=0.95,
                is_supporting=True,
            ),
            EvidenceItem(
                claim="",
                snippet="In 1793 during the French Revolution, the National Convention decreed the provisional metre.",
                source_name="Wikipedia: History of the metre",
                source_url="https://en.wikipedia.org/wiki/History_of_the_metre",
                similarity_score=0.60,
                is_supporting=True,
            ),
        ]
        fe_score, processed = engine.evaluate_claims_against_evidence([claim], evidence)
        statuses = getattr(engine, "last_claim_statuses", [])

        assert statuses == ["VERIFIED"], f"Expected VERIFIED but got {statuses}"
        assert fe_score <= 0.35, f"Expected fe_score <= 0.35 but got {fe_score}"
        # Provenance verification: History of the metre must be rejected as irrelevant
        hist_item = [e for e in processed if "History of the metre" in e.source_name][0]
        assert hist_item.relevance_score is not None and hist_item.relevance_score < 0.40
        assert hist_item.contradiction_prob == 0.0

    def test_condition_b_relevant_contradiction_irrelevant_support(self, engine):
        """Condition B: One relevant contradictory passage and one irrelevant supporting passage."""
        claim = "The speed of light is 150,000 km/s."
        evidence = [
            EvidenceItem(
                claim="",
                snippet="The speed of light in vacuum is a universal physical constant exactly equal to 299,792,458 m/s (approximately 300,000 km/s).",
                source_name="Wikipedia: Speed of light",
                source_url="https://en.wikipedia.org/wiki/Speed_of_light",
                similarity_score=0.90,
                is_supporting=True,
            ),
            EvidenceItem(
                claim="",
                snippet="The movie Speed was released in 1994 starring Keanu Reeves.",
                source_name="Wikipedia: Speed (1994 film)",
                source_url="https://en.wikipedia.org/wiki/Speed_(1994_film)",
                similarity_score=0.70,
                is_supporting=True,
            ),
        ]
        fe_score, processed = engine.evaluate_claims_against_evidence([claim], evidence)
        statuses = getattr(engine, "last_claim_statuses", [])

        assert statuses == ["CONTRADICTED"], f"Expected CONTRADICTED but got {statuses}"
        assert fe_score >= 0.70, f"Expected fe_score >= 0.70 but got {fe_score}"
        film_item = [e for e in processed if "film" in e.source_name][0]
        assert film_item.relevance_score is not None and film_item.relevance_score < 0.40

    def test_condition_c_multiple_relevant_supporting_passages(self, engine):
        """Condition C: Multiple relevant supporting passages."""
        claim = "Water has the chemical formula H2O."
        evidence = [
            EvidenceItem(
                claim="",
                snippet="Water is an inorganic compound with the chemical formula H2O.",
                source_name="Wikipedia: Water",
                source_url="https://en.wikipedia.org/wiki/Water",
                similarity_score=0.95,
                is_supporting=True,
            ),
            EvidenceItem(
                claim="",
                snippet="Water molecules have the chemical formula H2O with two hydrogen atoms bonded to one oxygen atom.",
                source_name="Wikipedia: Properties of water",
                source_url="https://en.wikipedia.org/wiki/Properties_of_water",
                similarity_score=0.92,
                is_supporting=True,
            ),
        ]
        fe_score, _ = engine.evaluate_claims_against_evidence([claim], evidence)
        statuses = getattr(engine, "last_claim_statuses", [])

        assert statuses == ["VERIFIED"]
        assert fe_score <= 0.35

    def test_condition_d_multiple_relevant_contradictory_passages(self, engine):
        """Condition D: Multiple relevant contradictory passages."""
        claim = "The Boeing 737 has four engines."
        evidence = [
            EvidenceItem(
                claim="",
                snippet="The Boeing 737 is a twin-engine narrow-body airliner produced by Boeing Commercial Airplanes.",
                source_name="Wikipedia: Boeing 737",
                source_url="https://en.wikipedia.org/wiki/Boeing_737",
                similarity_score=0.90,
                is_supporting=True,
            ),
            EvidenceItem(
                claim="",
                snippet="Powered by two turbofan engines mounted under the wings, the Boeing 737 has twin powerplants.",
                source_name="Wikipedia: Boeing 737 Classic",
                source_url="https://en.wikipedia.org/wiki/Boeing_737_Classic",
                similarity_score=0.88,
                is_supporting=True,
            ),
        ]
        fe_score, _ = engine.evaluate_claims_against_evidence([claim], evidence)
        statuses = getattr(engine, "last_claim_statuses", [])

        assert statuses == ["CONTRADICTED"]
        assert fe_score >= 0.70

    def test_condition_e_relevant_neutral_evidence_not_contradiction(self, engine):
        """Condition E: Relevant but neutral evidence must not automatically become contradiction."""
        claim = "Bengaluru has a pleasant climate throughout the year."
        evidence = [
            EvidenceItem(
                claim="",
                snippet="Bengaluru is the capital and largest city of Karnataka, located on the Deccan Plateau.",
                source_name="Wikipedia: Bangalore",
                source_url="https://en.wikipedia.org/wiki/Bangalore",
                similarity_score=0.75,
                is_supporting=True,
            ),
        ]
        fe_score, _ = engine.evaluate_claims_against_evidence([claim], evidence)
        statuses = getattr(engine, "last_claim_statuses", [])

        assert statuses == ["INSUFFICIENT_EVIDENCE"], f"Expected INSUFFICIENT_EVIDENCE but got {statuses}"
        assert fe_score == 0.50

    def test_condition_f_no_relevant_evidence_abstention(self, engine):
        """Condition F: No relevant evidence produces explicit abstention."""
        claim = "The quantum spin of hypothetical gravitons is 2."
        evidence = [
            EvidenceItem(
                claim="",
                snippet="The French Revolution began in 1789 and ended in the late 1790s.",
                source_name="Wikipedia: French Revolution",
                source_url="https://en.wikipedia.org/wiki/French_Revolution",
                similarity_score=0.20,
                is_supporting=True,
            ),
        ]
        fe_score, _ = engine.evaluate_claims_against_evidence([claim], evidence)
        statuses = getattr(engine, "last_claim_statuses", [])

        assert statuses == ["INSUFFICIENT_EVIDENCE"]
        assert fe_score == 0.50

    def test_conflicting_relevant_evidence_uncertainty_state(self, engine):
        """Conflicting relevant evidence produces an uncertainty state."""
        claim = "The ancient city was inhabited continuously for two thousand years."
        evidence = [
            EvidenceItem(
                claim="",
                snippet="Archaeological excavations prove the ancient city was inhabited continuously for two thousand years without interruption.",
                source_name="Archaeology Journal A",
                similarity_score=0.90,
                is_supporting=True,
            ),
            EvidenceItem(
                claim="",
                snippet="Stratigraphic radiocarbon dates demonstrate the ancient city was completely abandoned for several centuries.",
                source_name="Archaeology Journal B",
                similarity_score=0.88,
                is_supporting=True,
            ),
        ]
        fe_score, _ = engine.evaluate_claims_against_evidence([claim], evidence)
        statuses = getattr(engine, "last_claim_statuses", [])

        assert statuses == ["INSUFFICIENT_EVIDENCE"]
        assert fe_score == 0.50


# =====================================================================
# Phase 16E: Numerical Precision Policy Tests
# =====================================================================

class TestNumericalPrecisionPolicy:
    def test_exact_unit_conversion_strictness(self):
        """100 m and 101.4 m are NOT equivalent under unit conversion."""
        is_eq, domain, reason = NumericalPrecisionPolicy.evaluate_equivalence(
            100.0, "m", 101.4, "m", "length"
        )
        assert not is_eq
        assert domain == PrecisionDomain.EXACT_UNIT_CONVERSION

    def test_exact_unit_conversion_valid(self):
        """1 km equals 1000 m."""
        is_eq, domain, _ = NumericalPrecisionPolicy.evaluate_equivalence(
            1.0, "km", 1000.0, "m", "length"
        )
        assert is_eq
        assert domain == PrecisionDomain.EXACT_UNIT_CONVERSION

    def test_defined_physical_constant_exact(self):
        """c = 299,792,458 m/s exact match required."""
        is_eq, domain, _ = NumericalPrecisionPolicy.evaluate_equivalence(
            299792458.0, "m/s", 299792458.0, "m/s", "speed"
        )
        assert is_eq
        assert domain == PrecisionDomain.DEFINED_PHYSICAL_CONSTANT

    def test_defined_physical_constant_rejection(self):
        """150,000 km/s is rejected against speed of light."""
        is_eq, _, _ = NumericalPrecisionPolicy.evaluate_equivalence(
            150000.0, "km/s", 299792.458, "km/s", "speed"
        )
        assert not is_eq

    def test_rounded_representation_with_framing(self):
        """3.0e8 m/s allowed ONLY with approximation framing."""
        is_eq_approx, domain, _ = NumericalPrecisionPolicy.evaluate_equivalence(
            3.0e8, "m/s", 299792458.0, "m/s", "speed", text_context="The speed is approximately 3.0e8 m/s"
        )
        assert is_eq_approx
        assert domain == PrecisionDomain.ROUNDED_REPRESENTATION

        is_eq_exact, domain_exact, _ = NumericalPrecisionPolicy.evaluate_equivalence(
            3.0e8, "m/s", 299792458.0, "m/s", "speed", text_context="The speed is exactly 3.0e8 m/s"
        )
        assert not is_eq_exact
        assert domain_exact == PrecisionDomain.DEFINED_PHYSICAL_CONSTANT

    def test_measurement_uncertainty_molar_mass(self):
        """Molar mass of water 18.015 g/mol matches 18.01528 within empirical uncertainty."""
        is_eq, domain, _ = NumericalPrecisionPolicy.evaluate_equivalence(
            18.015, "g/mol", 18.01528, "g/mol", "molar_mass"
        )
        assert is_eq
        assert domain == PrecisionDomain.MEASUREMENT_UNCERTAINTY

    def test_absolute_temperature_evaluation(self):
        """0°C vs 1.4°C is NOT equivalent (evaluated on absolute degree scale, not Kelvin %)."""
        is_eq, domain, reason = NumericalPrecisionPolicy.evaluate_equivalence(
            0.0, "°c", 1.4, "°c", "temperature"
        )
        assert not is_eq
        assert domain == PrecisionDomain.ABSOLUTE_TEMPERATURE
        assert "differs by 1.40°C" in reason

    def test_discrete_count_exactness(self):
        """Discrete count 4 engines vs 2 engines rejected."""
        is_eq, domain, _ = NumericalPrecisionPolicy.evaluate_equivalence(
            4.0, "engines", 2.0, "engines", "count"
        )
        assert not is_eq
        assert domain == PrecisionDomain.DISCRETE_COUNT


# =====================================================================
# Phase 16F: Multi-Claim Verdict Aggregation
# =====================================================================

class TestMultiClaimAggregation:
    @pytest.fixture
    def pipeline(self):
        return HallucinationDetectionPipeline()

    def test_three_supported_claims(self, pipeline):
        """1. Three supported claims -> VERIFIED."""
        evidence = [
            EvidenceItem(claim="", snippet="Water freezes at 0°C.", source_name="P1", similarity_score=0.95, is_supporting=True),
            EvidenceItem(claim="", snippet="Water boils at 100°C under standard pressure.", source_name="P2", similarity_score=0.95, is_supporting=True),
            EvidenceItem(claim="", snippet="Water has the chemical formula H2O.", source_name="P3", similarity_score=0.95, is_supporting=True),
        ]
        text = "Water freezes at 0°C. Water boils at 100°C. Water has the chemical formula H2O."
        res = pipeline.analyze_response(text, provided_evidence=evidence)

        assert res.overall_risk_level == RiskLevel.VERIFIED
        assert res.overall_h_score <= 0.35

    def test_two_supported_one_contradicted(self, pipeline):
        """2. Two supported and one contradicted -> LIKELY_HALLUCINATED."""
        evidence = [
            EvidenceItem(claim="", snippet="Water freezes at 0°C.", source_name="P1", similarity_score=0.95, is_supporting=True),
            EvidenceItem(claim="", snippet="Water boils at 100°C under standard atmospheric pressure.", source_name="P2", similarity_score=0.95, is_supporting=True),
            EvidenceItem(claim="", snippet="The Boeing 737 is a twin-engine aircraft.", source_name="P3", similarity_score=0.95, is_supporting=True),
        ]
        text = "Water freezes at 0°C. Water boils at 100°C. The Boeing 737 has four engines."
        res = pipeline.analyze_response(text, provided_evidence=evidence)

        assert res.overall_risk_level == RiskLevel.LIKELY_HALLUCINATED
        # Claim-level results must remain independently visible
        assert len(res.sentence_analyses) == 3
        assert res.sentence_analyses[0].risk_level == RiskLevel.VERIFIED
        assert res.sentence_analyses[1].risk_level == RiskLevel.VERIFIED
        assert res.sentence_analyses[2].risk_level == RiskLevel.LIKELY_HALLUCINATED

    def test_two_supported_one_insufficient(self, pipeline):
        """3. Two supported and one insufficient -> ungrounded claim flagged at sentence level."""
        evidence = [
            EvidenceItem(claim="", snippet="Water freezes at 0°C.", source_name="P1", similarity_score=0.95, is_supporting=True),
            EvidenceItem(claim="", snippet="Water boils at 100°C under standard pressure.", source_name="P2", similarity_score=0.95, is_supporting=True),
        ]
        text = "Water freezes at 0°C. Water boils at 100°C. The ancient tablet had seventy runes."
        res = pipeline.analyze_response(text, provided_evidence=evidence)

        # Overall risk should reflect the presence of ungrounded claim without falsely marking contradiction
        assert res.overall_risk_level in (RiskLevel.NEEDS_VERIFICATION, RiskLevel.INSUFFICIENT_EVIDENCE)
        assert res.sentence_analyses[0].risk_level == RiskLevel.VERIFIED
        assert res.sentence_analyses[1].risk_level == RiskLevel.VERIFIED
        assert res.sentence_analyses[2].risk_level in (RiskLevel.INSUFFICIENT_EVIDENCE, RiskLevel.NEEDS_VERIFICATION)

    def test_one_contradicted_one_insufficient(self, pipeline):
        """4. One contradicted and one insufficient -> LIKELY_HALLUCINATED."""
        evidence = [
            EvidenceItem(claim="", snippet="The Boeing 737 is a twin-engine airliner.", source_name="P1", similarity_score=0.95, is_supporting=True),
        ]
        text = "The Boeing 737 has four engines. The ancient tablet had seventy runes."
        res = pipeline.analyze_response(text, provided_evidence=evidence)

        assert res.overall_risk_level == RiskLevel.LIKELY_HALLUCINATED
        assert res.sentence_analyses[0].risk_level == RiskLevel.LIKELY_HALLUCINATED

    def test_all_insufficient(self, pipeline):
        """5. All claims insufficient -> INSUFFICIENT_EVIDENCE."""
        evidence = [
            EvidenceItem(claim="", snippet="Napoleon was born in Corsica.", source_name="P1", similarity_score=0.20, is_supporting=True),
        ]
        text = "The ancient tablet had seventy runes. The second stone was found in a cave."
        res = pipeline.analyze_response(text, provided_evidence=evidence)

        assert res.overall_risk_level == RiskLevel.INSUFFICIENT_EVIDENCE

    def test_empty_claim_list(self, pipeline):
        """6. Empty claim list -> INSUFFICIENT_EVIDENCE."""
        res = pipeline.analyze_response("")
        assert res.overall_risk_level == RiskLevel.INSUFFICIENT_EVIDENCE

    def test_conversational_framing_with_factual_claims(self, pipeline):
        """7. Conversational framing with factual claim -> VERIFIED."""
        evidence = [
            EvidenceItem(claim="", snippet="Bengaluru is the capital of Karnataka.", source_name="P1", similarity_score=0.95, is_supporting=True),
        ]
        text = "Sure! Here is the answer: Bengaluru is the capital of Karnataka."
        res = pipeline.analyze_response(text, provided_evidence=evidence)

        assert res.overall_risk_level == RiskLevel.VERIFIED

    def test_repeated_claims(self, pipeline):
        """8. Repeated claims -> VERIFIED."""
        evidence = [
            EvidenceItem(claim="", snippet="Bengaluru is the capital of Karnataka.", source_name="P1", similarity_score=0.95, is_supporting=True),
        ]
        text = "Bengaluru is the capital of Karnataka. Bengaluru is the capital of Karnataka."
        res = pipeline.analyze_response(text, provided_evidence=evidence)

        assert res.overall_risk_level == RiskLevel.VERIFIED

    def test_duplicate_claims_with_conflicting_values(self, pipeline):
        """9. Duplicate claims with conflicting values -> LIKELY_HALLUCINATED."""
        evidence = [
            EvidenceItem(claim="", snippet="Bengaluru is the capital of Karnataka.", source_name="P1", similarity_score=0.95, is_supporting=True),
            EvidenceItem(claim="", snippet="Mumbai is the capital of Maharashtra.", source_name="P2", similarity_score=0.95, is_supporting=True),
        ]
        text = "Bengaluru is the capital of Karnataka. Bengaluru is the capital of Maharashtra."
        res = pipeline.analyze_response(text, provided_evidence=evidence)

        assert res.overall_risk_level == RiskLevel.LIKELY_HALLUCINATED
        assert res.sentence_analyses[0].risk_level == RiskLevel.VERIFIED
        assert res.sentence_analyses[1].risk_level == RiskLevel.LIKELY_HALLUCINATED


# =====================================================================
# Phase 16G: Provider and Pillar Status Invariants
# =====================================================================

class TestProviderAndPillarStatus:
    @pytest.fixture
    def pipeline(self):
        return HallucinationDetectionPipeline()

    def test_p2_unavailable_for_pasted_text(self, pipeline):
        """Pasted-text P2 remains unavailable without logprobs."""
        evidence = [
            EvidenceItem(claim="", snippet="Water has formula H2O.", source_name="P1", similarity_score=0.95, is_supporting=True),
        ]
        res = pipeline.analyze_response("Water has the chemical formula H2O.", provided_evidence=evidence)

        assert res.pillar2_summary.available is False
        assert res.pillar2_summary.status == "UNAVAILABLE"
        assert res.weights_used["beta_confidence_gap"] == 0.0

    def test_p3_unavailable_for_single_generation(self, pipeline):
        """Single-generation P3 remains unavailable without sample responses."""
        evidence = [
            EvidenceItem(claim="", snippet="Water has formula H2O.", source_name="P1", similarity_score=0.95, is_supporting=True),
        ]
        res = pipeline.analyze_response("Water has the chemical formula H2O.", provided_evidence=evidence)

        assert res.pillar3_summary.available is False
        assert res.pillar3_summary.status == "UNAVAILABLE"
        assert res.weights_used["gamma_consistency_failure"] == 0.0
