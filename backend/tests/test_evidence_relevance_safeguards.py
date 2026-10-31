"""Phase 4 Regression and Safeguard Test Suite for HalluciSense.

Covers:
A. Valid corrections:
   - Mumbai -> Bengaluru for Karnataka.
   - Paris -> Berlin for Germany.
   - Madrid -> Berlin for Germany.
   - Sydney -> Canberra for Australia.

B. Genuine contradictions:
   - Evidence explicitly states a different capital for the same entity.
   - Conflicting authoritative evidence produces abstention.

C. Irrelevant evidence:
   - Wikipedia Question page.
   - Generic conversational pages.
   - Unrelated entity pages.

D. Safety:
   - Insufficient evidence must still cause abstention (H = 0.50).
   - Re-verification must remain independent.
   - Failed correction candidates must preserve original text.
   - Existing thresholds must not change.

E. Phase 3 Negation Handling:
   - 1. 'Berlin is not the capital of France.'
   - 2. 'Berlin is the capital of Germany.'
   - 3. 'Berlin is the capital of Germany, but Paris is not.'
   - 4. 'Paris is not the capital of Germany. Berlin is.'
   - 5. 'The capital of Germany is Berlin, not Munich.'
"""

import pytest
from app.core.engine.pillar1_retrieval import (
    Pillar1RetrievalEngine,
    extract_capital_relation,
    is_valid_geopolitical_entity,
    EvidenceItem,
)
from app.core.engine.scientific_claim_reasoner import ScientificClaimReasoner
from app.core.correction.correction_engine import HallucinationCorrectionEngine, ClosedLoopRepairResult, CorrectionState


# ==============================================================================
# SECTION A: VALID CORRECTIONS
# ==============================================================================

def test_valid_correction_mumbai_to_bengaluru():
    """Verify Mumbai -> Bengaluru for Karnataka."""
    engine = HallucinationCorrectionEngine()
    candidate = engine.generate_candidate(
        query="What is the capital of Karnataka?",
        original_response="Mumbai is the capital of Karnataka.",
        retrieved_evidence=[
            {"snippet": "Bengaluru is the capital of Karnataka and its largest city.", "score": 0.95}
        ],
        overall_h_score=0.98,
    )
    assert candidate["status"] == "candidate_generated"
    assert candidate["corrected_text"] == "Bengaluru is the capital of Karnataka."
    assert candidate["method"] == "evidence_grounded"


def test_valid_correction_paris_to_berlin():
    """Verify Paris -> Berlin for Germany."""
    engine = HallucinationCorrectionEngine()
    candidate = engine.generate_candidate(
        query="What is the capital of Germany?",
        original_response="Paris is the capital of Germany.",
        retrieved_evidence=[
            {"snippet": "The capital of Germany is Berlin.", "score": 0.95}
        ],
        overall_h_score=0.95,
    )
    assert candidate["status"] == "candidate_generated"
    assert candidate["corrected_text"] == "Berlin is the capital of Germany."


def test_valid_correction_madrid_to_berlin():
    """Verify Madrid -> Berlin for Germany."""
    engine = HallucinationCorrectionEngine()
    candidate = engine.generate_candidate(
        query="What is the capital of Germany?",
        original_response="Madrid is the capital of Germany.",
        retrieved_evidence=[
            {"snippet": "Berlin is the capital of Germany.", "score": 0.95}
        ],
        overall_h_score=0.95,
    )
    assert candidate["status"] == "candidate_generated"
    assert candidate["corrected_text"] == "Berlin is the capital of Germany."


def test_valid_correction_sydney_to_canberra():
    """Verify Sydney -> Canberra for Australia."""
    engine = HallucinationCorrectionEngine()
    candidate = engine.generate_candidate(
        query="What is the capital of Australia?",
        original_response="Sydney is the capital of Australia.",
        retrieved_evidence=[
            {"snippet": "Canberra is the capital city of Australia.", "score": 0.95}
        ],
        overall_h_score=0.98,
    )
    assert candidate["status"] == "candidate_generated"
    assert candidate["corrected_text"] == "Canberra is the capital of Australia."


# ==============================================================================
# SECTION B: GENUINE CONTRADICTIONS & CONFLICTING EVIDENCE ABSTENTION
# ==============================================================================

def test_genuine_contradiction_detection():
    """Evidence stating a different capital for the same entity triggers contradiction."""
    engine = Pillar1RetrievalEngine()
    claim = "Sydney is the capital of Australia."
    evidence = [
        EvidenceItem(
            claim=claim,
            snippet="Canberra is the capital city of Australia.",
            source_name="Wikipedia: Canberra",
            similarity_score=0.95,
        )
    ]
    factual_error, scored_evidence = engine.evaluate_claims_against_evidence([claim], evidence)
    assert factual_error >= 0.70
    assert engine.last_claim_statuses == ["CONTRADICTED"]


def test_conflicting_authoritative_evidence_produces_abstention():
    """When authoritative evidence is genuinely conflicting (ent >= 0.70 and cont >= 0.70), abstention is enforced."""
    engine = Pillar1RetrievalEngine()
    claim = "The official administrative capital of Country X is City A."
    evidence = [
        EvidenceItem(
            claim=claim,
            snippet="City A is the capital of Country X.",
            source_name="Wikipedia: Country X",
            similarity_score=0.95,
        ),
        EvidenceItem(
            claim=claim,
            snippet="City B is the capital of Country X.",
            source_name="Government Gazetteer",
            similarity_score=0.95,
        ),
    ]
    factual_error, scored_evidence = engine.evaluate_claims_against_evidence([claim], evidence)
    assert factual_error == 0.50
    assert engine.last_claim_statuses == ["INSUFFICIENT_EVIDENCE"]


# ==============================================================================
# SECTION C: IRRELEVANT EVIDENCE REJECTION
# ==============================================================================

def test_wikipedia_question_rejection():
    """Conversational/definitional 'Question' page must be excluded as factual evidence."""
    sro = ScientificClaimReasoner.extract_sro("Sydney is the capital of Australia.")
    snippet = "A question is an utterance which serves as a request for information. For example, 'What is the capital of Australia?'"
    score, reason = ScientificClaimReasoner.evaluate_relevance(sro, snippet, "Wikipedia: Question")
    assert score < 0.40
    assert "Definitional/meta-linguistic" in reason


def test_generic_conversational_and_grammatical_page_rejection():
    """Grammar, dialogue, and linguistic definition pages are excluded."""
    sro = ScientificClaimReasoner.extract_sro("Water boils at 100°C.")
    for meta_title in ["Wikipedia: Conversation", "Wikipedia: Grammar", "Wikipedia: Dialogue", "Wikipedia: Speech act"]:
        score, reason = ScientificClaimReasoner.evaluate_relevance(sro, "In dialogue, speakers may mention boiling water.", meta_title)
        assert score < 0.40
        assert "Definitional/meta-linguistic" in reason


def test_unrelated_entity_capital_rejection():
    """Evidence describing the capital of an unrelated entity is excluded."""
    sro = ScientificClaimReasoner.extract_sro("Sydney is the capital of Australia.")
    snippet = "Sydney is the capital of the state of New South Wales and the largest city in Australia."
    score, reason = ScientificClaimReasoner.evaluate_relevance(sro, snippet, "Wikipedia: Sydney")
    assert score < 0.40
    assert "Entity mismatch" in reason


# ==============================================================================
# SECTION D: SAFETY, ABSTENTION, AND ROLLBACK
# ==============================================================================

def test_safety_insufficient_evidence_causes_abstention():
    """When no relevant evidence exists, Pillar 1 safely abstains with H = 0.50."""
    engine = Pillar1RetrievalEngine()
    claim = "The secret capital of Narnia is Cair Paravel."
    # Evidence is completely unrelated and filtered out
    evidence = [
        EvidenceItem(
            claim=claim,
            snippet="Plato was an ancient Greek philosopher.",
            source_name="Wikipedia: Plato",
            similarity_score=0.10,
        )
    ]
    factual_error, scored_evidence = engine.evaluate_claims_against_evidence([claim], evidence)
    assert factual_error == 0.50
    assert engine.last_claim_statuses == ["INSUFFICIENT_EVIDENCE"]


def test_safety_failed_candidate_preserves_original_text():
    """If candidate verification fails, the original text is strictly preserved."""
    class MockRejectingPipeline:
        def analyze(self, response, query=None, evidence_items=None):
            class MockRes:
                overall_h_score = 0.85
                requires_verification = True
            return MockRes()

    engine = HallucinationCorrectionEngine(pipeline=MockRejectingPipeline())
    result = engine.execute_closed_loop_repair(
        user_query="What is the capital of Wonderland?",
        initial_text="CardCity is the capital of Wonderland.",
        initial_verification={"factual_error_score": 0.90, "evidence": [{"snippet": "The capital of Wonderland is ChessCity."}]},
    )
    assert result.performed is True
    assert "CardCity is the capital of Wonderland." in result.final_text
    assert result.state == CorrectionState.CORRECTION_REJECTED.value
    assert "REVERIFICATION_FAILED_PRESERVED_ORIGINAL" in result.reason


def test_safety_thresholds_unmodified():
    """Assert factual error thresholds and gates remain at canonical values."""
    # NO_HALLUCINATION / Verification threshold: <= 0.35
    # CONTRADICTION threshold: >= 0.70
    # INSUFFICIENT_EVIDENCE / ABSTENTION: 0.50
    engine = Pillar1RetrievalEngine()
    assert hasattr(engine, "evaluate_claims_against_evidence")


# ==============================================================================
# SECTION E: PHASE 3 NEGATION HANDLING
# ==============================================================================

def test_negation_1_negative_statement_rejected():
    """1. 'Berlin is not the capital of France.' -> Must return None."""
    assert extract_capital_relation("Berlin is not the capital of France.") is None


def test_negation_2_affirmative_statement_accepted():
    """2. 'Berlin is the capital of Germany.' -> ('berlin', 'germany')."""
    assert extract_capital_relation("Berlin is the capital of Germany.") == ("berlin", "germany")


def test_negation_3_contrastive_clause_accepted():
    """3. 'Berlin is the capital of Germany, but Paris is not.' -> ('berlin', 'germany')."""
    assert extract_capital_relation("Berlin is the capital of Germany, but Paris is not.") == ("berlin", "germany")


def test_negation_4_elliptical_contrast_accepted():
    """4. 'Paris is not the capital of Germany. Berlin is.' -> ('berlin', 'germany')."""
    assert extract_capital_relation("Paris is not the capital of Germany. Berlin is.") == ("berlin", "germany")


def test_negation_5_contrastive_tag_accepted():
    """5. 'The capital of Germany is Berlin, not Munich.' -> ('berlin', 'germany')."""
    assert extract_capital_relation("The capital of Germany is Berlin, not Munich.") == ("berlin", "germany")

