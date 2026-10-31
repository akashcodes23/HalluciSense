"""Regression and Validation Test Suite for Capital Relation Extraction and Closed-Loop Correction.

Covers:
- Robust extraction patterns and edge-case handling in extract_capital_relation
- Prevention of generic grammatical entity captures (state, city, country, nation, the, a, an)
- Target-entity congruence validation in HallucinationCorrectionEngine
- Preservation of true assertions in multi-claim responses
- Closed-loop verification across target test cases:
    1. Mumbai is the capital of Karnataka. -> Corrected to Bengaluru is the capital of Karnataka.
    2. Bengaluru is the capital of Karnataka. -> Already verified / not_needed.
    3. Mumbai is the capital of Maharashtra. -> Already verified / not_needed.
    4. Paris is the capital of Germany. -> Corrected to Berlin is the capital of Germany.
    5. Madrid is the capital of Germany. -> Corrected to Berlin is the capital of Germany.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.engine.pillar1_retrieval import extract_capital_relation, is_valid_geopolitical_entity
from app.core.correction.correction_engine import HallucinationCorrectionEngine


@pytest.fixture
def client():
    return TestClient(app)


# ==============================================================================
# 1. CAPITAL RELATION EXTRACTION UNIT TESTS
# ==============================================================================

def test_extract_capital_relation_standard_patterns():
    """Verify standard positive capital extraction patterns."""
    # Pattern 2: The capital of Y is X
    res = extract_capital_relation("The capital of Germany is Berlin.")
    assert res == ("berlin", "germany")

    # Pattern 1: X is the capital of Y
    res = extract_capital_relation("Berlin is the capital of Germany.")
    assert res == ("berlin", "germany")

    # Pattern 5: X, the capital of Y, ...
    res = extract_capital_relation("Bengaluru, the capital of Karnataka, is a major technology hub.")
    assert res == ("bengaluru", "karnataka")

    # Pattern 3: The capital city of Y is X
    res = extract_capital_relation("The capital city of Australia is Canberra.")
    assert res == ("canberra", "australia")

    # Pattern: Y's capital is X
    res = extract_capital_relation("Germany's capital is Berlin.")
    assert res == ("berlin", "germany")


def test_extract_capital_relation_generic_placeholder_rejection():
    """Ensure generic grammatical placeholders are rejected as entities."""
    # The exact phrase from Wikipedia: Karnataka that caused production failure:
    # "The capital of the state, Bengaluru, is known as..."
    # 'state' is not a geopolitical entity.
    res = extract_capital_relation("The capital of the state, Bengaluru, is known as the Silicon Valley.")
    assert res is None

    # Generic country / city / nation placeholders
    assert extract_capital_relation("The capital of the country is the city.") is None
    assert extract_capital_relation("The capital of the nation is large.") is None
    assert extract_capital_relation("The capital of the province is situated nearby.") is None


def test_extract_capital_relation_modifiers_and_multiword():
    """Verify handling of optional modifiers and multiword geographical entities."""
    # Optional modifiers: official, state, national
    res = extract_capital_relation("Bengaluru is the official state capital of Karnataka.")
    assert res == ("bengaluru", "karnataka")

    # Multiword capital and entity names
    res = extract_capital_relation("New Delhi is the capital of India.")
    assert res == ("new delhi", "india")

    res = extract_capital_relation("The capital of United States is Washington.")
    assert res == ("washington", "united states")

    res = extract_capital_relation("The capital of the United States of America is Washington, D.C.")
    assert res is not None
    assert "washington" in res[0]


def test_extract_capital_relation_negative_and_ambiguous_rejection():
    """Verify that negative assertions and ambiguous statements are rejected."""
    # Negative assertions should not be extracted as factual relations
    assert extract_capital_relation("Paris is not the capital of Germany.") is None
    assert extract_capital_relation("Madrid is not the capital of Germany.") is None

    # Ambiguous or non-relational mentions
    assert extract_capital_relation("Capital punishment was abolished in Germany.") is None
    assert extract_capital_relation("Venture capital in Bengaluru has grown rapidly.") is None


def test_is_valid_geopolitical_entity():
    """Test entity validator stop words and minimum length requirements."""
    assert is_valid_geopolitical_entity("Karnataka") is True
    assert is_valid_geopolitical_entity("Germany") is True
    assert is_valid_geopolitical_entity("Berlin") is True
    assert is_valid_geopolitical_entity("New Delhi") is True

    # Generic stopwords must be rejected
    assert is_valid_geopolitical_entity("state") is False
    assert is_valid_geopolitical_entity("the state") is False
    assert is_valid_geopolitical_entity("country") is False
    assert is_valid_geopolitical_entity("city") is False
    assert is_valid_geopolitical_entity("the") is False
    assert is_valid_geopolitical_entity("an") is False
    assert is_valid_geopolitical_entity("it") is False
    assert is_valid_geopolitical_entity("") is False


# ==============================================================================
# 2. CORRECTION CANDIDATE GENERATION UNIT TESTS
# ==============================================================================

def test_correction_engine_capital_mumbai_karnataka():
    """Test candidate generation for Mumbai is the capital of Karnataka."""
    engine = HallucinationCorrectionEngine()
    res = engine.generate_candidate(
        query="What is the capital of Karnataka?",
        original_response="Mumbai is the capital of Karnataka.",
        retrieved_evidence=[
            "Bengaluru, the capital of Karnataka, is widely regarded as the Silicon Valley of India.",
            "The state of Karnataka has its capital in Bengaluru."
        ],
        overall_h_score=0.98,
    )
    assert res["status"] == "candidate_generated"
    assert res["method"] == "evidence_grounded"
    assert res["corrected_text"] == "Bengaluru is the capital of Karnataka."


def test_correction_engine_capital_paris_germany():
    """Test candidate generation for Paris is the capital of Germany."""
    engine = HallucinationCorrectionEngine()
    res = engine.generate_candidate(
        query="What is the capital of Germany?",
        original_response="Paris is the capital of Germany.",
        retrieved_evidence=[
            "The capital of Germany is Berlin, which is also the largest city in the country."
        ],
        overall_h_score=0.95,
    )
    assert res["status"] == "candidate_generated"
    assert res["method"] == "evidence_grounded"
    assert res["corrected_text"] == "Berlin is the capital of Germany."


def test_correction_engine_capital_madrid_germany():
    """Test candidate generation for Madrid is the capital of Germany."""
    engine = HallucinationCorrectionEngine()
    res = engine.generate_candidate(
        query="What is the capital of Germany?",
        original_response="Madrid is the capital of Germany.",
        retrieved_evidence=[
            "Berlin is the capital of Germany and one of its 16 states."
        ],
        overall_h_score=0.95,
    )
    assert res["status"] == "candidate_generated"
    assert res["method"] == "evidence_grounded"
    assert res["corrected_text"] == "Berlin is the capital of Germany."


def test_correction_engine_rejects_unrelated_target_entity():
    """Ensure candidate generation strictly matches target entity."""
    engine = HallucinationCorrectionEngine()
    # Claim is about Karnataka; evidence mentions Maharashtra capital
    res = engine.generate_candidate(
        query="What is the capital of Karnataka?",
        original_response="Mumbai is the capital of Karnataka.",
        retrieved_evidence=[
            "Mumbai is the financial capital of India and the capital of Maharashtra."
        ],
        overall_h_score=0.95,
    )
    # Evidence for Maharashtra should NOT produce a correction for Karnataka
    assert res["status"] == "abstained"
    assert res["corrected_text"] is None


def test_correction_engine_multi_claim_preservation():
    """Ensure true clauses in multi-claim sentences are preserved during correction."""
    engine = HallucinationCorrectionEngine()
    res = engine.generate_candidate(
        query="Tell me about chemistry and Maharashtra.",
        original_response="Water has the chemical formula H2O, and Bengaluru is the capital of Maharashtra.",
        retrieved_evidence=[
            "Mumbai is the capital of Maharashtra and the financial centre of India."
        ],
        overall_h_score=0.80,
    )
    assert res["status"] == "candidate_generated"
    assert res["method"] == "evidence_grounded"
    # The factual clause about water must remain intact!
    assert "Water has the chemical formula H2O" in res["corrected_text"]
    assert "Mumbai is the capital of Maharashtra" in res["corrected_text"]


def test_correction_engine_does_not_replace_subsequent_occurrences():
    """Verify that only the relevant occurrence of a city name is replaced."""
    engine = HallucinationCorrectionEngine()
    res = engine.generate_candidate(
        query="Tell me about Germany.",
        original_response="Paris is the capital of Germany, while Paris is a beautiful city in France.",
        retrieved_evidence=[
            "Berlin is the capital of Germany and its largest city."
        ],
        overall_h_score=0.85,
    )
    assert res["status"] == "candidate_generated"
    assert res["method"] == "evidence_grounded"
    # First occurrence 'Paris' is replaced by 'Berlin', but second occurrence in France remains 'Paris'
    assert res["corrected_text"] == "Berlin is the capital of Germany, while Paris is a beautiful city in France."


def test_correction_engine_abtains_when_evidence_ambiguous():
    """Ensure correction safely abstains when no valid capital evidence exists."""
    engine = HallucinationCorrectionEngine()
    res = engine.generate_candidate(
        query="What is the capital of Elbonia?",
        original_response="Dilbert is the capital of Elbonia.",
        retrieved_evidence=[
            "Elbonia is a fictional nation covered in mud."
        ],
        overall_h_score=0.90,
    )
    assert res["status"] == "abstained"
    assert res["corrected_text"] is None


# ==============================================================================
# 3. END-TO-END CLOSED-LOOP API VERIFICATION TESTS
# ==============================================================================

def test_api_correct_mumbai_karnataka(client):
    """End-to-end: Mumbai is the capital of Karnataka -> Bengaluru is the capital of Karnataka."""
    resp = client.post("/api/v1/correct", json={
        "query": "What is the capital of Karnataka?",
        "response": "Mumbai is the capital of Karnataka."
    })
    assert resp.status_code == 200
    data = resp.json()
    corr = data["correction"]
    assert corr["status"] == "verified"
    assert "Bengaluru is the capital of Karnataka." in corr["corrected_text"]
    assert corr["reverification"]["status"] == "VERIFIED"
    assert corr["reverification"]["overall_h_score"] <= 0.35


def test_api_correct_bengaluru_karnataka_not_needed(client):
    """End-to-end: Bengaluru is the capital of Karnataka is already true."""
    resp = client.post("/api/v1/correct", json={
        "query": "What is the capital of Karnataka?",
        "response": "Bengaluru is the capital of Karnataka."
    })
    assert resp.status_code == 200
    data = resp.json()
    corr = data["correction"]
    assert corr["status"] == "not_needed"
    assert corr["corrected_text"] == "Bengaluru is the capital of Karnataka."


def test_api_correct_mumbai_maharashtra_not_needed(client):
    """End-to-end: Mumbai is the capital of Maharashtra is already true."""
    resp = client.post("/api/v1/correct", json={
        "query": "What is the capital of Maharashtra?",
        "response": "Mumbai is the capital of Maharashtra."
    })
    assert resp.status_code == 200
    data = resp.json()
    corr = data["correction"]
    assert corr["status"] == "not_needed"
    assert corr["corrected_text"] == "Mumbai is the capital of Maharashtra."


def test_api_correct_paris_germany(client):
    """End-to-end: Paris is the capital of Germany -> Berlin is the capital of Germany."""
    resp = client.post("/api/v1/correct", json={
        "query": "What is the capital of Germany?",
        "response": "Paris is the capital of Germany."
    })
    assert resp.status_code == 200
    data = resp.json()
    corr = data["correction"]
    assert corr["status"] == "verified"
    assert "Berlin is the capital of Germany." in corr["corrected_text"]
    assert corr["reverification"]["status"] == "VERIFIED"
    assert corr["reverification"]["overall_h_score"] <= 0.35


def test_api_correct_madrid_germany(client):
    """End-to-end: Madrid is the capital of Germany -> Berlin is the capital of Germany."""
    resp = client.post("/api/v1/correct", json={
        "query": "What is the capital of Germany?",
        "response": "Madrid is the capital of Germany."
    })
    assert resp.status_code == 200
    data = resp.json()
    corr = data["correction"]
    assert corr["status"] == "verified"
    assert "Berlin is the capital of Germany." in corr["corrected_text"]
    assert corr["reverification"]["status"] == "VERIFIED"
    assert corr["reverification"]["overall_h_score"] <= 0.35


def test_api_correct_sydney_australia(client):
    """End-to-end: Sydney is the capital of Australia -> Canberra is the capital of Australia."""
    resp = client.post("/api/v1/correct", json={
        "query": "What is the capital of Australia?",
        "response": "Sydney is the capital of Australia."
    })
    assert resp.status_code == 200
    data = resp.json()
    corr = data["correction"]
    assert corr["status"] == "verified"
    assert "Canberra is the capital of Australia." in corr["corrected_text"]
    assert corr["reverification"]["status"] == "VERIFIED"
    assert corr["reverification"]["overall_h_score"] <= 0.35


def test_correction_engine_candidate_rejection_safety():
    """Unit test: Engine rejects unsupported candidates and preserves original text."""
    engine = HallucinationCorrectionEngine()
    res = engine.generate_candidate(
        query="What is the capital of Atlantis?",
        original_response="Poseidon is the capital of Atlantis.",
        retrieved_evidence=[
            "Atlantis is a mythical island mentioned in Plato's dialogues."
        ],
        overall_h_score=0.95,
    )
    # No verified ground truth for candidate relation -> abstains safely
    assert res["status"] == "abstained"
    assert res["corrected_text"] is None

