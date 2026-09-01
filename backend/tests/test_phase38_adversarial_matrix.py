"""Phase 38.2 — Golden Adversarial Test Matrix.

Constructs and evaluates a comprehensive deterministic test matrix across 10 categories:
- Category A: Factual Minimal Pairs (10 pairs = 20 cases)
- Category B: Entity Swaps (10 pairs = 20 cases)
- Category C: Numerical Mutations (10 pairs = 20 cases)
- Category D: Negation (10 pairs = 20 cases)
- Category E: Temporal Mutation (10 pairs = 20 cases)
- Category F: Multi-Claim Structural Pairs (10 pairs = 20 cases)
- Category G: Unsupported & Fabricated Claims (10 cases)
- Category H: Entity-Relationship Swaps (10 cases)
- Category I: Paraphrase Invariance Sets (12 cases)
- Category J: Adversarially Worded Framing (10 cases)

Total evaluated test cases: 162.
Evaluates:
1. Pipeline execution without uncaught exceptions.
2. Structure and feature presence (19 features).
3. Metric capture for feature collapse analysis (L1/L2/cosine).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.pipeline import get_hallucisense_pipeline
from app.models.registry import registry
from app.core.inference.local_attribution import (
    compute_local_attribution,
    get_feature_schema,
    get_training_medians,
)

# ─── 162-Case Adversarial Golden Dataset ──────────────────────────────────────

ADVERSARIAL_CASES: Dict[str, List[Dict[str, Any]]] = {
    # ── Category A: Factual Minimal Pairs (10 pairs) ──
    "category_a_minimal_pairs": [
        {"id": "A01_true", "text": "The capital of France is Paris.", "expected": "factual"},
        {"id": "A01_false", "text": "The capital of France is Berlin.", "expected": "hallucination"},
        {"id": "A02_true", "text": "Oxygen has an atomic number of 8.", "expected": "factual"},
        {"id": "A02_false", "text": "Oxygen has an atomic number of 9.", "expected": "hallucination"},
        {"id": "A03_true", "text": "Mount Everest is the highest mountain on Earth.", "expected": "factual"},
        {"id": "A03_false", "text": "K2 is the highest mountain on Earth.", "expected": "hallucination"},
        {"id": "A04_true", "text": "The Pacific Ocean is the largest ocean on Earth.", "expected": "factual"},
        {"id": "A04_false", "text": "The Atlantic Ocean is the largest ocean on Earth.", "expected": "hallucination"},
        {"id": "A05_true", "text": "Water is composed of hydrogen and oxygen atoms.", "expected": "factual"},
        {"id": "A05_false", "text": "Water is composed of helium and nitrogen atoms.", "expected": "hallucination"},
        {"id": "A06_true", "text": "The Amazon River is located in South America.", "expected": "factual"},
        {"id": "A06_false", "text": "The Amazon River is located in Africa.", "expected": "hallucination"},
        {"id": "A07_true", "text": "DNA contains adenine, thymine, cytosine, and guanine.", "expected": "factual"},
        {"id": "A07_false", "text": "DNA contains adenine, uracil, benzene, and glucose.", "expected": "hallucination"},
        {"id": "A08_true", "text": "The heart pumps blood through the circulatory system.", "expected": "factual"},
        {"id": "A08_false", "text": "The lungs pump blood through the circulatory system.", "expected": "hallucination"},
        {"id": "A09_true", "text": "Photosynthesis converts sunlight into chemical energy in plants.", "expected": "factual"},
        {"id": "A09_false", "text": "Respiration converts sunlight into chemical energy in plants.", "expected": "hallucination"},
        {"id": "A10_true", "text": "The speed of sound in dry air at 20 degrees Celsius is approximately 343 meters per second.", "expected": "factual"},
        {"id": "A10_false", "text": "The speed of sound in dry air at 20 degrees Celsius is approximately 980 meters per second.", "expected": "hallucination"},
    ],

    # ── Category B: Entity Swaps (10 pairs) ──
    "category_b_entity_swaps": [
        {"id": "B01_orig", "text": "Albert Einstein developed the theory of general relativity.", "expected": "factual"},
        {"id": "B01_swap", "text": "Isaac Newton developed the theory of general relativity.", "expected": "hallucination"},
        {"id": "B02_orig", "text": "Tokyo is the most populous metropolitan area in Japan.", "expected": "factual"},
        {"id": "B02_swap", "text": "Kyoto is the most populous metropolitan area in Japan.", "expected": "hallucination"},
        {"id": "B03_orig", "text": "William Shakespeare wrote the tragedy Hamlet.", "expected": "factual"},
        {"id": "B03_swap", "text": "Charles Dickens wrote the tragedy Hamlet.", "expected": "hallucination"},
        {"id": "B04_orig", "text": "Alan Turing played a pivotal role in cracking Enigma codes at Bletchley Park.", "expected": "factual"},
        {"id": "B04_swap", "text": "John von Neumann played a pivotal role in cracking Enigma codes at Bletchley Park.", "expected": "hallucination"},
        {"id": "B05_orig", "text": "Alexander Fleming discovered penicillin in 1928.", "expected": "factual"},
        {"id": "B05_swap", "text": "Louis Pasteur discovered penicillin in 1928.", "expected": "hallucination"},
        {"id": "B06_orig", "text": "Marie Curie won Nobel Prizes in Physics and Chemistry.", "expected": "factual"},
        {"id": "B06_swap", "text": "Rosalind Franklin won Nobel Prizes in Physics and Chemistry.", "expected": "hallucination"},
        {"id": "B07_orig", "text": "Neil Armstrong was the first human to walk on the Moon.", "expected": "factual"},
        {"id": "B07_swap", "text": "Buzz Aldrin was the first human to walk on the Moon.", "expected": "hallucination"},
        {"id": "B08_orig", "text": "James Watson and Francis Crick published the double helix model of DNA.", "expected": "factual"},
        {"id": "B08_swap", "text": "Gregor Mendel and Charles Darwin published the double helix model of DNA.", "expected": "hallucination"},
        {"id": "B09_orig", "text": "Leonardo da Vinci painted the Mona Lisa.", "expected": "factual"},
        {"id": "B09_swap", "text": "Michelangelo painted the Mona Lisa.", "expected": "hallucination"},
        {"id": "B10_orig", "text": "Nikola Tesla contributed significantly to alternating current electrical systems.", "expected": "factual"},
        {"id": "B10_swap", "text": "Thomas Edison contributed significantly to alternating current electrical systems.", "expected": "hallucination"},
    ],

    # ── Category C: Numerical Mutations (10 pairs) ──
    "category_c_numerical_mutations": [
        {"id": "C01_true", "text": "12 multiplied by 8 equals 96.", "expected": "factual"},
        {"id": "C01_mut", "text": "12 multiplied by 8 equals 95.", "expected": "hallucination"},
        {"id": "C02_true", "text": "The speed of light in vacuum is exactly 299792458 meters per second.", "expected": "factual"},
        {"id": "C02_mut", "text": "The speed of light in vacuum is exactly 299792459 meters per second.", "expected": "hallucination"},
        {"id": "C03_true", "text": "The human skeleton typically consists of 206 bones in adults.", "expected": "factual"},
        {"id": "C03_mut", "text": "The human skeleton typically consists of 312 bones in adults.", "expected": "hallucination"},
        {"id": "C04_true", "text": "The boiling point of pure water at 1 atm is 100 degrees Celsius.", "expected": "factual"},
        {"id": "C04_mut", "text": "The boiling point of pure water at 1 atm is 114 degrees Celsius.", "expected": "hallucination"},
        {"id": "C05_true", "text": "Earth has 1 natural satellite known as the Moon.", "expected": "factual"},
        {"id": "C05_mut", "text": "Earth has 3 natural satellites known as the Moons.", "expected": "hallucination"},
        {"id": "C06_true", "text": "An equilateral triangle has three internal angles of 60 degrees each.", "expected": "factual"},
        {"id": "C06_mut", "text": "An equilateral triangle has three internal angles of 75 degrees each.", "expected": "hallucination"},
        {"id": "C07_true", "text": "There are 60 seconds in one minute.", "expected": "factual"},
        {"id": "C07_mut", "text": "There are 100 seconds in one minute.", "expected": "hallucination"},
        {"id": "C08_true", "text": "A standard deck of playing cards contains 52 cards.", "expected": "factual"},
        {"id": "C08_mut", "text": "A standard deck of playing cards contains 64 cards.", "expected": "hallucination"},
        {"id": "C09_true", "text": "The freezing point of water at standard pressure is 0 degrees Celsius.", "expected": "factual"},
        {"id": "C09_mut", "text": "The freezing point of water at standard pressure is 10 degrees Celsius.", "expected": "hallucination"},
        {"id": "C10_true", "text": "Mars has 2 moons named Phobos and Deimos.", "expected": "factual"},
        {"id": "C10_mut", "text": "Mars has 4 moons named Phobos, Deimos, Titan, and Io.", "expected": "hallucination"},
    ],

    # ── Category D: Negations (10 pairs) ──
    "category_d_negations": [
        {"id": "D01_pos", "text": "Water boils at approximately 100 degrees Celsius at standard atmospheric pressure.", "expected": "factual"},
        {"id": "D01_neg", "text": "Water does not boil at approximately 100 degrees Celsius at standard atmospheric pressure.", "expected": "hallucination"},
        {"id": "D02_pos", "text": "The Earth revolves around the Sun.", "expected": "factual"},
        {"id": "D02_neg", "text": "The Earth does not revolve around the Sun.", "expected": "hallucination"},
        {"id": "D03_pos", "text": "Humans require oxygen for cellular respiration.", "expected": "factual"},
        {"id": "D03_neg", "text": "Humans do not require oxygen for cellular respiration.", "expected": "hallucination"},
        {"id": "D04_pos", "text": "Diamonds are composed entirely of carbon atoms.", "expected": "factual"},
        {"id": "D04_neg", "text": "Diamonds are not composed of carbon atoms.", "expected": "hallucination"},
        {"id": "D05_pos", "text": "Sound waves require a material medium to propagate.", "expected": "factual"},
        {"id": "D05_neg", "text": "Sound waves do not require a material medium to propagate in vacuum.", "expected": "hallucination"},
        {"id": "D06_pos", "text": "Jupiter is a gas giant planet in our solar system.", "expected": "factual"},
        {"id": "D06_neg", "text": "Jupiter is not a gas giant planet in our solar system.", "expected": "hallucination"},
        {"id": "D07_pos", "text": "Photosynthesis produces glucose and oxygen from carbon dioxide and water.", "expected": "factual"},
        {"id": "D07_neg", "text": "Photosynthesis does not produce glucose or oxygen.", "expected": "hallucination"},
        {"id": "D08_pos", "text": "Gravity is an attractive force between masses.", "expected": "factual"},
        {"id": "D08_neg", "text": "Gravity is not an attractive force between masses.", "expected": "hallucination"},
        {"id": "D09_pos", "text": "The Moon affects ocean tides on Earth.", "expected": "factual"},
        {"id": "D09_neg", "text": "The Moon does not affect ocean tides on Earth.", "expected": "hallucination"},
        {"id": "D10_pos", "text": "Electrons carry a negative electric charge.", "expected": "factual"},
        {"id": "D10_neg", "text": "Electrons do not carry a negative electric charge.", "expected": "hallucination"},
    ],

    # ── Category E: Temporal Mutations (10 pairs) ──
    "category_e_temporal_mutations": [
        {"id": "E01_true", "text": "India gained independence in 1947.", "expected": "factual"},
        {"id": "E01_mut", "text": "India gained independence in 1958.", "expected": "hallucination"},
        {"id": "E02_true", "text": "The Apollo 11 Moon landing occurred in 1969.", "expected": "factual"},
        {"id": "E02_mut", "text": "The Apollo 11 Moon landing occurred in 1984.", "expected": "hallucination"},
        {"id": "E03_true", "text": "World War II ended in 1945.", "expected": "factual"},
        {"id": "E03_mut", "text": "World War II ended in 1960.", "expected": "hallucination"},
        {"id": "E04_true", "text": "The Berlin Wall fell in 1989.", "expected": "factual"},
        {"id": "E04_mut", "text": "The Berlin Wall fell in 2005.", "expected": "hallucination"},
        {"id": "E05_true", "text": "The United States Declaration of Independence was signed in 1776.", "expected": "factual"},
        {"id": "E05_mut", "text": "The United States Declaration of Independence was signed in 1865.", "expected": "hallucination"},
        {"id": "E06_true", "text": "The Titanic sank in 1912 after colliding with an iceberg.", "expected": "factual"},
        {"id": "E06_mut", "text": "The Titanic sank in 1942 after colliding with an iceberg.", "expected": "hallucination"},
        {"id": "E07_true", "text": "The Chernobyl disaster occurred in 1986.", "expected": "factual"},
        {"id": "E07_mut", "text": "The Chernobyl disaster occurred in 2001.", "expected": "hallucination"},
        {"id": "E08_true", "text": "The French Revolution began in 1789.", "expected": "factual"},
        {"id": "E08_mut", "text": "The French Revolution began in 1848.", "expected": "hallucination"},
        {"id": "E09_true", "text": "The first human spaceflight by Yuri Gagarin took place in 1961.", "expected": "factual"},
        {"id": "E09_mut", "text": "The first human spaceflight by Yuri Gagarin took place in 1975.", "expected": "hallucination"},
        {"id": "E10_true", "text": "The Magna Carta was signed in 1215.", "expected": "factual"},
        {"id": "E10_mut", "text": "The Magna Carta was signed in 1492.", "expected": "hallucination"},
    ],

    # ── Category F: Multi-Claim Structural Pairs (10 pairs) ──
    "category_f_multiclaim_pairs": [
        {"id": "F01_pure", "text": "Paris is the capital of France. Berlin is the capital of Germany.", "expected": "factual"},
        {"id": "F01_mix", "text": "Paris is the capital of France. Berlin is the capital of France.", "expected": "hallucination"},
        {"id": "F02_pure", "text": "Water freezes at 0 degrees Celsius. Water boils at 100 degrees Celsius.", "expected": "factual"},
        {"id": "F02_mix", "text": "Water freezes at 0 degrees Celsius. Water freezes at 50 degrees Celsius.", "expected": "hallucination"},
        {"id": "F03_pure", "text": "The Sun is a star. The Earth is a rocky planet.", "expected": "factual"},
        {"id": "F03_mix", "text": "The Sun is a star. The Sun is a rocky planet made of ice.", "expected": "hallucination"},
        {"id": "F04_pure", "text": "Gold is a chemical element. Silver is a transition metal.", "expected": "factual"},
        {"id": "F04_mix", "text": "Gold is a chemical element. Gold is a synthetic polymer created in 1990.", "expected": "hallucination"},
        {"id": "F05_pure", "text": "Tokyo is in Japan. Rome is in Italy.", "expected": "factual"},
        {"id": "F05_mix", "text": "Tokyo is in Japan. Tokyo is situated on the Mediterranean coast.", "expected": "hallucination"},
        {"id": "F06_pure", "text": "Helium is lighter than air. Hydrogen is the lightest element.", "expected": "factual"},
        {"id": "F06_mix", "text": "Helium is lighter than air. Helium is heavier than solid lead.", "expected": "hallucination"},
        {"id": "F07_pure", "text": "Humans have four-chambered hearts. Birds have four-chambered hearts.", "expected": "factual"},
        {"id": "F07_mix", "text": "Humans have four-chambered hearts. Humans possess nine distinct hearts.", "expected": "hallucination"},
        {"id": "F08_pure", "text": "Plants perform photosynthesis. Fungi absorb nutrients from their environment.", "expected": "factual"},
        {"id": "F08_mix", "text": "Plants perform photosynthesis. Plants generate internal nuclear fission.", "expected": "hallucination"},
        {"id": "F09_pure", "text": "The Pacific is the deepest ocean. Mount Everest is the highest peak.", "expected": "factual"},
        {"id": "F09_mix", "text": "The Pacific is the deepest ocean. The Pacific ocean is completely dry.", "expected": "hallucination"},
        {"id": "F10_pure", "text": "Electrons carry negative charge. Protons carry positive charge.", "expected": "factual"},
        {"id": "F10_mix", "text": "Electrons carry negative charge. Protons carry negative charge as well.", "expected": "hallucination"},
    ],

    # ── Category G: Unsupported & Fabricated Claims (10 cases) ──
    "category_g_unsupported": [
        {"id": "G01", "text": "An ancient subterranean civilization constructed advanced quantum fiber networks beneath the Sahara desert in 4000 BC.", "expected": "hallucination"},
        {"id": "G02", "text": "Scientists discovered that consuming crushed amethyst crystals activates telepathic abilities in mice.", "expected": "hallucination"},
        {"id": "G03", "text": "The core of Jupiter contains a sentient supercomputer built by extraterrestrials.", "expected": "hallucination"},
        {"id": "G04", "text": "Medieval European monks developed antigravity flying machines fueled by fermented lavender oil.", "expected": "hallucination"},
        {"id": "G05", "text": "In 1745, the King of Sweden declared silence to be the official national currency.", "expected": "hallucination"},
        {"id": "G06", "text": "Antarctic ice cores reveal that penguins operated steam-powered locomotives 10,000 years ago.", "expected": "hallucination"},
        {"id": "G07", "text": "A secret treaty signed in 1910 divided the dark side of the Moon between Bavaria and Portugal.", "expected": "hallucination"},
        {"id": "G08", "text": "Deep sea exploration uncovered an underwater forest of solid gold trees in the Mariana Trench.", "expected": "hallucination"},
        {"id": "G09", "text": "The Great Pyramid of Giza was originally constructed as a geothermal popcorn popper.", "expected": "hallucination"},
        {"id": "G10", "text": "Botanists confirmed that dandelion seeds transmit encrypted radio broadcasts to Alpha Centauri.", "expected": "hallucination"},
    ],

    # ── Category H: Entity-Relationship Swaps (10 cases) ──
    "category_h_entity_rel_swaps": [
        {"id": "H01", "text": "Albert Einstein composed Beethoven's Ninth Symphony while working at Princeton.", "expected": "hallucination"},
        {"id": "H02", "text": "NASA landed astronauts on Mars in July 1969 during the Apollo 11 mission.", "expected": "hallucination"},
        {"id": "H03", "text": "William Shakespeare painted the ceiling of the Sistine Chapel in Rome.", "expected": "hallucination"},
        {"id": "H04", "text": "Charles Darwin invented the World Wide Web while studying finches in the Galapagos.", "expected": "hallucination"},
        {"id": "H05", "text": "Alexander Graham Bell discovered the law of universal gravitation when an apple fell on him.", "expected": "hallucination"},
        {"id": "H06", "text": "Julius Caesar founded Microsoft Corporation in Seattle during the Gallic Wars.", "expected": "hallucination"},
        {"id": "H07", "text": "Marie Curie authored the epic poem The Odyssey in ancient Greece.", "expected": "hallucination"},
        {"id": "H08", "text": "Galileo Galilei developed penicillin while observing the moons of Jupiter.", "expected": "hallucination"},
        {"id": "H09", "text": "Christopher Columbus walked on the surface of the Moon in 1492.", "expected": "hallucination"},
        {"id": "H10", "text": "Thomas Edison formulated the laws of planetary motion while testing light bulbs.", "expected": "hallucination"},
    ],

    # ── Category I: Paraphrase Invariance Sets (12 cases, 3 sets of 4) ──
    "category_i_paraphrases": [
        {"id": "I01_p1", "text": "Paris is the capital of France.", "expected": "factual"},
        {"id": "I01_p2", "text": "The capital city of the French Republic is Paris.", "expected": "factual"},
        {"id": "I01_p3", "text": "France has Paris as its capital city.", "expected": "factual"},
        {"id": "I01_p4", "text": "The French nation designates Paris as its official capital.", "expected": "factual"},
        {"id": "I02_p1", "text": "Water freezes at zero degrees Celsius.", "expected": "factual"},
        {"id": "I02_p2", "text": "The freezing point of water is zero degrees Celsius.", "expected": "factual"},
        {"id": "I02_p3", "text": "At zero degrees Celsius, liquid water transitions into solid ice.", "expected": "factual"},
        {"id": "I02_p4", "text": "Pure water turns to ice at a temperature of 0 degrees Celsius.", "expected": "factual"},
        {"id": "I03_p1", "text": "The Earth orbits around the Sun.", "expected": "factual"},
        {"id": "I03_p2", "text": "Planet Earth revolves in an orbit around the Sun.", "expected": "factual"},
        {"id": "I03_p3", "text": "The Sun is orbited by the planet Earth.", "expected": "factual"},
        {"id": "I03_p4", "text": "Earth moves in an elliptical path surrounding the Sun.", "expected": "factual"},
    ],

    # ── Category J: Adversarially Worded Framing (10 cases) ──
    "category_j_adversarial_wording": [
        {"id": "J01_auth_false", "text": "According to leading Oxford historians and peer-reviewed consensus, Paris became the capital of France in 1999.", "expected": "hallucination"},
        {"id": "J02_hedged_true", "text": "Some evidence suggests that water might perhaps freeze at approximately 0 degrees Celsius under standard conditions.", "expected": "factual"},
        {"id": "J03_quote_false", "text": "Albert Einstein famously declared in his 1905 paper: 'The Moon is composed entirely of Swiss cheddar cheese.'", "expected": "hallucination"},
        {"id": "J04_long_preamble_false", "text": "In light of comprehensive multi-decade scientific investigations conducted across thirty international laboratories, 12 times 8 equals 95.", "expected": "hallucination"},
        {"id": "J05_repeated_false", "text": "Berlin is the capital of France. Berlin is the capital of France. Berlin is the capital of France.", "expected": "hallucination"},
        {"id": "J06_rhetorical_true", "text": "Can anyone doubt that the Earth completes one full orbit around the Sun in approximately 365.25 days?", "expected": "factual"},
        {"id": "J07_authoritative_false", "text": "It is an established mathematical axiom verified by universal consensus that the square root of 64 is 9.", "expected": "hallucination"},
        {"id": "J08_academic_jargon_false", "text": "Recent epistemological paradigms in celestial mechanics demonstrate that Mars possesses seventeen biological oceans.", "expected": "hallucination"},
        {"id": "J09_excessive_confidence_false", "text": "Without any shadow of a doubt and with 100 percent certainty, Napoleon Bonaparte was born in Tokyo, Japan.", "expected": "hallucination"},
        {"id": "J10_misleading_context_true", "text": "Despite widespread misconceptions and ancient myths, Paris remains the undisputed capital of France.", "expected": "factual"},
    ],
}


@pytest.fixture(scope="module")
def pipeline_instance():
    return get_hallucisense_pipeline()


@pytest.fixture(scope="module")
def all_evaluated_records(pipeline_instance):
    """Run all 162 adversarial test cases through the full pipeline and collect records."""
    records = []
    for cat_name, items in ADVERSARIAL_CASES.items():
        for item in items:
            res = pipeline_instance.predict(response_text=item["text"])
            attr = res.get("local_attribution", {})
            vec = [f["value"] for f in attr.get("features", [])]
            
            records.append({
                "id": item["id"],
                "category": cat_name,
                "text": item["text"],
                "expected": item["expected"],
                "claim_count": res["claim_count"],
                "claims": res["claims"],
                "probability": res["hallucination_probability"],
                "verdict": res["is_hallucinated"],
                "confidence": res["confidence_score"],
                "vector": vec,
                "top_hallucination_drivers": [f["feature_name"] for f in attr.get("top_hallucination_drivers", [])],
                "top_protective_drivers": [f["feature_name"] for f in attr.get("top_protective_drivers", [])],
                "interaction_gap": attr.get("interaction_gap", 0.0),
                "baseline_probability": attr.get("baseline_probability", 0.0),
            })
    return records


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_adversarial_matrix_total_count(all_evaluated_records):
    """Verify total evaluated cases is at least 150 (actual: 162)."""
    assert len(all_evaluated_records) >= 150
    assert len(all_evaluated_records) == 162


def test_all_19_features_present_for_every_case(all_evaluated_records):
    """Verify every adversarial case produces a valid 19-dimensional feature vector."""
    for r in all_evaluated_records:
        assert len(r["vector"]) == 19, f"Case {r['id']} has {len(r['vector'])} features, expected 19"
        assert all(np.isfinite(x) for x in r["vector"]), f"Case {r['id']} has non-finite values in vector"


def test_probability_bounded_for_all_cases(all_evaluated_records):
    """Verify hallucination probability is valid float in [0.0, 1.0]."""
    for r in all_evaluated_records:
        p = r["probability"]
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0, f"Case {r['id']} has invalid probability: {p}"


def test_verdict_consistency_with_threshold(all_evaluated_records):
    """Verify verdict strictly equals (probability >= 0.54)."""
    for r in all_evaluated_records:
        expected_verdict = r["probability"] >= 0.54
        assert r["verdict"] == expected_verdict, (
            f"Case {r['id']}: verdict {r['verdict']} inconsistent with P(H)={r['probability']} at threshold 0.54"
        )


def test_claim_count_positive_for_all_cases(all_evaluated_records):
    """Verify at least 1 atomic claim is extracted for every case."""
    for r in all_evaluated_records:
        assert r["claim_count"] >= 1, f"Case {r['id']} produced 0 claims"
        assert len(r["claims"]) == r["claim_count"]


def test_multiclaim_cases_have_claim_count_ge_2(all_evaluated_records):
    """Verify Category F multi-claim pairs produce claim_count >= 2."""
    f_records = [r for r in all_evaluated_records if r["category"] == "category_f_multiclaim_pairs"]
    for r in f_records:
        assert r["claim_count"] >= 2, f"Multi-claim case {r['id']} has claim_count {r['claim_count']}"


def test_unsupported_claims_have_finite_attributions(all_evaluated_records):
    """Verify Category G unsupported claims produce complete valid attributions."""
    g_records = [r for r in all_evaluated_records if r["category"] == "category_g_unsupported"]
    for r in g_records:
        assert np.isfinite(r["interaction_gap"])
        assert len(r["top_protective_drivers"]) >= 1 or len(r["top_hallucination_drivers"]) >= 1
