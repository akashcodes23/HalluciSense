"""Generate Phase 6D Adversarial Temporal-Epistemic Benchmark & Counterfactual Pairs.

Constructs N=440 structured examples across 20 distinct temporal/epistemic categories.
Ensures exact 50/50 class balance (220 hallucinated / 220 factual).
Generates counterfactual pairs (PAIR A -> PAIR F) for mechanism verification.
Does NOT hardcode entity dates into production detection rules.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "external"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports" / "phase6d"
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def build_phase6d_dataset() -> List[Dict[str, Any]]:
    records = []
    
    # 20 distinct categories, 22 examples per category (11 factual, 11 hallucinated) = 440 total
    categories = [
        ("CAT01", "correct_historical_assertion", "history"),
        ("CAT02", "incorrect_historical_assertion", "history"),
        ("CAT03", "correct_future_prediction", "technology"),
        ("CAT04", "incorrect_future_prediction", "technology"),
        ("CAT05", "hypothetical_future", "science"),
        ("CAT06", "counterfactual_historical", "history"),
        ("CAT07", "conditional_temporal_statement", "economics"),
        ("CAT08", "negated_temporal_assertion", "law"),
        ("CAT09", "quoted_false_statement", "media"),
        ("CAT10", "meta_claim_debunking", "journalism"),
        ("CAT11", "fictional_temporal_statement", "literature"),
        ("CAT12", "multi_event_evidence", "astronomy"),
        ("CAT13", "evidence_irrelevant_dates", "medicine"),
        ("CAT14", "evidence_conflicting_dates", "history"),
        ("CAT15", "evidence_unrelated_entity_dates", "politics"),
        ("CAT16", "relative_temporal_expressions", "history"),
        ("CAT17", "event_ordering", "engineering"),
        ("CAT18", "date_range_contradiction", "climate"),
        ("CAT19", "multi_hop_temporal_relation", "history"),
        ("CAT20", "adversarial_evidence_ordering", "science"),
    ]

    domains = ["history", "medicine", "science", "astronomy", "technology", "law", "politics", "economics", "climate", "engineering"]

    idx = 1

    # Base templates for generating controlled 440 examples
    entities = [
        ("Apollo 11", "1969", "1972", "moon landing mission", "1969 moon landing"),
        ("James Webb Telescope", "2021", "2015", "space telescope launch", "2021 launch"),
        ("Human Genome Project", "2003", "1995", "sequencing completion", "2003 completion"),
        ("Large Hadron Collider", "2008", "2002", "first particle collisions", "2008 first beams"),
        ("Panama Canal", "1914", "1925", "canal opening", "1914 inauguration"),
        ("International Space Station", "1998", "2005", "first module launch", "1998 Zarya launch"),
        ("Curiosity Rover", "2012", "2008", "Mars landing", "2012 touchdown"),
        ("Hubble Space Telescope", "1990", "1985", "orbital deployment", "1990 launch"),
        ("Voyager 1", "1977", "1982", "interstellar probe launch", "1977 launch"),
        ("Spitzer Space Telescope", "2003", "2010", "infrared observatory launch", "2003 launch"),
        ("Kepler Space Telescope", "2009", "2004", "exoplanet search launch", "2009 launch"),
    ]

    for cat_code, cat_name, base_domain in categories:
        for i in range(11):
            ent_name, true_yr, false_yr, event_desc, ev_anchor = entities[i % len(entities)]
            dom = domains[(idx - 1) % len(domains)]

            # Factual record (gold_hallucination = False)
            if "future" in cat_name or "prediction" in cat_name:
                q_fact = f"When will {ent_name} undergo next upgrade?"
                r_fact = f"{ent_name} is targeted to undergo systems evaluation in {2028 + (i%5)}."
                ev_fact = f"Project roadmap indicates {ent_name} upgrade is scheduled for {2028 + (i%5)}."
                mod_fact = "PREDICTION"
            elif "hypothetical" in cat_name:
                q_fact = f"What if {ent_name} receives additional funding in 2030?"
                r_fact = f"If {ent_name} receives expanded funding by 2030, mission lifespan will extend."
                ev_fact = f"Planning documents outline potential extension if 2030 funding is approved."
                mod_fact = "HYPOTHETICAL"
            elif "counterfactual" in cat_name:
                q_fact = f"What if {ent_name} had been launched in {false_yr}?"
                r_fact = f"Had {ent_name} been deployed in {false_yr}, early testing would have occurred sooner."
                ev_fact = f"Historical proposal evaluated an alternative {false_yr} launch timeline."
                mod_fact = "COUNTERFACTUAL"
            elif "conditional" in cat_name:
                q_fact = f"Under what condition will {ent_name} operate in 2029?"
                r_fact = f"If power levels remain optimal, {ent_name} will continue observations into 2029."
                ev_fact = f"Engineering projections assume 2029 operations provided power levels hold."
                mod_fact = "CONDITIONAL"
            elif "negated" in cat_name:
                q_fact = f"Did {ent_name} experience failure in {false_yr}?"
                r_fact = f"{ent_name} did not experience system failure in {false_yr}."
                ev_fact = f"Official archives confirm {ent_name} operated without major failure in {false_yr}."
                mod_fact = "NEGATED_FACT"
            elif "quoted" in cat_name or "meta" in cat_name:
                q_fact = f"What was reported regarding {ent_name}?"
                r_fact = f"Recent reports falsely claimed that {ent_name} was cancelled in {false_yr}."
                ev_fact = f"Fact checkers debunked rumors claiming {ent_name} was cancelled in {false_yr}."
                mod_fact = "QUOTED_CLAIM"
            elif "fictional" in cat_name:
                q_fact = f"In the novel, when does {ent_name} appear?"
                r_fact = f"In the sci-fi novel, {ent_name} explores deep space in 2045."
                ev_fact = f"The literary narrative depicts {ent_name} operating in 2045."
                mod_fact = "FICTIONAL"
            else:
                q_fact = f"When occurred the {event_desc} of {ent_name}?"
                r_fact = f"The {event_desc} of {ent_name} occurred in {true_yr}."
                ev_fact = f"Historical record confirms {ent_name} {event_desc} took place in {true_yr}."
                mod_fact = "ASSERTED_FACT"

            records.append({
                "example_id": f"P6D_{idx:04d}",
                "domain": dom,
                "category": cat_name,
                "query": q_fact,
                "response": r_fact,
                "context": ev_fact,
                "gold_hallucination": False,
                "temporal_signal": "historical" if "historical" in cat_name else ("future" if "future" in cat_name else "general"),
                "query_modality": "ASSERTED_FACT",
                "response_modality": mod_fact,
                "temporal_relation": "point_date",
                "adversarial_type": cat_name,
                "evidence_noise_type": "clean" if "clean" in cat_name else "structured",
            })
            idx += 1

            # Hallucinated record (gold_hallucination = True)
            if "future" in cat_name and "incorrect" in cat_name:
                q_hall = f"When was {ent_name} completed?"
                r_hall = f"{ent_name} completed its operational mission in {2029 + (i%5)}." # Asserting past verb with future year
                ev_hall = f"Historical records confirm {ent_name} was completed in {true_yr}."
                mod_hall = "FUTURE_FACT_ASSERTION"
            elif "date_range" in cat_name:
                q_hall = f"What was the active period of {ent_name}?"
                r_hall = f"{ent_name} operated from {false_yr} to {true_yr}."
                ev_hall = f"Official timeline records show {ent_name} was active from {true_yr} to 2024."
                mod_hall = "ASSERTED_FACT"
            elif "irrelevant" in cat_name or "unrelated" in cat_name:
                q_hall = f"When did {ent_name} launch?"
                r_hall = f"{ent_name} was launched in {false_yr}."
                ev_hall = f"{ent_name} was launched in {true_yr}. (Note: Unrelated project B launched in {false_yr})."
                mod_hall = "ASSERTED_FACT"
            else:
                q_hall = f"When occurred the {event_desc} of {ent_name}?"
                r_hall = f"{ent_name} experienced {event_desc} in {false_yr}."
                ev_hall = f"Canonical records document that {ent_name} {event_desc} took place in {true_yr}."
                mod_hall = "ASSERTED_FACT"

            records.append({
                "example_id": f"P6D_{idx:04d}",
                "domain": dom,
                "category": cat_name,
                "query": q_hall,
                "response": r_hall,
                "context": ev_hall,
                "gold_hallucination": True,
                "temporal_signal": "historical",
                "query_modality": "ASSERTED_FACT",
                "response_modality": mod_hall,
                "temporal_relation": "mismatch",
                "adversarial_type": cat_name,
                "evidence_noise_type": "mismatch_or_noise",
            })
            idx += 1

    return records


def build_counterfactual_pairs() -> List[Dict[str, Any]]:
    pairs = [
        {
            "pair_id": "PAIR_A_POINT_DATE",
            "description": "Historical assertion date shift (2010 vs 2020)",
            "base_claim": "Apollo 11 landed on the Moon in 1969.",
            "variant_claim": "Apollo 11 landed on the Moon in 1975.",
            "evidence": "Apollo 11 landed on the Moon on July 20, 1969.",
            "base_gold_hallucination": False,
            "variant_gold_hallucination": True,
            "expected_mechanism": "Global Evidence-Date Alignment (DATE_MISMATCH detection)",
        },
        {
            "pair_id": "PAIR_B_FUTURE_ASSERTION",
            "description": "Assertion vs Prediction for future date (2030)",
            "base_claim": "Artemis IV landed on Mars in 2030.",
            "variant_claim": "Artemis IV is targeted to land on Mars in 2030.",
            "evidence": "Artemis IV mission roadmap targets Mars landing in 2030.",
            "base_gold_hallucination": True,
            "variant_gold_hallucination": False,
            "expected_mechanism": "Temporal-Epistemic Gate (FUTURE_IMPOSSIBLE_FACT vs PREDICTION protection)",
        },
        {
            "pair_id": "PAIR_C_NEGATION",
            "description": "Assertion vs Negated Assertion for historical date",
            "base_claim": "The bridge collapsed in 2018.",
            "variant_claim": "The bridge did not collapse in 2018.",
            "evidence": "Official inspection records confirm the bridge did not collapse in 2018.",
            "base_gold_hallucination": True,
            "variant_gold_hallucination": False,
            "expected_mechanism": "Epistemic Gate (NEGATED_FACT protection)",
        },
        {
            "pair_id": "PAIR_D_QUOTATION",
            "description": "Assertion vs Quoted/Debunked Meta-Claim",
            "base_claim": "The satellite crashed in 2022.",
            "variant_claim": "Reports falsely claimed that the satellite crashed in 2022.",
            "evidence": "Fact checkers debunked false reports claiming the satellite crashed in 2022.",
            "base_gold_hallucination": True,
            "variant_gold_hallucination": False,
            "expected_mechanism": "Epistemic Gate (QUOTED_CLAIM / META_CLAIM protection)",
        },
        {
            "pair_id": "PAIR_E_HYPOTHETICAL",
            "description": "Assertion vs Conditional Hypothetical",
            "base_claim": "Commercial fusion reached grid scale in 2038.",
            "variant_claim": "If commercial fusion reaches grid scale by 2038, carbon emissions will drop.",
            "evidence": "Energy projections estimate carbon reductions if fusion reaches scale in 2038.",
            "base_gold_hallucination": True,
            "variant_gold_hallucination": False,
            "expected_mechanism": "Temporal-Epistemic Gate (HYPOTHETICAL / CONDITIONAL protection)",
        },
        {
            "pair_id": "PAIR_F_COUNTERFACTUAL",
            "description": "Assertion vs Counterfactual Past",
            "base_claim": "Candidate A won the 2024 presidential election.",
            "variant_claim": "Had Candidate A won the 2024 election, economic policy would have differed.",
            "evidence": "Candidate B won the 2024 election. Analysts discussed counterfactual policy scenarios.",
            "base_gold_hallucination": True,
            "variant_gold_hallucination": False,
            "expected_mechanism": "Epistemic Gate (COUNTERFACTUAL protection)",
        },
    ]
    return pairs


def main():
    records = build_phase6d_dataset()
    dataset_file = DATA_DIR / "phase6d_adversarial_benchmark.json"
    with open(dataset_file, "w") as f:
        json.dump(records, f, indent=2)

    pos = sum(1 for r in records if r["gold_hallucination"])
    neg = len(records) - pos
    print(f"Saved Phase 6D Benchmark: {dataset_file}")
    print(f"Total: {len(records)} records | Positive: {pos} ({pos/len(records)*100:.1f}%) | Negative: {neg} ({neg/len(records)*100:.1f}%)")

    pairs = build_counterfactual_pairs()
    pairs_file = REPORTS_DIR / "phase6d_counterfactual_pairs.json"
    with open(pairs_file, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Saved Counterfactual Pairs: {pairs_file} ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
