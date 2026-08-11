# Phase 6D: Adversarial Temporal-Epistemic Dataset Card

**Dataset Name**: `phase6d_adversarial_benchmark.json`  
**Location**: `backend/data/external/phase6d_adversarial_benchmark.json`  
**Generated**: 2026-08-11  
**Total Records (N)**: 440  
**Class Distribution**: 220 Hallucinated (50.0%) / 220 Factual (50.0%) — Exact 50/50 Balance  
**Domains Covered**: 10 domains (history, medicine, science, astronomy, technology, law, politics, economics, climate, engineering)  

---

## Purpose & Scope

The `phase6d_adversarial_benchmark` dataset is a controlled, high-coverage research benchmark designed to evaluate temporal hallucination verification and epistemic modality protection. Unlike standard QA benchmarks (where 59.3% of examples contain no temporal content and 0% contain future predictions), this dataset specifically tests claims under temporal, relational, modal, and evidence-noise conditions.

---

## Category Distribution (20 Categories, N=22 per category)

| Category ID | Category Name | Modality | Hallucinated N | Factual N | Total N |
|:---|:---|:---:|:---:|:---:|:---:|
| CAT01 | `correct_historical_assertion` | ASSERTED_FACT | 0 | 11 | 11 |
| CAT02 | `incorrect_historical_assertion` | ASSERTED_FACT | 11 | 0 | 11 |
| CAT03 | `correct_future_prediction` | PREDICTION | 0 | 11 | 11 |
| CAT04 | `incorrect_future_prediction` | FUTURE_FACT_ASSERTION | 11 | 0 | 11 |
| CAT05 | `hypothetical_future` | HYPOTHETICAL | 0 | 11 | 11 |
| CAT06 | `counterfactual_historical` | COUNTERFACTUAL | 0 | 11 | 11 |
| CAT07 | `conditional_temporal_statement` | CONDITIONAL | 0 | 11 | 11 |
| CAT08 | `negated_temporal_assertion` | NEGATED_FACT | 0 | 11 | 11 |
| CAT09 | `quoted_false_statement` | QUOTED_CLAIM | 0 | 11 | 11 |
| CAT10 | `meta_claim_debunking` | QUOTED_CLAIM | 0 | 11 | 11 |
| CAT11 | `fictional_temporal_statement` | FICTIONAL | 0 | 11 | 11 |
| CAT12 | `multi_event_evidence` | ASSERTED_FACT | 11 | 11 | 22 |
| CAT13 | `evidence_irrelevant_dates` | ASSERTED_FACT | 11 | 11 | 22 |
| CAT14 | `evidence_conflicting_dates` | ASSERTED_FACT | 11 | 11 | 22 |
| CAT15 | `evidence_unrelated_entity_dates` | ASSERTED_FACT | 11 | 11 | 22 |
| CAT16 | `relative_temporal_expressions` | ASSERTED_FACT | 11 | 11 | 22 |
| CAT17 | `event_ordering` | ASSERTED_FACT | 11 | 11 | 22 |
| CAT18 | `date_range_contradiction` | ASSERTED_FACT | 11 | 11 | 22 |
| CAT19 | `multi_hop_temporal_relation` | ASSERTED_FACT | 11 | 11 | 22 |
| CAT20 | `adversarial_evidence_ordering` | ASSERTED_FACT | 11 | 11 | 22 |
| **TOTAL** | — | — | **220 (50%)** | **220 (50%)** | **440** |

---

## Controlled Counterfactual Pairs (N=6 Pairs)

| Pair ID | Base Claim | Variant Claim | Expected Mechanism Tested |
|:---|:---|:---|:---|
| `PAIR_A` | Apollo 11 landed in 1969. | Apollo 11 landed in 1975. | Global Evidence Alignment |
| `PAIR_B` | Artemis IV landed in 2030 (assertion). | Artemis IV is targeted to land in 2030 (prediction). | Epistemic Gate (PREDICTION) |
| `PAIR_C` | Bridge collapsed in 2018. | Bridge did not collapse in 2018. | Epistemic Gate (NEGATED_FACT) |
| `PAIR_D` | Satellite crashed in 2022. | Reports falsely claimed satellite crashed in 2022. | Epistemic Gate (QUOTED_CLAIM) |
| `PAIR_E` | Fusion reached grid scale in 2038. | If fusion reaches grid scale in 2038, emissions drop. | Epistemic Gate (HYPOTHETICAL) |
| `PAIR_F` | Candidate A won 2024 election. | Had Candidate A won 2024 election, policies would differ. | Epistemic Gate (COUNTERFACTUAL) |

---

## Schema Definition

```json
{
  "example_id": "P6D_0001",
  "domain": "history",
  "category": "correct_historical_assertion",
  "query": "When occurred the moon landing mission of Apollo 11?",
  "response": "The moon landing mission of Apollo 11 occurred in 1969.",
  "context": "Historical record confirms Apollo 11 moon landing mission took place in 1969.",
  "gold_hallucination": false,
  "temporal_signal": "historical",
  "query_modality": "ASSERTED_FACT",
  "response_modality": "ASSERTED_FACT",
  "temporal_relation": "point_date",
  "adversarial_type": "correct_historical_assertion",
  "evidence_noise_type": "structured"
}
```
