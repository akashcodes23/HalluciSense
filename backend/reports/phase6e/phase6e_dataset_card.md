# Phase 6E Independent Benchmark Dataset Card

**Dataset File**: `data/external/phase6e_independent_benchmark.json`  
**Total Records (N)**: 600  
**SHA-256**: `5909421a279ee8d48b2883fec9acdfbcc3be9285b8fbaada3317de81ba4afa30`  
**Independence Status**: PASS (Overlap count: 0)  

---

## Dataset Distribution Summary
- **Class Balance**: 300 Factual (50.0%) / 300 Hallucinated (50.0%)
- **Domains (10)**: history, medicine, science, astronomy, technology, law, economics, climate, engineering, politics (60 records each)
- **Epistemic Categories (10)**: ASSERTED_FACT, PREDICTION, HYPOTHETICAL, CONDITIONAL, NEGATED_FACT, QUOTED_CLAIM, COUNTERFACTUAL, TEMPORAL_CONTRADICTION, EVIDENCE_DATE_CONFOUNDING, NO_TEMPORAL_CONTROL (60 records each)
- **Noise Categories (9)**: N0_Clean_Evidence, N1_Irrelevant_Dates, N2_Historical_Background_Dates, N3_Multiple_Candidate_Years, N4_Conflicting_Dates_Across_Passages, N5_Correct_Date_Buried, N6_Irrelevant_Temporal_Anchors, N7_Missing_Temporal_Evidence, N8_Mixed_Relevant_Irrelevant
- **Construction Method**: Expert independent template generation with strict phrase-level independence verification.
