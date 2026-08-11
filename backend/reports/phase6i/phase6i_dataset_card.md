# Phase 6I Independent Benchmark Dataset Card

**Dataset File**: `data/external/phase6i_independent_benchmark.json`  
**Total Records (N)**: 500  
**SHA-256**: `f1866f286012080369a755c4f9457811d6a9b6d12414ef3d972f558684987bc7`  
**Independence Status**: PASS (Overlap count: 0)  

---

## Dataset Distribution Summary
- **Class Balance**: 300 Factual (60.0%) / 200 Hallucinated (40.0%) -> 500 total records
- **Multi-Claim Records**: 200 multi-claim responses (40.0%) / 300 single-claim responses (60.0%)
- **Domains (10)**: history, medicine, science, astronomy, technology, law, economics, climate, engineering, politics (50 records each)
- **Epistemic Categories (10)**: ASSERTED_FACT, PREDICTION, HYPOTHETICAL, CONDITIONAL, NEGATED_FACT, QUOTED_CLAIM, COUNTERFACTUAL, FICTIONAL, EVIDENCE_DATE_CONFOUNDING, MULTI_CLAIM_MIXED
- **Negative Controls (11)**: N0_Clean_Evidence, N1_Unrelated_Background_Dates, N2_Conflicting_Dates, N3_Irrelevant_Passages, N4_Entity_Distractors, N5_Multiple_Unrelated_Claims, N6_Future_Predictions, N7_Hypothetical_Statements, N8_Quoted_Claims, N9_Mixed_Modal_Response, N10_Multi_Claim_Mixed_Correctness
