# Phase 5 Benchmark Integrity & Audit Report

## 1. Executive Summary
This report presents a comprehensive benchmark integrity audit of all historical, temporal, ablation, and adversarial evaluation datasets across HalluciSense (Phase 1–Phase 4).

The audit evaluated 5 primary benchmark datasets:
1. **Phase 1 A–E Benchmark** (5 cases)
2. **Phase 1 7-Way Ablation Benchmark** (Ablation grid across 5 baseline cases)
3. **Phase 2 Temporal Benchmark** (`scripts/benchmark_temporal.py`, 20 cases)
4. **Phase 3 Temporal Generalization Benchmark** (`scripts/benchmark_temporal_generalization.py`, 55 cases)
5. **Phase 4 Adversarial Modality Benchmark** (`scripts/benchmark_temporal_adversarial.py`, 40 cases)

---

## 2. Architecture & Pipeline Dependency Map (Step 1)

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 User API / Engine Input                │
                  │              (text: Response, query: Query)            │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │          Claim Extraction & Sentence Segmentation      │
                  │   Extracts atomic claims & 4-digit years (1000–2100)  │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │               Pillar 1 Retrieval Engine                │
                  │  - Hybrid Vector/BM25 Retrieval                        │
                  │  - CrossEncoder NLI Factual Error Calculation          │
                  │  - Context-Aware TemporalClaimEngine Execution         │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
        [Temporal Penalty Active]                           [Protected / No Penalty]
     (Future Fact Assertion / Mismatch)                  (Prediction, Hypothetical, Conditional,
     FE = max(NLI_FE, Temporal_Score)                     Counterfactual, Negation, Fiction)
                                                          FE = Base NLI Factual Error
                    │                                                   │
                    └─────────────────────────┬─────────────────────────┘
                                              │
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │            Pillar 1 Factual Error (P1 Score)           │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
         [Pillar 1 Score (0.40)]   [Pillar 2 Conf (0.30)]    [Pillar 3 Cons (0.30)]
         Factual Error / Grounding  Entropy / Logprob Gap   Multi-sample NLI Paraphrase
                    │                         │                         │
                    └─────────────────────────┼─────────────────────────┘
                                              │
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │           Dynamic Three-Pillar Weight Renormalization  │
                  │           (Handles null/unavailable pillars safely)    │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │               Fused Hallucination Score (H)            │
                  │     H = alpha * P1 + beta * P2 + gamma * P3 (sums to 1)│
                  └───────────────────────────┬────────────────────────────┘
                                              │
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │              Risk Level Classification                 │
                  │ VERIFIED (<0.35) | NEEDS_VERIFICATION (<0.50)           │
                  │ MODERATE_RISK (<0.65) | LIKELY_HALLUCINATED (>=0.65)   │
                  └───────────────────────────┬────────────────────────────┘
```

---

## 3. Benchmark Integrity Findings & Discovered Vulnerabilities (Step 2)

| Issue ID | Benchmark Cases / Range | Description of Vulnerability | Severity | Optimistic Bias Risk | Proposed Remediation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **INT-01** | `T01`–`T06` vs `G01`, `G07`, `G24`, `G30`, `G34`, `G37` | **Cross-Benchmark Duplication**: 6 cases in Phase 3 are verbatim copies of Phase 2 cases (e.g. FIFA 2022/2027, Brazil 2030, France 2022). | **HIGH** | Over-inflates Phase 3 metrics by re-testing memorized Phase 2 samples. | Exclude Phase 2 duplicate strings from new holdout benchmarks. |
| **INT-02** | `G01`–`G55` vs `ADV01`–`ADV40` | **Cross-Phase Duplication**: 14 cases in Phase 4 are identical or near-identical rephrasings of Phase 3 cases (`ADV01`= `G48`, `ADV13` = `G34`, `ADV29` = `G37`, `ADV33` = `G16`). | **HIGH** | Artificially inflates Phase 4 adversarial benchmark accuracy ($95\%$). | Construct a completely isolated blind holdout dataset (`benchmark_temporal_holdout.py`). |
| **INT-03** | `T04`, `G30`, `ADV09` | **Implementation Rule Overfitting**: Keyword markers like `"Suppose Brazil wins"`, `"Imagine Apple buys"`, `"In the sci-fi story"` match keyword tuples directly in `temporal.py`. | **HIGH** | Model matches explicit keyword strings without full clause parsing. | Add non-keyword conditional/hypothetical structures in holdout dataset. |
| **INT-04** | `G07`–`G15`, `ADV01`–`ADV04` | **Trivial Date Heuristic**: All future factual assertions use years $> 2026$ (e.g. 2027, 2029, 2032, 2035). | **MEDIUM** | Year $> 2026$ acts as a trivial binary signal for future assertions when un-protected. | Include historical claims with future-sounding phrasing and relative dates. |
| **INT-05** | `G01`–`G55`, `ADV01`–`ADV40` | **Entity Over-Representation**: Over-reliance on entities: "FIFA World Cup" (14 cases), "iPhone" (6 cases), "Einstein" (5 cases), "George Washington" (4 cases). | **MEDIUM** | Domain bias towards sports, tech, and US history. | Broaden holdout domain distribution across medicine, climate, law, engineering, and astronomy. |
| **INT-06** | `test_temporal_benchmark.py` | **Unit Test Leakage**: Unit tests check exact benchmark strings (`"Brazil won the 2027 FIFA World Cup"`, `"George Washington was elected... in 2004"`). | **MEDIUM** | Risk of developer over-tuning code specifically for test case assertions. | Ensure holdout benchmark uses novel entities and sentences not found in unit tests. |

---

## 4. Remediation Plan & Holdout Strategy
To resolve all identified benchmark vulnerabilities, Phase 5 introduces a **Blind Holdout Benchmark** (`scripts/benchmark_temporal_holdout.py`) with:
1. **70 Completely Novel Test Cases** (0% overlap with Phase 2, 3, or 4).
2. **15 Categories (A–O)** including implied temporal contradictions without explicit years, relative temporal expressions, and multi-event ordering.
3. **13 Broad Domains** (medicine, climate, astronomy, law, engineering, economics, business, history, sports, technology, geography, entertainment, politics).
4. **Zero Hardcoded Entity Rules**: Evaluation must generalize purely by linguistic/semantic structure and factual retrieval.
