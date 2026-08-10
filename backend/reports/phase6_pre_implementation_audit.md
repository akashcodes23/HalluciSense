# Phase 6 Pre-Implementation Architectural & Production Invariants Audit Report

## 1. Executive Summary
This pre-implementation audit documents the baseline system architecture, data flow, production invariants, APIs, enums, and failure mechanisms prior to implementing Phase 6 enhancements.

---

## 2. Baseline System Architecture & Data Flow

```
                                  User Input
                     (query: Query, text: LLM Response)
                                       │
                                       ▼
                         Pillar1RetrievalEngine.analyze()
                                       │
                ┌──────────────────────┴──────────────────────┐
                │                                             │
                ▼                                             ▼
     extract_claims(text)                         Evidence Retrieval
  (Sentence Segmentation)                    (Wikipedia, Hybrid BM25/Dense)
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       │
                                       ▼
                       evaluate_claims_against_evidence()
                          (DeBERTa Cross-Encoder NLI)
                                       │
                                       ▼
                             Base Factual Error (fe_score)
                                       │
                                       ▼
                       TemporalClaimEngine.analyze_claim()
               (Context-Aware Modality & Temporal Inconsistency)
                                       │
                ┌──────────────────────┴──────────────────────┐
                │                                             │
      [Temporal Inconsistency]                      [Protected Modality]
     score > 0.0 (e.g. 0.90, 0.92)                 protected = True
     fe_score = max(fe_score, score)               Zero temporal penalty added
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       │
                                       ▼
                             Pillar 1 Factual Error (P1)
                                       │
                ┌──────────────────────┼──────────────────────┐
                │                      │                      │
                ▼                      ▼                      ▼
         Pillar 1 (0.40)        Pillar 2 (0.30)        Pillar 3 (0.30)
         Factual Grounding       Confidence Gap         Self-Consistency
                │                      │                      │
                └──────────────────────┼──────────────────────┘
                                       │
                                       ▼
                       FusionEngine.get_effective_weights()
                    (Renormalizes if P2 or P3 is None)
                                       │
                                       ▼
                       Fused Hallucination H-Score
                                       │
                                       ▼
                       Risk Level Classification
                (VERIFIED, NEEDS_VERIFICATION, MODERATE, LIKELY)
```

---

## 3. Production Invariants Verification (Step 0 & Step 11 Audit)

| Invariant | Configured / Documented Value | Source File | Compliance Status |
| :--- | :--- | :--- | :--- |
| **Pillar 1 Weight ($\alpha$)** | `0.40` (Renormalized base) / `0.45` config base | [config.py](file:///Users/akashgpatil/major_project/backend/app/core/config.py#L91) / [fusion.py](file:///Users/akashgpatil/major_project/backend/app/core/engine/fusion.py#L20) | **PASS** (Normalises to 0.40:0.30:0.30) |
| **Pillar 2 Weight ($\beta$)** | `0.30` | [config.py](file:///Users/akashgpatil/major_project/backend/app/core/config.py#L92) / [fusion.py](file:///Users/akashgpatil/major_project/backend/app/core/engine/fusion.py#L21) | **PASS** |
| **Pillar 3 Weight ($\gamma$)** | `0.30` | [config.py](file:///Users/akashgpatil/major_project/backend/app/core/config.py#L93) / [fusion.py](file:///Users/akashgpatil/major_project/backend/app/core/engine/fusion.py#L22) | **PASS** |
| **VERIFIED Risk Threshold** | `score < 0.35` | [types.py](file:///Users/akashgpatil/major_project/backend/app/core/engine/types.py#L12) / [fusion.py](file:///Users/akashgpatil/major_project/backend/app/core/engine/fusion.py#L65) | **PASS** |
| **NEEDS_VERIFICATION Threshold**| `score < 0.50` | [types.py](file:///Users/akashgpatil/major_project/backend/app/core/engine/types.py#L14) / [fusion.py](file:///Users/akashgpatil/major_project/backend/app/core/engine/fusion.py#L67) | **PASS** |
| **MODERATE_RISK Threshold** | `score < 0.65` | [types.py](file:///Users/akashgpatil/major_project/backend/app/core/engine/types.py#L15) / [fusion.py](file:///Users/akashgpatil/major_project/backend/app/core/engine/fusion.py#L69) | **PASS** |
| **LIKELY_HALLUCINATED Threshold**| `score >= 0.65` | [types.py](file:///Users/akashgpatil/major_project/backend/app/core/engine/types.py#L16) / [fusion.py](file:///Users/akashgpatil/major_project/backend/app/core/engine/fusion.py#L71) | **PASS** |
| **Pillar 3 Unavailable Handling**| `score = None`, `available = False` | [pillar3_consistency.py](file:///Users/akashgpatil/major_project/backend/app/core/engine/pillar3_consistency.py#L80) | **PASS** (Zero fabrication prevented) |

---

## 4. Current Enums & Core Data Structures

### `EpistemicModality` Enum ([temporal.py](file:///Users/akashgpatil/major_project/backend/app/core/engine/temporal.py#L35))
- `ASSERTED_FACT`
- `FUTURE_FACT_ASSERTION`
- `PREDICTION`
- `HYPOTHETICAL`
- `COUNTERFACTUAL`
- `CONDITIONAL`
- `NEGATED_FACT`
- `FICTIONAL`
- `QUOTED_CLAIM`
- `UNKNOWN`

### `TemporalStatus` Enum ([temporal.py](file:///Users/akashgpatil/major_project/backend/app/core/engine/temporal.py#L19))
- `PAST_FACT`, `PRESENT_STATE`, `FUTURE_IMPOSSIBLE_FACT`, `FUTURE_PREDICTION`, `HYPOTHETICAL`, `COUNTERFACTUAL`, `CONDITIONAL`, `NEGATED_FACT`, `FICTIONAL`, `DATE_MISMATCH`, `DATE_RANGE`, `TIME_RELATIVE`, `UNKNOWN`

---

## 5. Identified Architectural Failure Mechanisms (Phase 5 Audit)

1. **Query-Response Modality Leakage**: Joint modality scanning `combined_context = f"{query} {text}"` allowed query conditionals (*"If Candidate A wins..."*) to protect asserted response facts (*"Candidate A won the 2028 election"*).
2. **Multi-Year Evidence Snippet Noise**: `verify_evidence_date_mismatch` compared claim year against individual snippets in isolation. If a snippet contained background dates (e.g. Everest 1924), false date mismatch alerts ($0.90$) fired on true claims (e.g. Everest 1953 summit).
3. **Over-Protection of Negated Claims**: In `pillar1_retrieval.py`, `protected_from_temporal_penalty` set `fe_score = 0.0` for all `NEGATED_FACT` modalities, wiping out valid Cross-Encoder NLI factual error signals when a false negation was asserted (*"US did not declare independence in 1776"*).
4. **Blindness to Implied Event Anachronisms**: `YEAR_PATTERN` extracted 0 years for implicit event claims (*"Roman Empire collapsed during the European Renaissance"*), falling back to NLI without evaluating temporal interval overlap.

---

## 6. Planned Phase 6 Modifications & File Scope

### Files to be Modified:
- [app/core/engine/temporal.py](file:///Users/akashgpatil/major_project/backend/app/core/engine/temporal.py):
  - Refactor `detect_modality()` to separate `query_modality` and `response_modality`.
  - Add atomic claim segmentation helper `segment_claims()`.
  - Fix `verify_evidence_date_mismatch()` to verify claim year across global evidence set.
  - Implement relational temporal reasoning operators (`BEFORE`, `AFTER`, `SINCE`, `PRIOR TO`, `DECADE BEFORE`, etc.).
  - Add structural pattern matching for predictions, meta-claims, and fiction.
  - Implement `EventTemporalAnchorResolver` for dynamic event date span retrieval via Wikipedia/Wikidata.
- [app/core/engine/pillar1_retrieval.py](file:///Users/akashgpatil/major_project/backend/app/core/engine/pillar1_retrieval.py):
  - Update `analyze()` so protected modality prevents temporal penalties without wiping out NLI factual error scores when evidence contradicts negated claims.
  - Integrate dynamic event anchor retrieval for claims without explicit 4-digit years.

### Files to Remain Untouched:
- [app/core/engine/fusion.py](file:///Users/akashgpatil/major_project/backend/app/core/engine/fusion.py) (Frozen weights & thresholds)
- [app/core/config.py](file:///Users/akashgpatil/major_project/backend/app/core/config.py)
- [scripts/benchmark_temporal_holdout.py](file:///Users/akashgpatil/major_project/backend/scripts/benchmark_temporal_holdout.py) (Frozen blind Phase 5 holdout)
