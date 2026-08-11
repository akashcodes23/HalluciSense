# Phase 6D-A: Pre-Implementation Audit Report

**Date**: 2026-08-11  
**Target Architecture**: `app/core/engine/temporal.py` & `app/core/engine/pillar1_retrieval.py`  
**Purpose**: Systematically analyze current temporal-epistemic reasoning & global evidence-date alignment mechanics before formalizing and extending Phase 6D capabilities.

---

## 1. What Exactly Does the Current Temporal-Epistemic Gate Do?

The Temporal-Epistemic Gate in `TemporalClaimEngine` prevents non-assertion claims (such as future predictions, hypotheticals, counterfactuals, conditionals, quoted claims, fictional statements, and negated facts) from incurring temporal inconsistency penalties while retaining NLI factual grounding.

Specifically:
- It independently inspects the response/claim text for epistemic markers (`PREDICTION_PATTERNS`, `HYPOTHETICAL_PATTERNS`, `COUNTERFACTUAL_PATTERNS`, `FICTION_PATTERNS`, `META_CLAIM_PATTERNS`, `NEGATION_PATTERNS`).
- If a protected modality is detected, `analyze_claim()` returns `temporal_inconsistency_score = 0.0` and sets `protected_from_temporal_penalty = True`.
- In `pipeline.analyze()`, when a claim is protected, the overall claim score defaults to the NLI factual error score (Pillar 1) without adding temporal penalty penalties (such as `FUTURE_IMPOSSIBLE_FACT`=0.92 or `DATE_MISMATCH`=0.90).
- Crucially, it resolves query modality (`detect_query_modality`) and response modality (`detect_modality`) independently, ensuring that a user's hypothetical query (e.g. "What if...") does NOT accidentally protect an asserted factual response that contains a temporal hallucination.

---

## 2. What Exactly Does Global Evidence-Date Alignment Do?

Global Evidence-Date Alignment in `verify_evidence_date_mismatch()` prevents background dates in individual retrieved snippets from creating false temporal contradiction penalties.

Specifically:
- Instead of comparing a claim's year against each evidence snippet in isolation (where snippet A might contain an expedition date like 1957 and trigger a mismatch against a claim year like 1969), it extracts all 4-digit years from **all** retrieved evidence items into a global set `evidence_years`.
- If a claim year is found anywhere in `evidence_years` (global evidence support), the date is marked as supported and no mismatch penalty is applied.
- If a claim year is NOT in `evidence_years`, it checks for a minimum 3-year discrepancy against relevant snippet dates (matching shared lexical keywords) before assigning a `DATE_MISMATCH` score of 0.90.
- Comparative and relational temporal patterns (e.g., "before", "after", "since", "prior to") suppress naive year-matching mismatch logic entirely.

---

## 3. Which Code Paths Implement Each Mechanism?

| Mechanism | Main Class / Function | Secondary Dependencies |
|:---|:---|:---|
| **Query Modality Resolution** | `TemporalClaimEngine.detect_query_modality()` ([temporal.py:158](file:///Users/akashgpatil/major_project/backend/app/core/engine/temporal.py#L158)) | Pattern match lists |
| **Response Modality Resolution** | `TemporalClaimEngine.detect_modality()` ([temporal.py:175](file:///Users/akashgpatil/major_project/backend/app/core/engine/temporal.py#L175)) | Independent claim-text pattern matching |
| **Epistemic Gate Penalty Suppression** | `TemporalClaimEngine.analyze_claim()` ([temporal.py:285-312](file:///Users/akashgpatil/major_project/backend/app/core/engine/temporal.py#L285-L312)) | `protected_modalities` set |
| **Global Evidence-Date Alignment** | `TemporalClaimEngine.verify_evidence_date_mismatch()` ([temporal.py:205-266](file:///Users/akashgpatil/major_project/backend/app/core/engine/temporal.py#L205-L266)) | Global `evidence_years` aggregation |
| **Dynamic Event Anchoring** | `EventTemporalAnchorResolver` ([pillar1_retrieval.py:11-180](file:///Users/akashgpatil/major_project/backend/app/core/engine/pillar1_retrieval.py#L11-L180)) | Wikidata API lookup (`wbsearchentities` / `wbgetentities`) |

---

## 4. Which Parts Are Heuristic?

The current implementation contains several heuristic components:
1. **Regex Pattern Matching**: Modality patterns (e.g. `\bwill\b`, `\bexpected\b.{0,80}\bto\b`, `\bwhat if\b`) rely on surface keyword/regex presence rather than full semantic constituency parsing.
2. **Fixed Penalty Scores**: Hardcoded penalty scores (`FUTURE_IMPOSSIBLE_FACT` = 0.92, `DATE_MISMATCH` = 0.90).
3. **Keyword Overlap Thresholds**: In `verify_evidence_date_mismatch()`, filtering shared words using a hardcoded set of 10 stopwords (`"first", "second", "states"...`) and checking `len(filtered_common) >= 1`.
4. **4-Digit Year Extraction**: `YEAR_PATTERN` (`\b(1\d{3}|20\d{2}|2100)\b`) only extracts explicit 4-digit years between 1000 and 2100; relative time expressions like "3 years ago" or month/day dates without years are not parsed into temporal intervals.
5. **Wikidata Candidate Extraction**: Capitalized multi-word regex heuristics (`\b[A-Z][A-Za-z0-9'’-]+...\b`) for entity anchor extraction.

---

## 5. Which Parts Are Deterministic?

1. **Epistemic Modality Classification**: Given the same text string, `detect_modality()` and `detect_query_modality()` return identical `EpistemicModality` enum values 100% deterministically.
2. **Temporal Status Mapping**: Enum mapping and temporal score assignment (`analyze_claim`) are purely deterministic logic trees without random seeds or LLM calls.
3. **Global Date Set Union**: Aggregating `evidence_years = set(...)` and checking `claim_year in evidence_years` is fully deterministic.
4. **Pipeline Scoring Integration**: Combining P1 NLI score and temporal inconsistency score is fully deterministic.

---

## 6. Which Parts Depend on Retrieval?

1. **Global Evidence-Date Alignment**: Completely dependent on `provided_evidence` or retrieved evidence items. If no evidence items are provided (`evidence_items=None`), `verify_evidence_date_mismatch()` returns `None`.
2. **EventTemporalAnchorResolver**: Depends on network connectivity to Wikidata (`https://www.wikidata.org/w/api.php`). If offline or timed out (>0.8s), it falls back to 0 lookups and returns `None` (no anchor penalty).
3. **Factual Grounding (Pillar 1 NLI)**: Evaluates claim statements against snippets retrieved from Wikipedia or external search.

---

## 7. Where Can False Positives Still Occur?

1. **Unrecognized Assertion Modality**: If a prediction or hypothetical is expressed without standard pattern keywords (e.g. *"In all likelihood the project reaches fruition next decade"*), it will be misclassified as `ASSERTED_FACT` and penalised as a future impossible fact if a future year is mentioned.
2. **Unmatched Background Dates in Single Snippet**: If evidence contains a background date that is NOT in `evidence_years` for the specific claim, and keyword overlap passes (e.g. sharing general words like "company", "launched"), a false `DATE_MISMATCH` (0.90 penalty) can still occur.
3. **Complex Relational Constraints**: Expressions like *"X occurred 5 years before Y"* where Y=2010 (meaning X=2005) are currently bypassed by `RELATIONAL_PATTERNS` (returning score 0.0), but if relational patterns fail to trigger, a false mismatch can occur against 2010.

---

## 8. Where Can False Negatives Still Occur?

1. **Over-Protection of Assertions**: If a hallucinatory factual assertion contains an incidental modal word (e.g. *"The company will have built its headquarters in 1850"* matching `\bwill\b`), the epistemic gate will classify it as `PREDICTION` and set temporal penalty to 0.0, allowing a false negation of a temporal hallucination.
2. **Global Date Spurious Support**: If evidence contains year 1995 for Entity A, and the claim hallucinates year 1995 for Entity B in the same response, global evidence alignment sees `1995 in evidence_years` and marks it as supported, missing the cross-entity date swap hallucination.
3. **Subtle Relational Contradictions**: Statements like *"X was born in 1985 and graduated college in 1980"* currently pass temporal checks if no evidence contradicts 1985/1980 individually, because internal claim-level interval consistency (1985 > 1980) is not verified.

---

## 9. What Does Current Evidence Actually Establish?

1. **Exact Metric Reproducibility**: Metric consistency tests confirm that all Phase 6B confusion matrices (A0–A9) are 100% reproducible and mathematically exact.
2. **Mechanistic Functionality**: Targeted unit/adversarial tests (`test_phase6_architecture.py`, `test_temporal_benchmark.py`, `test_temporal_holdout.py`) prove that:
   - Modality protection correctly suppresses temporal penalties for predictions, hypotheticals, and fiction.
   - Query modality does not contaminate response modality.
   - Global evidence date alignment prevents false positives when a matching year is present anywhere in the evidence set.
3. **External Benchmark Equivalence**: On standard QA datasets (HaluBench, RAGTruth, HaluEval, N=550), M9 Full HalluciSense performs comparably to a pure NLI baseline (F1 0.5467 vs 0.5500, McNemar $p = 0.2864$, non-significant).

---

## 10. What Does It NOT Establish?

1. **General Benchmark Superiority**: Evidence does NOT establish that HalluciSense outperforms NLI baselines on standard, non-temporal QA benchmarks (59.3% of external records have no temporal expressions, 0% have future assertions).
2. **Statistical Significance on Noise Tests**: The previous Phase 6B evidence noise stress test (N=5) was a synthetic mechanistic demo with a strawman Phase 5 simulation; it does NOT establish statistical noise robustness.
3. **Deep Semantic Modality Understanding**: Regex-based modality detection does NOT establish deep linguistic or semantic understanding of complex sentences with nested clauses.
4. **Multi-Pillar Synergy**: Offline evaluations do NOT establish Pillar 2/3 performance, as LLM confidence gap and consistency evaluation are disabled offline.
