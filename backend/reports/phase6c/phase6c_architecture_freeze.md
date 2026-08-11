# Phase 6C: Architecture Freeze Record

**Freeze Date**: 2026-08-10
**Frozen Commit SHA**: cbe4de7f72b7c874727e1025acf348219136ed60 (short: cbe4de7)
**Branch**: main
**Repository**: akashcodes23/HalluciSense

This document records the exact frozen architecture used for all Phase 6C
publication experiments. NO production logic may be modified during final
evaluation unless a correctness bug is discovered and documented per Section 7.

---

## 1. Production Fusion Weights (FROZEN)

```
alpha (P1 weight) = 0.40   # Factual Verification / Evidence Grounding
beta  (P2 weight) = 0.30   # Predictive Confidence / Uncertainty
gamma (P3 weight) = 0.30   # Self-Consistency
```

Sum = 1.00. Dynamic renormalization applies when P2 or P3 are unavailable.

## 2. Production Risk Thresholds (FROZEN)

```
VERIFIED             < 0.35
NEEDS_VERIFICATION   < 0.50
MODERATE_RISK        < 0.65
LIKELY_HALLUCINATED >= 0.65
```

## 3. Pillar 3 Unavailable Semantics (FROZEN)

```python
score     = None       # NOT 0.0
available = False
```

When P3 is unavailable, its weight is redistributed to P1 and P2 proportionally.
Fabricating 0.0 as a score is strictly prohibited.

---

## 4. Pillar Architecture

### Pillar 1 — Retrieval + NLI Factual Verification
**File**: `app/core/engine/pillar1_retrieval.py`
- Wikipedia retrieval via MediaWiki API (top-k snippets)
- Cross-Encoder NLI (facebook/bart-large-mnli) for entailment scoring
- Atomic claim extraction and per-claim scoring
- Integration with TemporalClaimEngine for modality-aware temporal checks
- EventTemporalAnchorResolver for relational temporal verification via Wikidata
- Global evidence-set alignment across top-k retrieved snippets

### Pillar 2 — Predictive Confidence / Uncertainty
**Instance**: `self.p2_engine` in `HallucinationDetectionPipeline`
- Token-level confidence gap analysis
- Uncertainty quantification for response tokens

### Pillar 3 — Semantic Self-Consistency
**Instance**: `self.p3_engine` in `HallucinationDetectionPipeline`
- Alternate generation consistency via multiple response sampling
- Returns `score=None, available=False` when LLM sampling is unavailable

---

## 5. TemporalClaimEngine (Phase 6)
**File**: `app/core/engine/temporal.py`

### Modality Classes
```
EpistemicModality:
  ASSERTED_FACT
  FUTURE_FACT_ASSERTION
  PREDICTION
  HYPOTHETICAL
  COUNTERFACTUAL
  CONDITIONAL
  NEGATED_FACT
  FICTIONAL
  QUOTED_CLAIM
  UNKNOWN
```

### Temporal Status Classes
```
TemporalStatus:
  PAST_FACT
  PRESENT_STATE
  FUTURE_IMPOSSIBLE_FACT
  FUTURE_PREDICTION
  HYPOTHETICAL
  COUNTERFACTUAL
  CONDITIONAL
  NEGATED_FACT
  FICTIONAL
  DATE_MISMATCH
  DATE_RANGE
  TIME_RELATIVE
  UNKNOWN
```

### Key Patterns (Structural Only — No Entity/Date Hardcoding)
- `PREDICTION_PATTERNS`: `\bwill\b`, `\bexpected\b.{0,80}\bto\b`, etc.
- `HYPOTHETICAL_PATTERNS`: `\bif\b.{0,40}\b(were|had|would)\b`, etc.
- `COUNTERFACTUAL_PATTERNS`: `\bhad\b.{0,60}\b(happened|occurred)\b`, etc.
- `FICTION_PATTERNS`: `\b(in the (novel|story|film))\b`, etc.
- `META_CLAIM_PATTERNS`: `\b(falsely|debunked|incorrectly|supposedly)\b`, etc.
- `RELATIONAL_PATTERNS`: `\b(before|after|during|since|prior to)\b`, etc.
- `NEGATION_PATTERNS`: `\b(did not|didn't|never|not)\b`, etc.

### Key Behaviors
- Query modality resolved INDEPENDENTLY from response modality
- Evidence date matching performed globally across full evidence set (not per-snippet)
- Non-assertion modalities (PREDICTION, HYPOTHETICAL, etc.) are PROTECTED from temporal penalty
- CURRENT_YEAR = 2026

---

## 6. EventTemporalAnchorResolver
**File**: `app/core/engine/pillar1_retrieval.py` (also `scripts/run_novelty_experiments.py`)
- Queries Wikidata API for named entity temporal anchors
- TIME_PROPERTIES: P585, P580, P582, P571, P576, P575
- MAX_ANCHORS = 2 per claim
- TIMEOUT_SECONDS = 0.8
- No entity names or dates are hardcoded

---

## 7. Change Control Protocol

If a correctness bug is discovered during Phase 6C evaluation:

1. **Document** the bug with evidence in `reports/phase6c/bug_report_YYYYMMDD.md`
2. **Assess** whether it invalidates previous experiments
3. **Create a new version** with incremented architecture suffix
4. **Re-run** all affected experiments using the corrected version
5. **Do NOT silently mix** results from different architecture versions
6. **Update** the experiment_manifest.json with the corrected SHA

---

## 8. What Is NOT Permitted During Phase 6C Evaluation

- Modifying production fusion weights (alpha, beta, gamma)
- Modifying risk thresholds
- Adding entity-specific rules or hardcoded dates
- Adding benchmark-specific regexes
- Tuning thresholds against final test set
- Silently rewriting historical reports
- Fabricating P3 scores when P3 is unavailable
