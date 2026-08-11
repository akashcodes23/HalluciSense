# Phase 6C: Final Novelty Position

**Generated**: 2026-08-11
**Status**: Post-falsification, post-audit
**Based on**: Ablation audit, capability coverage, metric reconciliation, falsification report

---

## Executive Summary

After rigorous falsification, the defensible novel contribution of HalluciSense
is narrower than initially stated in Phase 6B, but genuine and identifiable.

The core contribution is NOT:
- "A new hallucination detection system" (many exist)
- "Integration of multiple signals" (common in literature)
- "High accuracy on benchmarks" (not demonstrated vs. baselines)

The core contribution IS:
- A specific operational mechanism for preventing false-positive hallucination
  verdicts caused by temporal expressions in non-asserted epistemic contexts

---

## Contribution 1: Temporal-Epistemic Gate

### Claim
A modality-aware temporal verification framework where the temporal inconsistency
signal is conditioned on the independently resolved epistemic modality of both
the query and the response. Non-assertion modalities (predictions, hypotheticals,
conditionals, counterfactuals, fiction, quotations) are exempted from temporal
penalty while retaining factual NLI grounding.

### Prior Art
- Temporal QA systems (Dhingra 2022, Zhao 2021): Do not model epistemic modality.
- Factuality annotators (Saurí 2012): Model factuality but not as operational gates in hallucination detection pipelines.
- NLI hallucination detectors (Honovich 2022, Laban 2022): Do not condition temporal signals on modality.

### Difference
No prior hallucination detection system found that explicitly conditions temporal
inconsistency verification on independently resolved epistemic modality of query
and response, creating a two-sided modality gate.

### Experimental Evidence
- Phase 6C Temporal Adversarial Benchmark (6C-K): 9 epistemic categories evaluated
- Phase 6B Stress Test (N=5, mechanistic only): Demonstrated FP reduction for future predictions
- Phase 6C Evidence Corruption (6C-J): Modality-Conflict condition tested

### Limitation
- Pattern-based modality detection (regex) is an approximation, not deep semantic understanding
- External benchmark has 0% future-year examples — primary benefit untestable on standard datasets
- Statistical evidence is limited to targeted adversarial benchmarks (small N per category)

### Confidence: MODERATE

---

## Contribution 2: Global Evidence-Set Temporal Alignment

### Claim
A date-aware evidence alignment mechanism that cross-references temporal expressions
in claims against the full retrieved evidence set, preventing temporal penalty from
being triggered by background dates (e.g., expedition dates, historical context)
that are temporally adjacent but semantically unrelated to the specific claim.

### Prior Art
- FEVER/fact-checking systems: Use sentence-level evidence without date-alignment
- RAG systems: Retrieve and aggregate evidence but do not distinguish claim-relevant vs. background dates

### Difference
The specific mechanism of distinguishing claim-relevant dates from background dates
within a retrieved evidence set, using a global alignment pass before per-claim
temporal scoring, is not found in the hallucination detection literature reviewed.

### Experimental Evidence
- Phase 6C Evidence Corruption conditions N1–N4 directly test this
- Phase 6B Stress Test E2–E4 (mechanistic, N=5)

### Limitation
- Implementation depends on `verify_evidence_date_mismatch` which requires explicit date expressions
- On datasets without temporal content (59.3% of external benchmark), this mechanism is inactive
- The evidence corruption benchmark uses synthetic cases

### Confidence: MODERATE

---

## Contribution 3: Interpretable Deterministic Risk Fusion

### Claim
A deterministic, interpretable multi-signal fusion framework that combines factual
grounding (P1), predictive uncertainty (P2), and self-consistency (P3) with fixed,
auditable weights and explicit pillar-unavailability semantics that prevent fabrication
of scores from unavailable pillars.

### Prior Art
- Multi-signal hallucination detection: Common approach (HADES, SelfCheckGPT)
- Fixed-weight fusion: Straightforward architecture choice
- Score fabrication prevention: Not emphasized in prior work but not novel per se

### Difference
The explicit design principle of using `score=None, available=False` for unavailable
pillars (rather than defaulting to 0.0) is a correctness guarantee. Its primary
value is engineering correctness, not research novelty.

### Experimental Evidence
- Architectural invariant — not tested separately
- Test suite enforces this guarantee

### Confidence: LOW (engineering correctness, not research novelty)

---

## What Should Be Omitted From Novelty Claims

The following should NOT be presented as research contributions:

1. **High accuracy on external benchmark**: Phase 6C M9 (full system) does not
   outperform M0 (NLI baseline) in F1 or MCC on the 550-case external benchmark.
   Claiming superior performance would be factually incorrect.

2. **"First system to combine NLI + temporal reasoning"**: This combination appears
   in prior work (e.g., temporal QA systems).

3. **"100% robustness on evidence noise"**: The stress test N=5, synthetic, strawman
   Phase 5. Not a valid statistical claim.

4. **"Validated on three external benchmarks"**: These benchmarks do not test the
   primary capabilities. Framing as "external validation" is overstated.

---

## Recommended Novelty Statement for Publication

**Title framing**: 
"Modality-Aware Temporal Verification for Hallucination Detection in LLM Responses"

**Abstract claim** (defensible):
> "We introduce a temporal-epistemic gate mechanism that conditions temporal
> inconsistency verification on the independently resolved epistemic modality of
> both the claim and the verification query. This prevents false-positive hallucination
> verdicts for valid predictions, hypotheticals, and conditionals that contain future
> temporal expressions — a systematic failure mode of naive temporal hallucination
> detectors. We combine this with global evidence-set date alignment to prevent
> background temporal references from contaminating claim-specific verification.
> On a targeted adversarial benchmark spanning 9 epistemic categories, the system
> reduces false-positive rates for non-assertion modalities while maintaining
> recall for factual hallucinations. On general-purpose external benchmarks
> (HaluBench, RAGTruth, HaluEval), where 59.3% of examples contain no temporal
> signal, the system performs comparably to an NLI baseline, consistent with the
> expectation that temporal components are inactive on non-temporal content."

---

## Publication Readiness by Contribution

| Contribution | Evidence Quality | Suitable For Publication? |
|:---|:---:|:---:|
| Temporal-epistemic gate mechanism | Moderate | YES — with scope disclosure |
| Global evidence-date alignment | Moderate | YES — with N disclosure |
| Interpretable risk fusion | Low (engineering) | As design feature only, not contribution |
| External benchmark superiority | Negative | NO — do not claim |
| Robustness to evidence noise | Mechanistic only | YES — as illustrative example, not claim |
