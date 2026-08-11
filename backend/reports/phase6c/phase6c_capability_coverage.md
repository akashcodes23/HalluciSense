# Phase 6C: Dataset Capability Coverage Audit

**Generated**: 2026-08-11
**Sampled**: 300 records (100 per dataset: HaluBench, RAGTruth, HaluEval)
**Method**: TemporalClaimEngine modality classification + regex keyword analysis

---

## Critical Findings

**59.3% of the combined external benchmark has NO temporal signal.**
The external benchmark (HaluBench + RAGTruth + HaluEval) was NOT designed
to evaluate temporal or epistemic reasoning. It primarily tests factual grounding.

**0% of records contain future years (2025+).**
The core Phase 5→6 improvement (handling future predictions) is NOT
testable on this benchmark.

---

## Modality Distribution (N=300 sampled)

| Modality | Count | % | Notes |
|:---|---:|---:|:---|
| ASSERTED_FACT | 250 | **83.3%** | Dominant. Pure NLI baseline is directly applicable. |
| QUOTED_CLAIM | 19 | 6.3% | Partial. Phase 6 meta-claim handling applicable. |
| PREDICTION | 19 | 6.3% | Partial. Phase 6 modality protection applicable. |
| CONDITIONAL | 5 | 1.7% | Very sparse. Not statistically testable. |
| COUNTERFACTUAL | 3 | 1.0% | Very sparse. Not statistically testable. |
| NEGATED_FACT | 2 | 0.7% | Very sparse. Not statistically testable. |
| FICTIONAL | 2 | 0.7% | Very sparse. Not statistically testable. |

**Consequence**: 83.3% of records are ASSERTED_FACTs where the temporal and
epistemic modality components of HalluciSense are NOT ACTIVE. These records
test only the NLI factual grounding component.

---

## Temporal Keyword Coverage (N=300 sampled)

| Capability | Count | % | Adequate for Testing? |
|:---|---:|---:|:---:|
| No temporal signal | 178 | **59.3%** | N/A (temporal components inactive) |
| Relational temporal (before/after/since) | 116 | 38.7% | ⚠️ Partial |
| Negation marker | 106 | 35.3% | ⚠️ Partial |
| Year mention (any) | 105 | 35.0% | ⚠️ Partial |
| Past year (1900–2019) | 105 | 35.0% | ✅ Applicable |
| Hypothetical marker (if/would/could) | 76 | 25.3% | ⚠️ Partial |
| Date mentioned (month name) | 68 | 22.7% | ✅ Applicable |
| Counterfactual marker | 68 | 22.7% | ⚠️ Partial (surface markers, not semantic) |
| Meta-claim marker (falsely/debunked) | 30 | 10.0% | ⚠️ Low but non-zero |
| **Future year (2025+)** | **0** | **0.0%** | ❌ NOT TESTABLE |

---

## Per-Capability Claim Validity

### Capability 1: Future-Year Assertion Detection
**HalluciSense claim**: Detects hallucinated future-fact assertions (e.g., "Brazil won the 2027 World Cup").  
**Coverage in external benchmark**: 0/300 = 0.0%  
**Verdict**: ❌ THE EXTERNAL BENCHMARK DOES NOT MEASURE THIS CAPABILITY.

This is the primary capability that distinguished Phase 5 from Phase 6. Its benefit
cannot be demonstrated on HaluBench/RAGTruth/HaluEval.

### Capability 2: Epistemic Modality Protection (Predictions, Hypotheticals)
**HalluciSense claim**: Prevents false-positive flagging of valid predictions and hypotheticals.  
**Coverage**: PREDICTION=19 (6.3%), CONDITIONAL=5 (1.7%), COUNTERFACTUAL=3 (1.0%)  
**Verdict**: ⚠️ SPARSE. Modality protection is present in ~9% of examples — too few
for isolated statistical analysis. The signal is present but underpowered.

### Capability 3: Relational Temporal Operator Parsing
**HalluciSense claim**: Parses "before/after/since/during" to verify temporal consistency.  
**Coverage**: 116/300 = 38.7% contain relational keywords  
**Verdict**: ⚠️ MODERATE COVERAGE but keyword presence ≠ temporal grounding needed.
Most relational keywords in these datasets are incidental ("since 2010 the company..."),
not requiring the specific verification logic.

### Capability 4: Global Evidence-Set Date Collision Prevention
**HalluciSense claim**: Prevents spurious temporal mismatch detection from background dates.  
**Coverage**: 105/300 = 35.0% contain year mentions  
**Verdict**: ⚠️ PLAUSIBLE COVERAGE for past-year scenarios but no specific
evidence-date collision cases in these QA datasets. RAGTruth contexts may contain
multiple dates, but this is not guaranteed.

### Capability 5: Meta-Claim / Fiction / Quotation Handling
**HalluciSense claim**: Protects falsely-framed or fictional claims from false positives.  
**Coverage**: QUOTED_CLAIM=19 (6.3%), FICTIONAL=2 (0.7%), meta-marker=30 (10%)  
**Verdict**: ⚠️ LOW-TO-MODERATE. Enough to demonstrate the mechanism but not for strong statistical claims.

### Capability 6: NLI Factual Grounding (Baseline)
**Coverage**: 250/300 = 83.3% are ASSERTED_FACTs  
**Verdict**: ✅ WELL-MEASURED. The NLI baseline is directly applicable and well-powered.

---

## Benchmark Compatibility Summary

| Component Tested | Benchmark Coverage | Can Be Statistically Validated? |
|:---|:---:|:---:|
| NLI factual grounding | 83.3% | ✅ YES |
| Past-year temporal verification | 35.0% | ⚠️ Partial |
| Relational temporal parsing | 38.7% | ⚠️ Partial (surface-level) |
| Meta-claim / quotation protection | 6–10% | ⚠️ Low power |
| Modality protection (predictions) | 6.3% | ⚠️ Low power |
| **Future-year assertion detection** | **0.0%** | ❌ NOT MEASURABLE |
| Counterfactual / conditional protection | <2% | ❌ NOT MEASURABLE |
| Evidence-date collision prevention | Unknown | ❌ NOT DESIGNED FOR THIS |

---

## Publication Implication

**This benchmark measures HalluciSense primarily as a factual NLI system.**
The novel temporal and epistemic components of Phase 6 are barely activated
on 59% of examples and completely untestable for the core future-year assertion use case.

**Recommended disclosure for publication**:
> "The external benchmarks (HaluBench, RAGTruth, HaluEval) were not designed to
> evaluate temporal or epistemic modality reasoning. Of 300 sampled records, 59.3%
> contain no temporal signal, and 0% contain future-year assertions (the primary
> scenario where Phase 6 provides mechanistic benefit over Phase 5). Benchmark
> performance on these datasets primarily reflects NLI factual grounding capability.
> The temporal and modality components are evaluated separately through targeted
> adversarial benchmarks (Section 6C-K) and mechanistic stress tests (Section 6C-J)."
