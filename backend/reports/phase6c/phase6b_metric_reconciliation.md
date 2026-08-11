# Phase 6B Metric Reconciliation Report

**Generated**: 2026-08-11
**Auditor**: Phase 6C canonical_evaluator.py
**Source**: backend/reports/phase6b_novelty_experiment_results.json
**Method**: All metrics recomputed from TP/TN/FP/FN using canonical_evaluator.py

---

## Section 1: Metric Verification Results (A0–A9)

**Verdict: ALL 10 CONFIGURATIONS MATCH. No numerical discrepancies found.**

Every reported metric (Accuracy, Precision, Recall, F1, Specificity, FPR, FNR) was
independently recomputed from the stored confusion matrices using the canonical evaluator.
Zero discrepancies were found at tolerance 1e-3.

| Config | TP | TN | FP | FN | Acc | F1 | MCC | Bal.Acc | Status |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| A0_NLI_Baseline | 121 | 231 | 46 | 152 | 0.6400 | 0.5500 | 0.3014 | 0.6386 | ✅ MATCH |
| A1_NLI_Retrieval | 121 | 231 | 46 | 152 | 0.6400 | 0.5500 | 0.3014 | 0.6386 | ✅ MATCH |
| A2_Plus_TemporalReasoning | 121 | 229 | 48 | 152 | 0.6364 | 0.5475 | 0.2925 | 0.6350 | ✅ MATCH |
| A3_Plus_ModalitySeparation | 0 | 275 | 2 | 273 | 0.5000 | 0.0000 | −0.0600 | 0.4964 | ✅ MATCH |
| A4_Plus_AtomicClaimDecomposition | 164 | 159 | 118 | 109 | 0.5873 | 0.5910 | 0.1748 | 0.5874 | ✅ MATCH |
| A5_Plus_GlobalEvidenceAlignment | 4 | 277 | 0 | 269 | 0.5109 | 0.0289 | 0.0862 | 0.5073 | ✅ MATCH |
| A6_Plus_RelationalTemporalParsing | 117 | 239 | 38 | 156 | 0.6473 | 0.5467 | 0.3238 | 0.6457 | ✅ MATCH |
| A7_Plus_MetaQuotationFiction | 121 | 234 | 43 | 152 | 0.6455 | 0.5538 | 0.3148 | 0.6440 | ✅ MATCH |
| A8_Plus_DynamicEventAnchoring | 123 | 231 | 46 | 150 | 0.6436 | 0.5566 | 0.3083 | 0.6422 | ✅ MATCH |
| A9_Full_HalluciSense | 123 | 223 | 54 | 150 | 0.6291 | 0.5467 | 0.2736 | 0.6278 | ✅ MATCH |

**MCC and Balanced Accuracy were NOT reported in Phase 6B** — these are newly added
for Phase 6C and reveal important patterns (see Section 3).

---

## Section 2: AUROC/AUPRC Verification

AUROC and AUPRC values stored in Phase 6B were computed from continuous prediction
scores during the original harness run. These **cannot be independently recomputed**
from confusion matrices alone (continuous scores not stored separately).

The stored values are treated as valid because:
- They were computed by the same `compute_auroc_auprc` function whose algorithm is correct
- The confusion matrix values match exactly, confirming the evaluation loop was correct
- No evidence of post-hoc modification was found

**AUROC/AUPRC status: ACCEPTED AS VALID (cannot recompute without raw scores)**

| Config | AUROC (reported) | AUPRC (reported) |
|:---|:---:|:---:|
| A0 | 0.6834 | 0.6654 |
| A1 | 0.6834 | 0.6654 |
| A2 | 0.6802 | 0.6614 |
| A3 | 0.6284 | 0.7007 |
| A4 | 0.6303 | 0.6102 |
| A5 | 0.6482 | 0.7428 |
| A6 | 0.7053 | 0.6810 |
| A7 | 0.6863 | 0.6690 |
| A8 | 0.6871 | 0.6682 |
| A9 | 0.6807 | 0.6526 |

---

## Section 3: Supplementary Metrics Not Previously Reported

The following were not in Phase 6B but are now computed for publication:

| Config | MCC | Balanced Accuracy |
|:---|:---:|:---:|
| A0 | 0.3014 | 0.6386 |
| A1 | 0.3014 | 0.6386 |
| A2 | 0.2925 | 0.6350 |
| A3 | **−0.0600** | **0.4964** |
| A4 | 0.1748 | 0.5874 |
| A5 | 0.0862 | 0.5073 |
| A6 | **0.3238** | 0.6457 |
| A7 | 0.3148 | 0.6440 |
| A8 | 0.3083 | 0.6422 |
| A9 | 0.2736 | 0.6278 |

**Key observation**: MCC for A9 (Full HalluciSense, 0.2736) is LOWER than A6 (0.3238).
The full system shows no MCC improvement over A6_Plus_RelationalTemporalParsing.
This must be disclosed in any publication.

---

## Section 4: Evidence Noise Stress Test — Critical Audit

### Reported Results (Phase 6B)
All 6 conditions showed identical: Baseline=100%, Phase5=40%, Phase6=100%

### Critical Findings

**FINDING 1: Identical result across ALL 6 conditions is impossible if the conditions differ.**

If evidence noise conditions E2–E6 add different content, then the NLI model and
TemporalClaimEngine produce different scores for different conditions. If ALL 6
conditions show identical results (5/5 → 5/5 for baseline), this implies the
evidence noise had NO effect on any of the 5 test cases under any condition — which
contradicts the purpose of the test.

**ROOT CAUSE**: The baseline (NLI) gives the same predictions across all conditions because:
- ST1 (Apollo 11 1969), ST3 (Constantinople 1453) — evidence always supports the claim
  regardless of noise
- ST2 (2024 election, "Candidate A") — no Wikipedia evidence returns verification
- ST4 (Artemis IV 2028), ST5 (fusion 2038) — future claims are NOT hallucinated (gold=False)

The baseline achieves 100% because ALL 5 test cases have gold=False (factual or
valid predictions) and the NLI model correctly scores them all below 0.50.

**FINDING 2: Phase 5 misclassification is correctly caused by the naive year check.**

Phase 5 hardcodes: `if "202" in response or "203" in response: score = max(score, 0.75)`

ST4 contains "2028" → flagged as hallucinated → incorrect (gold=False)
ST5 contains "2038" → flagged as hallucinated → incorrect (gold=False)

This gives 3/5 = 60% correct... but the report shows 40% (2/5). Counting again:
- ST1: 1969 — no 202x/203x trigger → NLI score used → below 0.5 → correct ✓
- ST2: "Candidate A" — no 202x trigger? Wait: "2024" contains "202" → TRIGGERS → pred=True → incorrect ✗
- ST3: 1453 — no trigger → correct ✓
- ST4: 2028 → triggers → pred=True → incorrect ✗
- ST5: 2038 → triggers → pred=True → incorrect ✗

So ST2 ALSO triggers the 202x check. That gives 2/5 correct = 40% Phase 5 accuracy.
This IS consistent with the reported 40%.

**FINDING 3: The "Phase 5" implementation is a 2-line approximation, not the actual Phase 5 code.**

The stress test constructs Phase 5 as:
```python
p5_score = nli_score
if "202" in response or "203" in response:
    p5_score = max(p5_score, 0.75)
```

This is a manually coded simulation of what Phase 5 MIGHT do, NOT the actual
Phase 5 pipeline. It is a strawman baseline. Any real comparison requires running
the actual Phase 5 system.

**CLASSIFICATION**: The evidence-noise stress test demonstrates a valid *mechanistic*
point — that naive substring year-matching causes false positives. However:
- The Phase 5 approximation is a strawman
- N=5 is too small for any statistical claim
- All 6 conditions showing identical results weakens the experimental design
- "100% robustness" language is NOT justified

**CORRECT REPORTING FOR PUBLICATION**:
> "In a controlled mechanistic stress test (N=5), the Phase 6 temporal framework
> correctly handled future predictions and conditional claims that a naive substring
> year-matching heuristic misclassified. The test demonstrates the mechanism but
> is insufficient as statistical evidence (N=5, synthetic conditions, strawman baseline)."

---

## Section 5: What Is Valid For Publication

| Result | Valid | Condition |
|:---|:---:|:---|
| A0–A9 confusion matrices | ✅ | Numerically verified |
| A0–A9 Accuracy, Precision, Recall, F1 | ✅ | Recomputed: exact match |
| A0–A9 MCC, Balanced Accuracy | ✅ | Newly computed; not in Phase 6B |
| A0–A9 AUROC/AUPRC | ⚠️ | Accepted but not independently recomputable |
| A0=A1 identical (known harness design issue) | ✅ | Must disclose |
| A3, A5 degenerate intermediate states | ✅ | Must explain as harness artifact |
| Evidence noise: mechanistic illustration | ✅ | Disclose N=5, strawman Phase 5 |
| Evidence noise: "100% robustness" claim | ❌ | INVALID FOR PUBLICATION |
| Full ablation showing A9 > A0 on F1 | ❌ | A9 F1=0.5467 = A0 F1=0.5500 (LOWER) |
| Full ablation showing A6 best MCC (0.3238) | ✅ | Defensible finding |
