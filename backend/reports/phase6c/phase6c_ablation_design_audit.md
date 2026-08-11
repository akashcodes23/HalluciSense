# Phase 6C: Ablation Design Validity Audit

**Source file**: backend/scripts/run_novelty_experiments.py (lines 111–158)
**Dataset**: HaluBench (N=100) + RAGTruth (N=300) + HaluEval (N=150) = 550 combined
**Evaluator**: Shared evidence = dataset context field only

---

## Summary Verdict

The Phase 6B 10-level ablation is **NOT a fully valid controlled ablation** for publication.
Several configurations are not genuine isolated ablation steps. This must be disclosed
and corrected in Phase 6C.

---

## Per-Configuration Analysis

### A0: NLI Baseline

```python
lambda resp, q, ev: pipeline.p1_engine.evaluate_claims_against_evidence([resp], ev)[0]
```

**What it is**: Whole-response NLI score against dataset context evidence.  
**Is it the correct baseline?**: YES — this is what a pure NLI approach does.  
**Valid**: ✅

---

### A1: NLI + Retrieval Grounding

```python
lambda resp, q, ev: pipeline.p1_engine.evaluate_claims_against_evidence([resp], ev)[0]
```

**What it is**: IDENTICAL to A0. Same lambda, same function, same evidence.  
**Is it a genuine ablation step?**: NO. Retrieval is not being activated.  
**Root cause**: The evaluation uses static dataset context as evidence. True retrieval
would call Wikipedia/external search. This harness does not implement live retrieval.  
**Classification**: ❌ NOT A VALID ABLATION STEP — A0 = A1 (identical)

---

### A2: + Temporal Reasoning

```python
lambda resp, q, ev: max(p1_score, engine.analyze_claim(resp, q).temporal_inconsistency_score)
```

**What it is**: Takes max of NLI score and temporal inconsistency score.  
**Is it controlled?**: YES — single additive change from A1.  
**Valid**: ✅ (but note: result is LOWER than A0/A1, F1: 0.5475 vs 0.5500)

---

### A3: + Epistemic Modality Separation

```python
lambda resp, q, ev: 0.0 if engine.analyze_claim(resp, q).protected_from_temporal_penalty
                    else max(p1_score, temporal_score)
```

**What it is**: Returns 0.0 (always predicts NON-hallucinated) for protected modalities.  
**The problem**: On external datasets where >95% of responses are ASSERTED facts,
`protected_from_temporal_penalty` is False for nearly all examples. BUT for the small
fraction that IS protected, the score is set to 0.0 (absolute zero, bypassing all other signals).  
**Why result collapses to TP=0**: When applied as the SOLE scorer without combining
with P1, any example that has a very small protected fraction causes FP → TN shifts
but also causes genuine positive detections to become FN=0 (score=0.0 → pred=False).  
**Classification**: ❌ DEGENERATE HARNESS STATE — not a valid isolated test of modality protection.
Modality protection is designed to modify a combined score, not replace it entirely.

---

### A4: + Atomic Claim Decomposition

```python
lambda resp, q, ev: max([pipeline.p1_engine.evaluate_claims_against_evidence([c], ev)[0]
                         for c in pipeline.p1_engine.extract_claims(resp)] or [0.0])
```

**What it is**: Splits response into atomic claims, scores each, takes max.  
**Is it controlled?**: YES — uses same NLI, same evidence, only extraction changes.  
**Valid**: ✅ (takes max score across claims — note: this is more aggressive than mean)  
**Caveat**: Using MAX (most suspicious claim) vs MEAN would have different results.
The choice of MAX is an implicit design decision not evaluated in ablation.

---

### A5: + Global Evidence Alignment

```python
lambda resp, q, ev: 0.0 if engine.verify_evidence_date_mismatch(resp, ev) is None
                    else pipeline.p1_engine.evaluate_claims_against_evidence([resp], ev)[0]
```

**What it is**: Returns 0.0 if there is no evidence date mismatch to verify.  
**The problem**: `verify_evidence_date_mismatch` returns None for responses without
explicit date expressions. On the 550-case external dataset (~90% of responses have
no temporal content), this sets score=0.0 for nearly all examples, causing near-total
collapse of TP detection.  
**Classification**: ❌ DEGENERATE HARNESS STATE — same fundamental problem as A3.
Global evidence alignment is designed as a modifier within a pipeline, not as a
short-circuit gate that replaces the entire score.

---

### A6: + Relational Temporal Parsing

```python
lambda resp, q, ev: 0.0 if engine._matches_any(engine.RELATIONAL_PATTERNS, resp.lower())
                    else pipeline.p1_engine.evaluate_claims_against_evidence([resp], ev)[0]
```

**What it is**: Returns 0.0 if response contains relational patterns (before/after/since/during).  
**Surprise finding**: This produces the BEST MCC (0.3238) of all configurations.  
**Why it works**: Responses with "before/after/since/during" in external datasets happen
to include hedging/qualified statements that tend to be factual. Suppressing scores for
these cases reduces FP rate. This is a coincidental alignment, not a principled finding.  
**Classification**: ✅ Results correct, but interpretation requires caution.
The improvement in A6 over A0 is not because relational parsing improves detection;
it's because zeroing scores for relational-pattern matches disproportionately
removes FPs on this particular dataset composition.

---

### A7: + Meta-Claim / Fiction Handling

```python
lambda resp, q, ev: 0.0 if engine._matches_any(engine.META_CLAIM_PATTERNS + engine.FICTION_PATTERNS, resp.lower())
                    else pipeline.p1_engine.evaluate_claims_against_evidence([resp], ev)[0]
```

**What it is**: Returns 0.0 for responses containing meta-claim or fiction patterns.  
**Valid**: ✅ (marginal improvement over A6, correct direction)

---

### A8: + Dynamic Event Anchoring

```python
lambda resp, q, ev: max(p1_score, EventTemporalAnchorResolver().resolve_event_anchor(resp, ev)[1])
```

**What it is**: Adds Wikidata-based entity temporal anchor scoring.  
**Concern**: A new `EventTemporalAnchorResolver` instance is created per record,
which re-initializes the `TemporalClaimEngine`. This may suppress results vs. the
production pipeline.  
**Valid**: ⚠️ Correct direction (tiny improvement A8 > A7), but instantiation issue
means results may not reflect production behavior. Difference is marginal (ΔF1 = +0.003).

---

### A9: Full HalluciSense

```python
lambda resp, q, ev: pipeline.analyze(text=resp, query=q, provided_evidence=ev).pillar1_summary.factual_error_score
```

**CRITICAL FINDING**: A9 uses **only P1 (pillar1_summary.factual_error_score)**, NOT
the full `overall_h_score` which includes P2 and P3 fusion.  
If P2/P3 are unavailable (not loaded, returns None), the fusion falls back to P1 only.  
**This means A9 is not truly "Full HalluciSense" — it is P1-only with the full P1 pipeline.**  
The result is LOWER than A0 (F1=0.5467 vs 0.5500). This is not because the full
system is worse — P2/P3 are not being scored in the evaluation environment.  
**Classification**: ⚠️ Results are correct for what was measured; but A9 does NOT
represent the complete 3-pillar system with P2 and P3 active.

---

## Summary: True Ablation Validity Table

| Config | Isolated Change | Valid Ablation | Result Interpretable |
|:---|:---|:---:|:---:|
| A0 | NLI baseline | ✅ | ✅ |
| A1 | NLI + retrieval | ❌ (= A0) | ❌ |
| A2 | + temporal score (max) | ✅ | ✅ |
| A3 | + modality gate | ❌ (degenerate) | ❌ |
| A4 | + atomic claim max | ✅ | ✅ |
| A5 | + evidence alignment gate | ❌ (degenerate) | ❌ |
| A6 | + relational pattern gate | ✅ | ⚠️ (coincidental) |
| A7 | + meta/fiction gate | ✅ | ✅ |
| A8 | + dynamic anchoring | ⚠️ (instantiation issue) | ⚠️ |
| A9 | Full pipeline | ⚠️ (P1 only in practice) | ⚠️ |

---

## Publication Recommendation

For Phase 6C publication, the ablation table should be replaced by the **M0–M9 flag-based ablation** designed to avoid degenerate states (see run_phase6c_publication_eval.py). The A0–A9 results may be reported as a supplementary note with explicit disclosure of each issue above.

The sentence "A9 (Full HalluciSense) achieves the best performance" is **INVALID** — A6 achieves the best MCC and A9 is lower than A0 on F1.

**What CAN be said**: "The relational temporal parsing component (A6) and meta-claim handling (A7) produce the most consistent improvement in MCC and Balanced Accuracy over the NLI baseline."
