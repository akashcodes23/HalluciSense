# HalluciSense Phase 6I — Claim-Level Retrieval Signal Reconstruction Report

## Executive Summary

Phase 6I claim-level retrieval signal reconstruction has completed.
- **LOCKED_FINAL_TEST Isolation**: `STRICTLY BLOCKED / 0 SAMPLES ACCESSED`
- **Candidate Status**: `NO_FEASIBLE_CANDIDATE`

---

## Key Finding

All 58,002 DEVELOPMENT and 12,483 VALIDATION predictions from Phase 6C.1 had `factual_error = null`
because benchmark evidence embedded in prompt text was never extracted and passed to the P1 NLI engine.

Phase 6I reconstructed P1 by:
1. Extracting context passages from dataset-specific prompt formats
2. Decomposing responses into atomic claims
3. Running NLI claim-by-claim against extracted evidence
4. Building rich claim-level features (entailment, contradiction, support margins)
5. Training regularized logistic regression on DEVELOPMENT only

---

## Development Results

| Metric | Value |
|--------|-------|
| MCC | `0.3563` |
| Balanced Accuracy | `0.6667` |
| Recall | `0.8333` |
| Specificity | `0.5` |
| ROC-AUC | `0.7083` |

## Validation Results

| Metric | Value |
|--------|-------|
| MCC | `0.0` |
| Balanced Accuracy | `0.5` |
| Recall | `1.0` |
| Specificity | `0.0` |
| ROC-AUC | `0.68` |
| MCC 95% CI | `[0.0, 0.0]` |

---

## Final Verdict

```
HALLUCISENSE PHASE 6I RETRIEVAL RECONSTRUCTION: NO FEASIBLE CANDIDATE
```
