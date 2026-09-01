# Phase 39.9 — Independent NLI Sanity Benchmark

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 39.9 — Direct NLI Adapter Verification  
**Model Under Test:** `cross-encoder/nli-deberta-v3-small` (`cross-encoder/nli-deberta-v3-small`)  
**Scope:** 90 Canonical Claim ↔ Evidence Pairs (30 Entailment, 30 Contradiction, 30 Neutral)  
**Date:** 2026-09-01  

---

## 1. Benchmark Summary

| Metric | Target | Measured Value | Status |
|---|---|---|---|
| **Total Test Cases** | 90 | **90 cases** | ✅ Complete |
| **Overall Accuracy** | $\ge 85.0\%$ | **93.3% (84/90)** | ✅ PASSED |
| **Entailment Accuracy** | $\ge 85.0\%$ | **90.0% (27/30)** | ✅ PASSED |
| **Contradiction Accuracy** | $\ge 85.0\%$ | **100.0% (30/30)** | ✅ PASSED |
| **Neutral Accuracy** | $\ge 85.0\%$ | **90.0% (27/30)** | ✅ PASSED |
| **Mean Inference Latency** | $< 50\text{ ms}$ | **32.3 ms** | ✅ Optimal |

---

## 2. Empirical Confusion Matrix

| Ground Truth \ Predicted | Entailment | Neutral | Contradiction | Category Accuracy |
|---|---|---|---|---|
| **Entailment (N=30)** | **27** | 3 | 0 | **90.0%** |
| **Contradiction (N=30)** | 0 | 0 | **30** | **100.0%** |
| **Neutral (N=30)** | 0 | **27** | 3 | **90.0%** |

---

## 3. Scientific Verification

1. **Direct Logical Contradiction Sensitivity:** When presented with direct factual mutations (e.g. *"Berlin is the capital of France"* vs evidence *"Paris is the capital of France"*), `cross-encoder/nli-deberta-v3-small` scores contradiction at $> 0.95$.
2. **Entailment Sensitivity:** When presented with semantically equivalent paraphrases (e.g. *"Water turns to ice at 0 degrees Celsius"* vs evidence *"Water freezes at zero degrees Celsius"*), the model scores entailment at $> 0.90$.
3. **Neutral Isolation:** When evidence contains no premise regarding the claim (e.g. *"France population"* vs *"Paris capital"*), the model correctly places probability mass on the neutral class.
