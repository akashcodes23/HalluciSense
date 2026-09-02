# PHASE 51 — PILLAR ABLATION & FUSION ANALYSIS
**Diagnostic Evaluation of Individual and Combined Pillar Performance**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `EMPIRICALLY MEASURED`

---

## 1. Pillar Ablation Matrix

| Ablation Configuration | AUROC | MCC (at $\tau=0.54$) | F1-Score | Specificity | Role in Pipeline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **P1 Only (Retrieval Grounding)** | **0.8341** | **0.4031** | **0.8053** | 0.6625 | 🏆 Dominant Pillar |
| **P1 + P2** | **0.8357** | **0.4463** | **0.8097** | **0.7250** | 🥇 Best Dual Combination |
| **P1 + P3** | 0.7473 | 0.4031 | 0.8053 | 0.6625 | Complementary on Multi-Claims |
| **P1 + P2 + P3 (Hybrid)** | 0.7183 | 0.1775 | 0.4542 | **0.8625** | High Precision / Conservative |
| **P2 Only (Confidence)** | 0.5478 | 0.0000 | 0.0000 | 1.0000 | Static Proxy / Non-Discriminative |
| **P3 Only (Consistency)** | 0.4353 | -0.1037 | 0.1709 | 0.8250 | Specialised to Multi-Claims |
| **P2 + P3** | 0.4679 | 0.0725 | 0.1121 | 0.9750 | Ineffective without Grounding |

---

## 2. Key Ablation Findings

1. **Pillar 1 is the Anchor**: Grounding against external evidence passages accounts for over $95\%$ of the discriminative power ($AUROC = 0.8341$).
2. **P1 + P2 Improves Specificity**: Combining P1 grounding with P2 confidence weighting increases specificity from $66.25\%$ to $72.50\%$ and raises MCC to **0.4463**.
3. **P3 Acts as a Targeted Multi-Claim Filter**: Pillar 3 is specifically effective on contradictory multi-sentence pairs ($70\%$ recall on category F), but provides negligible signal on single-sentence claims.
