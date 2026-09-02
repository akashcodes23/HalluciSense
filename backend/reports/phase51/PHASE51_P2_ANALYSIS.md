# PHASE 51 — PILLAR 2 SCIENTIFIC VALIDITY & DISTRIBUTION ANALYSIS
**Investigation of Predictive Confidence & Informational Value**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `EMPIRICALLY ANALYZED`

---

## 1. Pillar 2 Score Distributions by Ground Truth Label

| Metric | Factual Examples ($y=0, N=80$) | Hallucinated Examples ($y=1, N=200$) | Standardized Mean Diff (SMD) | Univariate AUROC |
| :--- | :--- | :--- | :--- | :--- |
| **Mean $P_{\text{P2}}$** | **0.0779** | **0.0815** | +0.2424 | **0.5478** |
| **Median $P_{\text{P2}}$** | 0.0875 | 0.0875 | — | — |
| **Standard Deviation** | 0.0162 | 0.0135 | — | — |
| **Interquartile Range** | 0.0228 | 0.0000 | — | — |

---

## 2. Scientific Diagnosis of Pillar 2

- **Informative Value**: In static verification mode (`STATIC_VERIFICATION_CONFIDENCE`), Pillar 2 operates as a proxy derived from retrieval relevance and coverage.
- **Distribution Separation**: As shown by the univariate AUROC of **0.5478** and SMD of **+0.2424**, static Pillar 2 provides slight positive correlation with uncertainty, but has a concentrated distribution around $0.075 - 0.0875$.
- **Conclusion**: Pillar 2 is non-harmful and adds stability to weighted fusion, but does not independently separate short factual from counterfactual claims without real white-box generation logprobs.
