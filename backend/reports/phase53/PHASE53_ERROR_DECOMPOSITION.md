# PHASE 53 — ERROR DECOMPOSITION REPORT (R1 TO R12)
**Comparative Root Cause Classification for Model 0 vs Model 2 on Independent Holdout ($N=200$)**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `MEASURED & CATALOGUED`

---

## 1. False Negative Root Cause Distribution ($N=100$ Hallucinations)

| Failure Code | Failure Description | Model 0: Frozen Baseline ($N_{\text{FN}} = 77$) | Model 2: Remediated Candidate ($N_{\text{FN}} = 19$) | Net Reduction |
| :--- | :--- | :--- | :--- | :---: |
| **R1** | Retrieval scope missingness | 0 | 2 | $+2$ (Identified) |
| **R2** | NLI ambiguity / soft neutral | 0 | 15 | $+15$ (Remaining NLI boundary) |
| **R4** | Classifier boundary sub-0.54 | 29 | 2 | **-27 (-93.1%)** 🏆 |
| **R7** | Polarity / fusion suppression | 29 | 0 | **-29 (-100.0%)** 🏆 |
| **R9** | Symbolic gateway integration missing | 19 | 0 | **-19 (-100.0%)** 🏆 |
| **TOTAL** | **Total False Negatives** | **77 / 100 (77.0%)** | **19 / 100 (19.0%)** | **-58 (-75.3%)** 🚀 |

---

## 2. Key Error Shift Insights

1. **Complete Eradication of R7 (Polarity Suppression)**: Polarity-induced false negatives dropped from **29 to 0**, proving that the remediated tree training resolved the cannibalization of P1 evidence.
2. **Complete Eradication of R9 (Symbolic Gateway Suppression)**: Numerical false negatives dropped from **19 to 0**, achieving 100% detection on arithmetic mutations.
3. **Primary Remaining Frontier (R2)**: The 15 remaining false negatives in Model 2 are soft NLI neutral classifications where Wikipedia documents do not mention obscure entities (e.g. fabricated dragon batteries or Martian pharaoh jet-skis).
