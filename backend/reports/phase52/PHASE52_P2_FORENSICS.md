# PHASE 52 — PILLAR 2 STRUCTURAL CONFIDENCE FORENSICS
**Analysis of Information Content & Static Verification Proxy**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `MEASURED & AUDITED`

---

## 1. P2 Score Distributions (Factual vs Hallucinated)

| Metric | Factual ($y=0, N=150$) | Hallucinated ($y=1, N=150$) | SMD | Univariate AUROC |
| :--- | :--- | :--- | :--- | :--- |
| **Mean $P_{\text{P2}}$** | **0.0762** | **0.0811** | +0.2812 | **0.5462** |
| **Median $P_{\text{P2}}$** | 0.0875 | 0.0875 | — | — |
| **Std Dev** | 0.0189 | 0.0154 | — | — |
| **IQR** | 0.0228 | 0.0000 | — | — |

---

## 2. Is `STATIC_VERIFICATION_CONFIDENCE` Meaningful?

- **Numerical Components**: In static verification mode (single generation, no token logprobs), P2 generates a confidence gap derived from knowledge retrieval score density and claim length:
  $$\text{gap} = 0.10 \times (1.0 - \text{retrieval\_density}) + 0.05 \times \min(1.0, \text{claim\_len}/50)$$
- **Informative Value**: Static P2 is nearly constant ($0.076$ vs $0.081$). When weighted with P1 in dual fusion (`0.60*P1 + 0.40*P2`), it acts as an effective regularizer that boosts specificity from $60.67\%$ to $66.00\%$ and AUROC from $0.8083$ to $0.8139$.
- **Standalone Value**: Evaluated alone, P2 is non-discriminative ($\text{AUROC} = 0.5462$).
