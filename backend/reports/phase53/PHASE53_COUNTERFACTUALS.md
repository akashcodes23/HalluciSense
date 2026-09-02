# PHASE 53 — CONTROLLED COUNTERFACTUAL MATCHED PAIRS REPORT
**Evaluating Model Sensitivity ($\Delta P_H$) Across 8 Semantic Mutation Categories**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `MEASURED & AUDITED`

---

## 1. Counterfactual Matched Pairs Sensitivity Matrix

| Category | True Claim vs False Claim | Frozen $P_H$ (T $\to$ F) | Frozen $\Delta P_H$ | Candidate B $P_H$ (T $\to$ F) | Cand B $\Delta P_H$ | Strategy S1 $\Delta P_H$ | Direction Correct? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **Arithmetic** | `14*5=70` vs `14*5=75` | $0.2738 \to 0.2738$ | $0.0000$ | $0.6002 \to 0.6002$ | $0.0000$ | **$+0.7500$** | 🏆 **CORRECT** |
| **Fact Swap** | Stockholm in Sweden vs Norway | $0.2623 \to 0.5316$ | $+0.2693$ | $0.0466 \to 0.8929$ | **$+0.8463$** | **$+0.8463$** | 🏆 **CORRECT** |
| **Entity Swap**| Kepler laws vs Curie in 1609 | $0.4053 \to 0.5880$ | $+0.1827$ | $0.7717 \to 0.9398$ | **$+0.1681$** | **$+0.1681$** | 🏆 **CORRECT** |
| **Negation** | Stockholm is vs is not capital | $0.2623 \to 0.2474$ | **$-0.0148$** (Inverted)| $0.0466 \to 0.5783$ | **$+0.5317$** | **$+0.5317$** | 🏆 **CORRECT** |
| **Temporal** | Mendeleev 1869 vs 300 BC | $0.5071 \to 0.2590$ | **$-0.2480$** (Inverted)| $0.3555 \to 0.4900$ | **$+0.1345$** | **$+0.1345$** | 🏆 **CORRECT** |
| **Direct Contradiction**| Water atoms vs zero atoms | $0.1612 \to 0.2119$ | $+0.0507$ | $0.2276 \to 0.4900$ | **$+0.2624$** | **$+0.2624$** | 🏆 **CORRECT** |
| **Causal** | Photosynthesis solar vs Mars cards| $0.0967 \to 0.4167$ | $+0.3200$ | $0.2587 \to 0.9141$ | **$+0.6554$** | **$+0.6554$** | 🏆 **CORRECT** |
| **Multi-Claim**| Single capital vs Dual capital | $0.4574 \to 0.7644$ | $+0.3070$ | $0.0848 \to 0.0805$ | $-0.0043$ | $-0.0043$ | ⚠️ Neutral |

---

## 2. Key Findings

- **Reversal of Frozen Inversions**: On negation (`"Stockholm is not the capital"`) and temporal mutation (`"Mendeleev in 300 BC"`), the frozen production model paradoxically assigned *lower* hallucination risk to the false claim ($\Delta P_H = -0.0148$ and $-0.2480$). Candidate B and Model 2 completely reverse this defect, properly increasing hallucination probability by **$+0.5317$** and **$+0.1345$**.
- **Massive Arithmetic Resolution**: On arithmetic mutation ($14 \times 5 = 75$), Model 2 jumps by **$+0.7500$**, moving from $0.20$ to $0.95$.
