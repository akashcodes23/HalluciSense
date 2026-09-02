# PHASE 52 — PILLAR ABLATION REPORT ($N=300$ BALANCED 50/50 SET)
**Comprehensive Multi-Configuration Ablation Matrix**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `EMPIRICALLY BENCHMARKED`

---

## 1. Complete Ablation Matrix on $N=300$ (150 Factual vs 150 Hallucinated)

| Configuration | AUROC | AUPRC | Accuracy | Precision | Recall | Specificity | F1 | MCC | Brier | ECE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A. P1 Only (Retrieval Grounding)** | **0.8083** | **0.8253** | **70.33%** | 0.6704 | **0.8000** | 0.6067 | **0.7295** | **0.4145** | 0.2448 | 0.2054 |
| **B. P2 Only (Structural Confidence)**| 0.5462 | 0.7275 | 50.33% | **1.0000** | 0.0067 | **1.0000** | 0.0132 | 0.0578 | 0.4214 | 0.4172 |
| **C. P3 Only (Consistency)** | 0.4428 | 0.4407 | 47.33% | 0.3947 | 0.1000 | 0.8467 | 0.1596 | -0.0802| 0.5225 | 0.5236 |
| **D. P1 + P2 (Grounding + Confidence)**| **0.8139** | **0.8327** | **71.67%** | **0.6946** | 0.7733 | **0.6600** | **0.7319** | **0.4361** | **0.1978** | **0.0795** |
| **E. P1 + P3 (Grounding + Consistency)**| 0.7358 | 0.6728 | 70.00% | 0.6744 | 0.7733 | 0.6267 | 0.7205 | 0.4044 | 0.2294 | 0.1537 |
| **F. P2 + P3 (Confidence + Consistency)**| 0.4832 | 0.4753 | 51.33% | 0.6667 | 0.0533 | 0.9733 | 0.0988 | 0.0680 | 0.4436 | 0.4292 |
| **G. P1 + P2 + P3 (Full Frozen Detector)**| **0.6905** | **0.5883** | **57.00%** | 0.6479 | **0.3067** | **0.8333** | **0.4163** | **0.1647** | 0.2516 | 0.2043 |

---

## 2. Definitive Scientific Insights from Ablation

1. **P1 Grounding is the Sole Anchor**: P1 alone achieves **0.8083 AUROC** and **0.8000 Recall** on the balanced dataset.
2. **P1 + P2 is the Highest Performing Combination**: Weighted dual fusion achieves the peak AUROC of **0.8139**, highest MCC of **0.4361**, and lowest calibration error ($\text{ECE} = 0.0795$).
3. **Full Hybrid Degradation**: Moving from P1+P2 to the full 19-feature frozen tree detector causes AUROC to drop from **0.8139** to **0.6905**, Recall to plummet from **77.33%** to **30.67%**, and MCC to fall from **0.4361** to **0.1647**.
4. **Why P3 and Fusion Hurt**: P3 is non-discriminative for atomic single claims (AUROC 0.4428). When combined into the frozen tree with polarity inversions, the composite model severely compresses the positive hallucination probabilities.
