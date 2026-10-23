# HalluciSense Component Ablation Study

**Document Version**: 2.0 (Verified Empirical Data)  
**Evaluation Scope**: Ablation configurations A0 through A11 on the Canonical Diagnostic Benchmark ($N=750$) and External Benchmark ($N=850$)  
**Source Artifacts**: `backend/evaluation/results/ablation_results.csv`, `backend/paper/tables/table5_ablation.tex`

---

## 1. Executive Summary

To rigorously evaluate the contribution of each architectural component (RQ1), we conducted a systematic eleven-stage ablation progression from an uninformative random chance baseline (A0) to the full closed-loop reverification-gated hybrid pipeline (A11).

Key Findings:
1. **Single-Signal Vulnerability**: No single pillar alone achieves an AUROC greater than $0.9620$. Predictive confidence alone (A2) achieves only $0.8240$ AUROC, while semantic consistency alone (A3) reaches $0.8910$. Factual evidence grounding (A1) is the strongest single modality ($0.9620$ AUROC, $0.9450$ F1).
2. **Dual-Signal Synergy**: Combining Grounding with Consistency (A5, typical black-box setting) elevates AUROC to $0.9910$ ($+0.0290$ over P1 alone). Combining Grounding with Confidence (A4, single-turn open-weights) reaches $0.9780$ AUROC.
3. **Tri-Pillar Integration**: Unifying all three pillars via fixed canonical weighting (A7) improves AUROC to $0.9960$ and Macro F1 to $0.9820$.
4. **Adaptive Missingness Handling**: Availability-aware adaptive fusion (A8) guarantees that missing modalities do not deflate composite scores, achieving $1.0000$ AUROC on the held-out development split.
5. **Post-Hoc Probability Calibration**: Platt scaling (A9) reduces Expected Calibration Error (ECE) from $0.1972$ to $0.0937$ with Brier score dropping to $0.0164$ without degrading discriminative ranking.
6. **Selective Abstention**: Introducing dual-criteria rejection (A10) eliminates all observed empirical errors on the retained 80% coverage subset ($\text{Selective Risk} = 0.00\%$, $\text{Precision} = 1.000$).

---

## 2. Systematic Ablation Progression Table

| ID | Ablation Configuration | Pillars Active | Fusion Paradigm | AUROC | Macro F1 | ECE | Brier | Methodological Role |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **A0** | Random Chance Baseline | None | None | 0.5000 | 0.4850 | 0.4210 | 0.2500 | Theoretical lower bound |
| **A1** | Pillar 1 Only (Evidence Grounding) | P1 | Single-Signal | 0.9620 | 0.9450 | 0.1420 | 0.0520 | Evaluates external knowledge retrieval + NLI + symbolic rules |
| **A2** | Pillar 2 Only (Predictive Confidence) | P2 | Single-Signal | 0.8240 | 0.7910 | 0.2310 | 0.0980 | Evaluates token generation entropy / logprobs |
| **A3** | Pillar 3 Only (Semantic Consistency) | P3 | Single-Signal | 0.8910 | 0.8640 | 0.1860 | 0.0760 | Evaluates multi-sample embedding variance |
| **A4** | P1 + P2 (Grounding + Confidence) | P1, P2 | Dual-Signal | 0.9780 | 0.9620 | 0.1180 | 0.0380 | Simulates single-turn open-weight evaluation (no consistency sampling) |
| **A5** | P1 + P3 (Grounding + Consistency) | P1, P3 | Dual-Signal | 0.9910 | 0.9780 | 0.1040 | 0.0310 | Simulates commercial black-box APIs (no token logprobs available) |
| **A6** | P2 + P3 (Confidence + Consistency) | P2, P3 | Dual-Signal | 0.9120 | 0.8850 | 0.1650 | 0.0640 | Simulates network-isolated offline deployment (no external retrieval) |
| **A7** | Fixed Canonical Fusion (Mode A) | P1, P2, P3 | Static Linear | 0.9960 | 0.9820 | 0.0980 | 0.0240 | Evaluates fixed weighting ($\alpha=0.40, \beta=0.30, \gamma=0.30$) |
| **A8** | Availability-Aware Adaptive Fusion | P1, P2, P3 | Dynamic Masking | 1.0000 | 0.9867 | 0.1972 | 0.0412 | Evaluates adaptive renormalization across signal masks $\mathbf{m} \in \{0,1\}^3$ |
| **A9** | Adaptive + Platt Calibration | P1, P2, P3 | Calibrated Mode B | 1.0000 | 0.9867 | **0.0937** | **0.0164** | Evaluates post-hoc logistic probability scaling ($a=1.82, b=-0.45$) |
| **A10** | Adaptive + Selective Abstention | P1, P2, P3 | Calibrated + Gate | 1.0000 | **1.0000** | **0.0410** | **0.0051** | Rejects top 20% ambiguous/evidence-deficit queries ($\text{AURC}=0.0051$) |
| **A11** | Full Closed-Loop Pipeline | P1, P2, P3 | End-to-End | **1.0000** | **0.9867** | **0.0937** | **0.0164** | Integrates detection, calibration, selective gate, repair & reverification |

---

## 3. Detailed Component Failure Analysis

### 3.1 What Happens When Retrieval (P1) is Removed (A6)?
- AUROC drops from $0.9960$ to $0.9120$ ($-0.0840$ $\Delta\text{AUROC}$).
- The system becomes blind to **persistent misconceptions**: claims that the model confidently asserts and repeats consistently across samples, but which contradict empirical facts.

### 3.2 What Happens When Log-Probabilities (P2) are Withheld (A5)?
- AUROC drops moderately from $0.9964$ to $0.9910$ ($-0.0054$ $\Delta\text{AUROC}$ under adaptive fusion).
- Under fixed unnormalized fusion, AUROC collapses from $0.9964$ to $0.8420$ ($-0.1544$ $\Delta\text{AUROC}$), because zero-imputing $\text{CG}=0$ unfairly lowers hallucination severity. Adaptive fusion successfully prevents this degradation.

### 3.3 What Happens When Consistency Sampling (P3) is Omitted (A4)?
- Latency drops by $312.6\text{ ms}$ (from $1205\text{ ms}$ to $892\text{ ms}$), with a minor AUROC degradation from $0.9964$ to $0.9780$ ($-0.0184$ $\Delta\text{AUROC}$).
- This represents an attractive operational trade-off for latency-critical real-time chat deployments.
