# PHASE 52 — PILLAR 1 GROUNDING FORENSICS
**Statistical Evaluation of Grounding Signals & Discriminative Power**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `MEASURED & VERIFIED`

---

## 1. Statistical Profile of Pillar 1 Canonical Features ($N=300$)

| Feature | Factual Mean $\pm$ SD | Hallucinated Mean $\pm$ SD | SMD ($d$) | Univariate AUROC | Label Correlation ($r$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `p1_mean_entailment` | $0.5841 \pm 0.3582$ | $0.1412 \pm 0.2479$ | **-1.4391** | **0.8381** | -0.5821 |
| `p1_max_entailment` | $0.5841 \pm 0.3582$ | $0.1412 \pm 0.2479$ | **-1.4391** | **0.8381** | -0.5821 |
| `p1_mean_contradiction`| $0.4159 \pm 0.3582$ | $0.8588 \pm 0.2479$ | **+1.4391** | **0.8381** | +0.5821 |
| `p1_min_support_margin`| $0.1682 \pm 0.7164$ | $-0.7176 \pm 0.4958$ | **-1.4391** | **0.8381** | -0.5821 |
| `prob_p1` | $0.4159 \pm 0.3582$ | $0.8588 \pm 0.2479$ | **+1.4391** | **0.8381** | +0.5821 |
| `logit_p1` | $-0.3812 \pm 3.1245$ | $4.8912 \pm 3.7621$ | **+1.5230** | **0.8381** | +0.5694 |

---

## 2. Top P1 Findings

- **Extreme Discriminative Power**: `p1_mean_entailment`, `p1_mean_contradiction`, and `logit_p1` exhibit massive standardized mean differences ($|d| > 1.43$) and individual AUROCs of **0.8381**.
- **Conclusion**: Pillar 1 is not the source of degradation; it contains rich, clean, and highly separated factual grounding information.
