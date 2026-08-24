# Phase 18 — Statistical Audit & Metric Verification

## 1. Executive Summary & Audited Metrics Ledger

| Metric | Benchmark Dataset | Sample Size ($N$) | Metric Value | 95% Confidence Interval | Denominator Definition | Statistical Test / Protocol | Source Artifact |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- | :--- |
| **AUROC** | Held-Out Test Split | 150 | `1.0000` | `[1.0000, 1.0000]` | Total test query-response pairs | Nonparametric Bootstrap ($B=500$) | `table2_main_results.csv` |
| **AUPRC** | Held-Out Test Split | 150 | `0.9967` | `[0.9920, 1.0000]` | Total positive (hallucinated) claims | Nonparametric Bootstrap ($B=500$) | `table2_main_results.csv` |
| **ECE (Platt)** | Held-Out Test Split | 150 | `0.0937` | `[0.0820, 0.1060]` | 10 uniform probability bins | Logistic Sigmoid Scaling ($a=1.82, b=-0.45$) | `table7_calibration.csv` |
| **Brier Score** | Held-Out Test Split | 150 | `0.0164` | `[0.0120, 0.0210]` | Total test instances | Mean Squared Probabilistic Error | `table7_calibration.csv` |
| **AUROC (Ext)** | Combined External | 850 | `0.9964` | `[0.9938, 0.9985]` | 5 public external datasets | Nonparametric Bootstrap ($B=500$) | `table3_external_generalization.csv` |
| **$\Delta\text{AUROC}$** | External Mask `[1, 0, 1]` | 850 | `+0.1490` | `[+0.1382, +0.1610]` | Paired per-instance bootstrap delta | Paired Wilcoxon Signed-Rank ($p < 0.001$) | `table6_availability_robustness.csv` |
| **Cohen's $d$** | External Mask `[1, 0, 1]` | 850 | **`1.42`** | `[1.31, 1.53]` | Standardized paired score difference $\frac{\bar{D}}{s_D}$ | Standard Paired Cohen's $d$ Formulation | `table6_availability_robustness.csv` |
| **AURC** | External Combined | 850 | `0.0051` | `[0.0041, 0.0062]` | Integrated risk over $[0, 1]$ coverage | Trapezoidal Rule on Risk-Coverage Curve | `table8_selective_abstention.csv` |
| **Selective Risk**| 80% Coverage Subset | 680 | `0.0000` | `[0.0000, 0.0000]` | Retained non-abstained queries | Empirical Error on Retained Subset | `table8_selective_abstention.csv` |
| **CSR** | Closed-Loop Repair | 350 | `88.4%` | `[85.1%, 91.7%]` | **Flagged-claim level** ($N=350$) | Deterministic Policy + Reverification | `table9_closed_loop_correction.csv` |
| **RPR** | Closed-Loop Repair | 350 | `91.2%` | `[88.2%, 94.2%]` | **Repair-attempt level** ($N=350$) | Downstream Gate ($H_{\text{post}} < 0.20$) | `table9_closed_loop_correction.csv` |
| **CIHR** | Closed-Loop Repair | 350 | `2.1%` | `[0.9%, 3.3%]` | **Accepted-repaired-claim level** ($N=319$) | Post-repair downstream contradiction check | `table9_closed_loop_correction.csv` |

---

## 2. Definitive Ruling on Cohen's $d$
- **Proper Standardized Effect Size:** Cohen's $d = \mathbf{1.42}$ (Paired difference score standardized by standard deviation).
- **Prohibited Overclaim:** Historical bootstrap mean $z$-score ($z = 25.69$) is strictly forbidden from being reported as Cohen's $d$ in the manuscript text and tables.
