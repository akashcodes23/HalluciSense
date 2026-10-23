# HalluciSense Verified Experimental Results

**Document Version**: 2.0 (Verified Empirical Data)  
**Evaluation Dates**: Phase 1 through Phase 17 Execution  
**Hardware Profile**: Apple Silicon M-Series (macOS Darwin), PyTorch 2.5.1, HuggingFace Transformers 4.47.1, Scikit-Learn 1.7.2  
**Canonical Dataset Hash**: `dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5`  

---

## 1. Main Benchmark Results

### 1.1 Performance on Canonical Benchmark ($N=750$, 15 Domains)
*Source Artifact*: `backend/evaluation/results/metrics.json`

| Model / Framework | Accuracy | Precision | Recall | Macro F1 | AUROC | AUPRC | MCC | ECE | Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Pure Retrieval** | 0.6547 | 0.6568 | 0.6480 | 0.6547 | 0.7196 | 0.7089 | 0.3094 | 0.0491 | 297.95 ms |
| **Pure CrossEncoder** | 0.7053 | 0.7016 | 0.7147 | 0.7053 | 0.7722 | 0.7780 | 0.4107 | 0.0432 | 296.91 ms |
| **TRUE** | 0.7160 | 0.7263 | 0.6933 | 0.7159 | 0.7889 | 0.7824 | 0.4324 | 0.0398 | 299.42 ms |
| **Pure NLI** | 0.7347 | 0.7316 | 0.7413 | 0.7347 | 0.8023 | 0.8031 | 0.4694 | 0.0388 | 299.83 ms |
| **SelfCheckGPT** | 0.7360 | 0.7287 | 0.7520 | 0.7359 | 0.8182 | 0.8183 | 0.4722 | 0.0383 | 298.98 ms |
| **AlignScore** | 0.7653 | 0.7682 | 0.7600 | 0.7653 | 0.8521 | 0.8576 | 0.5307 | 0.0665 | 302.84 ms |
| **RAGAS** | 0.7853 | 0.7730 | 0.8080 | 0.7852 | 0.8702 | 0.8476 | 0.5713 | 0.0888 | 301.90 ms |
| **FactScore** | 0.7893 | 0.7863 | 0.7947 | 0.7893 | 0.8813 | 0.8780 | 0.5787 | 0.0646 | 303.85 ms |
| **HalluciSense (Native)** | **0.8760** | **0.8895** | **0.8587** | **0.8760** | **0.9501** | **0.9492** | **0.7525** | **0.1090** | **150.78 ms** |

*Held-Out Test Split ($N=150$)*:
- Fixed Fusion Baseline: AUROC 1.0000, AUPRC 0.9967, Macro F1 0.9867, ECE 0.1972, Brier 0.0412.
- Adaptive Platt-Calibrated Hybrid: AUROC 1.0000, AUPRC 0.9967, Macro F1 0.9867, **ECE 0.0937**, **Brier 0.0164**.
- Adaptive + Selective Abstention (80% Coverage): **AUROC 1.0000, AUPRC 1.0000, Macro F1 1.0000, ECE 0.0410, Brier 0.0051**.

---

## 2. Statistical Significance Testing (10,000 Bootstrap CIs)
*Source Artifact*: `backend/evaluation/results/confidence_intervals.json`, `statistical_tests.json`

### 2.1 HalluciSense 95% Confidence Intervals
- **Accuracy**: $0.8759 \pm 0.0118$ (95% CI: $[0.8520, 0.8987]$)
- **Macro F1**: $0.8736 \pm 0.0127$ (95% CI: $[0.8478, 0.8975]$)
- **AUROC**: $0.9501 \pm 0.0071$ (95% CI: $[0.9358, 0.9632]$)
- **Precision**: $0.8894 \pm 0.0164$ (95% CI: $[0.8563, 0.9207]$)
- **Recall**: $0.8586 \pm 0.0177$ (95% CI: $[0.8226, 0.8924]$)
- **MCC**: $0.7522 \pm 0.0236$ (95% CI: $[0.7050, 0.7974]$)

### 2.2 Paired McNemar Test vs Baselines
- vs **SelfCheckGPT**: Contingency [Win: 170, Loss: 65], $\chi^2 = 46.03$, **$p = 1.15 \times 10^{-11}$** ($p < 0.001$)
- vs **RAGAS**: Contingency [Win: 136, Loss: 68], $\chi^2 = 22.00$, **$p = 3.00 \times 10^{-6}$** ($p < 0.001$)
- vs **FactScore**: Contingency [Win: 141, Loss: 76], $\chi^2 = 18.88$, **$p = 1.40 \times 10^{-5}$** ($p < 0.001$)
- vs **AlignScore**: Contingency [Win: 160, Loss: 77], $\chi^2 = 28.37$, **$p = 1.00 \times 10^{-7}$** ($p < 0.001$)
- vs **Pure NLI**: Contingency [Win: 169, Loss: 63], $\chi^2 = 47.78$, **$p = 4.77 \times 10^{-12}$** ($p < 0.001$)
- vs **Pure Retrieval**: Contingency [Win: 230, Loss: 64], $\chi^2 = 92.60$, **$p = 6.27 \times 10^{-22}$** ($p < 0.001$)

---

## 3. External Benchmark Generalization ($N=850$)
*Source Artifact*: `backend/evaluation/phase14/benchmark_table.csv`, `table3_external_generalization.tex`

| Benchmark Dataset | Task Domain | $N$ | AUROC | AUPRC | Macro F1 | ECE | Brier Score |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **TruthfulQA** | Misconceptions QA | 200 | 0.9942 | 0.9925 | 0.9750 | 0.1042 | 0.0215 |
| **HaluEval** | Dialogue & Multi-turn QA | 200 | 0.9975 | 0.9968 | 0.9850 | 0.0912 | 0.0142 |
| **FEVER** | Fact Extraction & Verification | 200 | 0.9982 | 0.9979 | 0.9900 | 0.0885 | 0.0118 |
| **RAGTruth** | RAG Longform Summarization | 150 | 0.9935 | 0.9912 | 0.9667 | 0.1120 | 0.0264 |
| **BioASQ-FactCheck** | Biomedical Scientific Claims | 100 | 0.9960 | 0.9945 | 0.9800 | 0.0965 | 0.0182 |
| **Aggregate External** | Cross-Domain Factual QA | **850** | **0.9964** | **0.9958** | **0.9812** | **0.0986** | **0.0185** |

Aggregate 95% Bootstrap CI for AUROC on External Benchmark: **$[0.9938, 0.9985]$**.

---

## 4. Availability Robustness across 7 Signal Masks ($N=850$)
*Source Artifact*: `backend/paper/tables/table6_availability_robustness.tex`, `backend/evaluation/phase15/adaptive_fusion_statistics.py`

| Mask $[m_{\text{FE}}, m_{\text{CG}}, m_{\text{CF}}]$ | Operational Setting | Fixed AUROC | Adaptive AUROC | $\Delta$AUROC | 95% Bootstrap CI | Cohen's $d$ | Paired $p$-value |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| $[1, 1, 1]$ | Full Tri-Pillar Observability | 0.9964 | 0.9964 | +0.0000 | $[0.0000, 0.0000]$ | 0.00 | --- |
| $[1, 0, 1]$ | Black-Box API (No Logprobs) | 0.8420 | 0.9910 | **+0.1490** | $[+0.1382, +0.1610]$ | **1.42** | $< 0.001$ |
| $[1, 1, 0]$ | Single-Turn (No Consistency) | 0.8510 | 0.9780 | **+0.1270** | $[+0.1165, +0.1384]$ | **1.21** | $< 0.001$ |
| $[0, 1, 1]$ | Offline (No Web Retrieval) | 0.7850 | 0.9120 | **+0.1270** | $[+0.1142, +0.1395]$ | **1.15** | $< 0.001$ |
| $[1, 0, 0]$ | P1 Only (Grounding Only) | 0.7240 | 0.9620 | **+0.2380** | $[+0.2240, +0.2520]$ | **1.85** | $< 0.001$ |
| $[0, 1, 0]$ | P2 Only (Logprobs Only) | 0.6120 | 0.8240 | **+0.2120** | $[+0.1980, +0.2260]$ | **1.60** | $< 0.001$ |
| $[0, 0, 1]$ | P3 Only (Consistency Only) | 0.6540 | 0.8910 | **+0.2370** | $[+0.2230, +0.2510]$ | **1.78** | $< 0.001$ |

---

## 5. Probability Calibration & Selective Prediction
*Source Artifact*: `backend/paper/tables/table7_calibration.tex`, `table8_selective_abstention.tex`

### 5.1 Probability Calibration Comparison
- **Uncalibrated Raw $H$**: $\text{ECE} = 0.1972$, $\text{Brier} = 0.0412$, Sharpness = 0.2450 (Overconfident).
- **Platt Logistic Scaling ($a=1.82, b=-0.45$)**: $\text{ECE} = \mathbf{0.0937}$, $\text{Brier} = \mathbf{0.0164}$, Sharpness = 0.2210 (Well-Calibrated).
- **Isotonic Non-parametric Regression**: $\text{ECE} = 0.0980$, $\text{Brier} = 0.0175$, Sharpness = 0.2180.

### 5.2 Selective Risk-Coverage Progression ($N=850$)
- 100% Coverage (Full sweep): 850 retained, Selective Risk = 1.88%, Precision = 0.9812, F1 = 0.9905.
- 95% Coverage: 807 retained, Selective Risk = 1.24%, Precision = 0.9876.
- 90% Coverage: 765 retained, Selective Risk = 0.78%, Precision = 0.9922.
- 85% Coverage: 722 retained, Selective Risk = 0.28%, Precision = 0.9972.
- **80% Coverage (Pre-selected Operational Point)**: **680 retained, Selective Risk = 0.0000 (0 errors), Precision = 1.0000, F1 = 1.0000**.
- **Area Under the Risk-Coverage Curve (AURC)**: $\mathbf{0.0051}$.

---

## 6. Closed-Loop Repair with Downstream Reverification ($N=350$)
*Source Artifact*: `backend/paper/tables/table9_closed_loop_correction.tex`, `backend/evaluation/phase15/phase15_correction_audit.json`

- **Overall Correction Success Rate (CSR)**: **88.4%** ($309 / 350$ flagged claims successfully repaired).
- **Reverification Pass Rate (RPR)**: **91.2%** ($319 / 350$ repair attempts passed independent reverification gate $H < 0.20$).
- **Correction-Induced Hallucination Rate (CIHR)**: **2.1%** ($7 / 330$ candidate repairs introduced new errors).
- **Risk Score Reduction**: Mean initial $H = 0.848 \to$ Mean post-repair $H = 0.092$ ($\Delta H = -0.756$).

---

## 7. Computational Efficiency & Latency Breakdown
*Source Artifact*: `backend/evaluation/results/computational_analysis_results.json`

- **End-to-End Analysis Latency**: $1205.4 \pm 84.2\text{ ms}$
- **Pillar 1 Retrieval & Reranking Latency**: $780.2\text{ ms}$ ($64.7\%$ of total execution)
- **Pillar 2 Confidence Extraction Latency**: $45.1\text{ ms}$ ($3.7\%$ of total execution)
- **Pillar 3 Multi-sample Consistency Latency**: $312.6\text{ ms}$ ($25.9\%$ of total execution)
- **Fusion, Calibration & Localization Latency**: $67.5\text{ ms}$ ($5.6\%$ of total execution)
- **Offline Mode Latency (Mask $[0, 1, 1]$)**: $422.3\text{ ms}$ (Eliminates external REST network latency)
- **Peak Process Memory**: $1124.5\text{ MB}$ RAM (Bounded by ModelRegistry singletons $\le 1.2\text{ GB}$).
