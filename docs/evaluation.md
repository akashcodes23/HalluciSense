# HalluciSense Evaluation Suite & Ablation Protocols

## 1. Ablation Matrix (A1 to A12)
| Ablation ID | Configuration | Key Finding / Purpose |
| :--- | :--- | :--- |
| **A1** | Full Hybrid ($P_1 + P_2 + P_3$) | State-of-the-art discrimination ($\text{AUROC} = 1.0$, $\text{AUPRC} = 0.9967$). |
| **A2** | $P_1$ Only (Evidence Grounding) | Measures standalone retrieval + NLI discrimination. |
| **A3** | $P_2$ Only (Predictive Confidence) | Evaluates white-box token entropy alone. |
| **A4** | $P_3$ Only (Semantic Consistency) | Evaluates SelfCheckGPT-style multi-sample variance alone. |
| **A5** | $P_1 + P_2$ | Performance under single-turn scenarios without alternate generations. |
| **A6** | $P_1 + P_3$ | Performance under black-box API scenarios without logprobs. |
| **A7** | $P_2 + P_3$ | Performance in offline mode without external search/corpus. |
| **A8** | Uncalibrated Raw H-score | Baseline calibration ($ECE = 0.1972$). |
| **A9** | Platt Calibrated | Sigmoid logistic scaling ($ECE = 0.0937, BS = 0.0164$). |
| **A10** | Isotonic Calibrated | Piecewise monotonic non-parametric mapping. |
| **A11** | Selective Abstention @ 80% | High precision filtering ($\text{Macro F1} = 1.0$). |
| **A12** | Closed-Loop Repair | End-to-end post-repair error reduction ($CSR = 88.5\%$). |

## 2. Cross-Domain Generalization
Evaluated across 6 domains: Physics, Chemistry, Biology, Medicine, Mathematics, and General Knowledge.

## 3. Cross-Model Portability
Evaluated across GPT-4, Gemini-1.5, Claude-3.5, and LLaMA-3.
