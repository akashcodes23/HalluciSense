# HalluciSense 19-Feature Provenance & Mathematical Audit

## 1. Overview

This document provides an independent trace of all 19 features consumed by the frozen `HistGradientBoostingClassifier` metadata model (`hybrid_meta_classifier.joblib`). 

Each feature is traced to its **source code file**, **producing component**, **exact mathematical formula**, and **0-indexed position** in the input vector $\mathbf{x} \in \mathbb{R}^{19}$.

---

## 2. Comprehensive 19-Feature Provenance Matrix

| Index | Feature Name | Source File | Producing Subsystem | Mathematical Definition | Risk Correlation |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **0** | `p1_mean_entailment` | `backend/app/core/engine/pillar1.py` | Pillar 1 (DeBERTa NLI) | $\frac{1}{K} \sum_{k=1}^K P(\text{Entailment} \mid c_k, e_{k}^*)$ | Lower $\to$ Higher Risk |
| **1** | `p1_max_entailment` | `backend/app/core/engine/pillar1.py` | Pillar 1 (DeBERTa NLI) | $\max_{k \in [1, K]} P(\text{Entailment} \mid c_k, e_{k}^*)$ | Lower $\to$ Higher Risk |
| **2** | `p1_mean_contradiction` | `backend/app/core/engine/pillar1.py` | Pillar 1 (DeBERTa NLI) | $\frac{1}{K} \sum_{k=1}^K P(\text{Contradiction} \mid c_k, e_{k}^*)$ | Higher $\to$ Higher Risk |
| **3** | `p1_min_support_margin` | `backend/app/core/engine/pillar1.py` | Pillar 1 (DeBERTa NLI) | $\min_k [P(\text{Entailment}) - P(\text{Contradiction})]$ | Lower $\to$ Higher Risk |
| **4** | `p1_num_claims` | `backend/app/core/engine/pillar1.py` | Claim Extractor | $K = \|\mathcal{C}\|$ (Count of propositional claims) | Contextual |
| **5** | `p2_max_pairwise_contradiction` | `backend/app/core/engine/pillar2.py` | Pillar 2 (Consistency) | $\max_{i \neq j} P(\text{Contradiction} \mid s_i, s_j)$ | Higher $\to$ Higher Risk |
| **6** | `p2_mean_pairwise_contradiction`| `backend/app/core/engine/pillar2.py` | Pillar 2 (Consistency) | $\frac{2}{N(N-1)} \sum_{i < j} P(\text{Contradiction} \mid s_i, s_j)$ | Higher $\to$ Higher Risk |
| **7** | `p2_max_pairwise_similarity` | `backend/app/core/engine/pillar2.py` | SentenceTransformer | $\max_{i \neq j} \cos(\mathbf{e}_{s_i}, \mathbf{e}_{s_j})$ | Inverted redundancy |
| **8** | `p2_fraction_contradictory_pairs`| `backend/app/core/engine/pillar2.py` | Pillar 2 (Consistency) | $\frac{1}{\binom{N}{2}} \sum_{i < j} \mathbb{I}[P(\text{Contra}) > 0.5]$ | Higher $\to$ Higher Risk |
| **9** | `p2_num_claims` | `backend/app/core/engine/pillar2.py` | Sentence Tokenizer | $N = \|\mathcal{S}\|$ (Count of sentences in response) | Contextual |
| **10** | `prob_p1` | `backend/app/models/registry.py` | Pillar 1 Base Model | $\sigma(\mathbf{w}_1^T \mathbf{x}_{P1} + b_1)$ (P1 Baseline Prob) | Higher $\to$ Higher Risk |
| **11** | `prob_p2` | `backend/app/models/registry.py` | Pillar 2 Base Model | $\sigma(\mathbf{w}_2^T \mathbf{x}_{P2} + b_2)$ (P2 Baseline Prob) | Higher $\to$ Higher Risk |
| **12** | `logit_p1` | `backend/app/models/registry.py` | Feature Synthesizer | $\ln\left(\frac{\text{prob\_p1} + \epsilon}{1 - \text{prob\_p1} + \epsilon}\right)$ | Higher $\to$ Higher Risk |
| **13** | `logit_p2` | `backend/app/models/registry.py` | Feature Synthesizer | $\ln\left(\frac{\text{prob\_p2} + \epsilon}{1 - \text{prob\_p2} + \epsilon}\right)$ | Higher $\to$ Higher Risk |
| **14** | `prob_disagreement_abs` | `backend/app/models/registry.py` | Cross-Pillar Delta | $\|\text{prob\_p1} - \text{prob\_p2}\|$ | Higher $\to$ Ambiguity |
| **15** | `prob_mean` | `backend/app/models/registry.py` | Unweighted Average | $\frac{\text{prob\_p1} + \text{prob\_p2}}{2}$ | Higher $\to$ Higher Risk |
| **16** | `prob_max` | `backend/app/models/registry.py` | Worst-Case Signal | $\max(\text{prob\_p1}, \text{prob\_p2})$ | Higher $\to$ Higher Risk |
| **17** | `prob_min` | `backend/app/models/registry.py` | Best-Case Signal | $\min(\text{prob\_p1}, \text{prob\_p2})$ | Higher $\to$ Higher Risk |
| **18** | `prob_ratio` | `backend/app/models/registry.py` | Cross-Pillar Ratio | $\frac{\text{prob\_p1} + \epsilon}{\text{prob\_p2} + \epsilon}$ ($\epsilon = 10^{-6}$) | Skewed $\to$ Ambiguity |

---

## 3. Preprocessing Transformation Pipeline

Before feeding $\mathbf{x} \in \mathbb{R}^{19}$ into `HistGradientBoostingClassifier`, the raw vector undergoes non-linear robust scaling via `RobustScaler` (`preprocessing.joblib`):

$$\tilde{x}_j = \frac{x_j - \text{median}(X_j)}{\text{IQR}(X_j)} = \frac{x_j - Q_2(X_j)}{Q_3(X_j) - Q_1(X_j)}$$

### Why `RobustScaler` is Mathematically Crucial:
- Features 12, 13 (`logit_p1`, `logit_p2`) and 18 (`prob_ratio`) produce long-tailed asymptotic values when probabilities approach 0 or 1.
- `StandardScaler` (mean/variance) would be distorted by outliers, pulling decision tree split candidates away from dense operational regimes.
- `RobustScaler` bounds the median IQR spread, ensuring numerical stability.
