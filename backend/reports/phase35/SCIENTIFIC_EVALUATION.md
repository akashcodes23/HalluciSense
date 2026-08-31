# HalluciSense Scientific Evaluation & Reproducibility Report

## 1. Executive Scientific Overview

HalluciSense was systematically evaluated using a multi-phase experimental protocol spanning benchmark construction, ablation studies, hyperparameter optimization, and calibration analysis.

This document compiles the **empirically established facts** from the repository artifacts and evaluation datasets.

---

## 2. Dataset Identity & Statistics

### A. Development & Training Dataset (`phase6m` / `phase6k`)
- **Total Training Samples**: `58,002` instances.
- **Partition**: Stratified development partition with cross-domain representation (Biomedical, Open-Domain QA, Entity Knowledge, Scientific Reasoning).
- **Label Convention**:
  - `0`: Factual / Non-Hallucinated ($y = 0$).
  - `1`: Hallucinated / Unsupported ($y = 1$).
- **Class Balance**: Balanced distribution (~50/50 factual vs hallucinated claims).

### B. Scientific Benchmark Dataset (`benchmark_dataset.jsonl`)
- **File Location**: `backend/evaluation/results/benchmark_dataset.jsonl`
- **File Size**: `295,354 bytes` (1,000 curated multi-domain evaluation instances).
- **SHA-256 Checksum**: `dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5`

---

## 3. Verified Model Performance Metrics

From `backend/evaluation_results/phase6m/final_hybrid_model/model_metadata.json`:

| Metric | Measured Value | Scientific Interpretation |
| :--- | :--- | :--- |
| **ROC-AUC** | **`0.7378`** | Area Under Receiver Operating Characteristic curve across probability thresholds |
| **F1 Score** | **`0.7100`** | Harmonic mean of precision and recall at optimal threshold $\tau^* = 0.54$ |
| **Accuracy** | **`0.6770`** | Overall classification accuracy on development validation set |
| **Matthews Correlation Coefficient (MCC)** | **`0.3466`** | Balanced correlation measure for binary classification |
| **Optimal Decision Threshold ($\tau^*$)** | **`0.5400`** | Threshold maximizing Youden's Index ($J = \text{Sensitivity} + \text{Specificity} - 1$) |

---

## 4. Pillar & Model Ablation Results

From `backend/evaluation_results/phase6k/` and `phase6ir/` historical audits:

| Model Architecture | Features Used | ROC-AUC | F1 Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Pillar 1 Standalone (Grounding Only)** | 5 features (`p1_*`) | `0.6842` | `0.6510` | Baseline / Fallback |
| **Pillar 2 Standalone (Consistency Only)**| 5 features (`p2_*`) | `0.6215` | `0.5980` | Subordinate Baseline |
| **Linear Fusion (Logistic Regression)** | 10 features | `0.7012` | `0.6740` | Intermediate |
| **Full Hybrid (HistGradientBoosting)** | **19 interaction features** | **`0.7378`** | **`0.7100`** | **Selected Production Model** |

### Key Scientific Takeaway:
Combining cross-pillar interaction features (probability disagreement $|P_1 - P_2|$, logit differences, probability ratios, and extreme values) with non-linear tree ensembles yielded a **+5.36% improvement in ROC-AUC** over the best standalone single pillar.

---

## 5. Decision Threshold Derivation ($\tau^* = 0.54$)

1. Standard naive classifiers apply a default cut-off of $\tau = 0.50$.
2. In real-world retrieval-augmented hallucination detection, imperfect retrieval precision (e.g., knowledge base omissions) causes slight upward bias in raw hallucination probabilities for factual claims.
3. Operating threshold analysis conducted across the interval $[0.30, 0.70]$ in increments of $0.01$ established that $\tau^* = 0.54$ minimizes false positives without degrading critical hallucination detection sensitivity.
4. Consequently:
   $$\hat{y} = \begin{cases} 1 \text{ (Hallucinated)}, & P(\text{hallucination} \mid \mathbf{x}) \ge 0.54 \\ 0 \text{ (Factual)}, & P(\text{hallucination} \mid \mathbf{x}) < 0.54 \end{cases}$$

---

## 6. Unreproduced / Unestablished Claims (Honest Scientific Boundaries)

To maintain scientific integrity during viva examination:

1. **Specific LLM Generation Benchmarks (e.g. GPT-4o, Claude 3.5 Sonnet specific accuracy breakdowns)**:
   - *Status*: The core 19-feature classifier was trained on standardized sentence-level claim pairs. Model-specific generation scores depend on dynamic prompt styles and are not claimed as fixed static constants.
2. **Cross-Language Generalization**:
   - *Status*: Not established. HalluciSense is validated primarily for English NLP corpora.
3. **Pillar 3 Active Weight in Single-Generation Mode**:
   - *Status*: In standard single-prompt inference, Pillar 3 (Multi-LLM Consensus) is omitted from the request payload to preserve sub-second latency; the fusion engine automatically renormalizes weights over available Pillars 1 and 2 ($\alpha=1.0$).
