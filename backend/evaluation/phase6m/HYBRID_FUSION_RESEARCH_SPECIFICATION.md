# HalluciSense Phase 6M.0 — Hybrid Fusion Research Specification & Architecture

**Document Version**: `1.0.0`  
**Status**: `FROZEN RESEARCH SPECIFICATION`  
**Author**: `Google DeepMind Advanced Agentic Coding Team`  
**Framework**: `HalluciSense: A Confidence-Aware Hybrid Framework for Detecting and Quantifying Hallucinations in LLMs`  

---

## Executive Summary

Phase 6K established that **Pillar 1** (Claim-Evidence Entailment) is scientifically validated ($\text{ROC-AUC} \approx 0.626$ on held-out validation with stable generalization). Phase 6L established that **Pillar 2** (Internal Structural Consistency) suffers from domain distribution shift when evaluated as an isolated standalone classifier ($\text{ROC-AUC} = 0.5784$, status: `NOT VALIDATED`).

Phase 6M.0 defines the formal scientific, mathematical, and architectural specification for **HalluciSense Hybrid Fusion**: uniting external evidence grounding (Pillar 1) and internal structural consistency (Pillar 2) into a unified, confidence-aware hallucination detection engine.

> [!IMPORTANT]
> **Strict Research Specification Mandate**:
> This document is a **100% READ-ONLY RESEARCH SPECIFICATION**. There is ZERO code implementation, feature extraction, model fitting, cross-validation, or held-out testing in this phase.

---

## 1. Scientific Problem Definition

### 1.1 Formal Statement

Let $x = (c, e, r)$ represent an LLM generation instance consisting of a set of atomic claims $c = \{c_1, c_2, \dots, c_K\}$, retrieved evidence passages $e = \{e_1, e_2, \dots, e_M\}$, and full response text $r$.

Let $Y \in \{0, 1\}$ represent the binary ground-truth hallucination label ($Y=0$ for factual/supported, $Y=1$ for hallucinated/unsupported).

- **Pillar 1 (Claim-Evidence Entailment)** estimates the probability $P_1(Y=1 \mid c, e)$ based on external factual grounding against evidence $e$.
- **Pillar 2 (Structural Consistency)** estimates the probability $P_2(Y=1 \mid c)$ based on internal inter-claim pairwise contradiction, agreement, and graph topology within claims $c$.

The **HalluciSense Hybrid Fusion Operator** $H(x)$ is defined as a mapping:

$$H(x) = f\Big( \mathbf{x}_{\text{P1}}, \mathbf{x}_{\text{P2}}, P_1(Y=1 \mid c, e), P_2(Y=1 \mid c), \mathbf{x}_{\text{meta}} \Big) \;\in\; [0, 1]$$

where:
- $\mathbf{x}_{\text{P1}} \in \mathbb{R}^{d_1}$ is the Pillar-1 feature vector (5 locked features).
- $\mathbf{x}_{\text{P2}} \in \mathbb{R}^{d_2}$ is the Pillar-2 feature vector (5 locked features or 24 full schema features).
- $P_1, P_2 \in [0, 1]$ are the calibrated out-of-fold probability estimates from the frozen Pillar-1 and Pillar-2 base models.
- $\mathbf{x}_{\text{meta}} \in \mathbb{R}^{d_m}$ represents response structural controls (e.g. claim count $N_{\text{claims}}$, claim length variance).

### 1.2 Scientific Rationale

1. **Complementarity of Errors**:
   - Pillar 1 fails when evidence passages are noisy, missing, or partially uninformative, but structural claims within the response are internally contradictory.
   - Pillar 2 fails when hallucinations are self-consistent (internally coherent false statements), but Pillar 1 detects their lack of external evidence support.
2. **Noise Regularization**:
   - Integrating Pillar-1 evidence grounding prevents Pillar 2 from suffering threshold collapse under structural distribution shifts.
   - Pillar 2 provides structural penalty constraints when Pillar 1 evidence scores fall into the ambiguous decision boundary ($P_1 \approx 0.50$).

### 1.3 Assumptions & Limitations

- **Assumption 1 (Conditional Partial Independence)**: Given ground-truth state $Y$, Pillar-1 evidence alignment errors and Pillar-2 structural contradiction errors exhibit non-identical error distributions.
- **Limitation 1 (Evidence Dependency)**: Hybrid fusion cannot override a total absence of evidence when claims require domain-specific verification.
- **Limitation 2 (Distribution Sensitivity)**: Meta-learners must be regularized against Pillar-2 feature variance to prevent over-indexing on shifted structural metrics.

---

## 2. Mathematical Formulations

Eight candidate mathematical formulations are defined for candidate evaluation during model selection in Phase 6M.2. No single formulation is selected *a priori*.

### Formulation 1: Weighted Linear Decision Fusion

$$H_{\text{linear}}(x) = \alpha \cdot P_1(x) + (1 - \alpha) \cdot P_2(x), \quad \alpha \in [0, 1]$$

- Parameter $\alpha$ represents the global evidence weight, optimized on DEV via grid search over $\alpha \in [0.0, 1.0]$.

### Formulation 2: Calibrated Logistic Meta-Learner

$$H_{\text{logistic}}(x) = \sigma\left( w_0 + w_1 \cdot z(P_1) + w_2 \cdot z(P_2) + w_3 \cdot |z(P_1) - z(P_2)| + \mathbf{w}_{\text{meta}}^T \mathbf{z}_{\text{meta}} \right)$$

where $\sigma(z) = (1 + e^{-z})^{-1}$ is the logistic sigmoid, $z(\cdot)$ is standard normalization, and $|z(P_1) - z(P_2)|$ captures probability divergence.

### Formulation 3: Gradient Boosting Meta-Learner (LightGBM / XGBoost)

$$H_{\text{gbm}}(x) = \sigma\left( \sum_{m=1}^M \gamma_m \cdot h_m(\mathbf{x}_{\text{hybrid}}) \right)$$

where $h_m(\cdot)$ are shallow regression decision trees (depth $\le 4$) trained on the joint hybrid feature vector $\mathbf{x}_{\text{hybrid}}$.

### Formulation 4: Random Forest Meta-Learner

$$H_{\text{rf}}(x) = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x}_{\text{hybrid}})$$

where $T_b$ are randomized decision trees trained with bootstrap bagging and feature subsampling.

### Formulation 5: Stacked Generalization (Two-Stage Meta-Learning)

$$H_{\text{stack}}(x) = M_2\Big( \hat{P}_{1,\text{OOF}}(x), \hat{P}_{2,\text{OOF}}(x), \mathbf{x}_{\text{locked}} \Big)$$

Stage 1 produces out-of-fold probability estimates $\hat{P}_{1,\text{OOF}}$ and $\hat{P}_{2,\text{OOF}}$ using 5-fold cross-validation. Stage 2 trains meta-classifier $M_2$ strictly on these OOF predictions to eliminate stacking leakage.

### Formulation 6: Naïve Bayes Probabilistic Fusion

$$P(Y=1 \mid P_1, P_2) = \frac{P(P_1 \mid Y=1) \cdot P(P_2 \mid Y=1) \cdot P(Y=1)}{\sum_{y \in \{0,1\}} P(P_1 \mid Y=y) \cdot P(P_2 \mid Y=y) \cdot P(Y=y)}$$

assuming class-conditional independence between base classifier probabilities.

### Formulation 7: Probabilistic Product and Union Operators

$$H_{\text{product}}(x) = P_1(x) \cdot P_2(x)$$

$$H_{\text{union}}(x) = P_1(x) + P_2(x) - P_1(x) \cdot P_2(x)$$

$$H_{\text{harmonic}}(x) = \frac{2 \cdot P_1(x) \cdot P_2(x)}{P_1(x) + P_2(x) + \epsilon}$$

### Formulation 8: Confidence-Gated Dynamic Fusion

$$H_{\text{gated}}(x) = g(x) \cdot P_1(x) + \big(1 - g(x)\big) \cdot P_2(x)$$

$$g(x) = \sigma\left( \mathbf{w}_g^T \mathbf{x}_{\text{meta}} \right)$$

where the gating network $g(x) \in [0, 1]$ dynamically allocates authority to Pillar 1 or Pillar 2 based on response structural controls (e.g. claim count $N_{\text{claims}}$).

---

## 3. Input Representation

The joint hybrid input vector $\mathbf{x}_{\text{hybrid}} \in \mathbb{R}^D$ fed to candidate meta-learners is defined as:

$$\mathbf{x}_{\text{hybrid}} = \big[ \mathbf{x}_{\text{P1}}, \; \mathbf{x}_{\text{P2}}, \; P_1, \; P_2, \; \Delta P, \; \mathbf{x}_{\text{calib}}, \; \mathbf{x}_{\text{meta}} \big]$$

### 3.1 Schema Breakdown

| Feature Name | Origin | Description | Data Type | Range |
| :--- | :--- | :--- | :---: | :---: |
| `mean_entailment` | Pillar 1 | Average claim-evidence NLI entailment score | `float64` | $[0, 1]$ |
| `max_entailment` | Pillar 1 | Maximum claim-evidence NLI entailment score | `float64` | $[0, 1]$ |
| `mean_contradiction` | Pillar 1 | Average claim-evidence NLI contradiction score | `float64` | $[0, 1]$ |
| `min_support_margin` | Pillar 1 | Minimum margin $(E - C)$ across response claims | `float64` | $[-1, 1]$ |
| `num_claims` | Pillar 1 / Control | Count of atomic claims in response | `float64` | $[1, \infty)$ |
| `max_pairwise_contradiction` | Pillar 2 | Maximum pairwise inter-claim contradiction | `float64` | $[0, 1]$ |
| `mean_pairwise_contradiction` | Pillar 2 | Average pairwise inter-claim contradiction | `float64` | $[0, 1]$ |
| `max_pairwise_similarity` | Pillar 2 | Maximum pairwise claim embedding similarity | `float64` | $[-1, 1]$ |
| `fraction_contradictory_pairs` | Pillar 2 | Proportion of pairs with contradiction $> 0.50$ | `float64` | $[0, 1]$ |
| `p1_probability_raw` | Pillar 1 Model | Frozen Pillar-1 predicted probability $P_1(Y=1 \mid x)$ | `float64` | $[0, 1]$ |
| `p2_probability_raw` | Pillar 2 Model | Frozen Pillar-2 predicted probability $P_2(Y=1 \mid x)$ | `float64` | $[0, 1]$ |
| `probability_divergence` | Derived | Absolute probability difference $|P_1 - P_2|$ | `float64` | $[0, 1]$ |
| `p1_probability_platt` | Derived | Platt-calibrated Pillar-1 probability | `float64` | $[0, 1]$ |
| `p2_probability_platt` | Derived | Platt-calibrated Pillar-2 probability | `float64` | $[0, 1]$ |
| `claim_length_variance` | Pillar 2 Control | Variance of character lengths across claims | `float64` | $[0, \infty)$ |

---

## 4. Candidate Fusion Strategies

Five strategy families will be systematically evaluated during Phase 6M.2 model selection:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CANDIDATE FUSION STRATEGIES                       │
└─────────────────────────────────────────────────────────────────────────┘
   │
   ├─► 1. LATE FUSION (Decision-Level)
   │     • Probability averaging / weighted linear combination of P1 and P2
   │     • Zero feature interaction; low parameter count
   │
   ├─► 2. MID-LEVEL FEATURE FUSION (Joint Feature Space)
   │     • Concatenation of Pillar-1 (5) + Pillar-2 (5) locked feature vectors
   │     • Enables feature-level cross-pillar interaction modeling
   │
   ├─► 3. EARLY FEATURE FUSION (Full Schema Joint Matrix)
   │     • Joint space of Pillar-1 (5) + Pillar-2 Full Schema (24) = 29 features
   │     • Maximum expressive capacity; requires strict regularization
   │
   ├─► 4. META-CLASSIFIER STACKING (Two-Stage OOF Learning)
   │     • Meta-learner trained on OOF predictions [P1_OOF, P2_OOF, ΔP, Meta]
   │     • Prevents data leakage via strict cross-validation splitting
   │
   └─► 5. CALIBRATION-AWARE GATING (Dynamic Gating Network)
         • Input-dependent gating weight g(x) driven by claim count & divergence
         • Pre-calibrates input probabilities via Isotonic / Platt regression
```

---

## 5. Model Candidates

Six meta-classifier algorithms are nominated for evaluation. Each algorithm will be paired with `StandardScaler` and `RobustScaler` preprocessing.

### 5.1 Algorithm Comparison Matrix

| Model Algorithm | Hyperparameter Search Space | Key Advantages | Primary Risks / Disadvantages |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | `penalty=['l1', 'l2']`, `C=[0.01, 0.1, 1.0, 10.0]`, `solver='liblinear'` | Linear interpretability, low risk of overfitting, strict monotonic probability outputs | Limited capacity to model non-linear cross-pillar interactions |
| **Random Forest** | `n_estimators=[50, 100, 200]`, `max_depth=[3, 5, 8]`, `min_samples_leaf=[5, 10]` | Robust to outliers, non-linear interaction modeling, automatic feature selection | Step-wise decision boundaries, probability compression near extremes |
| **Gradient Boosting** | `n_estimators=[50, 100]`, `learning_rate=[0.01, 0.05, 0.1]`, `max_depth=[3, 5]` | State-of-the-art tabular accuracy, strong gradient optimization | Higher risk of overfitting on small feature spaces |
| **XGBoost** | `n_estimators=[50, 100]`, `max_depth=[3, 4]`, `reg_alpha=[0.1, 1.0]`, `reg_lambda=[1.0, 5.0]` | Built-in L1/L2 regularization, exact split finding, robust performance | Requires careful regularization tuning to prevent DEV memorization |
| **LightGBM** | `n_estimators=[50, 100]`, `num_leaves=[7, 15, 31]`, `learning_rate=[0.03, 0.1]` | Fast leaf-wise tree growth, excellent handling of numerical features | Prone to overfitting on small sample sizes if leaf count is unconstrained |
| **Extra Trees** | `n_estimators=[100, 200]`, `max_depth=[4, 6, 8]`, `min_samples_split=[5, 10]` | Random split point choice reduces variance compared to standard RF | May yield higher bias if feature signals are sparse |

---

## 6. Experimental Hypotheses

Phase 6M.2 and 6M.3 will formally test five pre-declared scientific hypotheses:

- **Hypothesis $H_1$ (Superiority over Pillar 1)**:  
  $$\text{ROC-AUC}_{\text{Hybrid}} > \text{ROC-AUC}_{\text{Pillar1}} + 0.0100 \quad \text{on held-out VAL}$$  
  *Rationale*: Incorporating structural consistency constraints improves factual discrimination over evidence-only entailment.

- **Hypothesis $H_2$ (MCC Improvement)**:  
  $$\text{MCC}_{\text{Hybrid}} > \text{MCC}_{\text{Pillar1}} + 0.0200 \quad \text{at optimal operating threshold}$$  
  *Rationale*: Joint probability optimization improves decision boundary crispness and Matthews Correlation Coefficient.

- **Hypothesis $H_3$ (Calibration Reliability)**:  
  $$\text{ECE}_{\text{Hybrid}} < 0.0300 \quad \text{on held-out VAL}$$  
  *Rationale*: Pre-calibration and meta-learning eliminate probability compression, ensuring Expected Calibration Error stays below $3\%$.

- **Hypothesis $H_4$ (False Positive Reduction)**:  
  $$\text{FPR}_{\text{Hybrid}} \le 0.90 \cdot \text{FPR}_{\text{Pillar1}} \quad \text{at matched recall}$$  
  *Rationale*: Pillar-2 structural contradiction penalties eliminate false positives caused by weak or ambiguous evidence retrieval.

- **Hypothesis $H_5$ (Generalization Stability)**:  
  $$|\text{ROC-AUC}_{\text{DEV\_CV}} - \text{ROC-AUC}_{\text{VAL\_HeldOut}}| \le 0.0200 \quad (\text{Status: } \texttt{STABLE})$$  
  *Rationale*: Evidence-grounded Pillar-1 features regularize Pillar-2 structural feature variance, preventing held-out generalization degradation.

---

## 7. Ablation Study Design

Eight systematic ablation configurations are specified to isolate feature contribution and component impact during Phase 6M.2:

```
========================================================================================
                               ABLATION STUDY MATRIX
========================================================================================
Exp ID   Ablation Target                  Pillar-1 Feats   Pillar-2 Feats   Meta/Calib
──────   ─────────────────────────────    ──────────────   ──────────────   ──────────
EXP-1    Pillar 1 Only (Baseline)         Included (5)     Excluded         Excluded
EXP-2    Pillar 2 Only (Baseline)         Excluded         Included (5)     Excluded
EXP-3    Full Hybrid (Canonical)          Included (5)     Included (5)     Included
EXP-4    w/o Graph Topology               Included (5)     Excl Graph (3)   Included
EXP-5    w/o Structural Contradiction     Included (5)     Excl Contra (3)  Included
EXP-6    w/o Evidence Entailment          Excl Ent (2)     Included (5)     Included
EXP-7    w/o Calibration & Prob Signals   Included (5)     Included (5)     Excl Probs
EXP-8    w/o Response Structural Controls Included (5)     Included (5)     Excl Controls
========================================================================================
```

---

## 8. Statistical Analysis Plan

All Phase 6M evaluations must adhere to the following statistical rigor standards:

1. **Stratified Bootstrap Confidence Intervals**:
   - $B = 2,000$ stratified bootstrap resamples on held-out VAL ($N = 12,483$).
   - Export 95% percentile confidence intervals $[q_{0.025}, q_{0.975}]$ for ROC-AUC, PR-AUC, Accuracy, Balanced Accuracy, Precision, Recall, Specificity, F1, MCC, and Brier Score.

2. **DeLong Test for Paired ROC Curves**:
   - Compute exact non-parametric DeLong $Z$-statistic and $p$-value comparing $\text{ROC-AUC}_{\text{Hybrid}}$ vs $\text{ROC-AUC}_{\text{Pillar1}}$.
   - Declare statistical superiority if $p < 0.001$.

3. **McNemar's Test for Classification Concordance**:
   - Evaluate paired $2 \times 2$ contingency matrix of discordances ($n_{01}$ vs $n_{10}$) between Hybrid model and Pillar-1 model at matched operating thresholds.

4. **Net Reclassification Improvement (NRI)**:
   - Compute Continuous NRI and Category-Bound NRI to quantify correct upward/downward probability reclassifications for factual vs hallucinated instances.

5. **Decision Curve Analysis (DCA)**:
   - Calculate Net Benefit across threshold probability range $p_t \in [0.10, 0.90]$:
     $$\text{Net Benefit}(p_t) = \frac{\text{TP}}{N} - \frac{\text{FP}}{N} \cdot \left( \frac{p_t}{1 - p_t} \right)$$
   - Plot DCA curves comparing Hybrid Model vs "Treat All" and "Treat None" clinical/practical baselines.

---

## 9. Firewall Protocol & Data Isolation

Phase 6M strictly enforces the data isolation firewall established in Phase 6K and 6L:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA FIREWALL ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────────────────┘
  DEVELOPMENT PARTITION (DEV)                HELD-OUT VALIDATION (VAL)
  N = 58,002 Responses                       N = 12,483 Responses
  
  Used for:                                  Status:
  • Feature extraction                       • STRICTLY SEALED & READ-ONLY
  • Scaler fitting                           • Zero feature selection
  • Meta-learner training                    • Zero model tuning
  • 5-fold 3-repeat Cross-Validation         • Zero threshold optimization
  • Protocol definition & locking            • Evaluated EXACTLY ONCE in 6M.3
```

### 9.1 Protocol Locking Mandate
Before opening or running inference on VAL in Phase 6M.3, the winning configuration from Phase 6M.2 must be exported and locked to:

`evaluation_results/phase6m/final_hybrid_protocol.json`

The protocol JSON must store exact SHA-256 fingerprints of DEV and VAL datasets, locked feature names, scaler choice, model hyperparameters, and decision threshold.

---

## 10. Software Architecture Design

The Phase 6M software implementation will be organized under `backend/evaluation/phase6m/`:

```
backend/evaluation/phase6m/
├── __init__.py
├── config.py                # Directories, hyperparameter grids, feature schemas
├── dataset.py               # Joins Pillar-1 & Pillar-2 matrices with Phase 6I labels
├── fusion_models.py         # Candidate meta-learner factories and stacked wrappers
├── model_selection.py       # RepeatedStratifiedKFold (15 folds) CV engine over DEV
├── leakage.py               # 5-point automated data leakage & firewall verifier
├── protocol.py              # Protocol freeze engine (final_hybrid_protocol.json)
├── validation.py           # Held-out VAL validation engine (11 stages)
├── report_phase6m.py        # Figure plotting & Markdown report generator
└── run_phase6m.py           # Master orchestrator script
```

---

## 11. Artifact Inventory

Phase 6M will generate and export the following standardized artifacts into `evaluation_results/phase6m/`:

### 11.1 JSON Reports
1. `feature_matrix_validation.json` — Stage 1 DEV feature matrix integrity & label balance.
2. `collinearity_analysis.json` — Pearson, Spearman, Kendall matrices and VIF scores.
3. `stability_gate.json` — Preflight numerical stability gate audit.
4. `full_dev_candidate_comparison.json` — Stage 6 repeated 5-fold CV benchmarks across candidates.
5. `baseline_comparison.json` — Comparison against Majority, Random, and single-pillar baselines.
6. `leakage_audit.json` — 5-point data leakage verification status.
7. `final_hybrid_protocol.json` — Frozen model protocol prior to VAL access.
8. `heldout_validation_results.json` — Stage 3 held-out evaluation metrics on VAL ($N=12,483$).
9. `heldout_bootstrap_ci.json` — 2,000 stratified bootstrap confidence intervals.
10. `dev_val_generalization.json` — DEV OOF vs VAL generalization gap analysis.
11. `heldout_calibration.json` — ECE, MCE, and reliability bin metrics.
12. `delong_mcnemar_nri.json` — DeLong test, McNemar test, and NRI statistical results.

### 11.2 Publication Figures (300 DPI)
1. `phase6m_feature_correlation.png` — Joint hybrid feature correlation heatmap.
2. `phase6m_candidate_comparison.png` — CV metrics comparison across candidate fusion models.
3. `phase6m_heldout_roc.png` — Held-out ROC curve comparing Hybrid vs Pillar 1 vs Pillar 2.
4. `phase6m_heldout_pr.png` — Precision-Recall curves comparing Hybrid vs baselines.
5. `phase6m_calibration_diagram.png` — Reliability calibration diagram for Hybrid model.
6. `phase6m_confusion_matrix.png` — Confusion matrix heatmap at optimal operating threshold.
7. `phase6m_decision_curve_analysis.png` — Net Benefit Decision Curve Analysis plot.
8. `phase6m_ablation_comparison.png` — Bar chart of performance across 8 ablation experiments.

### 11.3 Frozen Model Artifacts
Directory: `evaluation_results/phase6m/final_model/`
- `preprocessing.joblib` — Fitted StandardScaler / RobustScaler.
- `hybrid_meta_classifier.joblib` — Fitted winning meta-classifier model.
- `feature_schema.json` — Locked feature schema and index mapping.
- `model_metadata.json` — Learned weights, hyperparameters, and environment metadata.

---

## 12. Threats to Validity

1. **Feature Shift Propagation**: If Pillar-2 features drift on VAL, tree-based meta-learners could propagate this shift unless regularized by Pillar-1 evidence features.
2. **Meta-Learner Overfitting**: Training high-capacity tree models (e.g. LightGBM with high depth) on meta-features risks memorizing DEV fold patterns; strict depth constraints ($\text{depth} \le 4$) are required.
3. **Probability Miscalibration**: Uncalibrated base model probabilities $P_1, P_2$ can distort meta-learner weight allocation; Platt pre-calibration is mandated.
4. **Class Imbalance Sensitivity**: Precision-Recall AUC and MCC must be monitored alongside ROC-AUC to prevent threshold bias toward the majority class.

---

## 13. Scientific Blueprint Approval & Stop Condition

This document constitutes the **frozen scientific specification** for Phase 6M. Execution is **STOPPED** per prompt instructions.

- **Phase 6M.0 Status**: `COMPLETED & FROZEN`
- **Next Step**: Await explicit approval before commencing Phase 6M.1 (Hybrid Feature Assembly & Preflight).
