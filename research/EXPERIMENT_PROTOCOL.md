# HalluciSense Experimental Protocol

**Document Version**: 2.0 (Post-Audit Verified Protocol)  
**Target Venue**: IEEE Conference on Artificial Intelligence / IEEE Transactions Standards  
**Status**: Active & Locked  

---

## 1. Experimental Objectives & Research Questions

This experimental evaluation investigates six core research questions:
- **RQ1 (Multi-Signal Synergy)**: Does fusing external retrieval grounding ($\text{FE}$), model predictive uncertainty ($\text{CG}$), and semantic consistency ($\text{CF}$) yield statistically superior hallucination discrimination over single-signal constituents?
- **RQ2 (Baseline Superiority)**: Does HalluciSense outperform established single- and dual-signal baselines (SelfCheckGPT, FActScore, RAGAS, Pure NLI, AlignScore) on standardized benchmarks under identical test conditions?
- **RQ3 (Probability Calibration & Selective Prediction)**: Can raw continuous $H$-Scores be calibrated into reliable posterior error probabilities via post-hoc scaling, and does selective abstention bound classification error?
- **RQ4 (Signal Availability Robustness)**: How does availability-aware adaptive fusion perform under severe signal missingness (e.g. black-box APIs lacking token logprobs, single-turn generations, or offline environments) compared to fixed unnormalized fusion?
- **RQ5 (Cross-Domain & External Generalization)**: Does the framework generalize zero-tuning across diverse task formulations and external public benchmarks (TruthfulQA, HaluEval, FEVER, RAGTruth, BioASQ)?
- **RQ6 (Operational Efficiency & Closed-Loop Repair)**: What is the end-to-end computational latency, memory footprint, and effectiveness of closed-loop claim repair with independent downstream reverification?

---

## 2. Benchmark Datasets & Partitioning Protocol

### 2.1 Canonical Benchmark Dataset ($N=750$)
- **Provenance**: Frozen diagnostic benchmark covering 15 knowledge-intensive domains (Medicine, Law, Finance, Science, History, Technology, Geography, Politics, Philosophy, Arts, Sports, Economics, Environment, Literature, Engineering).
- **Integrity**: Verified SHA-256 Checksum: `dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5`.
- **Partitioning Protocol**:
  - **Training Split (60%, $N=450$)**: Used strictly for fitting base signal weights and development exploration.
  - **Validation Split (20%, $N=150$)**: Used for hyperparameter tuning, Platt calibration parameter estimation ($a=1.82, b=-0.45$), and decision threshold selection ($\tau^* = 0.54$).
  - **Held-Out Test Split (20%, $N=150$)**: Sealed until final evaluation. Evaluated once without post-hoc threshold adjustment.

### 2.2 External Public Benchmarks ($N=850$)
To evaluate out-of-distribution transfer without fine-tuning, the framework was evaluated on five external peer-reviewed datasets:
1. **TruthfulQA ($N=200$)**: Adversarial question answering targeting human falsehoods and misconceptions.
2. **HaluEval ($N=200$)**: Dialogue and multi-turn question answering evaluating factual hallucinations.
3. **FEVER ($N=200$)**: Fact extraction and verification against Wikipedia claims.
4. **RAGTruth ($N=150$)**: Retrieval-augmented generation with fine-grained hallucination annotations.
5. **BioASQ-FactCheck ($N=100$)**: Highly specialized biomedical scientific assertions.

---

## 3. Evaluated Systems & Baselines

### 3.1 Evaluated Configurations (HalluciSense Family)
- **A1: Pillar 1 Only ($\text{FE}$)**: Hybrid BM25 + FAISS dense retrieval + DeBERTa-v3 cross-encoder NLI + deterministic symbolic verifier.
- **A2: Pillar 2 Only ($\text{CG}$)**: Token predictive confidence and Bernoulli entropy.
- **A3: Pillar 3 Only ($\text{CF}$)**: Stochastic multi-sample sentence embedding divergence and cross-NLI contradiction.
- **A7: Fixed Canonical Fusion (Mode A)**: Static linear combination $H = 0.40 \text{FE} + 0.30 \text{CG} + 0.30 \text{CF}$.
- **A8: Availability-Aware Adaptive Fusion (Mode B)**: Dynamic indicator masking $\mathbf{m} \in \{0, 1\}^3$ and reliability reweighting $\mathbf{r} \in (0, 1]^3$.
- **A9: Adaptive + Platt Calibration**: Post-hoc logistic calibration fitted on validation split.
- **A10: Adaptive + Selective Abstention**: Dual-criteria rejection gate ($S_{evidence} < 0.40$ or $|H - 0.40| < 0.08$).
- **A11: Full Closed-Loop Pipeline**: Detection + Calibration + Selective Gate + Symbolic Repair + Reverification Gate ($H < 0.20$).

### 3.2 Baseline Systems
1. **SelfCheckGPT**: Multi-sample stochastic consistency using sentence embeddings and cross-sample NLI.
2. **FactScore**: Atomic claim extraction and passage retrieval precision.
3. **RAGAS**: RAG evaluation measuring faithfulness and context relevance.
4. **AlignScore**: Unified alignment scoring using trained metric models.
5. **Pure NLI**: Standalone cross-encoder NLI evaluation against top-1 retrieved passage.
6. **Pure Retrieval**: Cosine similarity against BM25/dense retrieved passages without NLI reasoning.

---

## 4. Evaluation Metrics & Statistical Procedures

- **Discrimination**:
  - Area Under the Receiver Operating Characteristic Curve (**AUROC**).
  - Area Under the Precision-Recall Curve (**AUPRC**).
  - Classification Metrics: Accuracy, Precision, Recall, Macro F1, Matthews Correlation Coefficient (**MCC**).
- **Calibration**:
  - Expected Calibration Error (**ECE**, 10 equal-width bins).
  - **Brier Score** ($BS = \frac{1}{N}\sum (\hat{p}_i - y_i)^2$).
- **Selective Classification**:
  - Area Under the Risk-Coverage Curve (**AURC**).
  - Selective Risk ($\text{Error rate on retained queries}$).
- **Repair**:
  - Correction Success Rate (**CSR**).
  - Reverification Pass Rate (**RPR**).
  - Correction-Induced Hallucination Rate (**CIHR**).
- **Statistical Significance**:
  - 10,000-iteration non-parametric bootstrap resampling for 95% empirical percentile confidence intervals.
  - Paired **McNemar test** on classification contingency matrices ($2 \times 2$).
  - Paired **Wilcoxon signed-rank test** for continuous paired distributions.
  - Paired **Cohen's $d$** for effect size estimation.
  - Significance threshold locked at $\alpha = 0.05$.
