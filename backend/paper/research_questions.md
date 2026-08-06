# HalluciSense Formal Research Questions (RQ1 – RQ10)

**Document Version**: 1.0.0-Publication-Ready  
**Target Journal**: Elsevier *Information Fusion* / *Knowledge-Based Systems* / *Artificial Intelligence*  

---

## RQ1: Does adaptive hybrid fusion outperform static weighted fusion?
- **Hypothesis ($H_1^{(1)}$)**: Dynamically conditioning fusion weights $\alpha(q), \beta(q), \gamma(q), \delta(q)$ on query complexity and model uncertainty statistically significantly improves detection accuracy over fixed linear weights ($p < 0.001$).
- **Motivation**: Fixed weights fail when candidate passage retrieval is unavailable or when logit probabilities are masked by commercial API gateways.
- **Experimental Design**: Evaluate HalluciSense under `STATIC` ($\alpha=0.40, \beta=0.30, \gamma=0.30$) vs. `ADAPTIVE` Softmax Attention and Bayesian Gating across $N=750$ claims.
- **Datasets**: FEVER, TruthfulQA, HaluEval, SciFact, FactScore, FreshQA, RAGTruth.
- **Baselines**: Static linear fusion, equal weighting ($\frac{1}{3}, \frac{1}{3}, \frac{1}{3}$).
- **Metrics**: AUROC, AUPRC, F1-Score, MCC, ECE.
- **Statistical Tests**: McNemar's Test, DeLong ROC Test, 10,000-sample Bootstrap CIs.
- **Limitations**: Requires lightweight feature extraction overhead ($< 2$ ms).

---

## RQ2: Does confidence estimation improve hallucination detection?
- **Hypothesis ($H_1^{(2)}$)**: Integrating predictive entropy and aleatoric/epistemic logit uncertainty reduces false positive hallucinations by identifying unconfident model generations ($p < 0.001$).
- **Motivation**: Models generating text with high logit entropy are prone to speculative hallucination.
- **Experimental Design**: Compare Pillar 2 Confidence alone vs. Pillar 1 + Pillar 2 hybrid.
- **Metrics**: AUROC, ECE, Brier Score.
- **Statistical Tests**: DeLong test, Wilcoxon signed-rank test.

---

## RQ3: Does consistency reasoning reduce hallucination false positives?
- **Hypothesis ($H_1^{(3)}$)**: Constructing an NLI contradiction graph over stochastic paraphrase generations identifies self-contradictory claims when retrieval evidence is missing ($p < 0.001$).
- **Motivation**: Closed-book hallucinations often contradict alternative stochastic generations.
- **Metrics**: AUROC, Precision, Recall, Graph Consistency Index $C_G$.

---

## RQ4: Does token-level localization improve explainability?
- **Hypothesis ($H_1^{(4)}$)**: Sub-sentence token span localization provides fine-grained, actionable diagnostic heatmaps that accelerate human verification time by $> 40\%$.
- **Motivation**: Sentence-level scores fail to pinpoint exact character offsets of fabricated entities.
- **Metrics**: Token-level F1, Human Verification Time Saved (%).

---

## RQ5: Does calibration improve prediction reliability?
- **Hypothesis ($H_1^{(5)}$)**: Platt sigmoidal recalibration reduces Expected Calibration Error (ECE) below $0.0300$, ensuring output scores reflect true posterior probability of hallucination ($p < 0.001$).
- **Metrics**: ECE, MCE, Brier Score Loss, Reliability Diagrams.

---

## RQ6: Does the knowledge graph improve reasoning consistency?
- **Hypothesis ($H_1^{(6)}$)**: Directed multi-graph representations $G=(V, E)$ capture multi-hop claim-to-entity dependencies and prevent hallucination propagation ($p < 0.001$).
- **Metrics**: Graph Consistency Index $C_G$, Contradiction Edge Density.

---

## RQ7: Can HalluciSense generalize across multiple LLM families?
- **Hypothesis ($H_1^{(7)}$)**: HalluciSense maintains AUROC $> 0.9100$ across both white-box open-weights models and black-box API models.
- **Models Evaluated**: GPT-4, Gemini 1.5 Pro, Claude 3.5 Sonnet, Llama-3 70B, Mistral Large, Qwen 2.5 72B, DeepSeek V3, Phi-3 Medium.

---

## RQ8: What is the computational cost of HalluciSense?
- **Hypothesis ($H_1^{(8)}$)**: Fast passage filtering and linear graph construction maintain single-claim inference latency P50 $< 120$ ms with RSS RAM footprint $< 512$ MB SLA.
- **Metrics**: P50/P95/P99 Latency (ms), Memory Footprint (MB), CPU/GPU Utilization.

---

## RQ9: Which architectural components contribute most?
- **Hypothesis ($H_1^{(9)}$)**: Pillar 1 Evidence Grounding is the primary performance contributor (contributing $14.53\%$ AUROC drop when removed), followed by Pillar 2 Confidence ($6.96\%$) and Pillar 3 Consistency ($6.11\%$).
- **Metrics**: 10-Variant Component Ablation Degradation (%).

---

## RQ10: How robust is HalluciSense under adversarial conditions?
- **Hypothesis ($H_1^{(10)}$)**: HalluciSense retains AUROC $> 0.8800$ under 15 stress perturbations including prompt injection, citation spoofing, and zero-passage retrieval failures.
- **Metrics**: Stress AUROC, Metric Retention Rate (%).
