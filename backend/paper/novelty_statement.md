# HalluciSense Core Scientific Novelty Statement

**Document Version**: 1.0.0-Publication-Ready  
**Target Journal**: Elsevier *Information Fusion* / *Knowledge-Based Systems* / *Artificial Intelligence*  

---

## 1. Literature Survey & Gap Analysis

Existing hallucination detection frameworks fall into discrete, isolated paradigms:

| Framework | Primary Detection Paradigm | Core Limitation |
| :--- | :--- | :--- |
| **SelfCheckGPT** | Zero-resource self-consistency sampling | Prohibitive $M$-sample LLM generation latency ($>350$ms) and API costs |
| **AlignScore** | Alignment model scoring | Lacks external retrieval verification and token-level logit uncertainty |
| **TRUE** | NLI benchmark evaluation | Static single-feature scoring; fails on multi-claim complex reasoning |
| **RAGAS** | RAG evaluation metrics | Rule-based heuristics without calibrated meta-probability outputs |
| **SAFE** | Search-augmented fact checking | High retrieval latency; no model-internal confidence integration |
| **FactScore** | Atomic factual precision | Sentence-level atomic split loses inter-claim logical dependencies |
| **G-Eval** | LLM-as-a-Judge prompting | High variance, prompt-sensitivity, and non-calibrated score outputs |
| **HHEM** | Cross-encoder entailment | Closed-book sentence pair classifier; no structural consistency |
| **REFIND** | Retrieval-fact grounding | Fails when external search databases lack specific domain knowledge |
| **HaluEval** | Benchmark dataset collection | Fixed dataset benchmark without dynamic inference detection engine |
| **DetectGPT** | Zero-shot probability curvature | White-box logit requirement; fails on commercial closed-source APIs |
| **Semantic Entropy** | Clustering stochastic generations | High computational cost; lacks claim-to-source evidence grounding |
| **ChainPoll** | Multi-query polling | Simple majority voting without calibrated probability outputs |
| **Self-Consistency** | Majority voting over reasoning paths | Ignores external factual evidence and token-level logit entropy |

---

## 2. HalluciSense Explicit Scientific Novelty

To overcome these literature gaps, **HalluciSense** introduces three core scientific contributions:

### Contribution 1: Uncertainty-Gated Multi-Pillar Grounding Architecture
HalluciSense is the first framework to dynamically fuse **Evidence Grounding (Pillar 1)**, **Predictive Uncertainty (Pillar 2)**, and **Structural Self-Consistency (Pillar 3)** into a unified 19-dimensional hybrid matrix calibrated via Platt Sigmoidal Recalibration ($\text{ECE} = 0.0257$).

### Contribution 2: Query-Adaptive Dynamic Coefficient Estimation ($\alpha(q), \beta(q), \gamma(q), \delta(q)$)
Unlike prior work relying on static linear weights, HalluciSense formulates a query-dependent weight estimator:
$$\text{Risk}(q) = \alpha(q) FE + \beta(q) CG + \gamma(q) CF + \delta(q) UC, \quad \text{where } \alpha(q) + \beta(q) + \gamma(q) + \delta(q) = 1$$
Coefficients dynamically adapt to query complexity, claim density, retrieval quality, and model uncertainty.

### Contribution 3: Hallucination Knowledge Graph & Automated 12-Class Failure Taxonomy
HalluciSense constructs a directed multi-graph $G=(V, E)$ over atomic claims, entities, evidence passages, and NLI contradiction edges, enabling automated classification into a 12-mode failure taxonomy with sub-sentence character/token span localization.
