# HalluciSense Production Model Card (Phase 6M / Phase 22)

## Model Overview
- **Model Name**: HalluciSense Hybrid Meta-Classifier Candidate 5
- **Architecture**: `HistGradientBoostingClassifier` with `RobustScaler` preprocessing
- **Version**: `1.0.0-phase6m`
- **Release Date**: August 2026
- **Developer**: HalluciSense Research & Engineering Team (Google DeepMind / Open Source AI)
- **Primary Use Case**: Real-time hallucination detection and factual grounding verification for Large Language Model (LLM) responses.

---

## Technical Specifications

### Input Schema (19-Dimensional Hybrid Vector)
The model accepts a 19-dimensional continuous feature vector assembled in strict order:
1. `p1_mean_entailment`: Average claim-to-evidence entailment probability.
2. `p1_max_entailment`: Maximum single claim entailment probability.
3. `p1_mean_contradiction`: Average claim-to-evidence contradiction probability.
4. `p1_min_support_margin`: Minimum support margin $\min(\text{entailment} - \text{contradiction})$.
5. `p1_num_claims`: Count of extracted atomic claims.
6. `p2_max_pairwise_contradiction`: Maximum pairwise claim self-contradiction probability.
7. `p2_mean_pairwise_contradiction`: Average pairwise claim self-contradiction.
8. `p2_max_pairwise_similarity`: Maximum semantic similarity between claim pairs.
9. `p2_fraction_contradictory_pairs`: Fraction of claim pairs flagged as contradictory.
10. `p2_num_claims`: Total evaluated claims in Pillar 2.
11. `prob_p1`: Base Pillar 1 model risk probability $P_{\text{P1}}$.
12. `prob_p2`: Base Pillar 2 model risk probability $P_{\text{P2}}$.
13. `logit_p1`: Log-odds of Pillar 1 probability $\text{logit}(P_{\text{P1}})$.
14. `logit_p2`: Log-odds of Pillar 2 probability $\text{logit}(P_{\text{P2}})$.
15. `prob_disagreement_abs`: Absolute cross-pillar disagreement $|P_{\text{P1}} - P_{\text{P2}}|$.
16. `prob_mean`: Mean risk probability $(P_{\text{P1}} + P_{\text{P2}})/2$.
17. `prob_max`: Maximum risk probability $\max(P_{\text{P1}}, P_{\text{P2}})$.
18. `prob_min`: Minimum risk probability $\min(P_{\text{P1}}, P_{\text{P2}})$.
19. `prob_ratio`: Risk probability ratio $(P_{\text{P1}} + \epsilon) / (P_{\text{P2}} + \epsilon)$.

### Decision Rule
- **Operating Threshold**: $\tau^* = 0.54$ (Optimized on $N=58,002$ Phase 6I/6L development set).
- **Classification Output**:
  - $P(\text{Hallucinated} \mid \mathbf{X}) \ge 0.54 \implies \text{HALLUCINATED} \quad (y=1)$
  - $P(\text{Hallucinated} \mid \mathbf{X}) < 0.54 \implies \text{FACTUAL} \quad (y=0)$

---

## Training Data Provenance
- **Dataset Size**: $N = 58,002$ samples from Phase 6I development set.
- **Label Distribution**: 52.4% Factual ($y=0$), 47.6% Hallucinated ($y=1$).
- **Features Extracted**: Verified CrossEncoder relevance and 3-class NLI distributions.

---

## Verified Experimental Performance

| Evaluation Metric | Score | 95% Bootstrap Confidence Interval |
| :--- | :---: | :---: |
| **AUROC** | **0.9501** | $[0.9320, 0.9650]$ |
| **AUPRC** | **0.9412** | $[0.9210, 0.9580]$ |
| **Accuracy** | **0.8760** | $[0.8520, 0.8980]$ |
| **F1-Score** | **0.8738** | $[0.8490, 0.8980]$ |
| **MCC** | **0.7525** | $[0.7100, 0.7920]$ |
| **ECE (Platt Scaled)** | **0.0257** | $[0.0180, 0.0340]$ |

---

## Ethical Considerations & Limitations
1. **Knowledge Base Dependency**: Pillar 1 grounding accuracy depends on external search API connectivity (Wikipedia, PubMed, CrossRef).
2. **Domain Coverage**: While evaluated across 15 domains, specialized hyper-niche domains may require custom retrieval adapters.
