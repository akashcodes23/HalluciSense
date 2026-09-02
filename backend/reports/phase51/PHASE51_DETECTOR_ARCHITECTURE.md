# PHASE 51 — DETECTOR ARCHITECTURE & DATA FLOW SPECIFICATION
**HalluciSense Frozen Detection Pipeline Architecture**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `FROZEN ARCHITECTURAL SPECIFICATION`

---

## 1. End-to-End Mathematical & Inference Data Flow

```
[User Input / LLM Response Text]
                |
                v
       [Claim Extraction]
                | (Extracted Propositions {c_1, c_2, ... c_K})
                +-------------------+--------------------+
                |                   |                    |
                v                   v                    v
      [Pillar 1: Retrieval]    [Pillar 2: Conf]     [Pillar 3: Consist]
        - Evidence Search        - Token Entropy      - Lexical Jaccard
        - DeBERTa-v3 NLI         - Static Proxy       - DeBERTa-v3 NLI
        - Premise-Hypothesis     - Factual Evidence   - Claim Pairing
                |                   |                    |
                v                   v                    v
          [P1 Summary]        [P2 Summary]         [P3 Summary]
                \                   |                   /
                 \                  |                  /
                  v                 v                 v
                 [Canonical 19-Dimensional Feature Assembly]
                                    |
                                    v  X in R^{19}
                         [RobustScaler Preprocessing]
                                    | (Centered & scaled via RobustScaler.center_)
                                    v  X_scaled in R^{19}
                     [Frozen HistGradientBoostingClassifier]
                                    | (P(Hallucination | X) via 32 decision trees)
                                    v
                       [Calibrated Probability P_H]
                                    |
                                    v
                         [Decision Threshold tau* = 0.54]
                                    |
                     +--------------+--------------+
                     |                             |
                     v                             v
           [P_H < 0.54: FACTUAL]        [P_H >= 0.54: HALLUCINATED]
                     |                             |
                     v                             v
           [Status: VERIFIED]          [Status: NEEDS_VERIFICATION /
                                                LIKELY_HALLUCINATED]
```

---

## 2. Canonical 19-Feature Schema Definition

| Index | Feature Identifier | Source Layer | Description | Value Bounds |
| :--- | :--- | :--- | :--- | :--- |
| `[0]` | `p1_mean_entailment` | Pillar 1 (Retrieval) | Mean DeBERTa NLI entailment score across evidence passages | $[0.0, 1.0]$ |
| `[1]` | `p1_max_entailment` | Pillar 1 (Retrieval) | Maximum DeBERTa NLI entailment score across evidence passages | $[0.0, 1.0]$ |
| `[2]` | `p1_mean_contradiction` | Pillar 1 (Retrieval) | Mean DeBERTa NLI contradiction score across evidence passages | $[0.0, 1.0]$ |
| `[3]` | `p1_min_support_margin` | Pillar 1 (Retrieval) | Minimum margin $\min (P_{\text{entail}} - P_{\text{contrad}})$ | $[-1.0, 1.0]$ |
| `[4]` | `p1_num_claims` | Pillar 1 (Decomposition)| Number of claims extracted from candidate response | $[1, \infty)$ |
| `[5]` | `p2_max_pairwise_contradiction`| Pillar 2 / P3 Structure | Max pairwise contradiction across claim pairs | $[0.0, 1.0]$ |
| `[6]` | `p2_mean_pairwise_contradiction`| Pillar 2 / P3 Structure | Mean pairwise contradiction across claim pairs | $[0.0, 1.0]$ |
| `[7]` | `p2_max_pairwise_similarity` | Pillar 2 / P3 Structure | Max pairwise token/semantic similarity across claim pairs | $[0.0, 1.0]$ |
| `[8]` | `p2_fraction_contradictory_pairs`| Pillar 2 / P3 Structure| Fraction of evaluated pairs exceeding contradiction threshold | $[0.0, 1.0]$ |
| `[9]` | `p2_num_claims` | Pillar 2 / P3 Structure | Claim count consumed by structural consistency engine | $[1, \infty)$ |
| `[10]` | `prob_p1` | Pillar 1 Aggregator | Factual error probability derived from grounding | $[0.0, 1.0]$ |
| `[11]` | `prob_p2` | Pillar 2 Aggregator | Structural inconsistency / confidence gap probability | $[0.0, 1.0]$ |
| `[12]` | `logit_p1` | Mathematical Transform | Logit transformation of P1 probability: $\ln(p / (1-p))$ | $(-\infty, \infty)$ |
| `[13]` | `logit_p2` | Mathematical Transform | Logit transformation of P2 probability: $\ln(p / (1-p))$ | $(-\infty, \infty)$ |
| `[14]` | `prob_disagreement_abs` | Meta Fusion | Absolute disagreement: $\|P_{\text{P1}} - P_{\text{P2}}\|$ | $[0.0, 1.0]$ |
| `[15]` | `prob_mean` | Meta Fusion | Arithmetic mean: $(P_{\text{P1}} + P_{\text{P2}}) / 2$ | $[0.0, 1.0]$ |
| `[16]` | `prob_max` | Meta Fusion | Maximum probability: $\max(P_{\text{P1}}, P_{\text{P2}})$ | $[0.0, 1.0]$ |
| `[17]` | `prob_min` | Meta Fusion | Minimum probability: $\min(P_{\text{P1}}, P_{\text{P2}})$ | $[0.0, 1.0]$ |
| `[18]` | `prob_ratio` | Meta Fusion | Probability ratio: $(P_{\text{P1}} + \epsilon) / (P_{\text{P2}} + \epsilon)$ | $(0, \infty)$ |

---

## 3. Invariant Artifact Registry

- **Classifier File**: `backend/app/models/hybrid_meta_classifier.joblib` (or `backend/evaluation_results/phase6m/final_hybrid_model/hybrid_meta_classifier.joblib`)
- **Classifier Checksum**: SHA256 `089ebd2d277d1c21adc0541b71f1bf3e4cb5927d6e74f3ed96b1d00b15337cad`
- **Scaler File**: `backend/app/models/preprocessing.joblib` (or `backend/evaluation_results/phase6m/final_hybrid_model/preprocessing.joblib`)
- **Scaler Checksum**: SHA256 `bdbd42e3f386b7b2602e95b1fc32b6ded1ac404779498190442d17aec2f97e90`
- **Decision Threshold $\tau^*$**: `0.54`
