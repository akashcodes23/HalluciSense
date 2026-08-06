# HalluciSense Pillar-1: Model Limitations

*Generated: 2026-08-03T04:49:02.142488+00:00*  
*Phase: 6K (Frozen)*

---

## 1. Performance Gap

The model achieved a validation ROC-AUC of **0.6902**, falling short of the predefined
publication gate of 0.75. This gap indicates:
- Moderate discrimination ability
- Significant overlap in feature distributions between grounded and hallucinated responses
- The 5 NLI-based features alone are insufficient for high-confidence hallucination detection

## 2. Feature Limitations

### min_support_margin Dominance
The model relies heavily on `min_support_margin` (largest coefficient magnitude: 1.2485).
This creates a single point of failure: responses where all claims are weakly but not
contradictorily supported may be systematically misclassified.

### Binary NLI Aggregation
Aggregating NLI scores to scalars (mean, max, min) discards:
- Claim-level variance (some claims may be strongly supported, others not)
- Evidence source diversity
- Positional or order information

### num_claims Weakness
`num_claims` is a proxy for response length and has low discriminative power.
Hallucinated responses are not consistently longer or shorter than grounded ones.

## 3. Dataset Scope

- Training/validation data covers only **HaluBench**, **HaluEval**, and **RAGTruth**
- Out-of-distribution generalization to other domains (medical, legal, scientific) is untested
- Non-English responses are not supported

## 4. Calibration

- ECE (10-bin) on VAL: computed in Step 5
- The model is moderately calibrated; output probabilities can be used as confidence scores
  but should not be treated as exact posterior probabilities

## 5. Threshold Sensitivity

- The operating threshold of 0.56 was optimized on DEV; slight performance variation
  expected on other distributions
- No adaptive thresholding is implemented

## 6. NLI Model Dependency

Pillar-1 features are computed by `cross-encoder/nli-deberta-v3-small`:
- If the NLI model is updated or replaced, features must be recomputed
- NLI model errors propagate directly to Pillar-1 predictions
