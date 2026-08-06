# Phase 9 — Step 4: Error Analysis

**Generated**: 2026-08-03T04:48:29.684935+00:00

## 1. Quadrant Summary

| Quadrant | Count | % | Description |
| --- | --- | --- | --- |
| TP | 1095 | 31.3% | Correctly predicted hallucination |
| TN | 1286 | 36.7% | Correctly predicted grounded |
| FP | 567 | 16.2% | Incorrectly predicted hallucination (false alarm) |
| FN | 552 | 15.8% | Missed hallucination (miss) |

## 2. Feature Statistics by Quadrant

| Feature | TP Mean | TN Mean | FP Mean | FN Mean |
| --- | --- | --- | --- | --- |
| `mean_entailment` | 0.2036 | 0.0985 | 0.1359 | 0.1143 |
| `max_entailment` | 0.2460 | 0.1608 | 0.1795 | 0.1726 |
| `mean_contradiction` | 0.6974 | 0.0410 | 0.6915 | 0.0594 |
| `min_support_margin` | -0.6113 | 0.0349 | -0.6873 | 0.0312 |
| `num_claims` | 5.3680 | 4.0047 | 5.8748 | 3.7409 |

## 3. Failure Cluster Analysis

### FP (567 samples → 3 clusters)
- **Cluster 1** (288 samples, 50.8%): dominant=`num_claims`, mean_prob=0.611
- **Cluster 3** (186 samples, 32.8%): dominant=`num_claims`, mean_prob=0.684
- **Cluster 2** (93 samples, 16.4%): dominant=`max_entailment`, mean_prob=0.594

### FN (552 samples → 3 clusters)
- **Cluster 2** (450 samples, 81.5%): dominant=`mean_entailment`, mean_prob=0.460
- **Cluster 3** (54 samples, 9.8%): dominant=`mean_entailment`, mean_prob=0.544
- **Cluster 1** (48 samples, 8.7%): dominant=`mean_contradiction`, mean_prob=0.525

## 4. Systematic Weaknesses

- **FP vs TN | mean_contradiction**: FP samples have mean_contradiction = 0.691 vs TN = 0.041. Model over-predicts hallucination when this feature differs.
- **FP vs TN | min_support_margin**: FP samples have min_support_margin = -0.687 vs TN = 0.035. Model over-predicts hallucination when this feature differs.
- **FP vs TN | num_claims**: FP samples have num_claims = 5.875 vs TN = 4.005. Model over-predicts hallucination when this feature differs.
- **FN vs TP | mean_contradiction**: FN samples have mean_contradiction = 0.059 vs TP = 0.697. Model misses hallucinations when this feature differs.
- **FN vs TP | min_support_margin**: FN samples have min_support_margin = 0.031 vs TP = -0.611. Model misses hallucinations when this feature differs.

## 5. Recommendations

- **[R1] Feature Engineering**: min_support_margin is the strongest predictor. Pillar 2/3 should engineer richer support margin signals (e.g., per-document, per-sentence margins) to reduce FN rate.
- **[R2] Threshold Calibration**: The 0.56 threshold was optimized for balanced F1/MCC. For applications requiring higher recall (catching more hallucinations), lower threshold to 0.50 at the cost of precision.
- **[R3] Feature Expansion**: mean_entailment and max_entailment show weak discriminative power. Consider claim-level aggregation variants or attention-weighted NLI scores to capture subtle entailment patterns in longer responses.
- **[R4] Ensemble Strategy**: Pillar-1 misses hallucinations with low contradiction scores. A Pillar-3 semantic similarity Pillar could complement Pillar-1 for hallucinations expressed without direct contradiction.
- **[R5] Dataset Expansion**: The 0.47:0.53 class ratio in VAL differs from DEV (0.46:0.54). Future work should evaluate on more diverse benchmarks beyond HaluBench/HaluEval/RAGTruth to assess generalization.

## 6. Figures

- `step4_quadrant_feature_distributions.png` — Feature box plots by quadrant
- `step4_probability_by_quadrant.png` — Predicted probability histograms
- `step4_feature_heatmap.png` — Mean feature value heatmap
- `step4_fp_clusters.png` — FP cluster scatter (min_support_margin vs mean_contradiction)
- `step4_fn_clusters.png` — FN cluster scatter