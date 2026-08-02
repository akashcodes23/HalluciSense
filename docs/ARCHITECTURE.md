# HalluciSense System Architecture

```
User Input / Response Text
            │
            ▼
   Claim Extraction (Atomic Claims)
            │
            ▼
   Evidence Retrieval (BM25 / Dense)
            │
    ┌───────┴───────┐
    ▼               ▼
[Pillar 1 Engine] [Pillar 2 Engine]
(Evidence Grounding) (Pairwise Structure)
    │               │
    └───────┬───────┘
            ▼
   19-Dimensional Hybrid Feature Vector
            │
            ▼
  RobustScaler Preprocessing
            │
            ▼
  HistGradientBoosting Meta-Classifier
            │
            ▼
  Operating Threshold (τ* = 0.54)
            │
            ▼
  Confidence-Aware Output & Explanation
```
