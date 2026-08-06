# Phase 9 — Step 2: Prediction Explainability

**Generated**: 2026-08-03T04:48:06.766751+00:00

## Explainability Method

Each prediction is explained using the frozen logistic regression coefficients:

```
contribution_i = coef_i × RobustScaled(feature_i)
logit = Σ contribution_i + intercept
probability = sigmoid(logit)
prediction = (probability ≥ 0.56)
```

No SHAP, no gradient-based attribution, no black-box approximations.

## Quadrant Distribution (VAL 3,500 samples)

| Quadrant | Count | % |
| --- | --- | --- |
| TP | 1095 | 31.3% |
| TN | 1286 | 36.7% |
| FP | 567 | 16.2% |
| FN | 552 | 15.8% |

## Confidence Category Distribution

| Category | Count | % |
| --- | --- | --- |
| Moderate | 1771 | 50.6% |
| Low | 1587 | 45.3% |
| High | 132 | 3.8% |
| Very High | 10 | 0.3% |

## Aggregate Feature Contribution Statistics

| Feature | Mean Contrib | Std | |Contrib| Mean | Positive | Negative |
| --- | --- | --- | --- | --- | --- |
| `mean_entailment` | 0.3266 | 0.6772 | 0.3303 | 1896 | 1582 |
| `max_entailment` | -0.1100 | 0.2000 | 0.1115 | 1576 | 1911 |
| `mean_contradiction` | -0.1731 | 0.2334 | 0.1887 | 1685 | 1815 |
| `min_support_margin` | 0.3357 | 0.7531 | 0.6048 | 1798 | 1702 |
| `num_claims` | 0.0588 | 0.1201 | 0.0822 | 1401 | 1879 |

## Spot-Check Examples

### TP Examples (Top 3 by confidence margin)

- **Sample #2188** | True: `HALLUCINATED` | Predicted: `HALLUCINATED` | prob=0.8871 | margin=0.3271 | conf=Very High
  > Prediction: HALLUCINATED (p=0.887, threshold=0.56). Very High confidence with margin 0.327. Key supporting signals: [mean_entailment (+1.721), min_support_margin (+1.218)]. Key opposing signals: [max_entailment (-0.571), mean_contradiction (-0.115)].

- **Sample #2705** | True: `HALLUCINATED` | Predicted: `HALLUCINATED` | prob=0.8870 | margin=0.3270 | conf=Very High
  > Prediction: HALLUCINATED (p=0.887, threshold=0.56). Very High confidence with margin 0.327. Key supporting signals: [mean_entailment (+1.745), min_support_margin (+1.195)]. Key opposing signals: [max_entailment (-0.574), mean_contradiction (-0.113)].

- **Sample #3090** | True: `HALLUCINATED` | Predicted: `HALLUCINATED` | prob=0.8852 | margin=0.3252 | conf=Very High
  > Prediction: HALLUCINATED (p=0.885, threshold=0.56). Very High confidence with margin 0.325. Key supporting signals: [mean_entailment (+1.743), min_support_margin (+1.182)]. Key opposing signals: [max_entailment (-0.576), mean_contradiction (-0.114)].

### TN Examples (Top 3 by confidence margin)

- **Sample #1081** | True: `GROUNDED` | Predicted: `GROUNDED` | prob=0.4001 | margin=-0.1599 | conf=High
  > Prediction: GROUNDED (p=0.400, threshold=0.56). High confidence with margin 0.160. Key supporting signals: [mean_entailment (+0.256), num_claims (+0.153)]. Key opposing signals: [max_entailment (-0.575), min_support_margin (-0.023)].

- **Sample #1078** | True: `GROUNDED` | Predicted: `GROUNDED` | prob=0.4024 | margin=-0.1576 | conf=High
  > Prediction: GROUNDED (p=0.402, threshold=0.56). High confidence with margin 0.158. Key supporting signals: [mean_entailment (+0.206), num_claims (+0.196)]. Key opposing signals: [max_entailment (-0.560), min_support_margin (-0.020)].

- **Sample #2077** | True: `GROUNDED` | Predicted: `GROUNDED` | prob=0.4056 | margin=-0.1544 | conf=High
  > Prediction: GROUNDED (p=0.406, threshold=0.56). High confidence with margin 0.154. Key supporting signals: [mean_entailment (+0.301), num_claims (+0.131)]. Key opposing signals: [max_entailment (-0.574), min_support_margin (-0.025)].

### FP Examples (Top 3 by confidence margin)

- **Sample #2194** | True: `GROUNDED` | Predicted: `HALLUCINATED` | prob=0.9172 | margin=0.3572 | conf=Very High
  > Prediction: HALLUCINATED (p=0.917, threshold=0.56). Very High confidence with margin 0.357. Key supporting signals: [mean_entailment (+1.934), min_support_margin (+1.234)]. Key opposing signals: [max_entailment (-0.575), mean_contradiction (-0.061)].

- **Sample #2182** | True: `GROUNDED` | Predicted: `HALLUCINATED` | prob=0.8860 | margin=0.3260 | conf=Very High
  > Prediction: HALLUCINATED (p=0.886, threshold=0.56). Very High confidence with margin 0.326. Key supporting signals: [mean_entailment (+1.963), min_support_margin (+0.737)]. Key opposing signals: [max_entailment (-0.574), mean_contradiction (-0.014)].

- **Sample #1656** | True: `GROUNDED` | Predicted: `HALLUCINATED` | prob=0.7981 | margin=0.2381 | conf=High
  > Prediction: HALLUCINATED (p=0.798, threshold=0.56). High confidence with margin 0.238. Key supporting signals: [min_support_margin (+1.228), mean_entailment (+0.966)]. Key opposing signals: [max_entailment (-0.576), mean_contradiction (-0.161)].

### FN Examples (Top 3 by confidence margin)

- **Sample #952** | True: `HALLUCINATED` | Predicted: `GROUNDED` | prob=0.4092 | margin=-0.1508 | conf=High
  > Prediction: GROUNDED (p=0.409, threshold=0.56). High confidence with margin 0.151. Key supporting signals: [mean_entailment (+0.299), num_claims (+0.131)]. Key opposing signals: [max_entailment (-0.574), min_support_margin (-0.006)].

- **Sample #997** | True: `HALLUCINATED` | Predicted: `GROUNDED` | prob=0.4123 | margin=-0.1477 | conf=Moderate
  > Prediction: GROUNDED (p=0.412, threshold=0.56). Moderate confidence with margin 0.148. Key supporting signals: [num_claims (+0.262), mean_entailment (+0.166)]. Key opposing signals: [max_entailment (-0.567)].

- **Sample #352** | True: `HALLUCINATED` | Predicted: `GROUNDED` | prob=0.4123 | margin=-0.1477 | conf=Moderate
  > Prediction: GROUNDED (p=0.412, threshold=0.56). Moderate confidence with margin 0.148. Key supporting signals: [mean_entailment (+0.254), num_claims (+0.131)]. Key opposing signals: [max_entailment (-0.504), min_support_margin (-0.020)].
