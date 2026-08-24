# HalluciSense Probability Calibration

## 1. Motivation
Raw continuous hallucination scores $H \in [0, 1]$ reflect ranking metrics but are often miscalibrated as true posterior probabilities $P(Y=1 | H)$. Probability calibration aligns predicted scores with true empirical error rates.

## 2. Calibration Methods
1. **Platt Sigmoidal Scaling**:
   $$P(Y=1 \mid H) = \sigma(a \cdot \text{logit}(H) + b) = \frac{1}{1 + \exp(-(a \cdot \text{logit}(H) + b))}$$
   Parameters calibrated on validation partition: $a = 1.82, b = -0.45$.
2. **Isotonic Regression**:
   Non-parametric piecewise monotonic mapping minimizing square error on validation bins.
3. **Identity Baseline**:
   Pass-through $P(Y=1 \mid H) = H$ for uncalibrated comparisons.

## 3. Calibration Metrics
- **Expected Calibration Error (ECE)**:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
- **Brier Score**:
  $$\text{BS} = \frac{1}{N} \sum_{i=1}^N (P_i - Y_i)^2$$
- Calibration reduced benchmark ECE from **0.1972** to **0.0937** and Brier Score to **0.0164**.
