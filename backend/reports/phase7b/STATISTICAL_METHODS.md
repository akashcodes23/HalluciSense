# HalluciSense Phase 7B — Statistical Methods & Testing Protocols

## 1. Primary Metrics Formulations
* **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
* **Balanced Accuracy**: $\frac{1}{2} \left(\frac{TP}{TP + FN} + \frac{TN}{TN + FP}\right)$
* **Precision**: $\frac{TP}{TP + FP}$
* **Recall (Sensitivity)**: $\frac{TP}{TP + FN}$
* **Specificity**: $\frac{TN}{TN + FP}$
* **$F_1$ Score**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
* **Matthews Correlation Coefficient (MCC)**:
  $$\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$
* **Expected Calibration Error (ECE)**:
  $$\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{N} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$$
* **Brier Score**: $\frac{1}{N} \sum_{i=1}^{N} (H_i - y_i)^2$

---

## 2. Hypothesis Testing Protocols
1. **McNemar's Test for Paired Classifications**:
   Used to test whether disagreement between detectors ($b$ vs $c$) is statistically significant:
   $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}, \quad \text{df} = 1$$
2. **Wilcoxon Signed-Rank Test for Continuous Risk Scores**:
   Non-parametric paired comparison between predicted continuous scores $H_{\text{Full}}$ vs $P_1$.
3. **Bootstrap Confidence Intervals**:
   Computed using empirical percentile bootstrap with $B = 2000$ resamples and random seed $42$. Confidence intervals are reported at the 95% level ($\alpha = 0.05$).
4. **Held-Out Calibration Protocol**:
   Data is partitioned into 70% validation ($N=525$) and 30% test ($N=225$). All threshold sweeping, Platt scaling, and Isotonic regression are learned solely on the validation partition and evaluated strictly on the unseen test partition.
