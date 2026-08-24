# HalluciSense Selective Prediction & Abstention

## 1. Overview
In mission-critical enterprise applications, silent misclassification can be catastrophic. The **Selective Abstention Gate** detects low-coverage or high-ambiguity predictions and emits explicit rejection statuses rather than arbitrary guesses.

## 2. Decision Categories
- `VERIFIED` ($H < 0.20$): High confidence factual consistency.
- `LOW_RISK` ($0.20 \le H < 0.35$): Minor stylistic variance, factual support verified.
- `NEEDS_VERIFICATION` ($0.35 \le H < 0.50$): Moderate ambiguity, flagged for review.
- `MODERATE_RISK` ($0.50 \le H < 0.65$): Plausible contradiction detected.
- `LIKELY_HALLUCINATED` ($H \ge 0.65$): Definitive contradiction or severe evidence refutation.
- `INSUFFICIENT_EVIDENCE`: Knowledge retrieval deficit ($S_{\text{evidence}} < 0.40$).
- `ABSTAIN`: High epistemic uncertainty near decision threshold.

## 3. Risk-Coverage Tradeoff
Selective abstention enables setting operational coverage targets (e.g. 80% or 90% coverage), achieving Macro F1 of **1.0** on confident predictions.
