# HalluciSense Natural Language Inference (NLI) Diagnostic Report

## Model Configuration
* **NLI Model**: `cross-encoder/nli-deberta-v3-small`
* **Evaluated Pairs**: `1`

## Performance Metrics

| Diagnostic Metric | Value | Target | Status |
|:---|:---:|:---:|:---:|
| **Mean Entailment Probability** | `0.9981` | - | - |
| **Mean Neutral Probability** | `0.0018` | - | - |
| **Mean Contradiction Probability** | `0.0001` | - | - |
| **False Entailment Rate** | `0.0000` | $\le 0.05$ | ✅ |
| **False Contradiction Rate** | `0.0000` | $\le 0.05$ | ✅ |
| **Neutral Ambiguity Rate** | `0.0000` | $\le 0.15$ | ✅ |
