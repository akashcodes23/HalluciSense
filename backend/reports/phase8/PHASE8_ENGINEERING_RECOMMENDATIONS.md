# Phase 8 Engineering Recommendations

1. **Deploy Claim Decomposition in Production**: Adopt `ClaimDecomposer` as the default front-end before Pillar 1 NLI scoring.
2. **Standardize Numeric and Unit Normalization**: Enforce `NumericUnitChecker` to intercept orders of magnitude discrepancies before token-level embeddings.
3. **Implement Dynamic Label Verification**: Discontinue using static hallucination labels for evaluating generative pipelines in favor of response-grounded verification.
