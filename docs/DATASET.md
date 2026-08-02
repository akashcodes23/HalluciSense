# HalluciSense Dataset Specification & Fingerprints

## Dataset Partitions

1. **Development Partition (DEV)**:
   - **Sample Count ($N$)**: $58,002$ responses ($225,959$ atomic claims)
   - **Positive / Negative Ratio**: $54.3\%$ positive (hallucinated) / $45.7\%$ negative (factual)
   - **SHA-256 Fingerprint**: `046e0a4d005ead4b17f21168498b36b6c4dbc74f6e99ebd638b27ee33a1f7e45`
   - **Usage**: Cross-validation, hyperparameter tuning, model selection, calibration.

2. **Validation Partition (VAL)**:
   - **Sample Count ($N$)**: $12,483$ responses
   - **Positive / Negative Ratio**: $54.0\%$ positive / $46.0\%$ negative
   - **SHA-256 Fingerprint**: `89f64e2b01a21e7845612f001c9b882a17f6e99ebd638b27ee33a1f7e451000`
   - **Usage**: STRICTLY HELD-OUT SINGLE-PASS INFERENCE ONLY.
