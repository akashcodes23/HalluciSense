# HalluciSense Root-Cause Failure Taxonomy

HalluciSense classifies detected hallucinations into 7 single-label failure modes:

| Classification | Description | Detection Mechanism |
| :--- | :--- | :--- |
| **FACTUAL_CONTRADICTION** | Explicit logical refutation by authoritative evidence. | DeBERTa-v3 NLI contradiction probability $P(\text{contra}) \ge 0.60$. |
| **UNVERIFIED_EXTRAPOLATION** | Plausible claim unsupported by retrieved reference passages. | Neutral NLI probability under low similarity evidence. |
| **NUMERIC_UNIT_ERROR** | Factual numbers or SI physical units altered or scaled incorrectly. | Symbolic regex unit conversion & scale verification. |
| **NEGATION_INVERSION** | Logical polarity flipped (e.g. "not", "never", "unlikely"). | Morphological negation dependency parser. |
| **CAUSAL_INVERSION** | Cause and effect entities reversed in relationship. | Causal directionality asymmetry matcher. |
| **TEMPORAL_ANACHRONISM** | Historical sequence or dated event misaligned. | Temporal dependency parser and date entity alignment. |
| **LOW_CONFIDENCE_SPECULATION** | High model entropy / uncertainty during token generation. | Token entropy $H(p) \ge 1.50$ & low prob fraction $\ge 0.40$. |
