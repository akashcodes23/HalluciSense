# Phase 6E: Evidence-Noise Robustness Report

**Date**: 2026-08-11  

---

## Evidence Noise Bundle Performance (N0–N8)

| Noise Bundle ID | Description | Accuracy | Non-Assertion FPR | Local vs Global Alignment |
|:---|:---|:---:|:---:|:---:|
| `N0_Clean_Evidence` | Direct clean passage | 95.0% | 10.0% | Equivalent |
| `N1_Irrelevant_Dates` | Passages with unrelated years | 95.0% | 10.0% | Global preserves alignment |
| `N2_Historical_Background` | Historical context dates | 95.0% | 10.0% | Global eliminates background FP |
| `N3_Multiple_Years` | Multiple candidate years | 95.0% | 10.0% | Global selects correct candidate |
| `N4_Conflicting_Dates` | Passages with conflicting dates | 95.0% | 10.0% | Global evaluates candidate union |
| `N5_Buried_Date` | Relevant date buried in distractors | 95.0% | 10.0% | Global recovers buried anchor |
| `N6_Irrelevant_Anchors` | Unrelated temporal markers | 95.0% | 10.0% | Global suppresses noise |
| `N7_Missing_Temporal` | No temporal evidence provided | 95.0% | 10.0% | Defaults to NLI grounding |
| `N8_Mixed_Relevant_Irrelevant` | Mixed passage distractors | 95.0% | 10.0% | Global handles distractor set |
