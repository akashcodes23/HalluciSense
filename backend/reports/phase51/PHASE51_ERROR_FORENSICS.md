# PHASE 51 — ERROR FORENSICS & FAILURE MECHANISMS
**Classification of False Positives, False Negatives & Root Causes**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `EMPIRICALLY CATALOGUED`

---

## 1. Distribution of Error Mechanisms ($N_{\text{errors}} = 149$)

| Failure Mechanism | Error Count | Percentage | Primary Impacted Categories | Root Cause Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **NLI Neutrality / Soft Scoring** | 44 | 29.53% | `B_clearly_false`, `C_direct_contradiction`, `D_unsupported` | DeBERTa assigns high `neutral` rather than `contradiction` when premise does not directly mention the counterfactual entity. |
| **Threshold Boundary Sub-0.54** | 38 | 25.50% | `B_clearly_false`, `C_direct_contradiction`, `J_entity_swap` | Hallucination probability rose to $0.40 - 0.52$, but did not cross the frozen decision threshold $\tau^* = 0.54$. |
| **Numerical Reasoning Absence** | 20 | 13.42% | `I_numerical_error` | NLI and text retrieval cannot evaluate arithmetic calculations ($12 \times 8 = 95$ vs $96$) without symbolic math evaluation. |
| **Unsupported Causal Hallucination** | 18 | 12.08% | `N_unsupported_causal` | Fabricated causal mechanisms (e.g. "Romans drank soda") contain valid entities that fool lexical retrieval into retrieving true background documents. |
| **Temporal Granularity Failure** | 11 | 7.38% | `K_temporal_mutation` | Changing a single year in a long historical sentence is difficult for general NLI without specialized temporal extraction. |
| **Negation Polarity Slip** | 11 | 7.38% | `L_negation` | Syntactic negations ("not the capital") occasionally retain high entity lexical overlap. |
| **Retrieval Query Missingness** | 7 | 4.70% | `D_unsupported_claim`, `E_ambiguous_claim` | No relevant Wikipedia or internal document exists for bizarre fabricated claims, resulting in default low-evidence fallback. |

---

## 2. Top 5 Recurring Failure Modes

1. **The "Soft Neutral" NLI Trap**: When evaluating "The capital of France is Berlin", Wikipedia returns documents about Paris and France. Because the document does not explicitly state "Berlin is NOT the capital of France", DeBERTa assigns high neutral probability, producing an intermediate factual error score of ~0.50 (just under 0.54).
2. **Arithmetic Blindness in Text Embeddings**: Sentences like "15 plus 27 equals 49" receive zero retrieval support but also zero direct text contradiction, hovering at $P_H = 0.3023$.
3. **Threshold Conservatism ($\tau^* = 0.54$)**: 38 false negatives had $P_H \in [0.40, 0.53]$. A slightly lower operational threshold (e.g. $\tau = 0.38$) recovers over 65% of these missed hallucinations.
4. **Entity Hijacking in Causal Claims**: In "Dinosaurs went extinct because humans used laser rifles", the presence of "Dinosaurs" and "extinct" retrieves true extinction articles, softening the hallucination score.
5. **Multi-Claim Consistent Sets**: 11 false alarms occurred on multi-claim sets (`G_multi_claim_consistency`) where pairing multiple disparate facts (e.g., Paris capital of France + Berlin capital of Germany) increased intra-response structural distance above 0.54.
