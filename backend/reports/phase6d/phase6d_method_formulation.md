# Phase 6D-B: Method Formalization & Mathematical Framework

**Title**: Formalization of Modality-Aware Temporal Verification and Global Evidence Alignment  
**Date**: 2026-08-11  

---

## 1. Problem Definition & Operational Input

Let a verification request be defined as a tuple $(Q, R, E)$, where:
- $Q \in \Sigma^*$ is the user query string.
- $R \in \Sigma^*$ is the LLM-generated response text.
- $E = \{e_1, e_2, \dots, e_m\}$ is the set of retrieved evidence passages, where each passage $e_j = (t_j, s_j)$ consists of text $t_j$ and metadata/source $s_j$.

The objective is to compute a calibrated hallucination risk score $H(Q, R, E) \in [0, 1]$ and an associated risk classification $L \in \{\text{VERIFIED}, \text{NEEDS\_VERIFICATION}, \text{MODERATE\_RISK}, \text{LIKELY\_HALLUCINATED}\}$.

---

## 2. Epistemic Modality Ontology

We define a formal epistemic modality ontology $\mathcal{M}$:

$$\mathcal{M} = \begin{cases}
\text{ASSERTED\_FACT} & \text{Direct factual claim about past or present reality} \\
\text{FUTURE\_FACT\_ASSERTION} & \text{Claim asserting a completed past action with a future date} \\
\text{PREDICTION} & \text{Forward-looking expectation, forecast, or planned action} \\
\text{HYPOTHETICAL} & \text{Supposition or imagined scenario without factual commitment} \\
\text{COUNTERFACTUAL} & \text{Alternative scenario contrary to historical fact} \\
\text{CONDITIONAL} & \text{Dependent proposition conditioned on a premise} \\
\text{NEGATED\_FACT} & \text{Explicit denial of an event or state} \\
\text{QUOTED\_CLAIM} & \text{Attributed statement or reported claim} \\
\text{META\_CLAIM} & \text{Statement analyzing or debunking another claim} \\
\text{FICTIONAL} & \text{Claim anchored within a creative/fictional narrative domain} \\
\text{UNKNOWN} & \text{Uncertain or ambiguous epistemic stance}
\end{cases}$$

---

## 3. Independent Dual-Modality Resolution

To prevent query context contamination (e.g. a user asking a hypothetical question "What if X happened in 2030?" protecting a response asserting "X happened in 2030 as a fact"), query modality $M_q$ and claim modality $M(c_i)$ are resolved independently:

$$M_q = \Phi_{\text{modality}}(Q)$$
$$M(c_i) = \Phi_{\text{modality}}(c_i)$$

where $\Phi_{\text{modality}}: \Sigma^* \to \mathcal{M}$ maps a text sequence to its primary epistemic modality.

---

## 4. Atomic Claim Decomposition

Response $R$ is decomposed into a set of atomic claims $C = \{c_1, c_2, \dots, c_n\}$ using dependency-based syntactic claim extraction $\Psi_{\text{claim}}: R \to 2^{\Sigma^*}$.

For each atomic claim $c_i \in C$, we extract:
1. **Temporal Representation**: $T(c_i) = (Y_{c_i}, I_{c_i}, R_{c_i})$, where $Y_{c_i}$ is the set of explicit 4-digit years, $I_{c_i}$ represents interval bounds, and $R_{c_i}$ represents relational operators (e.g., `before`, `after`, `during`, `since`).
2. **Epistemic Frame**: $M(c_i) \in \mathcal{M}$.
3. **Local Evidence Subset**: $E(c_i) = \{e_j \in E \mid \text{relevance}(c_i, e_j) > \tau_{\text{rel}}\}$.

---

## 5. Epistemic Gating Function

We define the Epistemic Gating Function $G(M_q, M(c_i)) \in \{0, 1\}$:

$$G(M_q, M(c_i)) = \begin{cases}
0 & \text{if } M(c_i) \in \{\text{PREDICTION}, \text{HYPOTHETICAL}, \text{COUNTERFACTUAL}, \text{CONDITIONAL}, \text{NEGATED\_FACT}, \text{QUOTED\_CLAIM}, \text{FICTIONAL}\} \\
1 & \text{if } M(c_i) \in \{\text{ASSERTED\_FACT}, \text{FUTURE\_FACT\_ASSERTION}, \text{UNKNOWN}\}
\end{cases}$$

**Invariant**: $G(M_q, M(c_i))$ depends strictly on the claim modality $M(c_i)$. The query modality $M_q$ is recorded for diagnostics but CANNOT override $M(c_i) = \text{ASSERTED\_FACT}$.

---

## 6. Global Evidence-Date Alignment

Let $Y_E = \bigcup_{e_j \in E} \text{years}(e_j)$ be the global set of all temporal anchors present across the entire evidence set $E$.

For a claim $c_i$ with extracted years $Y_{c_i}$:

1. **Global Support Criterion**:
   $$\text{Supported}(Y_{c_i}, Y_E) = \bigwedge_{y \in Y_{c_i}} \mathbb{I}(y \in Y_E)$$

2. **Temporal Mismatch Score**:
   $$S_{\text{temporal}}(c_i, Y_E) = \begin{cases}
   0.92 & \text{if } M(c_i) = \text{FUTURE\_FACT\_ASSERTION} \\
   0.90 & \text{if } \neg\text{Supported}(Y_{c_i}, Y_E) \land R_{c_i} = \emptyset \land \text{Discrepancy}(Y_{c_i}, Y_E) \ge 3 \text{ yrs} \\
   0.0 & \text{otherwise}
   \end{cases}$$

---

## 7. Composite Risk Fusion

For atomic claim $c_i$, the composite verification score $H(c_i)$ combines factual NLI grounding $S_{\text{NLI}}(c_i, E(c_i))$ and gated temporal inconsistency:

$$H(c_i) = \max \left( S_{\text{NLI}}(c_i, E(c_i)), \; G(M_q, M(c_i)) \cdot S_{\text{temporal}}(c_i, Y_E) \right)$$

The response-level risk score is the supremum over all atomic claims:

$$H(R) = \max_{c_i \in C} H(c_i)$$

Risk level mapping follows frozen production thresholds:
$$L(H) = \begin{cases}
\text{VERIFIED} & \text{if } H < 0.35 \\
\text{NEEDS\_VERIFICATION} & \text{if } 0.35 \le H < 0.50 \\
\text{MODERATE\_RISK} & \text{if } 0.50 \le H < 0.65 \\
\text{LIKELY\_HALLUCINATED} & \text{if } H \ge 0.65
\end{cases}$$

---

## 8. Taxonomy of Verification Failure Modes

| Error Category | Symbol | Description | Corrective Mechanism |
|:---|:---:|:---|:---|
| **Factual Contradiction** | $E_{\text{fact}}$ | Response directly contradicts evidence facts | NLI Entailment ($S_{\text{NLI}}$) |
| **Temporal Mismatch** | $E_{\text{temp}}$ | Past event assigned incorrect year | Global Evidence Alignment ($Y_E$) |
| **Future Fact Assertion** | $E_{\text{fut}}$ | Completed action assigned future date | Future Fact Assertion Detector ($S_{\text{temporal}}=0.92$) |
| **False Modality Penalty** | $E_{\text{modal\_FP}}$ | Valid prediction/hypothetical penalized as hallucination | Epistemic Gate ($G=0$) |
| **Evidence Date Contamination** | $E_{\text{noise\_FP}}$ | Background date in single snippet triggers false penalty | Global Evidence Set Union ($Y_E$) |
| **Relational Misinterpretation** | $E_{\text{rel\_FP}}$ | Naive date difference applied to "before/after" clause | Relational Operator Protection ($R_{c_i}$) |
