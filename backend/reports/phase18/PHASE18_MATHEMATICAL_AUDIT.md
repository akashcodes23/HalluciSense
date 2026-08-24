# Phase 18 — Mathematical Formulation & Edge-Case Audit

## 1. Equation Formulation & Dimensional Analysis
The availability-aware adaptive fusion formula is defined as:
$$H_{\text{adaptive}} = \frac{\sum_{i \in \{\text{FE}, \text{CG}, \text{CF}\}} m_i r_i w_i S_i}{\sum_{i \in \{\text{FE}, \text{CG}, \text{CF}\}} m_i r_i w_i}$$
where:
- $\mathbf{w} = [w_{\text{FE}}, w_{\text{CG}}, w_{\text{CF}}] = [0.40, 0.30, 0.30]$ with $\sum w_i = 1.0$.
- $\mathbf{m} = [m_{\text{FE}}, m_{\text{CG}}, m_{\text{CF}}] \in \{0, 1\}^3$.
- $\mathbf{r} = [r_{\text{FE}}, r_{\text{CG}}, r_{\text{CF}}] \in (0, 1]^3$.
- $\mathbf{S} = [S_{\text{FE}}, S_{\text{CG}}, S_{\text{CF}}] \in [0, 1]^3$.

### Dimensional Consistency
Since $m_i \in \{0, 1\}$, $r_i \in (0, 1]$, $w_i > 0$, and $S_i \in [0, 1]$, the numerator is a non-negative scalar $\sum m_i r_i w_i S_i \ge 0$, and the denominator is $\sum m_i r_i w_i > 0$ for any non-zero mask. Because $S_i \le 1.0$, the quotient is strictly bounded in $[0, 1.0]$.

---

## 2. Exhaustive Edge-Case Mask Analysis (8 Configurations)

| Mask $\mathbf{m}$ | Operational Scenario | Active Denominator | Renormalized Effective Weights $(w_{\text{FE}}^*, w_{\text{CG}}^*, w_{\text{CF}}^*)$ | Mathematical Behavior | Denominator Safety |
| :---: | :--- | :---: | :---: | :--- | :---: |
| `[1, 1, 1]` | Complete Tri-Pillar Observability | $0.40 r_1 + 0.30 r_2 + 0.30 r_3$ | $(0.40, 0.30, 0.30)$ (assuming $r_i=1$) | Fully balanced multi-signal triangulation | **SAFE** |
| `[1, 0, 1]` | Black-Box API (No Logprobs) | $0.40 r_1 + 0.30 r_3$ | $(0.5714, 0.0000, 0.4286)$ | Renormalizes cleanly across grounding & consistency | **SAFE** |
| `[1, 1, 0]` | Single-Turn (No Alternate Samples) | $0.40 r_1 + 0.30 r_2$ | $(0.5714, 0.4286, 0.0000)$ | Renormalizes cleanly across grounding & confidence | **SAFE** |
| `[0, 1, 1]` | Offline (No Retrieval / Grounding) | $0.30 r_2 + 0.30 r_3$ | $(0.0000, 0.5000, 0.5000)$ | Equal split between internal predictive channels | **SAFE** |
| `[1, 0, 0]` | Single-Turn Black-Box (P1 Only) | $0.40 r_1$ | $(1.0000, 0.0000, 0.0000)$ | Pure evidence grounding | **SAFE** |
| `[0, 1, 0]` | Token Logprobs Only (P2 Only) | $0.30 r_2$ | $(0.0000, 1.0000, 0.0000)$ | Pure token entropy predictive uncertainty | **SAFE** |
| `[0, 0, 1]` | Alternate Samples Only (P3 Only) | $0.30 r_3$ | $(0.0000, 0.0000, 1.0000)$ | Pure semantic consistency variance | **SAFE** |
| `[0, 0, 0]` | Complete Signal Absence / Total Failure | $0.0000$ | $(0.0000, 0.0000, 0.0000)$ | **Triggers explicit `INSUFFICIENT_EVIDENCE` / `ABSTAIN` fallback** | **TRAPPED & SAFE** |

---

## 3. Total Missingness Trapping ($m=[0, 0, 0]$)
In [`backend/app/core/engine/fusion.py`](file:///Users/akashgpatil/major_project/backend/app/core/engine/fusion.py), if $\sum m_i = 0$, the engine traps the zero denominator, sets `H_score = None`, emits an error log, and downstream handlers route the claim directly to `Category.INSUFFICIENT_EVIDENCE` with `requires_abstention = True`.
