# HalluciSense Mathematical Fusion Formulations

## 1. Canonical Baseline Formulation (Mode A)
When all three signals ($\text{FE}, \text{CG}, \text{CF}$) are observed:

$$H_{\text{canonical}} = \alpha \cdot \text{FE} + \beta \cdot \text{CG} + \gamma \cdot \text{CF}$$

subject to:
$$\alpha + \beta + \gamma = 1.0, \quad \alpha, \beta, \gamma \ge 0$$

Default baseline weights learned from validation data:
- $\alpha = 0.40$ (Evidence Grounding)
- $\beta = 0.30$ (Predictive Confidence)
- $\gamma = 0.30$ (Semantic Consistency)

---

## 2. Availability-Aware Adaptive Fusion (Mode B)
In real-world deployment, black-box APIs omit logprobs ($m_{\text{CG}} = 0$) or single-turn prompts omit alternate samples ($m_{\text{CF}} = 0$).

$$H_{\text{adaptive}} = \frac{\sum_{i=1}^3 m_i \cdot r_i \cdot w_i \cdot S_i}{\sum_{i=1}^3 m_i \cdot r_i \cdot w_i}$$

where:
- $\mathbf{S} = [\text{FE}, \text{CG}, \text{CF}]^T \in [0, 1]^3$ is the signal score vector.
- $\mathbf{m} = [m_{\text{FE}}, m_{\text{CG}}, m_{\text{CF}}]^T \in \{0, 1\}^3$ is the signal availability indicator mask.
- $\mathbf{w} = [\alpha, \beta, \gamma]^T$ are the base importance coefficients.
- $\mathbf{r} = [r_{\text{FE}}, r_{\text{CG}}, r_{\text{CF}}]^T \in (0, 1]^3$ is the empirical signal reliability vector:
  * $r_{\text{FE}} = \max(0.1, \text{retrieval\_similarity})$
  * $r_{\text{CG}} = \max(0.1, \text{token\_calibration\_score})$
  * $r_{\text{CF}} = \max(0.1, \text{semantic\_agreement\_consistency})$

This dynamic re-normalization prevents artificial score deflation and eliminates the need for arbitrary zero-imputation.
