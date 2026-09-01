# Phase 46.4 — Adaptive Fusion Hardening Audit

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 46.4 — Adaptive Fusion Availability & Weighting  
**Production Commit:** `41e1186`  
**Date:** 2026-09-01  

---

## 1. Mathematical Formulation

$$H_{\text{adaptive}} = \frac{\sum_{i=1}^3 m_i \cdot r_i \cdot w_i \cdot S_i}{\sum_{i=1}^3 m_i \cdot r_i \cdot w_i}$$

where:
- $S = [FE, CG, CF]$ is the pillar hallucination signal vector.
- $m \in \{0, 1\}^3$ is the explicit availability mask.
- $r \in (0, 1]^3$ is the empirical signal reliability vector.
- $w = [\alpha, \beta, \gamma] = [0.45, 0.30, 0.25]$ are base feature weights.

---

## 2. Invariants Preserved

1. **Unavailable $\ne$ Zero:** A missing pillar is omitted from the denominator and renormalized, rather than assumed to contribute zero hallucination risk.
2. **Canonical 19-Feature Schema Preserved:** Feature ordering and scaler transformations remain unchanged.
3. **Threshold Invariant:** Operating threshold $\tau^* = 0.54$ remains strictly unchanged.
