# Phase 18 — Prioritized Manuscript Revision Plan

## 1. Priority P0 — Mandatory Pre-Submission Verifications
- [x] **P0.1 — Baseline Comparability Separation:** Verify Table 4 in `main.tex` explicitly distinguishes Category A (Directly Evaluated) from Category C (Published Literature Reference).
- [x] **P0.2 — Statistical Effect Size Precision:** Ensure all manuscript mentions of Cohen's $d$ reference the paired per-instance effect size ($d = 1.42$) and that bootstrap $z$-scores ($25.69$) are not reported as Cohen's $d$.
- [x] **P0.3 — Canonical Benchmark Invariance:** Verify canonical benchmark SHA-256 hash matches `dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5`.
- [x] **P0.4 — Zero-Logit Safety Guarantee:** Verify that black-box API interactions never manufacture dummy token probabilities.

---

## 2. Priority P1 — Strongly Recommended Enhancements
- [x] **P1.1 — Retrieval Latency Disclosure:** Document in Section 4 and Section 12 that external Wikipedia retrieval accounts for $\sim 65\%$ ($780\text{ ms}$) of pipeline latency.
- [x] **P1.2 — Abstention Tradeoff Discussion:** Explicitly discuss the operational implications of rejecting $20\%$ of high-ambiguity queries to achieve zero empirical selective risk.
- [x] **P1.3 — Edge-Case Denominator Safety:** Ensure Section 4.2 documents explicit trapping of $\sum m_i = 0$.

---

## 3. Priority P2 — Optional Future Extensions
- [ ] **P2.1 — Multilingual Grounding:** Expand reference indexing to multilingual Wikipedia dumps (future work).
- [ ] **P2.2 — Computer Algebra System (CAS) Integration:** Integrate SymPy for arbitrary symbolic calculus (future work).
