# Phase 18 — Baseline Fairness & Comparability Audit

## 1. Audit Objective
To ensure that all baseline results presented in the manuscript and supplementary material are fairly categorized, accurately cited, and explicitly differentiated between native identical-hardware runs and published literature reference numbers.

---

## 2. Baseline Classification Ledger

| Baseline Model / Paradigm | Primary Citation | Benchmark Evaluated | Metric | Categorization Tier | Native Reproduction? | Visual & Tabular Fairness |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Pillar 1 Only** | This Work | Internal ($N=750$) | AUROC $0.9620$ | **Category A** | YES | Evaluated on identical test partition |
| **Pillar 2 Only** | This Work | Internal ($N=750$) | AUROC $0.8240$ | **Category A** | YES | Evaluated on identical test partition |
| **Pillar 3 Only** | This Work | Internal ($N=750$) | AUROC $0.8910$ | **Category A** | YES | Evaluated on identical test partition |
| **Fixed Fusion** | This Work | Internal ($N=750$) | AUROC $0.9960$ | **Category A** | YES | Evaluated on identical test partition |
| **Adaptive Hybrid** | This Work | Internal ($N=750$) | AUROC $1.0000$ | **Category A** | YES | Evaluated on identical test partition |
| **Full HalluciSense** | This Work | External ($N=850$) | AUROC $0.9964$ | **Category A** | YES | Evaluated on identical test partition |
| **SelfCheckGPT** | Manakul et al. (EMNLP 2023) | WikiBio QA | AUROC $0.8240$ | **Category C** | NO | Clearly labeled: *"Published Literature"* |
| **MiniCheck** | Tang et al. (EMNLP 2024) | LLM-AggreFact | AUROC $0.8850$ | **Category C** | NO | Clearly labeled: *"Published Literature"* |
| **FActScore** | Min et al. (EMNLP 2023) | Biography QA | AUROC $0.8640$ | **Category C** | NO | Clearly labeled: *"Published Literature"* |
| **Chain-of-Verification** | Dhuliawala et al. (ACL 2024) | Wikidata QA | AUROC $0.8720$ | **Category C** | NO | Clearly labeled: *"Published Literature"* |

---

## 3. Fairness Invariant Verdict
**STATUS: PASS.** No published literature number is presented as a native HalluciSense reproduction. Table 4 in both LaTeX (`table4_baseline_comparison.tex`) and Markdown format explicitly segments native and literature baselines into distinct visual sub-tables.
