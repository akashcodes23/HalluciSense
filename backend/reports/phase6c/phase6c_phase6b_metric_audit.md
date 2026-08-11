# Phase 6C: Phase 6B Metric Audit Report

**Generated**: 2026-08-10
**Auditor**: Phase 6C Pre-Implementation Inspection
**Raw predictions available**: YES (TP/TN/FP/FN per config in phase6b_novelty_experiment_results.json)

---

## Section 1: A0-A9 Metric Verification

All ten Phase 6B ablation configurations were independently re-verified by reconstructing y_true
and y_pred from reported confusion matrices and recomputing using evaluation/metrics.py.

### Result: 0 DISCREPANCIES FOUND. All A0-A9 metrics are internally consistent and correct.

| Config | TP | TN | FP | FN | F1 (rep) | F1 (calc) | MCC | Status |
|:---|---:|---:|---:|---:|---:|---:|---:|:---:|
| A0_NLI_Baseline | 121 | 231 | 46 | 152 | 0.5500 | 0.5500 | 0.3014 | VALID |
| A1_NLI_Retrieval | 121 | 231 | 46 | 152 | 0.5500 | 0.5500 | 0.3014 | VALID |
| A2_Plus_Temporal | 121 | 229 | 48 | 152 | 0.5475 | 0.5475 | 0.2925 | VALID |
| A3_Plus_Modality | 0 | 275 | 2 | 273 | 0.0000 | 0.0000 | -0.0600 | VALID |
| A4_Plus_AtomicClaim | 164 | 159 | 118 | 109 | 0.5910 | 0.5910 | 0.1748 | VALID |
| A5_Plus_GlobalAlign | 4 | 277 | 0 | 269 | 0.0289 | 0.0289 | 0.0862 | VALID |
| A6_Plus_Relational | 117 | 239 | 38 | 156 | 0.5467 | 0.5467 | 0.3238 | VALID |
| A7_Plus_MetaFiction | 121 | 234 | 43 | 152 | 0.5538 | 0.5538 | 0.3148 | VALID |
| A8_Plus_DynAnchor | 123 | 231 | 46 | 150 | 0.5566 | 0.5566 | 0.3083 | VALID |
| A9_Full_HalluciSense | 123 | 223 | 54 | 150 | 0.5467 | 0.5467 | 0.2736 | VALID |

---

## Section 2: Issues Classified

### ISSUE 1 — HaluBench: 100% Hallucinated Dataset
- HaluBench (N=100) = positive-only; combined 550-case is ~balanced (49.6%/50.4%)
- Classification: VALID for combined; NOT_REPRODUCIBLE per-dataset standalone
- Safe to cite: YES with disclosure

### ISSUE 2 — Evidence Noise Stress Test: N=5 Per Condition
- E1-E6 evaluated on exactly 5 manually selected test cases
- Phase 5 "40%" = 2/5 correct; one sample difference cannot support statistical claim
- Classification: VALID as mechanistic illustration; INVALID_FOR_PUBLICATION as statistical result
- Safe to cite: Only as qualitative illustration with N=5 disclosed

### ISSUE 3 — Phase 6 Ablation Table Mixes Datasets
- phase6_ablation.md rows A-F: 70-case Phase 5 holdout
- Row G (Full Phase 6): 105-case unseen benchmark (DIFFERENT dataset)
- Accuracy drops from 87.62% to 53.33% NOT because system degrades,
  but because two different datasets are compared in the same table
- Classification: REPRODUCIBLE_BUT_REPORTED_INCORRECTLY
- Safe to cite: NO without explicit dataset labeling per row

### ISSUE 4 — A0 and A1 Identical Results
- Both use evaluate_claims_against_evidence([resp], ev)[0] with identical evidence construction
- Retrieval is not ablated separately in the harness design
- Classification: VALID (correctly computed); requires disclosure in publication
- Safe to cite: YES with disclosure that A0=A1 and retrieval not separately ablated

### ISSUE 5 — A3/A5 Degenerate Intermediate States
- A3: modality gate as replacement scorer -> 0.0 for most examples -> F1=0.0
- A5: alignment gate as replacement scorer -> 0.0 for temporal-free examples -> F1=0.03
- These are harness artifacts, NOT findings that components degrade performance
- Classification: VALID (correctly computed); REPRODUCIBLE_BUT_REPORTED_INCORRECTLY (framing)
- Safe to cite: Only with explicit explanation of harness design artifact

### ISSUE 6 — Phase 6 Unseen Benchmark Accuracy Discrepancy
- phase6_architectural_evaluation.md: Accuracy=89.52% (94/105)
- phase6_unseen_benchmark.json: Accuracy=53.33% (56/105) — SAME 105 cases
- Cause: 89.52% = temporal engine local-only (pre-freeze config); 53.33% = full pipeline (frozen)
- The 53.33% is the authoritative frozen-system measurement
- Classification: REPRODUCIBLE_BUT_REPORTED_INCORRECTLY
- Safe to cite: 53.33% ONLY. The 89.52% figure MUST NOT be cited as the Phase 6 system accuracy.

---

## Section 3: Summary

| Result | Classification | Safe to Cite |
|:---|:---:|:---:|
| Phase 6B A0-A9 confusion matrices | VALID | YES |
| Phase 6B A0-A9 metric calculations | VALID | YES |
| Phase 6B AUROC/AUPRC | VALID | YES |
| Evidence noise stress test as statistical evidence | INVALID_FOR_PUBLICATION | NO |
| Evidence noise stress test as mechanistic illustration | VALID (disclose N=5) | With disclosure |
| Phase 6 ablation table (mixed datasets) | REPRODUCIBLE_BUT_REPORTED_INCORRECTLY | Only with correction |
| A0=A1 identical results | VALID (disclose harness) | With disclosure |
| A3/A5 degenerate intermediate states | VALID (disclose harness artifact) | With framing |
| Phase 6 unseen benchmark Acc=89.52% | NOT_REPRODUCIBLE (pre-freeze config) | NO |
| Phase 6 unseen benchmark Acc=53.33% | VALID (frozen system) | YES |
| Phase 6 blind holdout (70 cases, 88.57%) | VALID | YES |
