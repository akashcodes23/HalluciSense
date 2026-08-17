# Phase 8D Scientific Integrity Report

## 1. Non-Optimization and Pre-Registration
- **Dataset Frozen**: `dataset_8a.jsonl` was SHA-256 verified prior to evaluation. No records were added, removed, or edited.
- **Phase 6 Canonical Intact**: SHA-256 hash verified as `dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5`.
- **Fixed Decision Threshold**: Canonical comparison strictly conducted at $T=0.50$.
- **No Inconvenient Failures Omitted**: Every false positive, false negative, and regression case is preserved in `phase8d_paired_results.csv` and `phase8d_manual_review.csv`.

## 2. Multiple-Testing Disclosure
- All category-level hypothesis tests include Benjamini-Hochberg FDR-adjusted $q$-values to guard against family-wise Type I error inflation.
