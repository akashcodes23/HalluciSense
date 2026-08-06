# HalluciSense Automated Error Analysis Report

## False Positive & False Negative Failure Clusters

### False Positive (FP) Failure Clusters
1. **Ambiguous Medical Acronyms** (Count: 14) — Misinterpreting context-specific medical abbreviations.
2. **Legal Citation Overlap** (Count: 11) — State vs Federal court case numbering confusion.
3. **Mathematical Notation Variants** (Count: 8) — Alternative latex notation parsed as discrepancy.

### False Negative (FN) Failure Clusters
1. **Stereochemistry Descriptor Swap** (Count: 9) — R/S stereocenter inversion in chemical names.
2. **Historical Date Precision** (Count: 7) — Plausible date shifts within 12 months.

## Domain Error Rates
- Clinical Medicine: 3.8%
- Legal Jurisprudence: 4.2%
- Organic Chemistry: 4.5%
- Mathematics: 3.5%
- World History: 2.9%
