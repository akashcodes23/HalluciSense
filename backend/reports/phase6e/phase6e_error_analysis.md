# Phase 6E: Error Taxonomy & Analysis Report

**Date**: 2026-08-11  
**Total Errors**: 30 out of 600 records (5.0% error rate).

---

## Error Category Breakdown

| Error Code | Category Description | Count | Percentage |
|:---|:---|:---:|:---:|
| `E1` | NLI Baseline Failure | 0 | 0.0% |
| `E2` | Modality Resolution Failure | 30 | 100.0% |
| `E3`–`E13` | Other Categories | 0 | 0.0% |

All 30 errors occurred on false positive predictions where NLI cross-encoder scored high uncertainty on complex non-assertional phrasing. Zero factual assertions were misclassified.
