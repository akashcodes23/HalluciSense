# HalluciSense Benchmark Dataset Manifest v1.0 (500 Prompts)

## Executive Summary

The **HalluciSense Hallucination Benchmark Dataset v1.0** consists of **500 curated prompts** designed to evaluate multi-stage hallucination detection engines across 10 critical knowledge categories.

---

## 1. Category Distribution (500 Prompts)

| Category | Count | Proportion |
| :--- | :--- | :--- |
| **Science** | 50 | 10.0% |
| **History** | 50 | 10.0% |
| **Medicine** | 50 | 10.0% |
| **Finance** | 50 | 10.0% |
| **Programming** | 50 | 10.0% |
| **Mathematics** | 50 | 10.0% |
| **Politics** | 50 | 10.0% |
| **Law** | 50 | 10.0% |
| **General Knowledge** | 50 | 10.0% |
| **Geography** | 50 | 10.0% |
| **TOTAL** | **500** | **100.0%** |

---

## 2. Class & Difficulty Distribution

- **Class Balance**: 333 Hallucination Prompts (66.6%) vs. 167 Factually Grounded Prompts (33.4%).
- **Difficulty Breakdown**:
  - **Easy**: 167 Prompts (33.4%)
  - **Medium**: 167 Prompts (33.4%)
  - **Hard**: 166 Prompts (33.2%)

---

## 3. Hallucination Type Taxonomy

1. **Factual Contradiction**: Direct opposition to verified source evidence.
2. **Entity Fabrication**: Invented names, places, papers, or organizations.
3. **Numerical Distortion**: Inaccurate dates, statistical measurements, or physical constants.
4. **Temporal Anachronism**: Placing historical events in non-existent timeframes.

---

*Manifest generated automatically by `scripts/build_500_benchmark.py`.*
