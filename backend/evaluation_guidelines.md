# HalluciSense Human Evaluation & Annotation Guidelines

## Overview
This document provides instructions for human domain experts evaluating AI-generated responses alongside HalluciSense H-Scores.

---

## 1. Annotation Labels
Reviewers must assign one of the following labels to each response:
1. **True**: Factually accurate and fully supported by reference domain evidence.
2. **False / Hallucinated**: Contains clear factual errors, fabricated entities, or contradicted claims.
3. **Partially Hallucinated**: Mix of true facts and unverified/contradicted claims.
4. **Uncertain**: Evidence is insufficient to verify truth value.

---

## 2. Review Process
1. Inspect the `prompt` and `gemini_response`.
2. Compare claims against trusted domain sources.
3. Fill `human_label`, `human_confidence` (1-5 scale), and explanatory `comments`.
