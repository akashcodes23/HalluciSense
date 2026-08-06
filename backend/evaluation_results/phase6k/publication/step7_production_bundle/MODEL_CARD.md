---
model_name: hallucisense-pillar1-logistic
version: 1.0.0
language: en
license: cc-by-nc-4.0
tags:
  - hallucination-detection
  - NLI
  - logistic-regression
  - RAG
  - evaluation
---

# HalluciSense Pillar-1: Logistic Regression Hallucination Detector

## Model Summary

HalluciSense Pillar-1 is a 5-feature logistic regression classifier trained to detect
hallucinations in RAG (Retrieval-Augmented Generation) system outputs using NLI-based
claim-evidence alignment signals.

**Model Version**: 1.0.0  
**Algorithm**: LogisticRegression (liblinear, L2, C=1.0)  
**Validation Verdict**: PILLAR 1 VALIDATED WITH LIMITATIONS  
**Generated**: 2026-08-03T04:48:55.143897+00:00

## Performance (Held-Out VAL — 3,500 samples)

| Metric | Value |
| --- | --- |
| ROC-AUC | 0.6902 |
| PR-AUC (AP) | 0.6311 |
| F1 (τ=0.56) | 0.6618 |
| MCC (τ=0.56) | 0.3587 |
| Accuracy (τ=0.56) | 0.6803 |
| Brier Score | 0.2332 |
| ECE (10-bin) | see publication |

## Input Features

| Feature | Description | Range |
| --- | --- | --- |
| `mean_entailment` | Mean NLI entailment score across claims | [0, 1] |
| `max_entailment` | Maximum NLI entailment score | [0, 1] |
| `mean_contradiction` | Mean NLI contradiction score | [0, 1] |
| `min_support_margin` | Minimum support margin | [-1, 1] |
| `num_claims` | Number of atomic claims | ≥ 0 |

## Preprocessing

Input features must be passed through the paired **RobustScaler** before inference.
See `robust_scaler.joblib` in the artifact directory.

## Operating Threshold

Default threshold: **0.56** (optimized for balanced F1/MCC on DEV set).  
For higher recall, threshold can be lowered to 0.50 (see threshold analysis).

## Limitations & Known Issues

- ROC-AUC of 0.69 indicates moderate discrimination — above baseline but not publication-perfect.
- The model relies on NLI signals only; factual grounding beyond textual entailment is not captured.
- Performance may degrade on out-of-distribution domains not represented in HaluBench/HaluEval/RAGTruth.
- `num_claims` feature has relatively low importance (see permutation importance analysis).
- Logit values are bounded [-3, 3]; probability outputs are well-constrained.

## Intended Use

- Automated hallucination detection in RAG pipelines.
- Screening of LLM outputs for downstream verification.
- Research benchmark for claim-level NLI alignment methods.

## Out-of-Scope Uses

- High-stakes medical/legal decision-making without human review.
- Generative model replacement or factual grounding.

## Training Data

- **Development set**: 58,002 samples (HaluBench + HaluEval + RAGTruth)
- **Validation set**: 3,500 held-out samples (protocol-locked before training)
- **NLI model**: `cross-encoder/nli-deberta-v3-small`

## Artifact Hashes

| Artifact | SHA-256 |
| --- | --- |
| `pillar1_logistic_model.joblib` | `cf5199567b880c292d5c6b4f7dc5e63ee6e6be03b14e5965662da563152dbfb5` |
| `robust_scaler.joblib` | `89d54d65bc1b015d4fefcb514eb8bf37339e6d8b499652f67c375161b8e763d8` |

## Citation

If you use this model, please cite the HalluciSense paper (forthcoming, Elsevier).
