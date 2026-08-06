# HalluciSense Phase 10 — Pillar 2 System Architecture

*Generated: 2026-08-03T05:01:37.151249+00:00*

## 1. Overview

HalluciSense Pillar 2 is an Evidence-Aware Multi-LLM Hallucination Verification Engine.
It extends Pillar 1 statistical probability by extracting atomic claims, building knowledge graphs,
retrieving multi-provider evidence, orchestrating parallel LLM verifications, and computing
statistical consensus and contradiction metrics.

## 2. Pipeline Flow

```
User Prompt -> LLM Response
  ↓
Claim Extraction Engine (Module 10.1)
  ↓
Semantic Entity/Relation Graph (Module 10.2)
  ↓
Multi-Provider Evidence Retrieval (Module 10.3: Wikipedia, PubMed, CrossRef, etc.)
  ↓
Multi-LLM Parallel Verification (Module 10.4: Gemini, GPT-4, Claude)
  ↓
Consensus Engine (Module 10.5: Majority/Weighted Vote, Entropy, Variance)
  ↓
Contradiction Analyzer (Module 10.6: Contradiction Graph)
  ↓
10 Evidence Features (Module 10.7)
  ↓
Unified H-Score Fusion (Module 10.8: Fuses frozen Pillar 1 prob + Pillar 2)
  ↓
Explainability Engine (Module 10.9) -> Verification Report & Dashboard UI
```

## 3. Pillar 1 Immutable Dependency

Pillar 1 model (`pillar1_logistic_model.joblib`, `robust_scaler.joblib`) is treated as an
immutable dependency. Its output probability is passed into Module 10.8 without modification.
