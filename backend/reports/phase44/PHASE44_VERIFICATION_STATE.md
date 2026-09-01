# Phase 44.2 — Verification State Semantics Report

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 44.2 — Typed Verification States & Modality Mapping  
**Date:** 2026-09-01  

---

## 1. Formal Verification State Contract

| State Identifier | Semantic Meaning | Pre-conditions |
|---|---|---|
| **VERIFIED** | Claim is supported by explicit evidence or computation | High NLI entailment ($\ge 0.80$) OR symbolic equality |
| **CONTRADICTED** | Claim is refuted by evidence or computation | High NLI contradiction ($\ge 0.80$) OR symbolic inequality |
| **INSUFFICIENT_EVIDENCE** | No matching or conclusive evidence retrieved | Neutral NLI score OR empty retrieval |
| **NOT_APPLICABLE** | Claim is a subjective or stylistic statement | Non-verifiable linguistic structure |
| **ERROR** | Subsystem timeout or parser exception | Gracefully caught execution failure |
