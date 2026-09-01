# Phase 42.1 — Forensic Baseline: Retrieval & Verification Gateway

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 42.1 — Forensic Baseline Audit & Problem Framing  
**Date:** 2026-09-01  

---

## 1. System State & Motivation

Phase 41 revealed that **61.5% of remaining verification errors (R1)** stemmed from retrieval scope limitations. Wikipedia retrieval fails on:
1. **Dynamic Arithmetic:** *"12 x 8 = 95"* (No Wikipedia article lists all multiplication combinations).
2. **Unit Conversions:** *"100 km/h is 500 m/s"* (Dimensional mismatches).
3. **Temporal Math:** *"2024 is 10 years after 2020"* (Calendar arithmetic).

---

## 2. Target Evidence Intelligence Topology

```
User Claim
    │
    ▼
Claim Type Classifier (Regex / AST / Token patterns)
    │
    ├── ARITHMETIC ────────► Safe AST Symbolic Verifier ─────┐
    ├── UNIT_CONVERSION ───► Unit Conversion Engine ─────────┤
    ├── TEMPORAL_MATH ─────► Temporal Logic Engine ──────────┼──► Structured Grounding
    └── TEXTUAL_FACT ──────► Hybrid Wikipedia + DeBERTa NLI ─┘
```

---

## 3. Guiding Scientific Boundary

- **Symbolic Verifiers** check internal mathematical and unit consistency without external network calls.
- **Textual Retrievers** provide encyclopedic evidence for empirical factual statements.
- **DeBERTa Cross-Encoder** performs sentence-level natural language inference.
- Production classifier (`HistGradientBoostingClassifier`, $\tau^*=0.54$) remains frozen and authoritative.
