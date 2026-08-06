# HalluciSense Phase 6B.2 — Partition & Leakage Control Report

## Executive Summary

Phase 6B.2 partition generation completed successfully using deterministic group-aware partitioning.
- **Fixed Partition Seed**: `2026`
- **Total Canonical Examples**: `82690`
- **Partition Ratios**:
  - **DEVELOPMENT**: `58002` samples
  - **VALIDATION**: `12483` samples
  - **LOCKED_FINAL_TEST**: `12205` samples

---

## Per-Dataset Partition Breakdown

### Dataset: HaluBench (Total: 14900)

- **Processed File**: `processed/halubench/benchmark.jsonl`
- **Manifest SHA-256**: `510ad7baea018f4a8df2b786b5d6647b16a2da49ede5c2f476337762248fb01b`
- **Partitions**:
  - `DEVELOPMENT`: `10451` samples (Factual: `5437`, Hallucinated: `5014`)
  - `VALIDATION`: `2233` samples (Factual: `1163`, Hallucinated: `1070`)
  - `LOCKED_FINAL_TEST`: `2216` samples (Factual: `1130`, Hallucinated: `1086`)
- **Leakage Status**:
  - Shared Example IDs: `0`
  - Shared Group IDs: `0`

### Dataset: RAGTruth (Total: 17790)

- **Processed File**: `processed/ragtruth/benchmark.jsonl`
- **Manifest SHA-256**: `a987d6266da13d7cfbd98a50fb8a2aa456d2964b701af4b852265ecb66867f68`
- **Partitions**:
  - `DEVELOPMENT`: `12558` samples (Factual: `7077`, Hallucinated: `5481`)
  - `VALIDATION`: `2688` samples (Factual: `1546`, Hallucinated: `1142`)
  - `LOCKED_FINAL_TEST`: `2544` samples (Factual: `1503`, Hallucinated: `1041`)
- **Leakage Status**:
  - Shared Example IDs: `0`
  - Shared Group IDs: `0`

### Dataset: HaluEval (Total: 50000)

- **Processed File**: `processed/halueval/benchmark.jsonl`
- **Manifest SHA-256**: `46bbdc99254df731b45b42fcceb4661d5e8690600458ca769857495b9342036f`
- **Partitions**:
  - `DEVELOPMENT`: `34993` samples (Factual: `13986`, Hallucinated: `21007`)
  - `VALIDATION`: `7562` samples (Factual: `3028`, Hallucinated: `4534`)
  - `LOCKED_FINAL_TEST`: `7445` samples (Factual: `2986`, Hallucinated: `4459`)
- **Leakage Status**:
  - Shared Example IDs: `0`
  - Shared Group IDs: `0`

