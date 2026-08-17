# HalluciSense Phase 7B — Pillar 2 Provider Capability Blocker Report

## 1. Executive Summary
Under the core research directive **`SCIENCE > VISUAL POLISH | HONEST PROVENANCE > FABRICATED COMPLETENESS`**, Pillar 2 (Epistemic / Token Confidence) was set to **`UNAVAILABLE`** in Phase 7 live testing rather than generating synthetic confidence scores.

This document discloses the technical reasons why live token-level probabilities could not be parsed from the active runtime environment.

---

## 2. Provider Capability Analysis

| Provider / Model | Protocol | Logprob Capability | Active Status | Root Blocker |
|---|---|---|---|---|
| **Ollama (`qwen2.5-coder:1.5b`)** | Local REST (`11434`) | Omitted by standard API | **ACTIVE** | Default `/api/chat` endpoint does not serialize per-token logit distributions. |
| **OpenAI (`gpt-4o-mini`)** | Cloud REST | Supported (`logprobs=True`) | **BLOCKED** | API quota / billing threshold exceeded (`HTTP 429 Insufficient Quota`). |
| **Google Gemini (`gemini-2.0-flash`)** | Cloud SDK | Not exposed in standard SDK | **BLOCKED** | `google.generativeai` standard SDK does not provide raw per-token log probabilities. |

---

## 3. Methodological Integrity Guard
* **Zero Synthetic Fillers**: HalluciSense strictly refuses to manufacture pseudo-random or heuristic confidence values when genuine token logprobs are omitted.
* **Availability-Aware Fusion**: When $P_2$ is unavailable, the fusion engine dynamically renormalizes over active signals ($w_{\text{eff}} = [0.6429, 0.0, 0.3571]$) rather than injecting a deceptive zero or proxy score.
