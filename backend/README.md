# 🛡️ HalluciSense — Enterprise AI Hallucination Detection & Verification SaaS Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-green.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14.0-black.svg)](https://nextjs.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Production-Ready-success.svg)](#)

> **HalluciSense** is a confidence-aware, multi-stage AI hallucination detection platform engineered around **Google Gemini**, local **DeBERTa NLI cross-encoders**, and token logit uncertainty evaluation.

---

## 🌟 Executive Features

- 🎯 **Single Gemini Call Guarantee**: Zero-waste execution graph consuming **exactly 1 Gemini API call** per user prompt.
- ⚡ **Tri-Pillar Hallucination Detection**:
  - **Pillar 1 (Factual Grounding)**: Local DeBERTa cross-encoder entailment against retrieved evidence.
  - **Pillar 2 (Confidence Gap)**: Token logit softmax probability and entropy evaluation.
  - **Pillar 3 (Consistency Failure)**: Lazy semantic consistency drift analysis.
- 🛡️ **Global Quota Circuit Breaker**: Instantly trips on HTTP 429 quota exhaustion, skipping downstream operations.
- 📊 **Zero-NaN Frontend Metric Display**: Safe formatting guarantees clean UI score rendering (`PillarCard` & `CircularGauge`).
- 💎 **Modern Glassmorphism Interface**: Real-time WebSocket token streaming with sentence-level risk highlighting and evidence links.

---

## 🏗️ Architecture Overview

```
User Prompt ──► WebSocket / HTTP API ──► LLMOrchestrator ──► GeminiProvider (1 LLM Call)
                                                                 │
                                                                 ▼
Background Verification ◄── PostgreSQL / Upstash Redis ◄── Tri-Pillar Pipeline
```

---

## 🚀 Quickstart & Docker Deployment

```bash
# Clone the repository
git clone https://github.com/akashcodes23/HalluciSense.git
cd HalluciSense

# Launch full-stack using Docker Compose
docker compose up -d --build

# Access services:
# Frontend API: http://localhost:3000
# Backend Docs: http://localhost:8000/docs
# Health Probe: http://localhost:8000/health
```

---

## 📄 License & Citation

Licensed under the **MIT License**.
