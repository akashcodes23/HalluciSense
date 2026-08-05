# HalluciSense v1.0.0 Open Source Release

**Release Tag**: `v1.0.0`  
**Release Date**: August 5, 2026  
**License**: MIT License  

---

## What's Included in v1.0.0
- **Hybrid Meta-Classifier**: Phase 6M candidate model delivering **0.9501 AUROC**, **0.8738 F1**, and **0.0257 ECE** via Platt scaling.
- **12 Public Benchmark Adapters**: HaluEval, TruthfulQA, FEVER, SciFact, PubHealth, FreshQA, FActScore across 15 domains ($N=750$ claims).
- **FastAPI Core App**: Async endpoints with CORS, rate limiting, structured JSON logging, and OpenTelemetry tracing.
- **Next.js 14 Web Dashboard**: Responsive dashboard with dark mode, interactive SHAP attributions, claim graphs, and benchmark leaderboards.
- **DevEx & One-Command Launch**: Run `python run_all_experiments.py` to reproduce all figures, tables, and metrics.
