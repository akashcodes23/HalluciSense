# Professional Resume Bullet Points — HalluciSense Project

- **Architected HalluciSense**, an open-source, hybrid multi-pillar hallucination detection framework for LLMs, achieving **0.9501 AUROC** ($95\text{\% CI: } [0.9320, 0.9650]$) and **0.0257 ECE** across $N=750$ claims in 15 research domains.
- **Engineered a 19-Dimensional Hybrid Meta-Classifier** combining external web evidence grounding (Pillar 1) and intra-model self-consistency contradiction matrices (Pillar 2) using a robustly scaled `HistGradientBoostingClassifier`.
- **Built Production Backend & SRE Architecture** in FastAPI and Docker ($218\text{ MB}$ image size), deployed on Railway PaaS with OpenTelemetry tracing, Prometheus `/metrics`, and sub-150ms P90 inference latency.
- **Designed Interactive Explainability UX** using Next.js 14, Tailwind CSS, SHAP feature attributions, and topological claim-evidence graphs for real-time risk breakdown.
