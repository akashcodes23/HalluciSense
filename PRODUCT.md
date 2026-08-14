# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary users are academic scholars, AI researchers, and peer reviewers evaluating empirical hallucination detection benchmarks, multi-pillar calibration metrics, and explainable uncertainty methodologies for scientific publication (e.g. Elsevier *Information Fusion*, *Knowledge-Based Systems*, *Artificial Intelligence*). Secondary users include ML engineers inspecting model reliability and diagnostic verification traces.

## Product Purpose

HalluciSense is a confidence-aware hybrid AI verification framework designed to detect and quantify hallucinations in Large Language Model responses with rigorous scientific calibration. It replaces single-metric heuristics with a multi-signal verification architecture that combines external retrieval grounding, white-box uncertainty/entropy estimation, and semantic consistency reasoning into an explainable, Platt-calibrated confidence score. Success means providing reproducible, mathematically grounded, and token-localized hallucination assessments with zero unsubstantiated claims.

## Positioning

Unlike conventional hallucination detectors that rely exclusively on LLM-as-a-judge prompting or simple semantic vector similarity, HalluciSense fuses three orthogonal mathematical signals:
1. **Factual Evidence Retrieval ($FE$):** BM25 sparse matching + dense embeddings + cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) with NLI entailment.
2. **Confidence Estimation ($CG$):** White-box token logprobs, predictive entropy $H(Y)$, mutual information $I(Y;W)$, attention entropy, and epistemic/aleatoric uncertainty decomposition.
3. **Consistency Reasoning ($CF$):** Multi-prompt paraphrase sampling, pairwise SBERT similarity matrices, and claim-aligned NLI contradiction graphs.

Combined via calibrated hybrid fusion ($H = \alpha FE + \beta CG + \gamma CF$) with Platt scaling (achieving empirical ECE $\le 0.0257$).

## Operating Context

- Interactive verification workspace (`/verify` and `/analyze`) for inspecting prompts, responses, and generated token-level risk spans.
- Benchmark and evaluation suites (`/benchmark` and `/metrics`) for cross-model comparative runs against HaluEval, TruthfulQA, and FEVER datasets.
- Real-time diagnostic telemetry and WebSocket streaming for live multi-step pipeline inspection.
- Publication-ready chart exports, LaTeX-compatible metric tables, and statistical significance reports ($p$-values, Wilcoxon signed-rank tests, Cohen's $d$).

## Capabilities and Constraints

- **Stack:** Next.js 14 App Router, React 18, TypeScript, Tailwind CSS, Framer Motion frontend; FastAPI, Python 3.11, PyTorch, HuggingFace Transformers, PostgreSQL, Redis backend.
- **4-Tier Risk Taxonomy:**
  - **Verified** (`#10B981` / Emerald green)
  - **Needs Verification** (`#F59E0B` / Amber yellow)
  - **Moderate Risk** (`#F97316` / Orange)
  - **Likely Hallucinated** (`#EF4444` / Rose red)
- **Mathematical Integrity:** All displayed metrics, confusion matrices, ROC-AUC, PR-AUC, and calibration curves must reflect real computational outputs from the backend pipeline.

## Brand Commitments

- **Name:** HalluciSense — Scientific Hallucination Detection
- **Tone & Voice:** Rigorous, empirical, authoritative, clear, and academic.
- **Visual Foundation:** Deep space dark aesthetic (`#050816` canvas, `#0B0F19` surface cards), subtle glassmorphism (`backdrop-blur`), vibrant scientific accents (Purple/Indigo primary `#A855F7` / `#6366F1`, Blue `#3B82F6`), monospace typography for quantitative readouts (`font-mono`).

## Evidence on Hand

- Research benchmarks and empirical test suites in `backend/tests/` and `backend/benchmarks/`.
- Pre-trained calibration models, hybrid fusion weights, and statistical validation scripts in `backend/scripts/`.
- Real API contracts and WebSocket telemetry endpoints in `backend/app/api/`.

## Product Principles

1. **Empirical Over Assertive:** Present calibrated probabilities, confidence bounds, and retrieved passages rather than binary black-box verdicts.
2. **Transparent Decomposition:** Always expose individual pillar attributions ($FE$, $CG$, $CF$) alongside the fused score so researchers can diagnose failure modes.
3. **Scannable Precision:** Surface high-level risk summaries immediately, with deep token-level heatmap inspection and raw tensor logs available on demand.
4. **Reproducible Integrity:** Visualizations and telemetry must remain strictly synchronized with underlying backend statistical computations.
