# HalluciSense Demonstration Walkthrough

This document outlines the recommended demonstration sequence for showcasing HalluciSense during final-year project evaluations, recruitment reviews, and technical presentations.

---

## 1. Prerequisites for Demonstration

- **Live Deployed URL**: `https://hallucisense-production.up.railway.app` (or local development instance on `http://localhost:3000`).
- **Target Audience**: Technical evaluators, faculty members, machine learning engineers, and recruiters.
- **Estimated Duration**: 5 to 7 minutes.

---

## 2. Demonstration Flow & Core Scenarios

### Scenario A: Overview Dashboard & System Observability
1. **Navigate to `/overview`**:
   - Point out the **Live Production Telemetry** banner, displaying real-time metrics (Total Verifications, Hallucinations Caught, Verified Safe, Success Rate, Average H-Score, and End-to-End Latency).
   - Show the **Pipeline Component Status** panel confirming that shared singleton engines (`ModelRegistry`, `Retriever`, `NLI Cross-Encoder`, `Fusion Engine`) are active.
   - Note the **Verification Outcomes** bar chart breaking down live query risk distributions.

### Scenario B: Verifying a Factual Statement (P1 Grounding)
1. **Navigate to `/verify`**:
2. Enter the query:
   - **Question**: `What is the capital of Karnataka?`
   - **Response**: `The capital of Karnataka is Bengaluru.`
3. Click **Verify** (or press `⌘+Enter`):
4. **Observe Output**:
   - **Verdict Banner**: Displays `VERIFIED` with green badge and a low Hallucination Score ($H \approx 13.3\%$).
   - **Pillar Signals**:
     - *Pillar 1 (Evidence Grounding)*: $13.3\%$ (Active).
     - *Pillar 2 (Confidence)*: `Unavailable — Token log-probabilities not provided`.
     - *Pillar 3 (Consistency)*: `Unavailable — Multiple generations not available`.
   - **Fusion Decomposition**: Shows adaptive renormalization mode with effective weight $\alpha = 1.0$, confirming the system did not treat missing signals as zero risk.
   - **Claim Analysis**: Displays the decomposed claim matched against retrieved Wikipedia passages.

### Scenario C: Catching an Entity Hallucination (Entity Linking Failure)
1. In `/verify`, enter the corrupted statement:
   - **Question**: `What is the capital of Karnataka?`
   - **Response**: `The capital of Karnataka is Mumbai.`
2. Click **Verify**:
3. **Observe Output**:
   - **Verdict Banner**: Displays `LIKELY_HALLUCINATED` with red warning badge and $H \approx 99.1\%$.
   - **Root Cause Taxonomy**: Flags `Entity Linking Failure`.
   - **Token Heatmap**: Highlights `Mumbai` in red as the suspect hallucination token.
   - **Attributed Evidence**: Shows retrieved passages explicitly stating Bengaluru is the capital, triggering an NLI contradiction.

### Scenario D: Distributed Pipeline Traces
1. **Navigate to `/traces`**:
   - Open the latest trace generated from Scenario C.
   - Inspect the **Pipeline Execution Waterfall**:
     - `Evidence Retrieval` (BM25 sparse search + FAISS dense indexing).
     - `NLI Entailment Scoring` (DeBERTa-v3 cross-encoder evaluation).
     - `Adaptive Fusion & Calibration` (< 1ms).
   - Highlight the **Measured Timings** breakdown verifying execution performance.

### Scenario E: Closed-Loop AI Chat with Evidence-Grounded Repair
1. **Navigate to `/chat`**:
2. Submit a complex domain query:
   - `What causes Type 1 diabetes mellitus?`
3. **Observe Pipeline in Action**:
   - Real-time multi-stage progress indicator: Draft generation $\to$ Evidence Retrieval $\to$ Claim Decomposition $\to$ NLI Check $\to$ Re-Verification.
   - **Result**: Displays the verified explanation accompanied by 5 attributed peer-reviewed Wikipedia references.
   - Expand **Verification Details** to show the low risk score ($H \approx 1.03\%$) and trace metadata.

### Scenario F: Scientific Lab & Research Transparency
1. **Navigate to `/scientific`**:
   - Show that benchmark evaluation metrics ($AUROC = 0.9964$, $AUPRC = 0.9958$, $ECE = 0.0986$) are segregated from live runtime metrics.
   - Explain the ablation charts proving the $+0.1490$ AUROC advantage of availability-aware adaptive fusion over naive fixed fusion.

---

## 3. Key Takeaways to Emphasize

1. **Zero-Signal Safety**: HalluciSense never assumes missing signals imply safety.
2. **Explainability**: Every verdict includes atomic claim breakdowns, token heatmaps, and cited evidence snippets.
3. **Statistical Rigor**: Scores are calibrated through empirical Platt scaling and backed by formal selective abstention bounds.
