# HalluciSense — Final Viva Demonstration Script (10 Minutes)

**Repository:** akashcodes23/HalluciSense  
**Presentation Target:** Major Project Final Defense & Examination  
**System URL:** `https://hallucisense-production.up.railway.app` (or `http://localhost:3000`)  
**Date:** 2026-09-01  

---

## 1. Demonstration Sequence

### Case 1: Verified Factual Grounding (1.5 mins)
- **Input:** *"The James Webb Space Telescope was launched on December 25, 2021 aboard an Ariane 5 rocket."*
- **Expected Outcome:**
  - **Verdict:** `FACTUAL` ($P(H) = 0.08 < 0.54$).
  - **Evidence Trace:** Wikipedia passage retrieved with NLI Entailment $= 0.96$.
  - **Verification State:** `VERIFIED`.

### Case 2: Direct Factual Contradiction (1.5 mins)
- **Input:** *"The capital of France is Berlin."*
- **Expected Outcome:**
  - **Verdict:** `HALLUCINATED` ($P(H) = 0.82 > 0.54$).
  - **Evidence Trace:** Wikipedia *Paris* retrieved; DeBERTa Contradiction $= 0.98$.
  - **Verification State:** `CONTRADICTED`.
  - **Local Attribution:** Highlights `p1_mean_contradiction` ($+0.28$) as top driver.

### Case 3: Deterministic Arithmetic Verification (2.0 mins)
- **Input:** *"12 multiplied by 8 equals 95."*
- **Expected Outcome:**
  - **Modality:** Dispatched to **Safe Symbolic AST Verifier**.
  - **Audit Note:** *"Computed 12 * 8 = 96 (Claim stated 95)"*.
  - **Verification State:** `CONTRADICTED` (Shows arithmetic callout in UI).
  - **Verdict:** `HALLUCINATED` ($P(H) = 0.88 > 0.54$).

### Case 4: Insufficient Evidence Disambiguation (2.0 mins)
- **Input:** *"An obscure ancient subterranean civilization built fiber-optic networks in 3000 BC."*
- **Expected Outcome:**
  - **Verification State:** `INSUFFICIENT_EVIDENCE` (No matching encyclopedic facts).
  - **UX Note:** Clearly explains *lack of evidence* rather than false contradiction.

### Case 5: Multi-Claim Decomposed Response (2.0 mins)
- **Input:** *"Paris is the capital of France. Steve Jobs invented gravity in 1999."*
- **Expected Outcome:**
  - Decomposes into 2 distinct atomic claims.
  - Claim 1: `VERIFIED`.
  - Claim 2: `CONTRADICTED`.
  - Primary Status: `CONTAINS_CONTRADICTION`.

### Case 6: Production Health & Observability (1.0 min)
- Show `/health` and `/ready` telemetry.
- Point to **538 MB steady RSS memory** (484 MB free headroom under 1024 MB container limit).
