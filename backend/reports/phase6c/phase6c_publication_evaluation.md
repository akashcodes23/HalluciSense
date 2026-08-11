# Phase 6C: Publication-Grade Evaluation Report

**Generated**: 2026-08-11
**Git SHA**: cbe4de7f72b7c874727e1025acf348219136ed60
**Evaluation seed**: 42
**Threshold**: 0.50
**Bootstrap samples**: 5000
**Canonical evaluator**: evaluation/canonical_evaluator.py (34/34 tests pass)

> [!IMPORTANT]
> P2 (predictive uncertainty) and P3 (self-consistency) require live Gemini API.
> In this evaluation environment, P2 and P3 are unavailable. All results reflect
> the P1 + temporal reasoning pipeline only. The "Full HalluciSense" label refers
> to the full temporal/epistemic reasoning within Pillar 1.

---

## Section 1: Evaluation Datasets

| Dataset | Source | N | Positive | Negative | Balance | Role |
|:---|:---|---:|---:|---:|:---|:---|
| HaluBench | PatronusAI/HaluBench | 100 | 100 | 0 | 100% pos | FINAL TEST |
| RAGTruth | ParticleMedia/RAGTruth | 300 | 98 | 202 | 32.7%/67.3% | FINAL TEST |
| HaluEval | RUCAIBox/HaluEval | 150 | 75 | 75 | 50%/50% | FINAL TEST |
| **Combined** | — | **550** | **273** | **277** | **49.6%/50.4%** | FINAL TEST |

**Note on HaluBench**: Domain label is "drop" in normalized records. Contains only
hallucinated examples (N=100, 100% positive). Specificity, FPR, and MCC are
**undefined** for HaluBench in isolation — they cannot be computed without negative examples.
Do NOT report these metrics for HaluBench alone.

---

## Section 2: Per-Dataset Results

### HaluBench (N=100, 100% hallucinated)

> [!WARNING]
> Specificity, FPR, and MCC are undefined here (no negative examples). Accuracy = Recall.

| Method | Acc | Precision | Recall | F1 | AUROC |
|:---|:---:|:---:|:---:|:---:|:---:|
| P1 NLI Baseline | 0.7200 | 1.0000 | 0.7200 | 0.8372 | — |
| M9 Full System | 0.7200 | 1.0000 | 0.7200 | 0.8372 | — |

**Observation**: No difference between P1 baseline and M9 on HaluBench.
72% of hallucinated examples are detected; 28% are missed.

---

### HaluEval (N=150, 50%/50% balanced)

| Method | TP | TN | FP | FN | Acc | F1 | MCC | Spec | FPR |
|:---|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|
| P1 NLI Baseline | 43 | 40 | 35 | 32 | 0.5533 | 0.5621 | 0.1068 | 0.5333 | 0.4667 |
| M9 Full System | 41 | 40 | 35 | 34 | 0.5400 | 0.5430 | 0.0800 | 0.5333 | 0.4667 |

**Observation**: M9 is marginally LOWER than P1 baseline on HaluEval.
MCC (0.0800 vs 0.1068) confirms marginal regression. F1 95% CI overlaps heavily.

---

### RAGTruth (N=300, 32.7%/67.3% imbalanced)

| Method | TP | TN | FP | FN | Acc | F1 | MCC | Spec | FPR |
|:---|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|
| P1 NLI Baseline | 6 | 189 | 13 | 92 | 0.6500 | 0.1000 | 0.0137 | 0.9356 | 0.0644 |
| M9 Full System | 10 | 183 | 19 | 88 | 0.6433 | 0.1575 | 0.0127 | 0.9059 | 0.0941 |

**Observation**: M9 detects 4 more TPs but introduces 6 more FPs on RAGTruth.
Net MCC effect is negligible (0.0127 vs 0.0137). F1 improvement (0.1575 vs 0.1000)
reflects increased recall at cost of specificity. Both are very low — neither system
performs well on RAGTruth.

**Important**: RAGTruth Accuracy (~65%) is dominated by the majority negative class.
Balanced accuracy is more appropriate: P1 = 0.529, M9 = 0.524 (effectively random on this dataset).

---

## Section 3: Combined Results (N=550, Primary Comparison)

| Method | TP | TN | FP | FN | Acc | Precision | Recall | F1 | MCC | BAcc | Spec | FPR | FNR | AUROC | AUPRC |
|:---|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| P1 NLI Baseline | 121 | 231 | 46 | 152 | 0.6400 | 0.7246 | 0.4432 | 0.5500 | 0.3014 | 0.6386 | 0.8339 | 0.1661 | 0.5568 | 0.6834 | 0.6654 |
| M9 Full System | 123 | 223 | 54 | 150 | 0.6291 | 0.6949 | 0.4505 | 0.5467 | 0.2736 | 0.6278 | 0.8051 | 0.1949 | 0.5495 | 0.6807 | 0.6526 |

### Bootstrap 95% Confidence Intervals (F1, 5000 samples, seed=42)

| Method | F1 | Lower | Upper |
|:---|:---:|:---:|:---:|
| P1 NLI Baseline | 0.5500 | 0.4942 | 0.6041 |
| M9 Full System | 0.5467 | 0.4894 | 0.6009 |

**Both confidence intervals substantially overlap. The difference is NOT statistically significant.**

---

## Section 4: Statistical Significance Tests

### McNemar's Test (P1 Baseline vs M9 Full System, N=550)

| Metric | Value |
|:---|:---:|
| Both correct | 338 |
| Both wrong | 190 |
| b (P1 right, M9 wrong) | **14** |
| c (P1 wrong, M9 right) | **8** |
| McNemar χ² (with continuity correction) | 1.1364 |
| p-value | **0.2864** |
| Significant at α=0.05 | **NO** |

> [!CAUTION]
> The difference between P1 NLI Baseline and M9 Full HalluciSense is **NOT
> statistically significant** (p=0.2864 > 0.05). This means we cannot claim
> that HalluciSense performs statistically differently from the NLI baseline
> on this combined external benchmark.
>
> M9 makes 14 predictions that P1 got right but M9 got wrong, and 8 predictions
> that P1 got wrong but M9 got right. Net discordant: 14 vs 8. This is not sufficient
> evidence of a systematic difference.

---

## Section 5: What This Means

### Honest Interpretation

HalluciSense's full temporal and epistemic reasoning pipeline (M9) does NOT
significantly outperform a pure NLI baseline (P1) on the combined
HaluBench + RAGTruth + HaluEval benchmark (N=550).

This is expected, given that:
- 59.3% of examples have no temporal signal (temporal components inactive)
- 0% of examples contain future-year assertions (primary novel use case)
- 83.3% of examples are ASSERTED_FACTs (NLI baseline is directly applicable)

The benchmark is measuring primarily the NLI factual grounding component, which
is shared between P1 baseline and M9. The temporal/epistemic components that
differentiate the systems are barely activated on these datasets.

### What Would Have Been Different With Temporal Datasets

If the evaluation benchmark contained:
- Future-year assertions (2025–2030)
- Explicit hypotheticals and conditionals
- Evidence snippets with background historical dates
- Counterfactual claims

...then the temporal-epistemic gate mechanism would be activated and FPR differences
would be measurable. The current benchmark cannot produce this signal.

---

## Section 6: Complete Phase 6B Ablation Results (Audited)

These are the Phase 6B results, verified against canonical evaluator. See
`phase6c_ablation_design_audit.md` for design validity notes.

| Config | TP | TN | FP | FN | Acc | F1 | MCC | BAcc | AUROC | Note |
|:---|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|:---|
| A0 NLI Baseline | 121 | 231 | 46 | 152 | 0.640 | 0.550 | 0.301 | 0.639 | 0.683 | Valid baseline |
| A1 NLI+Retrieval | 121 | 231 | 46 | 152 | 0.640 | 0.550 | 0.301 | 0.639 | 0.683 | = A0 (invalid step) |
| A2 +Temporal | 121 | 229 | 48 | 152 | 0.636 | 0.548 | 0.293 | 0.635 | 0.680 | Valid (marginal regression) |
| A3 +Modality | 0 | 275 | 2 | 273 | 0.500 | 0.000 | −0.060 | 0.496 | 0.628 | ❌ Degenerate |
| A4 +Atomic | 164 | 159 | 118 | 109 | 0.587 | 0.591 | 0.175 | 0.587 | 0.630 | Valid |
| A5 +Alignment | 4 | 277 | 0 | 269 | 0.511 | 0.029 | 0.086 | 0.507 | 0.648 | ❌ Degenerate |
| A6 +Relational | 117 | 239 | 38 | 156 | 0.647 | 0.547 | 0.324 | 0.646 | 0.705 | ✅ Best MCC |
| A7 +Meta | 121 | 234 | 43 | 152 | 0.645 | 0.554 | 0.315 | 0.644 | 0.686 | Valid |
| A8 +Anchoring | 123 | 231 | 46 | 150 | 0.644 | 0.557 | 0.308 | 0.642 | 0.687 | ⚠️ Slight improvement |
| A9 Full | 123 | 223 | 54 | 150 | 0.629 | 0.547 | 0.274 | 0.628 | 0.681 | ⚠️ P1-only (P2/P3 absent) |

**Key finding**: A6 (relational temporal parsing) achieves the best MCC (0.324) on
the external benchmark. A9 (full system) performs below A6 and below A0 on F1.

---

## Section 7: Primary Evidence Table for Publication

This is the authoritative table for the paper. Only verified, valid results included.

| Method | Dataset | N | Acc | F1 | MCC | BAcc | AUROC | Notes |
|:---|:---|---:|:---:|:---:|:---:|:---:|:---:|:---|
| NLI Baseline | HaluBench | 100 | 0.720 | 0.837 | N/A | N/A | — | No negatives |
| Full System | HaluBench | 100 | 0.720 | 0.837 | N/A | N/A | — | No negatives |
| NLI Baseline | HaluEval | 150 | 0.553 | 0.562 | 0.107 | 0.553 | — | Balanced |
| Full System | HaluEval | 150 | 0.540 | 0.543 | 0.080 | 0.540 | — | Balanced |
| NLI Baseline | RAGTruth | 300 | 0.657 | 0.104 | 0.014 | 0.529 | — | Imbalanced |
| Full System | RAGTruth | 300 | 0.643 | 0.158 | 0.013 | 0.524 | — | Imbalanced |
| NLI Baseline | Combined | 550 | 0.640 | 0.550 | 0.301 | 0.639 | 0.683 | F1 CI: [0.494, 0.604] |
| Full System | Combined | 550 | 0.629 | 0.547 | 0.274 | 0.628 | 0.681 | F1 CI: [0.489, 0.601] |

**Statistical test**: McNemar χ²=1.14, p=0.286 — **NOT SIGNIFICANT**

---

## Section 8: Negative Result

The full HalluciSense system does not outperform an NLI baseline on any of the
three standard external hallucination detection benchmarks. The difference is
not statistically significant (McNemar p=0.286).

**This is a scientifically useful finding, not a failure.**

It demonstrates that:
1. The NLI baseline is strong on factual QA datasets
2. Temporal and epistemic enhancements do not harm performance
3. General-purpose benchmarks are insufficient for evaluating temporal/epistemic components
4. A benchmark specifically designed for temporal/epistemic hallucination is needed

This negative result should be reported honestly and prominently in any publication.
