# Phase 6C: Novelty Falsification Report

**Purpose**: Adversarial assessment of HalluciSense's research contribution.
**Standard**: Every component is compared against known prior art. A combination
is NOT claimed as novel unless it can survive the challenge:
"This is merely an engineering combination of NLI + temporal regex."

---

## Falsification Question

Could a reviewer reasonably write:

> "HalluciSense is an engineering combination of existing NLI-based hallucination
> detection, temporal consistency checking, and epistemic modality classification —
> none of which are individually novel — assembled without a clear principled framework."

**Answer**: This challenge is PARTIALLY VALID. HalluciSense does combine known techniques.
The question is whether the specific combination, the mechanism of interaction between components,
or the resulting behavioral properties constitute a defensible contribution.

---

## Component-by-Component Classification

| Component | Known In Literature | HalluciSense Version | Classification |
|:---|:---:|:---|:---:|
| NLI-based hallucination detection | Yes | Used directly (bart-large-mnli) | KNOWN |
| Wikipedia retrieval for evidence | Yes | Standard retrieval | KNOWN |
| Hallucination scoring from NLI | Yes | Direct application | KNOWN |
| Temporal year extraction | Yes | Regex-based | KNOWN |
| Future-year detection heuristic | Yes (as simple heuristic) | Integrated into modality engine | KNOWN ADAPTATION |
| Epistemic modality classification | Yes (linguistic literature) | Applied to hallucination detection | KNOWN ADAPTATION |
| Claim decomposition / atomic splitting | Yes (FactScore, 2023) | Applied to response verification | KNOWN ADAPTATION |
| Multi-passage evidence alignment | Yes (RAG literature) | Date-alignment specific | KNOWN ADAPTATION |
| Meta-claim / quotation detection | Yes (NLP coreference/attribution) | Pattern-based | KNOWN ADAPTATION |
| Counterfactual detection | Yes (conditional generation lit.) | Pattern-based | KNOWN ADAPTATION |
| Weighted score fusion | Yes (FEVER, hallucination fusion) | Fixed alpha/beta/gamma | KNOWN |
| Deterministic pipeline | N/A | Architecture choice | ARCHITECTURAL |
| **Independent query-response modality** | Not found | Resolves modality separately per-side | POTENTIALLY NOVEL |
| **Global evidence-date alignment** | Not found in exact form | Prevents date collisions across full context | POTENTIALLY NOVEL |
| **Temporal-epistemic interaction gate** | Not found | Blocks temporal penalty for non-assertion modalities | POTENTIALLY NOVEL |

---

## Prior Art Analysis

### NLI-based hallucination detection
**Representative work**: Honovich et al. (2022), "True: Re-evaluating Factual Consistency"; 
Laban et al. (2022), "SummaC"; Maynez et al. (2020) (Faithfulness in Abstractive Summarization).
**Difference**: HalluciSense applies NLI for Q&A hallucination verification, not summarization
faithfulness. The setup is different but the core NLI application is well-known.

### Temporal hallucination detection
**Representative work**: Dhingra et al. (2022) "Time-Sensitive QA"; Kasner & Dusek (2020)
"Text-to-Text Generation for Factual Temporal Reasoning"; Zhao et al. (2021).
**Difference**: Prior work focuses on detecting temporal inconsistencies in time-sensitive QA.
HalluciSense extends this by conditioning temporal checking on modality.

### FactScore-style claim decomposition
**Representative work**: Min et al. (2023) "FActScoring: Fine-grained Atomic Evaluation of
Factual Precision in Long-Form Text Generation".
**Difference**: FactScore uses GPT-based claim decomposition + Wikipedia retrieval. HalluciSense
uses a simpler rule-based claim extractor and applies NLI per claim. The architecture is simpler.

### Epistemic modality in NLP
**Representative work**: Baker et al. (2012) "Modality Annotation"; Saurí & Pustejovsky (2012)
"Are You Sure That This Happened? Assessing the Factuality Degree of Events in Text".
**Difference**: These works focus on event factuality annotation. HalluciSense applies modality
classification as an operational gate in a hallucination detection pipeline — a different use.

### Multi-passage evidence alignment
**Representative work**: FEVER (Thorne et al., 2018); Guo et al. (2022) "Survey on Automated
Fact-Checking".
**Difference**: FEVER uses sentence-level claims; HalluciSense applies date-specific alignment
across the full evidence set to prevent temporal contamination. The date-alignment mechanism
is a more specific operationalization.

---

## The Potentially Novel Mechanism

After falsification analysis, the strongest candidate for novelty is the following
specific mechanism, which does not appear in the above prior art:

**"Temporal-Epistemic Gate"**: When verifying a claim that contains temporal expressions,
the system first resolves the epistemic modality of BOTH the query AND the response
independently. If either resolves to a non-assertion modality (prediction, hypothetical,
counterfactual, conditional, fiction, quotation), the temporal inconsistency signal
is suppressed and only factual grounding is applied.

This is different from:
- Simply detecting future dates (done in Phase 5, and in prior heuristic systems)
- Modality detection as annotation (prior NLP work)
- Evidence retrieval (retrieval systems don't model modality)

The specific mechanism of using modality as a conditional gate on temporal verification,
applied independently to query and response, does not appear in prior hallucination
detection literature as of the knowledge cutoff of this report.

**Strength of this claim**: MODERATE. It requires:
1. A literature search confirming no prior work implements this exact gate
2. Experimental evidence that this gate measurably reduces FPR on non-assertion claims
3. Acknowledgment that pattern-based modality detection is an approximation

---

## Adversarial Counterarguments

**Counterargument 1**: "The modality gate just suppresses score=0.0, which any threshold
would achieve with a lower threshold."  
**Response**: Not equivalent. A lower threshold would suppress ALL scores, not
selectively suppress temporal penalty while retaining factual grounding.

**Counterargument 2**: "The external benchmark shows no improvement over NLI baseline."  
**Response**: VALID (see Phase 6C evaluation results). The benchmark is not designed
to test this capability (0% future-year examples). The improvement is demonstrated
only on the targeted temporal adversarial benchmark.

**Counterargument 3**: "Pattern-based modality detection is too simple to be a
principled contribution."  
**Response**: PARTIALLY VALID. The patterns are interpretable and auditable, which
is a design choice with real value. However, the depth of semantic understanding is
limited. This limitation must be disclosed.

**Counterargument 4**: "The system is not evaluated on a true blind holdout."  
**Response**: The 550-case external benchmark was acquired independently and not
used during development. The Phase 5 holdout (70 cases) was used during development
and is correctly excluded from final test reporting.

---

## Verdict

The NLI, retrieval, temporal, and modality components are individually known.
The specific **temporal-epistemic gate mechanism** — suppressing temporal verification
penalty based on independently resolved epistemic modality — represents a potentially
novel operational combination that is:

1. Not directly equivalent to any single prior work found
2. Mechanistically motivated and interpretable
3. Partially supported by targeted adversarial benchmarks
4. NOT supported by the general-purpose external benchmark

**Novel combination claim confidence**: MODERATE  
**Statistical support on external benchmark**: WEAK (capability untestable)  
**Statistical support on targeted benchmark**: MODERATE (limited N)

**The claim is defensible but must be scoped narrowly.**
