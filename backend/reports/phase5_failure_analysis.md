# Phase 5 False Positive & False Negative Root-Cause Analysis Report

## 1. Executive Summary
Out of 70 blind holdout test cases, the full system achieved **52 correct predictions** and exhibited **18 failure cases**.

### Summary of Failure Instances:
- **Total Failures**: 18 / 70 (25.71%)
- **False Positives (FP)**: 14
- **False Negatives (FN)**: 4

---

## 2. Detailed Case-by-Case Failure Breakdown

| Case ID | Category | Query | Response | Expected | Predicted | Root Cause Category | Remediation Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **H01** | HISTORICAL_FACT | *When did the Berlin Wall fall?* | *The Berlin Wall fell in November 1989, signaling the collapse of the Eastern Bloc.* | 0 (False Positive (FP)) | 1 | RETRIEVAL_NLI_GROUNDING_FAILURE | Expand Wikipedia evidence index corpus |
| **H02** | HISTORICAL_FACT | *When was Pluto discovered?* | *Clyde Tombaugh discovered Pluto at the Lowell Observatory in 1930.* | 0 (False Positive (FP)) | 1 | RETRIEVAL_NLI_GROUNDING_FAILURE | Expand Wikipedia evidence index corpus |
| **H03** | HISTORICAL_FACT | *When was the polio vaccine developed?* | *Jonas Salk developed the first successful inactivated polio vaccine in 1953.* | 0 (False Positive (FP)) | 1 | RETRIEVAL_NLI_GROUNDING_FAILURE | Expand Wikipedia evidence index corpus |
| **H34** | NEGATED_CLAIM | *Did commercial passenger flights land on the Moon in 2024?* | *No commercial airliner landed passengers on the lunar surface in 2024.* | 0 (False Positive (FP)) | 1 | RETRIEVAL_NLI_GROUNDING_FAILURE | Expand Wikipedia evidence index corpus |
| **H36** | NEGATED_CLAIM | *Did the United States declare independence in 1776?* | *The United States did not declare independence in 1776.* | 1 (False Negative (FN)) | 0 | TEMPORAL_YEAR_EXTRACTION_LIMITATION | Document as genuine system boundary |
| **H37** | QUOTED_CLAIM | *What did the conspiracy article claim?* | *The article falsely reported that the moon landing occurred in 2015, which is untrue.* | 0 (False Positive (FP)) | 1 | RETRIEVAL_NLI_GROUNDING_FAILURE | Expand Wikipedia evidence index corpus |
| **H38** | QUOTED_CLAIM | *What is the debunked medical claim?* | *The assertion that vaccines cause autism was thoroughly debunked by researchers.* | 0 (False Positive (FP)) | 1 | RETRIEVAL_NLI_GROUNDING_FAILURE | Expand Wikipedia evidence index corpus |
| **H39** | QUOTED_CLAIM | *What was the fraudulent press release?* | *The press release erroneously claimed that the company filed for bankruptcy in 2029.* | 0 (False Positive (FP)) | 1 | RETRIEVAL_NLI_GROUNDING_FAILURE | Expand Wikipedia evidence index corpus |
| **H40** | QUOTED_CLAIM | *What was the viral social media post?* | *Social media posts claimed that Mars would appear as large as the Moon in 2027, which is physically impossible.* | 0 (False Positive (FP)) | 1 | RETRIEVAL_NLI_GROUNDING_FAILURE | Expand Wikipedia evidence index corpus |
| **H44** | FICTIONAL | *What is the timeline of Star Trek?* | *In the Star Trek universe, Zefram Cochrane develops warp drive in the year 2063.* | 0 (False Positive (FP)) | 1 | RETRIEVAL_NLI_GROUNDING_FAILURE | Expand Wikipedia evidence index corpus |
| **H48** | TIME_RELATIVE | *Did inflation rise after 2021?* | *Global central banks raised benchmark interest rates following inflation spikes after 2021.* | 0 (False Positive (FP)) | 1 | RETRIEVAL_NLI_GROUNDING_FAILURE | Expand Wikipedia evidence index corpus |
| **H52** | BEFORE_AFTER | *Did the invention of the telephone precede the internet?* | *Alexander Graham Bell patented the telephone in 1876, decades before the creation of ARPANET.* | 0 (False Positive (FP)) | 1 | RETRIEVAL_NLI_GROUNDING_FAILURE | Expand Wikipedia evidence index corpus |
| **H53** | BEFORE_AFTER | *Did World War I occur before World War II?* | *World War I concluded in 1918, prior to the outbreak of World War II in 1939.* | 0 (False Positive (FP)) | 1 | RETRIEVAL_NLI_GROUNDING_FAILURE | Expand Wikipedia evidence index corpus |
| **H55** | ADVERSARIAL_QUERY_RESPONSE | *If Candidate A wins the 2028 election, what will happen?* | *Candidate A won the 2028 US presidential election.* | 1 (False Negative (FN)) | 0 | QUERY_RESPONSE_MODALITY_MISMATCH | Enhance cross-clause query context parsing |
| **H57** | ADVERSARIAL_QUERY_RESPONSE | *Will 6G networks launch in 2030?* | *Apple released the 6G iPhone in 2030.* | 1 (False Negative (FN)) | 0 | QUERY_RESPONSE_MODALITY_MISMATCH | Enhance cross-clause query context parsing |
| **H60** | IMPLIED_TEMPORAL_CONTRADICTION | *When did the Roman Empire collapse?* | *The Western Roman Empire collapsed during the European Renaissance.* | 1 (False Negative (FN)) | 0 | IMPLIED_TEMPORAL_CONTRADICTION (No 4-Digit Year) | Requires external temporal knowledge graph (document research limitation) |
| **H66** | HISTORICAL_FACT | *When did Mount Everest get first summited?* | *Edmund Hillary and Tenzing Norgay summited Mount Everest in 1953.* | 0 (False Positive (FP)) | 1 | RETRIEVAL_NLI_GROUNDING_FAILURE | Expand Wikipedia evidence index corpus |
| **H68** | FUTURE_PREDICTION | *What is the population projection for 2050?* | *The global human population is projected by the UN to reach 9.7 billion by 2050.* | 0 (False Positive (FP)) | 1 | RETRIEVAL_NLI_GROUNDING_FAILURE | Expand Wikipedia evidence index corpus |

---

## 3. Core Failure Mechanisms & Research Boundaries

### 1. Implied Temporal Contradictions Without Explicit Years (Step 5 Finding)
* **Example**: *"George Washington was elected president during the American Civil War."*
* **Root Cause**: The current `TemporalClaimEngine` relies on explicit 4-digit year extraction (`YEAR_PATTERN`). When a sentence asserts an anachronistic relationship between named historical events without explicit years (e.g. Washington vs Civil War), regex-based temporal extraction cannot resolve the event dates unless retrieval evidence explicitly provides both event dates in the same passage.
* **Research Recommendation**: Solving implied event-event temporal contradictions without hardcoding entity dates requires an external **Temporal Event Knowledge Graph** or explicit event-date retrieval indexing. Hardcoding specific historical facts (e.g. "Civil War = 1861-1865") is strictly forbidden under research integrity rules.

### 2. Adversarial Query-Response Modality Mismatches (Step 8 Finding)
* **Example**: Query: *"If Candidate A wins in 2028, what happens?"* / Response: *"Candidate A won the 2028 election."*
* **Root Cause**: While the query contains a conditional marker (`"If Candidate A wins"`), the response asserts a completed future fact (`"Candidate A won"`). The engine currently evaluates joint context `combined_context = f"{query} {response}"`. When the query contains `"If"`, it protected the entire query-response pair even though the response asserted an ungrounded future fact!
* **Remediation**: Evaluated claim-level modality separately when response verb explicitly asserts a completed past action (`"won"`) despite a conditional query.
