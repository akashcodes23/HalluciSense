# Phase 8 Scientific Integrity Report

## 1. Provenance and Ground Truth Verification
Every record in Dataset 8A (N=175) is grounded in authoritative scientific sources (Wikipedia, NIST CODATA, WHO guidelines, peer-reviewed literature).
- **Ground Truth Independence**: Ground truth labels were authored independently of HalluciSense outputs.
- **Dataset Hash**: SHA-256 manifest frozen at `dataset_8a.jsonl`.

## 2. Circularity Audits and Mitigation
In sub-experiment 8B, responses were categorized based on Phase 7B NLI evidence grounding.
- **Explicit Disclosure**: Any metric derived from comparing P1 against Dataset B is labeled `CIRCULAR` and excluded from primary detector efficacy claims.
- **Purpose**: Documented as an empirical refutation of static benchmark assumptions on non-deterministic LLMs.

## 3. Honest Robustness Reporting
Sub-experiment 8C reports genuine pipeline performance without post-hoc threshold adjustment or synthetic boosting.
