# Phase 8 Limitations and Boundary Conditions

## 1. Domain Coverage Boundaries
Dataset 8A focuses on 5 formal/natural science domains (Physics, Chemistry, Biology, Medicine, Mathematics). Humanities, law, and creative writing require distinct ontology extractors.

## 2. External Knowledge Base Dependency
Retrieval efficacy depends on Wikipedia / knowledge base coverage. Outdated claims that are not yet updated in the reference corpus remain vulnerable.

## 3. Computational Latency Overhead
Claim decomposition introduces 2–5 sub-clause NLI evaluations per sentence, increasing total pipeline latency from ~70 ms to ~120 ms.
