# HalluciSense Pillar-1: Training Pipeline

*Generated: 2026-08-03T04:49:02.142488+00:00*  
*Phase: 6K (Frozen)*

---

## 1. Data Partitioning (Phase 6B)

The HalluciSense dataset comprises three benchmarks:
- **HaluBench**: RAG hallucination benchmark
- **HaluEval**: General LLM hallucination evaluation
- **RAGTruth**: RAG faithfulness dataset

Partitioning strategy:
- **Development set**: 58,002 samples (used for all training and CV)
- **Validation set**: 3,500 samples (protocol-locked before any model fitting)
- **Partition integrity**: Verified via SHA-256 checksums; 0% leakage confirmed

## 2. Feature Reconstruction (Phase 6I)

NLI features were reconstructed via full inference over all 61,502 samples using
`cross-encoder/nli-deberta-v3-small`. This stage produces:
- `evaluation_results/phase6i/claim_evidence_features_development.jsonl` (58,002 rows)
- `evaluation_results/phase6i/claim_evidence_features_validation.jsonl` (3,500 rows)

## 3. Feature Engineering & Collinearity Audit (Phase 6M)

From an initial feature universe of 11 NLI aggregations:
1. **Variance threshold**: Removed zero-variance features
2. **Correlation filter**: Removed features with Pearson |r| > 0.85
3. **Discriminative test**: Removed features with single-feature ROC-AUC < 0.52
4. **Final feature set**: SET_D — 5 features: `['mean_entailment', 'max_entailment', 'mean_contradiction', 'min_support_margin', 'num_claims']`

## 4. Preprocessing

- **Scaler**: RobustScaler (median and IQR-based, robust to outliers)
- **Fit**: On DEV set only; applied to both DEV and VAL
- **Stored artifact**: `final_model/robust_scaler.joblib`

## 5. Cross-Validation Protocol (Phase 6K)

- **CV strategy**: Stratified K-Fold (k=5)
- **Metric**: ROC-AUC (primary), Brier Score, F1, MCC (secondary)
- **Candidates evaluated**: 4 feature set variants × multiple C values
- **Final candidate**: Candidate 3 (SET_D + RobustScaler + liblinear, C=1.0)

## 6. Stability Gate (Phase 6K)

Before final selection, all candidates passed a **32-iteration bootstrap stability gate**:
- Each iteration: sample 80% of DEV, retrain, evaluate
- Gate criterion: Coefficient sign consistency > 95%
- **Result**: 32/32 PASS for the final candidate

## 7. Solver Consistency Test

Four solvers (lbfgs, liblinear, newton-cg, saga) were benchmarked:
- `liblinear` and `saga` produce zero numerical warnings
- **Selected solver**: `liblinear` (coordinate descent; warning-free)

## 8. Final Model Fitting

- **Data**: Full DEV set (58,002 samples)
- **Algorithm**: LogisticRegression(solver='liblinear', penalty='l2', C=1.0)
- **Scaler**: RobustScaler (fitted on DEV)
- **Operating threshold**: 0.56 (MCC-optimized on DEV)
- **Protocol lock timestamp**: 2026-08-03T04:22:00Z
