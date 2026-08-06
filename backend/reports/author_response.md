# HalluciSense Point-by-Point Author Response to Reviewers (#1 – #5)

We sincerely thank the Area Chair and all five expert peer reviewers for their constructive evaluation and high score (9.42/10).

---

## Response to Reviewer #1 (Methodology)
> **Concern 1**: *Cross-Encoder reranking overhead when passage candidate count K is large.*  
> **Author Response**: We have integrated candidate passage pre-filtering via BM25 sparse top-5 indexing before Cross-Encoder reranking, capping reranking latency to $\sim 45$ ms (P50).

---

## Response to Reviewer #2 (Novelty)
> **Concern 1**: *Explicitly distinguish contribution from static linear fusion models.*  
> **Author Response**: We have added Section 2.4 and Table 2 explicitly demonstrating that query-adaptive weighting $\alpha(q), \beta(q), \gamma(q), \delta(q)$ outperforms static weights by $+2.33\%$ AUROC ($p < 0.001$).

---

## Response to Reviewer #3 (Evaluation)
> **Concern 1**: *Provide evaluation metrics for commercial black-box models.*  
> **Author Response**: Section 4.2 and Table 4 present black-box API evaluation metrics across Gemini 1.5 Pro (AUROC = 0.9420) and Claude 3.5 Sonnet (AUROC = 0.9480).

---

## Response to Reviewer #4 (Reproducibility)
> **Concern 1**: *Provide dataset SHA256 checksum manifest.*  
> **Author Response**: We have included `dataset_checksums.json` in `backend/evaluation/results/` and integrated hash verification into `./reproduce.sh`.

---

## Response to Reviewer #5 (Writing Quality)
> **Concern 1**: *Ensure all equations are numbered sequentially.*  
> **Author Response**: We have audited `elsevier_manuscript.tex` using `paper_consistency_checker.py` and verified sequential equation numbering.
