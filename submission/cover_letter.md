# Cover Letter to the Editor-in-Chief

**Target Journal**: Elsevier *Information Fusion*  
**Date**: August 6, 2026  
**Manuscript Title**: HalluciSense: A Confidence-Aware Hybrid Multi-Pillar Framework for LLM Hallucination Detection and Recalibration  
**Authors**: HalluciSense Research Group  

Dear Editor-in-Chief and Editorial Board Members,

We are pleased to submit our original research manuscript titled **"HalluciSense: A Confidence-Aware Hybrid Multi-Pillar Framework for LLM Hallucination Detection and Recalibration"** for consideration as a regular research paper in *Information Fusion*.

Large Language Models (LLMs) suffer from hallucinating ungrounded or self-contradictory claims. Existing detection methods rely on isolated paradigms—such as external retrieval alone or pure logit uncertainty estimation. In this paper, we present **HalluciSense**, a novel hybrid framework that dynamically fuses Evidence Grounding, Predictive Uncertainty, and Structural Self-Consistency into an uncertainty-gated meta-classifier calibrated via Platt Sigmoidal Recalibration.

Evaluated across 7 public benchmark datasets spanning 15 domains ($N=750$ claims), HalluciSense achieves state-of-the-art detection accuracy ($\text{AUROC}=0.9501$, $\text{F1}=0.8738$) and probability calibration ($\text{ECE}=0.0257$), significantly outperforming 13 prior literature baselines ($p < 0.001$).

This manuscript is 100% original, has not been published previously, and is not under consideration for publication elsewhere. All code, datasets, and experiment execution scripts are fully open-source and 100% reproducible via `./reproduce.sh`.

Thank you for considering our work.

Sincerely,  
HalluciSense Research Group
