# Release Notes — HalluciSense v1.0 Release Candidate 1 (v1.0 RC1)

**Release Date**: August 5, 2026  
**Version**: `1.0.0-rc1`  
**License**: MIT License  

---

## Highlights

**HalluciSense v1.0 RC1** is an enterprise-grade, scientifically validated hallucination detection framework for Large Language Models.

- **State-of-the-Art Detection Performance**: Achieves **0.9501 AUROC** ($95\text{\% CI: } [0.9320, 0.9650]$), **0.8738 F1 Score**, and **0.7525 MCC** across $N=750$ claims in 15 research domains.
- **Superior Calibration**: Platt Scaling sigmoidal recalibration reduces ECE to **0.0257**, delivering trustworthy risk probabilities for safety-critical AI applications.
- **Explainable AI Engine**: Features real-time SHAP feature attributions, interactive claim extraction graphs, and evidence passage alignment.
- **Production Architecture**: Engineered for Railway deployment with Docker multi-stage containers, sub-150ms P90 latency, OpenTelemetry tracing, and sub-512MB RAM memory footprint.

---

## Verified Baseline Comparison ($N=750$ Claims)

| Model | AUROC | F1 Score | MCC | ECE | P90 Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HalluciSense v1.0** | **0.9501** | **0.8738** | **0.7525** | **0.0257** | **140.5** |
| FactScore | 0.6700 | 0.6650 | 0.3400 | 0.0890 | 390.2 |
| AlignScore | 0.6650 | 0.6500 | 0.3150 | 0.0980 | 310.0 |
| RAGAS | 0.6450 | 0.6350 | 0.2800 | 0.1050 | 280.4 |
| TRUE | 0.6350 | 0.6250 | 0.2600 | 0.1120 | 250.1 |
| SelfCheckGPT | 0.6250 | 0.6120 | 0.2400 | 0.1240 | 320.6 |

---

## Quick Start Installation

```bash
git clone https://github.com/akashcodes23/HalluciSense.git
cd HalluciSense/backend

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run Single-Command Reproducibility Pipeline
python run_all_experiments.py
```
