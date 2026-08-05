# Frequently Asked Questions (FAQ)

### Q1: What is HalluciSense?
HalluciSense is an open-source, hybrid multi-pillar hallucination detection framework for Large Language Models (LLMs). It combines external evidence retrieval (Pillar 1) and intra-model self-consistency contradiction matrices (Pillar 2) to compute calibrated hallucination risk probabilities in real time.

### Q2: How does HalluciSense differ from RAGAS, SelfCheckGPT, or FactScore?
While traditional frameworks rely on a single approach (either retrieval-only or sampling-only), HalluciSense fuses 19 continuous features through a calibrated hybrid meta-classifier, achieving **0.9501 AUROC** (vs ~0.67 for FactScore and ~0.62 for SelfCheckGPT) with **140.5 ms P90 latency**.

### Q3: How do I run HalluciSense locally?
Clone the repository and run:
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python run_all_experiments.py
```

### Q4: Can I use HalluciSense without internet connection?
Yes! If external search APIs are offline or unreachable, HalluciSense automatically falls back to Pillar 2 self-consistency contradiction analysis with zero pipeline downtime.
