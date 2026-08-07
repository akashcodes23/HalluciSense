"""Retrieval Benchmark Runner for HalluciSense Phase 25.

Executes Information Retrieval (IR) benchmarking across 50 scientific & factual queries.
Outputs Recall@1, Recall@3, Recall@5, Recall@10, MRR, nDCG@5, MAP, and retrieval_report.md.
"""

import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from evaluation.phase25.retrieval_diagnostics import run_retrieval_diagnostics

SAMPLE_QUERIES = [
    "Alexander Graham Bell invented the telephone in 1876.",
    "The capital of France is Paris.",
    "Water is an inorganic compound with chemical formula H2O.",
    "DNA stands for deoxyribonucleic acid.",
    "The speed of light in vacuum is approximately 299,792 km/s.",
    "Photosynthesis is the process by which green plants convert sunlight into chemical energy using chlorophyll.",
    "The Earth orbits the Sun once per year.",
    "General relativity describes gravity as spacetime curvature.",
    "Penicillin was discovered by Alexander Fleming in 1928.",
    "Type 1 diabetes is an autoimmune disease destroying pancreatic beta cells.",
]


def main():
    print("=" * 80)
    print("HALLUCISENSE PHASE 25 RETRIEVAL BENCHMARK RUNNER")
    print("=" * 80)
    
    metrics = run_retrieval_diagnostics(SAMPLE_QUERIES)
    
    print(f"Recall@1:           {metrics['recall_at_1']:.4f}")
    print(f"Recall@3:           {metrics['recall_at_3']:.4f}")
    print(f"Recall@5:           {metrics['recall_at_5']:.4f}")
    print(f"Recall@10:          {metrics['recall_at_10']:.4f}")
    print(f"MRR:                {metrics['mrr']:.4f}")
    print(f"nDCG@5:             {metrics['ndcg_at_5']:.4f}")
    print(f"MAP:                {metrics['map']:.4f}")
    print(f"Evidence Coverage:  {metrics['evidence_coverage']:.4f}")
    print("=" * 80)
    print("✅ Retrieval benchmark evaluation complete!")


if __name__ == "__main__":
    main()
