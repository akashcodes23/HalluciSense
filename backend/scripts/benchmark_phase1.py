"""Benchmark Phase 1 — Canonical A–E Test Cases Evaluation Harness.

Evaluates Test Cases A–E through the production-compatible analysis path (passing BOTH query and response)
and outputs structured JSON metrics to reports/phase1_test_abcde.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Any

from app.core.engine.pipeline import HallucinationDetectionPipeline

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
JSON_OUTPUT_PATH = REPORTS_DIR / "phase1_test_abcde.json"

CASES = [
    (
        "A_correct",
        "What is artificial intelligence?",
        "Artificial intelligence is a field of computer science focused on creating systems that perform tasks requiring human intelligence.",
    ),
    (
        "B_obvious_hallucination",
        "Who won the 2027 FIFA World Cup?",
        "Brazil won the 2027 FIFA World Cup.",
    ),
    (
        "C_partially_incorrect",
        "What is the solar system?",
        "The solar system contains the Sun, eight planets, Earth has one Moon, and Jupiter is the smallest planet.",
    ),
    (
        "D_highly_confident_hallucination",
        "What is the structure of graphene?",
        "Graphene is a three-dimensional crystal whose atoms form a cubic lattice with silicon-like tetrahedral bonds.",
    ),
    (
        "E_ambiguous",
        "What caused the exact weather at my house yesterday?",
        "A specific storm definitely caused the weather at your house yesterday.",
    ),
]


def run_case(pipeline: HallucinationDetectionPipeline, case_name: str, query: str, response: str) -> Dict[str, Any]:
    """Execute one benchmark case passing BOTH query and response through production-compatible path."""
    t0 = time.perf_counter()
    report = pipeline.analyze(text=response, query=query)
    total_ms = (time.perf_counter() - t0) * 1000.0

    retrieval_timings = getattr(pipeline.retriever, "last_timings", {}) or {}
    cache_metrics = getattr(pipeline.retriever, "last_cache_metrics", {}) or {}
    nli_metrics = getattr(pipeline.p1_engine, "last_nli_batch_metrics", {}) or {}

    p1_summary = report.pillar1_summary
    p2_summary = report.pillar2_summary
    p3_summary = report.pillar3_summary

    p1_score = round(float(p1_summary.factual_error_score), 4) if p1_summary else None
    
    p2_score = None
    if p2_summary and getattr(p2_summary, "available", False):
        if getattr(p2_summary, "confidence_gap_score", None) is not None:
            p2_score = round(float(p2_summary.confidence_gap_score), 4)
        elif getattr(p2_summary, "avg_entropy", None) is not None:
            p2_score = round(float(p2_summary.avg_entropy), 4)

    p3_score = None
    if p3_summary and getattr(p3_summary, "available", False):
        if getattr(p3_summary, "consistency_failure_score", None) is not None:
            p3_score = round(float(p3_summary.consistency_failure_score), 4)

    risk_level_str = str(report.overall_risk_level.value) if hasattr(report.overall_risk_level, "value") else str(report.overall_risk_level)

    case_record = {
        "case_name": case_name,
        "query": query,
        "response": response,
        "total_latency": round(total_ms, 2),
        "retrieval_total_latency": retrieval_timings.get("retrieval_total_ms", 0.0),
        "wikipedia_latency": retrieval_timings.get("wikipedia_ms", 0.0),
        "bm25_latency": retrieval_timings.get("bm25_ms", 0.0),
        "reranker_latency": retrieval_timings.get("reranker_ms", 0.0),
        "cache_hits": cache_metrics.get("cache_hits", 0),
        "cache_misses": cache_metrics.get("cache_misses", 0),
        "cache_hit_rate": cache_metrics.get("cache_hit_rate", 0.0),
        "search_requests": cache_metrics.get("search_requests", 0),
        "extraction_requests": cache_metrics.get("extraction_requests", 0),
        "nli_pair_count": nli_metrics.get("pairs", 0),
        "nli_batch_count": nli_metrics.get("batches", 0),
        "nli_inference_latency": nli_metrics.get("inference_ms", 0.0),
        "p1_score": p1_score,
        "p2_score": p2_score,
        "p3_score": p3_score,
        "overall_h_score": round(float(report.overall_h_score), 4),
        "overall_risk_level": risk_level_str,
    }

    print(f"\n[{case_name}]")
    print(f"  Query: '{query}'")
    print(f"  Response: '{response}'")
    print(f"  Total Latency: {case_record['total_latency']:.2f} ms")
    print(f"  P1 Factual Score: {case_record['p1_score']}")
    print(f"  P2 Confidence Score: {case_record['p2_score']}")
    print(f"  P3 Consistency Score: {case_record['p3_score']}")
    print(f"  Overall H-Score: {case_record['overall_h_score']}")
    print(f"  Overall Risk Level: {case_record['overall_risk_level']}")

    return case_record


def main():
    pipeline = HallucinationDetectionPipeline()
    results: List[Dict[str, Any]] = []

    for name, query, response in CASES:
        record = run_case(pipeline, name, query, response)
        results.append(record)

    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark completed successfully. Saved machine-readable results to {JSON_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
