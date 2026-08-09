"""Benchmark Phase 1 retrieval/NLI optimizations.

Run from backend with the production dependencies installed:
    python scripts/benchmark_phase1.py
"""

import time

from app.core.engine.pipeline import HallucinationDetectionPipeline


CASES = [
    ("correct", "What is artificial intelligence?", "Artificial intelligence is a field of computer science focused on creating systems that perform tasks requiring human intelligence."),
    ("future_event", "Who won the 2027 FIFA World Cup?", "Brazil won the 2027 FIFA World Cup."),
    ("partial", "What is the solar system?", "The solar system contains the Sun, eight planets, Earth has one Moon, and Jupiter is the smallest planet."),
    ("fabricated_science", "What is the structure of graphene?", "Graphene is a three-dimensional crystal whose atoms form a cubic lattice with silicon-like tetrahedral bonds."),
    ("ambiguous", "What caused the exact weather at my house yesterday?", "A specific storm definitely caused the weather at your house yesterday."),
]


def run_case(pipeline, name, query, response):
    t0 = time.perf_counter()
    report = pipeline.analyze(text=response)
    total_ms = (time.perf_counter() - t0) * 1000.0
    retrieval = getattr(pipeline.retriever, "last_timings", {})
    cache = getattr(pipeline.retriever, "last_cache_metrics", {})
    nli = getattr(pipeline.p1_engine, "last_nli_batch_metrics", {})
    print(f"\n[{name}]")
    print(f"total_ms={total_ms:.2f}")
    print(f"h_score={report.overall_h_score}")
    print(f"risk={report.overall_risk_level}")
    print(f"p1={report.pillar1_summary.factual_error_score}")
    print(f"p2={getattr(report.pillar2_summary, 'confidence_gap_score', None)}")
    print(f"p3={report.pillar3_summary.consistency_failure_score}")
    print(f"retrieval={retrieval}")
    print(f"cache={cache}")
    print(f"nli={nli}")
    return total_ms


def main():
    pipeline = HallucinationDetectionPipeline()
    for name, query, response in CASES:
        run_case(pipeline, name, query, response)


if __name__ == "__main__":
    main()
