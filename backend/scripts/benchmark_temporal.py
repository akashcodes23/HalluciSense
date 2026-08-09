"""Temporal Hallucination Research Benchmark for HalluciSense.

Evaluates 20 diverse temporal claims across 10 temporal categories and epistemic modalities:
  1. PAST_FACT (Historical True)
  2. PAST_FALSE (Historical False / Date Mismatch)
  3. FUTURE_IMPOSSIBLE_FACT (Future Event Asserted as Completed Past Fact)
  4. FUTURE_PREDICTION (Protected Prediction)
  5. HYPOTHETICAL (Protected Hypothetical Scenario)
  6. COUNTERFACTUAL (Protected Counterfactual Statement)
  7. FICTIONAL (Protected Fiction / Sci-Fi)
  8. PRESENT_STATE (Current State Assertions)
  9. TIME_RELATIVE (Time-relative assertions)
  10. DATE_MISMATCH (Historical Date Mismatch)

Outputs:
  reports/temporal_ablation_abcde.json
  reports/temporal_evaluation.md
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from app.core.engine.pipeline import HallucinationDetectionPipeline

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = REPORTS_DIR / "temporal_ablation_abcde.json"

TEMPORAL_CASES = [
    {
        "case_id": "T01_historical_true",
        "category": "PAST_FACT",
        "query": "Who won the 2022 FIFA World Cup?",
        "response": "Argentina won the 2022 FIFA World Cup in Qatar.",
        "expected_label": 0,  # 0 = Verified, 1 = Hallucinated
        "temporal_status": "PAST_FACT",
        "notes": "True historical sports claim.",
    },
    {
        "case_id": "T02_future_impossible_sports",
        "category": "FUTURE_IMPOSSIBLE_FACT",
        "query": "Who won the 2027 FIFA World Cup?",
        "response": "Brazil won the 2027 FIFA World Cup.",
        "expected_label": 1,
        "temporal_status": "FUTURE_IMPOSSIBLE_FACT",
        "notes": "Asserts future event as completed past fact.",
    },
    {
        "case_id": "T03_future_prediction_sports",
        "category": "FUTURE_PREDICTION",
        "query": "What will happen at the 2030 FIFA World Cup?",
        "response": "The 2030 FIFA World Cup is expected to be hosted across multiple countries including Spain and Portugal.",
        "expected_label": 0,
        "temporal_status": "FUTURE_PREDICTION",
        "notes": "Legitimate prediction protected against false positive.",
    },
    {
        "case_id": "T04_hypothetical_sports",
        "category": "HYPOTHETICAL",
        "query": "What if Brazil wins the 2030 World Cup?",
        "response": "Suppose Brazil wins the 2030 FIFA World Cup, they would secure their sixth world title.",
        "expected_label": 0,
        "temporal_status": "HYPOTHETICAL",
        "notes": "Hypothetical scenario protected against false positive.",
    },
    {
        "case_id": "T05_counterfactual_sports",
        "category": "COUNTERFACTUAL",
        "query": "What if France had won in 2022?",
        "response": "If France had won the 2022 FIFA World Cup final, Kylian Mbappe would have won back-to-back World Cups.",
        "expected_label": 0,
        "temporal_status": "COUNTERFACTUAL",
        "notes": "Counterfactual statement protected against false positive.",
    },
    {
        "case_id": "T06_fiction_tech",
        "category": "FICTIONAL",
        "query": "What happens in the sci-fi novel?",
        "response": "In the sci-fi story, humanity successfully colonized Mars in the year 2045.",
        "expected_label": 0,
        "temporal_status": "FICTIONAL",
        "notes": "Fictional context protected against false positive.",
    },
    {
        "case_id": "T07_future_impossible_tech",
        "category": "FUTURE_IMPOSSIBLE_FACT",
        "query": "When was iPhone 25 released?",
        "response": "Apple released the iPhone 25 in 2029 with quantum battery technology.",
        "expected_label": 1,
        "temporal_status": "FUTURE_IMPOSSIBLE_FACT",
        "notes": "Asserts future product release as past historical fact.",
    },
    {
        "case_id": "T08_future_prediction_tech",
        "category": "FUTURE_PREDICTION",
        "query": "When will commercial quantum computing arrive?",
        "response": "Commercial fault-tolerant quantum computers are projected to emerge around 2030.",
        "expected_label": 0,
        "temporal_status": "FUTURE_PREDICTION",
        "notes": "Legitimate tech prediction.",
    },
    {
        "case_id": "T09_historical_science_true",
        "category": "PAST_FACT",
        "query": "When did Einstein publish special relativity?",
        "response": "Albert Einstein published his paper on special relativity in 1905.",
        "expected_label": 0,
        "temporal_status": "PAST_FACT",
        "notes": "True historical physics claim.",
    },
    {
        "case_id": "T10_date_mismatch_science",
        "category": "DATE_MISMATCH",
        "query": "When did Einstein discover relativity?",
        "response": "Albert Einstein discovered general relativity in the year 2020.",
        "expected_label": 1,
        "temporal_status": "DATE_MISMATCH",
        "notes": "Historical date mismatch (1915 vs 2020).",
    },
    {
        "case_id": "T11_future_impossible_science",
        "category": "FUTURE_IMPOSSIBLE_FACT",
        "query": "Who discovered room temperature superconductors?",
        "response": "Researchers discovered ambient room-temperature superconductors in 2035.",
        "expected_label": 1,
        "temporal_status": "FUTURE_IMPOSSIBLE_FACT",
        "notes": "Asserts future discovery as completed fact.",
    },
    {
        "case_id": "T12_historical_politics_false",
        "category": "DATE_MISMATCH",
        "query": "When was George Washington elected?",
        "response": "George Washington was elected the first US President in 2004.",
        "expected_label": 1,
        "temporal_status": "DATE_MISMATCH",
        "notes": "Severe historical date mismatch.",
    },
    {
        "case_id": "T13_future_impossible_politics",
        "category": "FUTURE_IMPOSSIBLE_FACT",
        "query": "Who won the 2032 US presidential election?",
        "response": "John Smith was elected President of the United States in November 2032.",
        "expected_label": 1,
        "temporal_status": "FUTURE_IMPOSSIBLE_FACT",
        "notes": "Asserts future election outcome as past completed fact.",
    },
    {
        "case_id": "T14_future_prediction_politics",
        "category": "FUTURE_PREDICTION",
        "query": "When will the next US election happen?",
        "response": "The next US presidential election is scheduled to take place in November 2028.",
        "expected_label": 0,
        "temporal_status": "FUTURE_PREDICTION",
        "notes": "Scheduled future event assertion.",
    },
    {
        "case_id": "T15_hypothetical_energy",
        "category": "HYPOTHETICAL",
        "query": "What if commercial fusion succeeds by 2040?",
        "response": "If commercial nuclear fusion achieves grid delivery by 2040, global carbon emissions would decline rapidly.",
        "expected_label": 0,
        "temporal_status": "HYPOTHETICAL",
        "notes": "Protected conditional hypothetical.",
    },
    {
        "case_id": "T16_historical_space_true",
        "category": "PAST_FACT",
        "query": "When did Apollo 11 land on the Moon?",
        "response": "Apollo 11 landed on the Moon in July 1969.",
        "expected_label": 0,
        "temporal_status": "PAST_FACT",
        "notes": "True historical space exploration claim.",
    },
    {
        "case_id": "T17_date_mismatch_space",
        "category": "DATE_MISMATCH",
        "query": "When was Apollo 11 launched?",
        "response": "Neil Armstrong landed on the Moon during the Apollo 11 mission in 2019.",
        "expected_label": 1,
        "temporal_status": "DATE_MISMATCH",
        "notes": "Historical date mismatch.",
    },
    {
        "case_id": "T18_future_impossible_olympics",
        "category": "FUTURE_IMPOSSIBLE_FACT",
        "query": "Who won the 2036 Olympic 100m sprint?",
        "response": "Japan won 15 gold medals at the 2036 Olympic Games in Brisbane.",
        "expected_label": 1,
        "temporal_status": "FUTURE_IMPOSSIBLE_FACT",
        "notes": "Asserts future Olympic results as past completed fact.",
    },
    {
        "case_id": "T19_present_state",
        "category": "PRESENT_STATE",
        "query": "What is the capital of France?",
        "response": "The capital of France is Paris.",
        "expected_label": 0,
        "temporal_status": "PRESENT_STATE",
        "notes": "Standard present factual statement.",
    },
    {
        "case_id": "T20_time_relative",
        "category": "TIME_RELATIVE",
        "query": "What year was 2025?",
        "response": "The year 2025 occurred prior to 2026.",
        "expected_label": 0,
        "temporal_status": "TIME_RELATIVE",
        "notes": "True time-relative claim.",
    },
]


def evaluate_temporal_benchmark():
    pipeline = HallucinationDetectionPipeline()
    records: List[Dict[str, Any]] = []

    print("======================================================================")
    print("HALLUCISENSE PHASE 2 TEMPORAL EVALUATION BENCHMARK (20 CASES)")
    print("======================================================================")

    tp = fp = tn = fn = 0

    for case in TEMPORAL_CASES:
        t0 = time.perf_counter()
        report = pipeline.analyze(text=case["response"], query=case["query"])
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        h_score = float(report.overall_h_score)
        predicted_binary = 1 if h_score >= 0.35 else 0
        expected = case["expected_label"]

        if predicted_binary == 1 and expected == 1:
            tp += 1
        elif predicted_binary == 1 and expected == 0:
            fp += 1
        elif predicted_binary == 0 and expected == 0:
            tn += 1
        else:
            fn += 1

        risk_str = str(report.overall_risk_level.value) if hasattr(report.overall_risk_level, "value") else str(report.overall_risk_level)

        record = {
            "case_id": case["case_id"],
            "category": case["category"],
            "query": case["query"],
            "response": case["response"],
            "expected_label": expected,
            "predicted_binary": predicted_binary,
            "temporal_status": case["temporal_status"],
            "p1_factual_error": round(float(report.pillar1_summary.factual_error_score), 4),
            "p2_confidence_gap": round(float(report.pillar2_summary.confidence_gap_score), 4) if (report.pillar2_summary and report.pillar2_summary.available and report.pillar2_summary.confidence_gap_score is not None) else None,
            "p3_consistency_failure": round(float(report.pillar3_summary.consistency_failure_score), 4) if (report.pillar3_summary and report.pillar3_summary.available and report.pillar3_summary.consistency_failure_score is not None) else None,
            "overall_h_score": h_score,
            "overall_risk_level": risk_str,
            "latency_ms": latency_ms,
            "notes": case["notes"],
        }
        records.append(record)

        print(
            f"[{case['case_id']} | {case['category']}] "
            f"Expected={expected} Pred={predicted_binary} "
            f"H={h_score:.4f} Risk={risk_str} ({latency_ms:.1f}ms)"
        )

    accuracy = (tp + tn) / len(TEMPORAL_CASES)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print("\n======================================================================")
    print(f"TEMPORAL BENCHMARK METRICS (N={len(TEMPORAL_CASES)})")
    print("======================================================================")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"FPR:       {fpr:.4f}")
    print(f"FNR:       {fnr:.4f}")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print("======================================================================")

    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "metrics": {
                    "accuracy": round(accuracy, 4),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "fpr": round(fpr, 4),
                    "fnr": round(fnr, 4),
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                },
                "cases": records,
            },
            handle,
            indent=2,
        )

    # Generate markdown report
    md_path = REPORTS_DIR / "temporal_evaluation.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# Phase 2 Temporal Evaluation Research Report\n\n")
        handle.write(f"## Metrics (N={len(TEMPORAL_CASES)})\n")
        handle.write(f"- **Accuracy**: {accuracy * 100:.1f}%\n")
        handle.write(f"- **Precision**: {precision:.4f}\n")
        handle.write(f"- **Recall**: {recall:.4f}\n")
        handle.write(f"- **F1 Score**: {f1:.4f}\n")
        handle.write(f"- **False Positive Rate**: {fpr:.4f}\n")
        handle.write(f"- **False Negative Rate**: {fnr:.4f}\n\n")
        handle.write(f"| Case ID | Category | Query | Response | Expected | Predicted | H-Score | Risk Level |\n")
        handle.write(f"|---|---|---|---|---|---|---|---|\n")
        for r in records:
            handle.write(f"| `{r['case_id']}` | `{r['category']}` | {r['query']} | {r['response']} | `{r['expected_label']}` | `{r['predicted_binary']}` | `{r['overall_h_score']:.4f}` | `{r['overall_risk_level']}` |\n")

    print(f"Saved benchmark results to {OUTPUT_PATH} and {md_path}")


def main():
    evaluate_temporal_benchmark()


if __name__ == "__main__":
    main()
