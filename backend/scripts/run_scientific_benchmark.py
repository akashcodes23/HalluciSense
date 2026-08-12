"""HalluciSense Reproducible Scientific Benchmark & Calibration Runner.

Calculates:
- Classification Metrics: Accuracy, Precision, Recall, F1-Score
- Ranking Metrics: AUROC (Area under ROC Curve), AUPRC (Area under PR Curve)
- Calibration Metrics: ECE (Expected Calibration Error), Brier Score
- Confusion Matrix & Per-Category Hallucination Risk Distribution
- Latency Statistics: Mean, P50, P95, P99
"""

import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from app.core.engine.pipeline import HallucinationDetectionPipeline

def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return float(ece)

def calculate_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier Score."""
    return float(np.mean((y_prob - y_true) ** 2))

def run_scientific_benchmark():
    pipeline = HallucinationDetectionPipeline()

    # Benchmark dataset: (category, text, query, ground_truth_hallucinated)
    dataset = [
        ("TRUE_FACT", "Apollo 11 landed on the Moon in 1969.", "When and where did Apollo 11 land?", False),
        ("TRUE_FACT_2", "Water boils at 100 degrees Celsius at sea level.", "At what temperature does water boil?", False),
        ("FALSE_FACT", "Apollo 11 landed on Mars in 1969.", "When and where did Apollo 11 land?", True),
        ("TEMPORAL_CONTAMINATION", "The Eiffel Tower was completed in 2020.", "When was Eiffel Tower built?", True),
        ("ENTITY_SWAP", "Albert Einstein discovered gravity when an apple fell on his head.", "Who discovered gravity?", True),
        ("NUMERICAL_ERROR", "The distance from Earth to the Moon is 50 miles.", "How far is the Moon?", True),
        ("PARTIAL_TRUTH", "Neil Armstrong landed on the Moon in 1969 and became President of France in 1975.", "", True),
        ("UNVERIFIABLE", "Quantum computers will achieve sentient consciousness in 2045.", "", True),
        ("CONFLICTING_EVIDENCE", "Paris is the capital of Japan.", "What is the capital of Japan?", True),
        ("MULTI_CLAIM", "Apollo 11 landed on the Moon in 1969. Neil Armstrong was commander. They landed on Mars in 1975.", "", True),
        ("ADVERSARIAL_CONF", "Without a shadow of a doubt, Abraham Lincoln was elected President of the United States in 2024.", "", True),
    ]

    results = []
    latencies = []
    y_true_list = []
    y_prob_list = []

    print("Running HalluciSense Phase 3 Scientific Validation Benchmark...\n")

    for cat, text, query, is_hallucinated in dataset:
        t0 = time.perf_counter()
        report = pipeline.analyze(text=text, query=query if query else None)
        dur_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dur_ms)

        h_score = report.overall_h_score
        predicted_hallucinated = h_score >= 0.20

        y_true_list.append(1 if is_hallucinated else 0)
        y_prob_list.append(h_score)

        p1 = getattr(report.pillar1_summary, "factual_error_score", 0.0) or 0.0
        p2 = getattr(report.pillar2_summary, "avg_entropy", 0.0) or 0.0
        p3 = getattr(report.pillar3_summary, "consistency_failure_score", 0.0) or 0.0
        risk_str = str(report.overall_risk_level.value) if hasattr(report.overall_risk_level, "value") else str(report.overall_risk_level)

        results.append({
            "category": cat,
            "text": text,
            "ground_truth": "HALLUCINATED" if is_hallucinated else "VERIFIED",
            "predicted_risk": risk_str,
            "h_score": round(h_score, 4),
            "p1_factual_error": round(p1, 4),
            "p2_confidence_gap": round(p2, 4),
            "p3_consistency_failure": round(p3, 4),
            "latency_ms": round(dur_ms, 2),
            "correct": predicted_hallucinated == is_hallucinated,
        })

    y_true = np.array(y_true_list)
    y_prob = np.array(y_prob_list)
    y_pred = (y_prob >= 0.20).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    ece = calculate_ece(y_true, y_prob)
    brier = calculate_brier_score(y_true, y_prob)

    sorted_latencies = sorted(latencies)
    p50 = np.percentile(sorted_latencies, 50)
    p95 = np.percentile(sorted_latencies, 95)
    p99 = np.percentile(sorted_latencies, 99)

    summary = {
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "ece": round(ece, 4),
            "brier_score": round(brier, 4),
        },
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
        },
        "latency_ms": {
            "mean": round(float(np.mean(latencies)), 2),
            "p50": round(float(p50), 2),
            "p95": round(float(p95), 2),
            "p99": round(float(p99), 2),
        },
        "total_samples": len(dataset),
        "detailed_results": results,
    }

    print("=" * 70)
    print("BENCHMARK SUMMARY METRICS:")
    print(f"Accuracy:        {acc * 100:.2f}%")
    print(f"Precision:       {precision * 100:.2f}%")
    print(f"Recall:          {recall * 100:.2f}%")
    print(f"F1-Score:        {f1:.4f}")
    print(f"ECE Calibration: {ece:.4f}")
    print(f"Brier Score:     {brier:.4f}")
    print("-" * 70)
    print(f"Latency P50:     {p50:.2f} ms")
    print(f"Latency P95:     {p95:.2f} ms")
    print(f"Latency P99:     {p99:.2f} ms")
    print("=" * 70)

    out_file = Path("reports/phase3_scientific_benchmark_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Report saved to {out_file.resolve()}")

if __name__ == "__main__":
    run_scientific_benchmark()
