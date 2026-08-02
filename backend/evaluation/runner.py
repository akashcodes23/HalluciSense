"""Offline Evaluation Runner for HalluciSense Phase 6A.

Wraps production HallucinationDetectionPipeline to evaluate benchmark datasets offline.
Performs metric calculation, threshold sweeps, pillar ablation, availability analysis,
calibration curves, category breakdown, error extraction, and JSON persistence.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.core.engine.types import HallucinationReport
from evaluation.ablation import run_ablation_study
from evaluation.calibration import analyze_calibration, analyze_score_distributions
from evaluation.dataset import BenchmarkSample, DatasetLoader
from evaluation.metrics import (
    compute_all_metrics,
    compute_confusion_matrix,
    compute_f1,
    compute_precision,
    compute_recall,
    compute_specificity,
)


@dataclass
class EvaluationSampleResult:
    sample_id: str
    ground_truth: int
    predicted_risk: str
    h_score: float
    p1_factual_error: float
    p1_available: bool
    p2_confidence_gap: Optional[float]
    p2_available: bool
    p3_consistency_failure: Optional[float]
    p3_available: bool
    effective_weights: Dict[str, float]
    processing_time_ms: float
    category: str


class EvaluationRunner:
    """Offline Evaluation Runner executing benchmark datasets through production pipeline."""

    def __init__(
        self,
        pipeline: Optional[HallucinationDetectionPipeline] = None,
        default_threshold: float = 0.35,
    ):
        self.pipeline = pipeline if pipeline is not None else HallucinationDetectionPipeline()
        self.default_threshold = default_threshold

        # Offline evaluation mock for correction step to guarantee network isolation
        self.pipeline._generate_correction = lambda text, sentence_analyses, evidence_items: (
            "Offline evaluation response",
            sentence_analyses,
        )

    def evaluate_sample(
        self, sample: BenchmarkSample
    ) -> EvaluationSampleResult:
        """Evaluates a single BenchmarkSample using the production pipeline."""
        start_time = time.perf_counter()

        ev_items = None
        if sample.evidence:
            from app.core.engine.types import EvidenceItem
            ev_items = [
                EvidenceItem(
                    claim=e.get("claim", sample.response),
                    snippet=e.get("snippet", ""),
                    source_name=e.get("source_name", "Ground Truth KB"),
                    source_url=e.get("source_url", None),
                    similarity_score=float(e.get("similarity_score", 1.0)),
                    is_supporting=bool(e.get("is_supporting", True)),
                )
                for e in sample.evidence
            ]

        # Execute production pipeline
        result: HallucinationReport = self.pipeline.analyze_response(
            full_text=sample.response,
            token_probabilities=sample.metadata.get("token_probs", None)
            if sample.metadata
            else None,
            evidence_items=ev_items,
        )

        processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        risk_str = (
            result.overall_risk_level.value
            if hasattr(result.overall_risk_level, "value")
            else str(result.overall_risk_level)
        )

        p1_fe = result.pillar1_summary.factual_error_score
        p1_avail = True

        p2_cg = (
            result.pillar2_summary.confidence_gap_score
            if result.pillar2_summary is not None
            else None
        )
        p2_avail = (
            getattr(result.pillar2_summary, "available", False)
            if result.pillar2_summary is not None
            else False
        )

        p3_cf = (
            result.pillar3_summary.consistency_failure_score
            if result.pillar3_summary is not None
            else None
        )
        p3_avail = (
            getattr(result.pillar3_summary, "available", False)
            if result.pillar3_summary is not None
            else False
        )

        return EvaluationSampleResult(
            sample_id=sample.id,
            ground_truth=sample.ground_truth_label,
            predicted_risk=risk_str,
            h_score=result.overall_h_score,
            p1_factual_error=p1_fe,
            p1_available=p1_avail,
            p2_confidence_gap=p2_cg,
            p2_available=p2_avail,
            p3_consistency_failure=p3_cf,
            p3_available=p3_avail,
            effective_weights=result.weights_used,
            processing_time_ms=processing_time_ms,
            category=sample.category,
        )

    def evaluate_dataset(
        self, samples: List[BenchmarkSample]
    ) -> Dict[str, Any]:
        """Evaluates an entire dataset of BenchmarkSamples and generates complete evaluation metrics."""
        sample_results: List[EvaluationSampleResult] = [
            self.evaluate_sample(s) for s in samples
        ]

        y_true = [sr.ground_truth for sr in sample_results]
        scores = [sr.h_score for sr in sample_results]

        # Binary prediction mapping: VERIFIED (0), NEEDS_VERIFICATION/LIKELY_HALLUCINATED (1)
        y_pred_default = [
            0 if sr.predicted_risk == "VERIFIED" else 1 for sr in sample_results
        ]

        # 1. Overall Metrics
        overall_metrics = compute_all_metrics(y_true, y_pred_default, scores)

        # 2. Confusion Matrix
        tp, tn, fp, fn = compute_confusion_matrix(y_true, y_pred_default)
        confusion_matrix = {
            "matrix": [[tn, fp], [fn, tp]],
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

        # 3. Threshold Sweep (0.00 -> 1.00 step 0.01)
        threshold_sweep = self.run_threshold_sweep(y_true, scores)

        # 4. Pillar Ablation Study
        p1_fe_list = [sr.p1_factual_error for sr in sample_results]
        p2_cg_list = [sr.p2_confidence_gap for sr in sample_results]
        p3_cf_list = [sr.p3_consistency_failure for sr in sample_results]
        ablation_results = run_ablation_study(
            y_true, p1_fe_list, p2_cg_list, p3_cf_list, threshold=self.default_threshold
        )

        # 5. Availability Analysis
        availability_analysis = self.run_availability_analysis(sample_results)

        # 6. Score Distribution
        score_distributions = analyze_score_distributions(y_true, scores)

        # 7. Calibration Analysis
        calibration_results = analyze_calibration(y_true, scores)

        # 8. Category Breakdown & Error Analysis
        category_metrics, error_analysis = self.run_error_analysis(
            samples, sample_results
        )

        return {
            "dataset_metadata": {
                "total_samples": len(samples),
                "factual_count": sum(1 for y in y_true if y == 0),
                "hallucinated_count": sum(1 for y in y_true if y == 1),
            },
            "overall_metrics": overall_metrics,
            "confusion_matrix": confusion_matrix,
            "threshold_analysis": threshold_sweep,
            "ablation_results": ablation_results,
            "availability_analysis": availability_analysis,
            "score_distributions": score_distributions,
            "calibration_results": calibration_results,
            "category_metrics": category_metrics,
            "error_analysis": error_analysis,
            "samples": [asdict(sr) for sr in sample_results],
        }

    def run_threshold_sweep(
        self, y_true: List[int], scores: List[float], step: float = 0.01
    ) -> Dict[str, Any]:
        """Sweeps threshold from 0.00 to 1.00 and identifies F1-optimal and Youden-J optimal thresholds."""
        sweep_points = []
        num_steps = int(round(1.0 / step)) + 1

        best_f1_val = -1.0
        best_f1_threshold = 0.35

        best_youden_val = -2.0
        best_youden_threshold = 0.35

        for i in range(num_steps):
            thresh = round(i * step, 2)
            preds = [1 if s >= thresh else 0 for s in scores]

            tp, tn, fp, fn = compute_confusion_matrix(y_true, preds)
            prec = compute_precision(tp, fp)
            rec = compute_recall(tp, fn)
            spec = compute_specificity(tn, fp)
            f1 = compute_f1(prec, rec)

            fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            fnr = (fn / (fn + tp)) if (fn + tp) > 0 else 0.0
            youden_j = (rec + spec - 1.0) if (rec is not None and spec is not None) else None

            point = {
                "threshold": thresh,
                "precision": round(prec, 4) if prec is not None else None,
                "recall": round(rec, 4) if rec is not None else None,
                "specificity": round(spec, 4) if spec is not None else None,
                "f1": round(f1, 4) if f1 is not None else None,
                "false_positive_rate": round(fpr, 4),
                "false_negative_rate": round(fnr, 4),
                "youden_j": round(youden_j, 4) if youden_j is not None else None,
            }
            sweep_points.append(point)

            if f1 is not None and f1 > best_f1_val:
                best_f1_val = f1
                best_f1_threshold = thresh

            if youden_j is not None and youden_j > best_youden_val:
                best_youden_val = youden_j
                best_youden_threshold = thresh

        return {
            "sweep_points": sweep_points,
            "optimal_f1_threshold": best_f1_threshold,
            "optimal_f1_score": round(best_f1_val, 4) if best_f1_val >= 0 else None,
            "optimal_youden_j_threshold": best_youden_threshold,
            "optimal_youden_j_score": (
                round(best_youden_val, 4) if best_youden_val > -2.0 else None
            ),
            "current_production_threshold": 0.35,
        }

    def run_availability_analysis(
        self, sample_results: List[EvaluationSampleResult]
    ) -> Dict[str, Any]:
        """Groups samples by pillar availability and computes metrics for each condition."""
        groups: Dict[str, List[EvaluationSampleResult]] = {
            "all_pillars_available": [],
            "p2_unavailable": [],
            "p3_unavailable": [],
            "p2_and_p3_unavailable": [],
        }

        for sr in sample_results:
            if sr.p2_available and sr.p3_available:
                groups["all_pillars_available"].append(sr)
            elif not sr.p2_available and sr.p3_available:
                groups["p2_unavailable"].append(sr)
            elif sr.p2_available and not sr.p3_available:
                groups["p3_unavailable"].append(sr)
            elif not sr.p2_available and not sr.p3_available:
                groups["p2_and_p3_unavailable"].append(sr)

        analysis = {}
        for group_name, items in groups.items():
            if not items:
                analysis[group_name] = {
                    "sample_count": 0,
                    "mean_h_score": None,
                    "metrics": None,
                }
            else:
                y_true = [item.ground_truth for item in items]
                scores = [item.h_score for item in items]
                preds = [0 if item.predicted_risk == "VERIFIED" else 1 for item in items]
                mean_h = sum(scores) / len(scores)
                metrics = compute_all_metrics(y_true, preds, scores)

                analysis[group_name] = {
                    "sample_count": len(items),
                    "mean_h_score": round(mean_h, 4),
                    "metrics": metrics,
                }

        return analysis

    def run_error_analysis(
        self,
        samples: List[BenchmarkSample],
        sample_results: List[EvaluationSampleResult],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extracts false positives, false negatives, and computes category-level metrics."""
        sample_map = {s.id: s for s in samples}

        false_positives = []
        false_negatives = []
        by_category: Dict[str, List[EvaluationSampleResult]] = {}

        for sr in sample_results:
            cat = sr.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(sr)

            pred_binary = 0 if sr.predicted_risk == "VERIFIED" else 1
            sample = sample_map.get(sr.sample_id)

            if sr.ground_truth == 0 and pred_binary == 1:
                # False Positive: factual statement flagged as hallucinated
                false_positives.append(
                    {
                        "sample_id": sr.sample_id,
                        "prompt": sample.prompt if sample else "",
                        "response": sample.response if sample else "",
                        "ground_truth": 0,
                        "predicted_risk": sr.predicted_risk,
                        "h_score": sr.h_score,
                        "factual_error": sr.p1_factual_error,
                        "confidence_gap": sr.p2_confidence_gap,
                        "consistency_failure": sr.p3_consistency_failure,
                        "effective_weights": sr.effective_weights,
                        "category": sr.category,
                    }
                )
            elif sr.ground_truth == 1 and pred_binary == 0:
                # False Negative: hallucinated statement marked verified
                false_negatives.append(
                    {
                        "sample_id": sr.sample_id,
                        "prompt": sample.prompt if sample else "",
                        "response": sample.response if sample else "",
                        "ground_truth": 1,
                        "predicted_risk": sr.predicted_risk,
                        "h_score": sr.h_score,
                        "factual_error": sr.p1_factual_error,
                        "confidence_gap": sr.p2_confidence_gap,
                        "consistency_failure": sr.p3_consistency_failure,
                        "effective_weights": sr.effective_weights,
                        "category": sr.category,
                    }
                )

        category_metrics = {}
        for cat, items in by_category.items():
            y_true = [item.ground_truth for item in items]
            scores = [item.h_score for item in items]
            preds = [0 if item.predicted_risk == "VERIFIED" else 1 for item in items]
            metrics = compute_all_metrics(y_true, preds, scores)
            category_metrics[cat] = {
                "sample_count": len(items),
                "metrics": metrics,
            }

        error_analysis = {
            "false_positive_count": len(false_positives),
            "false_negative_count": len(false_negatives),
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }

        return category_metrics, error_analysis

    def save_results(
        self, results: Dict[str, Any], output_path: Union[str, Path]
    ) -> Path:
        """Persists evaluation results dictionary as formatted JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return path
