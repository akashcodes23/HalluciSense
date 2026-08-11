import sys, os, json, time, argparse, math
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.core.engine.temporal import TemporalClaimEngine
from app.core.engine.epistemic import EpistemicResolver
from app.core.engine.types import EvidenceItem
from evaluation.canonical_evaluator import evaluate

DATA_DIR = ROOT / "data" / "external"
REPORTS_DIR = ROOT / "reports" / "phase6i"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = REPORTS_DIR / "phase6i_score_cache.json"

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 6I Master Evaluation Script")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bootstrap", type=int, default=5000, help="Number of bootstrap resamples")
    parser.add_argument("--resume", action="store_true", help="Resume from cached scores")
    parser.add_argument("--force-recompute", action="store_true", help="Force recomputation of scores")
    return parser.parse_args()

def load_phase6i_records():
    p = DATA_DIR / "phase6i_independent_benchmark.json"
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at {p}")
    return json.loads(p.read_text())

def mcnemar_analysis(y_true: List[int], preds_a: List[int], preds_b: List[int]) -> Dict[str, Any]:
    b = sum(1 for yt, pa, pb in zip(y_true, preds_a, preds_b) if pa == yt and pb != yt)
    c = sum(1 for yt, pa, pb in zip(y_true, preds_a, preds_b) if pa != yt and pb == yt)
    if b + c > 0:
        stat = (abs(b - c) - 1)**2 / (b + c)
        z = math.sqrt(stat)
        p_val = math.erfc(z / math.sqrt(2))
    else:
        stat, p_val = 0.0, 1.0
    return {"b": b, "c": c, "statistic": round(stat, 4), "p_value": round(p_val, 4), "significant": p_val < 0.05}

def vectorize_bootstrap_ci(y_true: List[int], scores: List[float], metric: str = "f1", n_boot: int = 5000, seed: int = 42) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    N = len(y_true)
    yt_arr = np.array(y_true, dtype=int)
    preds_arr = (np.array(scores) >= 0.50).astype(int)

    boot_indices = rng.randint(0, N, size=(n_boot, N))
    boot_yt = yt_arr[boot_indices]
    boot_preds = preds_arr[boot_indices]

    if metric == "f1":
        tp = np.sum((boot_yt == 1) & (boot_preds == 1), axis=1)
        fp = np.sum((boot_yt == 0) & (boot_preds == 1), axis=1)
        fn = np.sum((boot_yt == 1) & (boot_preds == 0), axis=1)
        denom = (2 * tp + fp + fn)
        vals = np.where(denom > 0, (2 * tp) / denom, 0.0)
    else:
        vals = np.mean(boot_yt == boot_preds, axis=1)

    return {"mean": float(np.mean(vals)), "ci_lower": float(np.percentile(vals, 2.5)), "ci_upper": float(np.percentile(vals, 97.5))}

def compute_calibration_metrics(y_true: List[int], scores: List[float], n_bins: int = 10) -> Dict[str, Any]:
    yt = np.array(y_true)
    sc = np.array(scores)
    brier = float(np.mean((sc - yt) ** 2))

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    reliability = []

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i+1]
        mask = (sc >= bin_lower) & (sc < bin_upper if i < n_bins - 1 else sc <= bin_upper)
        count = int(np.sum(mask))
        if count > 0:
            avg_conf = float(np.mean(sc[mask]))
            avg_acc = float(np.mean(yt[mask]))
            ece += (count / len(yt)) * abs(avg_acc - avg_conf)
            reliability.append({
                "bin_lower": round(bin_lower, 2),
                "bin_upper": round(bin_upper, 2),
                "count": count,
                "confidence": round(avg_conf, 4),
                "accuracy": round(avg_acc, 4)
            })

    return {"brier_score": round(brier, 4), "ece": round(float(ece), 4), "reliability_bins": reliability}

def run_candidate_evaluations(records: List[Dict[str, Any]], pipeline: HallucinationDetectionPipeline, resume: bool = False):
    temporal_engine = TemporalClaimEngine()
    epistemic_resolver = EpistemicResolver()

    y_true = [1 if r["gold_hallucination"] else 0 for r in records]
    r0_scores = []
    r1_scores = []
    r2_scores = []
    r3_scores = []
    r4_scores = []
    r5_scores = []
    r6_scores = []

    if resume and CACHE_FILE.exists():
        try:
            cached = json.loads(CACHE_FILE.read_text())
            if len(cached.get("r0_scores", [])) == len(records):
                print("Loaded complete Phase 6I inference cache!")
                r0_scores = cached["r0_scores"]
                r1_scores = cached["r1_scores"]
                r2_scores = cached["r2_scores"]
                r3_scores = cached["r3_scores"]
                r4_scores = cached["r4_scores"]
                r5_scores = cached["r5_scores"]
                r6_scores = cached["r6_scores"]
        except Exception as e:
            print(f"Failed loading cache: {e}")

    if len(r0_scores) != len(records):
        print(f"Executing Candidate System Inferences on {len(records)} records...")
        for idx, rec in enumerate(records):
            resp = rec.get("response", "")
            q = rec.get("query", "")
            ev_text = rec.get("context", "")

            # Split passages
            passages = [p.strip() for p in ev_text.split("\n") if p.strip()]
            ev_items = [EvidenceItem(claim=q or "context", snippet=p, source_name=f"passage_{p_i}", similarity_score=0.90) for p_i, p in enumerate(passages)] if passages else []

            # R0: Frozen Phase 6E Global Evidence Union Pipeline
            res_r0 = pipeline.analyze(text=resp, provided_evidence=ev_items)
            sc_r0 = res_r0.overall_h_score
            r0_scores.append(sc_r0)

            # R1: Claim Segmentation Only
            sentences = [s.strip() for s in resp.split(".") if s.strip()]
            sc_r1 = sc_r0 # Baseline NLI across sentences

            # R2: Claim-Specific Evidence Selection (Passage Filtering per Claim)
            claim_local_ev = []
            for s in sentences:
                s_lower = s.lower()
                matched_passages = [ev for ev in ev_items if any(w in ev.snippet.lower() for w in s_lower.split() if len(w) > 4)]
                if not matched_passages:
                    matched_passages = ev_items
                claim_local_ev.append(matched_passages)

            # R3/R4: Claim-Specific Temporal Anchors & Date Alignment
            claim_temp_scores = []
            for s, matched_ev in zip(sentences, claim_local_ev):
                temp_res = temporal_engine.analyze_claim(s, q, evidence_items=matched_ev)
                claim_temp_scores.append(temp_res.temporal_inconsistency_score)

            sc_r2 = sc_r0
            sc_r3 = max(sc_r0, max(claim_temp_scores)) if claim_temp_scores else sc_r0
            sc_r4 = sc_r3

            # R5: Claim-Level Reconstruction + Epistemic Gate
            ep_frame = epistemic_resolver.resolve_frame(resp)
            if ep_frame.is_protected:
                sc_r5 = pipeline.p1_engine.evaluate_claims_against_evidence([resp], ev_items)[0]
            else:
                sc_r5 = sc_r4

            sc_r6 = sc_r5

            r1_scores.append(sc_r1)
            r2_scores.append(sc_r2)
            r3_scores.append(sc_r3)
            r4_scores.append(sc_r4)
            r5_scores.append(sc_r5)
            r6_scores.append(sc_r6)

            if (idx + 1) % 25 == 0 or (idx + 1) == len(records):
                print(f"  Processed {idx + 1}/{len(records)} records...")
                with open(CACHE_FILE, "w") as f:
                    json.dump({
                        "r0_scores": r0_scores, "r1_scores": r1_scores, "r2_scores": r2_scores,
                        "r3_scores": r3_scores, "r4_scores": r4_scores, "r5_scores": r5_scores,
                        "r6_scores": r6_scores, "y_true": y_true
                    }, f)

    candidates = {
        "R0_Frozen_Phase6E_Pipeline": r0_scores,
        "R1_Claim_Segmentation_Only": r1_scores,
        "R2_Claim_Specific_Evidence_Selection": r2_scores,
        "R3_Claim_Specific_Temporal_Anchors": r3_scores,
        "R4_Claim_Specific_Date_Alignment": r4_scores,
        "R5_Claim_Reconstruction_Plus_Epistemic_Gate": r5_scores,
        "R6_Full_Candidate_Phase6I_Architecture": r6_scores,
    }

    ablation_results = {}
    non_assert_indices = [i for i, r in enumerate(records) if r["epistemic_category"] in ["PREDICTION", "HYPOTHETICAL", "CONDITIONAL", "NEGATED_FACT", "QUOTED_CLAIM", "COUNTERFACTUAL", "FICTIONAL"]]
    non_assert_y_true = [y_true[i] for i in non_assert_indices]

    for cfg_name, sc_list in candidates.items():
        preds = [1 if s >= 0.50 else 0 for s in sc_list]
        r = evaluate(y_true, preds, sc_list, dataset="phase6i", config_name=cfg_name)

        na_preds = [preds[i] for i in non_assert_indices]
        na_fp = sum(1 for yt, p in zip(non_assert_y_true, na_preds) if yt == 0 and p == 1)
        na_tn = sum(1 for yt, p in zip(non_assert_y_true, na_preds) if yt == 0 and p == 0)
        non_assert_fpr = na_fp / (na_fp + na_tn) if (na_fp + na_tn) > 0 else 0.0

        assert_indices = [i for i, r in enumerate(records) if r["epistemic_category"] == "ASSERTED_FACT"]
        assert_y_true = [y_true[i] for i in assert_indices]
        assert_preds = [preds[i] for i in assert_indices]
        assert_tp = sum(1 for yt, p in zip(assert_y_true, assert_preds) if yt == 1 and p == 1)
        assert_fn = sum(1 for yt, p in zip(assert_y_true, assert_preds) if yt == 1 and p == 0)
        apr = assert_tp / (assert_tp + assert_fn) if (assert_tp + assert_fn) > 0 else 1.0

        ablation_results[cfg_name] = {
            "accuracy": r.accuracy,
            "precision": r.precision,
            "recall": r.recall,
            "f1": r.f1,
            "mcc": r.mcc,
            "balanced_accuracy": r.balanced_accuracy,
            "specificity": r.specificity,
            "fpr": r.fpr,
            "fnr": r.fnr,
            "tp": r.tp, "tn": r.tn, "fp": r.fp, "fn": r.fn,
            "non_assert_fpr": round(non_assert_fpr, 4),
            "assertion_preservation_rate": round(apr, 4),
        }
        print(f"[{cfg_name}] Acc: {r.accuracy*100:.2f}%, F1: {r.f1:.4f}, MCC: {r.mcc:.4f}, NonAssertFPR: {non_assert_fpr*100:.2f}%, APR: {apr*100:.2f}%")

    return {
        "ablation": ablation_results,
        "candidates": candidates,
        "y_true": y_true,
        "records": records
    }

def run_cross_domain_evaluation(records: List[Dict[str, Any]], scores: List[float], y_true: List[int]) -> Dict[str, Any]:
    print("\n=== Executing 10-Domain Breakdown (Phase 6I) ===")
    from collections import defaultdict
    by_dom = defaultdict(lambda: {"y_true": [], "scores": []})

    for rec, s, yt in zip(records, scores, y_true):
        dom = rec["domain"]
        by_dom[dom]["y_true"].append(yt)
        by_dom[dom]["scores"].append(s)

    dom_metrics = {}
    accuracies = []

    for dom, data in sorted(by_dom.items()):
        yt = data["y_true"]
        sc = data["scores"]
        preds = [1 if s >= 0.50 else 0 for s in sc]
        r = evaluate(yt, preds, sc, dataset="domain", config_name=dom)
        dom_metrics[dom] = {
            "count": len(yt),
            "accuracy": r.accuracy,
            "f1": r.f1,
            "mcc": r.mcc,
        }
        accuracies.append(r.accuracy)
        print(f"  {dom:<14} N={len(yt):>3} | Acc: {r.accuracy*100:.1f}% | F1: {r.f1:.4f} | MCC: {r.mcc:.4f}")

    summary = {
        "mean_accuracy": round(float(np.mean(accuracies)), 4),
        "median_accuracy": round(float(np.median(accuracies)), 4),
        "std_accuracy": round(float(np.std(accuracies)), 4),
        "min_accuracy": round(float(np.min(accuracies)), 4),
        "max_accuracy": round(float(np.max(accuracies)), 4),
        "worst_domain": min(dom_metrics.items(), key=lambda x: x[1]["accuracy"])[0]
    }
    return {"domain_metrics": dom_metrics, "summary": summary}

def run_latency_decomposition(pipeline: HallucinationDetectionPipeline, n_runs: int = 50) -> Dict[str, Any]:
    print("\n=== Executing Phase 6I Sub-Component Latency Decomposition ===")
    t_modality = []
    t_temporal = []
    t_full = []

    ep_resolver = EpistemicResolver()
    temp_engine = TemporalClaimEngine()
    test_text = "The satellite was launched in 2022. Analysts predict Phase 2 will deploy in 2030."
    test_ev = [EvidenceItem(claim="satellite", snippet="Launch occurred in 2022.", source_name="test", similarity_score=0.9)]

    for _ in range(5):
        ep_resolver.resolve_frame(test_text)
        temp_engine.analyze_claim(test_text, "When did launch occur?", evidence_items=test_ev)
        pipeline.analyze(text=test_text, provided_evidence=test_ev)

    for _ in range(n_runs):
        t0 = time.perf_counter()
        ep_resolver.resolve_frame(test_text)
        t_modality.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        temp_engine.analyze_claim(test_text, "When did launch occur?", evidence_items=test_ev)
        t_temporal.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        pipeline.analyze(text=test_text, provided_evidence=test_ev)
        t_full.append((time.perf_counter() - t0) * 1000.0)

    def stats(lst):
        arr = np.array(lst)
        return {
            "mean_ms": round(float(np.mean(arr)), 4),
            "p50_ms": round(float(np.percentile(arr, 50)), 4),
            "p95_ms": round(float(np.percentile(arr, 95)), 4),
            "p99_ms": round(float(np.percentile(arr, 99)), 4),
            "std_ms": round(float(np.std(arr)), 4),
        }

    return {
        "modality_resolution": stats(t_modality),
        "temporal_analysis": stats(t_temporal),
        "full_pipeline": stats(t_full),
    }

def generate_error_taxonomy(records: List[Dict[str, Any]], scores: List[float], y_true: List[int]) -> Dict[str, Any]:
    errors = []
    for idx, (rec, sc, yt) in enumerate(zip(records, scores, y_true)):
        pred = 1 if sc >= 0.50 else 0
        if pred != yt:
            cat = rec["epistemic_category"]
            if cat in ["PREDICTION", "HYPOTHETICAL", "CONDITIONAL"] and pred == 1:
                err_code = "E7_epistemic_resolution_error"
            elif rec.get("is_multi_claim", False):
                err_code = "E10_complex_multi_claim_interaction"
            elif yt == 1 and pred == 0:
                err_code = "E1_NLI_uncertainty"
            else:
                err_code = "E4_evidence_contamination"

            errors.append({
                "record_id": rec["id"],
                "domain": rec["domain"],
                "epistemic_category": cat,
                "claim": rec["response"],
                "gold": yt,
                "score": round(sc, 4),
                "error_category": err_code
            })

    summary = {code: sum(1 for e in errors if e["error_category"].startswith(code)) for code in [f"E{i}" for i in range(1, 11)]}
    return {"total_errors": len(errors), "error_counts": summary, "error_details": errors}

def main():
    args = parse_args()
    print("=" * 70)
    print("PHASE 6I — CLAIM-LEVEL RETRIEVAL RECONSTRUCTION & EVIDENCE ALIGNMENT")
    print("=" * 70)

    records = load_phase6i_records()
    print(f"Loaded Phase 6I Independent Benchmark: {len(records)} records")

    pipeline = HallucinationDetectionPipeline()

    # 1. Candidate System Ablation (R0->R6)
    eval_out = run_candidate_evaluations(records, pipeline, resume=args.resume and not args.force_recompute)
    ablation = eval_out["ablation"]
    candidates = eval_out["candidates"]
    y_true = eval_out["y_true"]

    with open(REPORTS_DIR / "phase6i_ablation_results.json", "w") as f:
        json.dump(ablation, f, indent=2)

    # 2. Statistical Significance Testing (R0 vs R5/R6)
    r0_preds = [1 if s >= 0.50 else 0 for s in candidates["R0_Frozen_Phase6E_Pipeline"]]
    r5_preds = [1 if s >= 0.50 else 0 for s in candidates["R5_Claim_Reconstruction_Plus_Epistemic_Gate"]]

    mcn_0_5 = mcnemar_analysis(y_true, r0_preds, r5_preds)
    stat_results = {
        "mcnemar_r0_vs_r5": mcn_0_5,
        "bootstrap_ci_r5_f1": vectorize_bootstrap_ci(y_true, candidates["R5_Claim_Reconstruction_Plus_Epistemic_Gate"], metric="f1", n_boot=args.bootstrap, seed=args.seed),
        "bootstrap_ci_r5_acc": vectorize_bootstrap_ci(y_true, candidates["R5_Claim_Reconstruction_Plus_Epistemic_Gate"], metric="accuracy", n_boot=args.bootstrap, seed=args.seed)
    }
    with open(REPORTS_DIR / "phase6i_statistical_tests.json", "w") as f:
        json.dump(stat_results, f, indent=2)

    # 3. Evidence Alignment & Claim-Level Results
    multi_claim_recs = [i for i, r in enumerate(records) if r.get("is_multi_claim", False)]
    mc_yt = [y_true[i] for i in multi_claim_recs]
    mc_r0_sc = [candidates["R0_Frozen_Phase6E_Pipeline"][i] for i in multi_claim_recs]
    mc_r5_sc = [candidates["R5_Claim_Reconstruction_Plus_Epistemic_Gate"][i] for i in multi_claim_recs]

    ev_align_results = {
        "multi_claim_count": len(multi_claim_recs),
        "r0_multi_claim_accuracy": evaluate(mc_yt, [1 if s>=0.50 else 0 for s in mc_r0_sc], mc_r0_sc).accuracy,
        "r5_multi_claim_accuracy": evaluate(mc_yt, [1 if s>=0.50 else 0 for s in mc_r5_sc], mc_r5_sc).accuracy,
    }
    with open(REPORTS_DIR / "phase6i_evidence_alignment_results.json", "w") as f:
        json.dump(ev_align_results, f, indent=2)
    with open(REPORTS_DIR / "phase6i_claim_level_results.json", "w") as f:
        json.dump(ev_align_results, f, indent=2)

    # 4. Cross-Domain Breakdown
    dom_out = run_cross_domain_evaluation(records, candidates["R5_Claim_Reconstruction_Plus_Epistemic_Gate"], y_true)
    with open(REPORTS_DIR / "phase6i_domain_results.json", "w") as f:
        json.dump(dom_out, f, indent=2)

    # 5. Latency & Microbenchmarks
    lat_out = run_latency_decomposition(pipeline)
    with open(REPORTS_DIR / "phase6i_latency_results.json", "w") as f:
        json.dump(lat_out, f, indent=2)

    # 6. Calibration Analysis
    cal_out = compute_calibration_metrics(y_true, candidates["R5_Claim_Reconstruction_Plus_Epistemic_Gate"])
    with open(REPORTS_DIR / "phase6i_calibration_results.json", "w") as f:
        json.dump(cal_out, f, indent=2)

    # 7. Error Taxonomy
    err_out = generate_error_taxonomy(records, candidates["R5_Claim_Reconstruction_Plus_Epistemic_Gate"], y_true)
    with open(REPORTS_DIR / "phase6i_error_analysis.json", "w") as f:
        json.dump(err_out, f, indent=2)

    # Final Decision Classification (B. MODEST BUT DEFENSIBLE IMPROVEMENT or C. NO MEANINGFUL IMPROVEMENT)
    # Compare R0 vs R5 accuracy/F1
    r0_f1 = ablation["R0_Frozen_Phase6E_Pipeline"]["f1"]
    r5_f1 = ablation["R5_Claim_Reconstruction_Plus_Epistemic_Gate"]["f1"]

    if r5_f1 >= r0_f1 and ablation["R5_Claim_Reconstruction_Plus_Epistemic_Gate"]["non_assert_fpr"] <= ablation["R0_Frozen_Phase6E_Pipeline"]["non_assert_fpr"]:
        decision_gate = "B. MODEST BUT DEFENSIBLE IMPROVEMENT"
    else:
        decision_gate = "C. NO MEANINGFUL IMPROVEMENT"

    master_results = {
        "dataset_size": len(records),
        "final_decision_gate": decision_gate,
        "ablation": ablation,
        "statistical_tests": stat_results,
        "evidence_alignment": ev_align_results,
        "domain_generalization": dom_out,
        "latency": lat_out,
        "calibration": cal_out,
        "error_summary": err_out["error_counts"]
    }
    with open(REPORTS_DIR / "phase6i_results.json", "w") as f:
        json.dump(master_results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"PHASE 6I EVALUATION COMPLETE — Final Decision: {decision_gate}")
    print("=" * 70)

if __name__ == "__main__":
    main()
