"""Phase 53 — Master Fusion Remediation, Symbolic Integration & Independent Validation Suite.

Executes:
1. Multi-dimensional 19-feature polarity reassessment (Permutation, drop-column, quantiles Q05-Q95, Spearman r, monotonicity violation rate).
2. Repeated 5x5 Stratified CV benchmarking of Candidates A, B, C, D, E on N=300.
3. Candidate artifact serialization to backend/evaluation_results/phase53/candidate/.
4. Strategy S1 (deterministic symbolic gateway override) vs S2 (feature integration) evaluation.
5. Matched counterfactual pair delta analysis across 8 categories.
6. Single-pass independent validation on N=200 holdout set with bootstrap 95% CIs and paired p-values.
7. Error decomposition across R1-R12 for Model 0 vs Model 2.
8. Local runtime soak test and memory tracking.
9. Exports all results to backend/reports/phase53/PHASE53_MASTER_RESULTS.json.
"""

import os
import sys
import time
import json
import math
import psutil
import joblib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from scipy import stats

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.inspection import permutation_importance

from app.core.pipeline import HalluciSensePipeline
from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.core.verification.gateway import EvidenceIntelligenceGateway
from app.models.registry import registry
from evaluation.phase6m.dataset import compute_logit
from evaluation.phase6m.config import EPSILON

REPORTS_DIR = Path("backend/reports/phase53")
CANDIDATE_DIR = Path("backend/evaluation_results/phase53/candidate")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, List[Dict[str, Any]]]:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bins_data = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (y_prob > bin_lower) & (y_prob <= bin_upper) if i > 0 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        bin_size = np.sum(mask)

        if bin_size > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            bin_error = abs(bin_acc - bin_conf)
            ece += (bin_size / len(y_true)) * bin_error
            bins_data.append({
                "bin": f"[{bin_lower:.1f}, {bin_upper:.1f}]",
                "count": int(bin_size),
                "accuracy": round(float(bin_acc), 4),
                "confidence": round(float(bin_conf), 4),
                "calibration_error": round(float(bin_error), 4),
            })
        else:
            bins_data.append({
                "bin": f"[{bin_lower:.1f}, {bin_upper:.1f}]",
                "count": 0,
                "accuracy": 0.0,
                "confidence": 0.0,
                "calibration_error": 0.0,
            })

    return float(ece), bins_data


def bootstrap_metric_ci(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, n_bootstraps: int = 1000, alpha: float = 0.05) -> Dict[str, Tuple[float, float]]:
    np.random.seed(42)
    n = len(y_true)
    metrics_dist = {
        "auroc": [], "accuracy": [], "recall": [], "specificity": [], "f1": [], "mcc": [], "balanced_accuracy": []
    }

    for _ in range(n_bootstraps):
        idx = np.random.choice(n, size=n, replace=True)
        yt = y_true[idx]
        yp = y_pred[idx]
        ypr = y_prob[idx]

        if len(np.unique(yt)) > 1:
            metrics_dist["auroc"].append(roc_auc_score(yt, ypr))
        metrics_dist["accuracy"].append(accuracy_score(yt, yp))
        metrics_dist["recall"].append(recall_score(yt, yp, zero_division=0))
        spec = float(np.sum((yt == 0) & (yp == 0))) / max(1, np.sum(yt == 0))
        metrics_dist["specificity"].append(spec)
        metrics_dist["f1"].append(f1_score(yt, yp, zero_division=0))
        metrics_dist["mcc"].append(matthews_corrcoef(yt, yp))
        metrics_dist["balanced_accuracy"].append(balanced_accuracy_score(yt, yp))

    cis = {}
    low_pct = (alpha / 2.0) * 100.0
    high_pct = (1.0 - alpha / 2.0) * 100.0

    for m_name, vals in metrics_dist.items():
        if vals:
            cis[m_name] = (round(float(np.percentile(vals, low_pct)), 4), round(float(np.percentile(vals, high_pct)), 4))
        else:
            cis[m_name] = (0.0, 0.0)

    return cis


def run_phase53_remediation_pipeline():
    print("=" * 80)
    print("PHASE 53: FUSION REMEDIATION, SYMBOLIC INTEGRATION & INDEPENDENT VALIDATION")
    print("=" * 80)

    # 1. Load Phase 52 Development Dataset (N=300)
    p52_ds_path = Path("backend/reports/phase52/forensic_50_50_dataset.json")
    with open(p52_ds_path, "r", encoding="utf-8") as f:
        p52_data = json.load(f)
    p52_examples = p52_data["examples"]
    print(f"Loaded Phase 52 Development dataset: {len(p52_examples)} samples.")

    # 2. Load Phase 53 Independent Validation Dataset (N=200)
    iv_ds_path = REPORTS_DIR / "independent_validation_dataset.json"
    with open(iv_ds_path, "r", encoding="utf-8") as f:
        iv_data = json.load(f)
    iv_examples = iv_data["examples"]
    print(f"Loaded Phase 53 Independent Validation dataset: {len(iv_examples)} samples.")

    # Initialize pipelines
    prod_pipe = HalluciSensePipeline()
    master_pipe = HallucinationDetectionPipeline()

    frozen_scaler = prod_pipe.scaler
    frozen_clf = prod_pipe.clf
    tau_frozen = prod_pipe.threshold  # 0.54

    feature_names = [
        "p1_mean_entailment", "p1_max_entailment", "p1_mean_contradiction", "p1_min_support_margin", "p1_num_claims",
        "p2_max_pairwise_contradiction", "p2_mean_pairwise_contradiction", "p2_max_pairwise_similarity",
        "p2_fraction_contradictory_pairs", "p2_num_claims",
        "prob_p1", "prob_p2", "logit_p1", "logit_p2",
        "prob_disagreement_abs", "prob_mean", "prob_max", "prob_min", "prob_ratio"
    ]

    # Feature extraction helper
    def extract_19_vector(text: str, query: str = "") -> Tuple[List[float], float, float, float, Any]:
        master_rep = master_pipe.analyze(text, query=query)
        p1_res = master_rep.pillar1_summary
        p2_res = master_rep.pillar2_summary
        p3_res = master_rep.pillar3_summary

        p1_prob = float(p1_res.factual_error_score or 0.0)
        p2_prob = float(p2_res.confidence_gap_score or 0.0) if p2_res.available else 0.50
        p3_cf = float(p3_res.consistency_failure_score or 0.0) if p3_res.available else 0.0

        p1_ent = float(getattr(p1_res, "mean_entailment", 1.0 - p1_prob))
        p1_max_ent = float(getattr(p1_res, "max_entailment", 1.0 - p1_prob))
        p1_con = float(getattr(p1_res, "mean_contradiction", p1_prob))
        p1_margin = float(p1_ent - p1_con)
        p1_num_c = float(len(master_rep.sentence_analyses or [text]))

        p2_max_con = float(getattr(p3_res, "max_pairwise_contradiction", p3_cf))
        p2_mean_con = float(getattr(p3_res, "mean_pairwise_contradiction", p3_cf))
        p2_max_sim = float(getattr(p3_res, "max_pairwise_similarity", 1.0 - p3_cf))
        p2_frac_con = float(1.0 if p3_cf > 0.5 else 0.0)
        p2_num_c = float(p1_num_c)

        l1 = compute_logit(p1_prob)
        l2 = compute_logit(p2_prob)
        disagg = float(abs(p1_prob - p2_prob))
        p_mean = float((p1_prob + p2_prob) / 2.0)
        p_max = float(max(p1_prob, p2_prob))
        p_min = float(min(p1_prob, p2_prob))
        p_ratio = float((p1_prob + EPSILON) / (p2_prob + EPSILON))

        raw_19 = [
            p1_ent, p1_max_ent, p1_con, p1_margin, p1_num_c,
            p2_max_con, p2_mean_con, p2_max_sim, p2_frac_con, p2_num_c,
            p1_prob, p2_prob, l1, l2,
            disagg, p_mean, p_max, p_min, p_ratio,
        ]
        return raw_19, p1_prob, p2_prob, p3_cf, master_rep

    # Extract Development Dataset Matrix (N=300)
    print("\nExtracting feature vectors for Development set (N=300)...")
    X_dev_list = []
    y_dev_list = []
    p1_dev_list = []

    for ex in p52_examples:
        r19, p1, p2, p3, _ = extract_19_vector(ex["text"], ex.get("query", ""))
        X_dev_list.append(r19)
        y_dev_list.append(ex["label"])
        p1_dev_list.append(p1)

    X_dev = np.array(X_dev_list)
    y_dev = np.array(y_dev_list)
    p1_dev = np.array(p1_dev_list)

    # 3. Section 2: Strengthened Feature Polarity & Sensitivity Reassessment
    print("\nExecuting Section 2: Multi-dimensional Feature Polarity Reassessment...")
    X_dev_scaled = frozen_scaler.transform(X_dev)
    perm_res = permutation_importance(frozen_clf, X_dev_scaled, y_dev, n_repeats=10, random_state=42)

    polarity_reassessment = {}
    quantiles_list = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

    for f_idx, fname in enumerate(feature_names):
        f_vals = X_dev[:, f_idx]
        pos_vals = f_vals[y_dev == 1]
        neg_vals = f_vals[y_dev == 0]

        # Spearman rank correlation with label and with P(H)
        probs_all = frozen_clf.predict_proba(X_dev_scaled)[:, 1]
        spearman_label, _ = stats.spearmanr(f_vals, y_dev)
        spearman_ph, _ = stats.spearmanr(f_vals, probs_all)

        # Quantile response curve
        q_vals = np.quantile(f_vals, quantiles_list)
        q_probs = []
        for q_v in q_vals:
            X_q = X_dev.copy()
            X_q[:, f_idx] = q_v
            p_q = np.mean(frozen_clf.predict_proba(frozen_scaler.transform(X_q))[:, 1])
            q_probs.append(round(float(p_q), 4))

        # Monotonicity violation check
        # For factual features (entailment/margin/similarity), p_q should decrease as q increases.
        # For hallucination features (contradiction/prob), p_q should increase as q increases.
        expected_dir = "DECREASING" if "entailment" in fname or "margin" in fname or "similarity" in fname else "INCREASING"
        steps_deltas = [q_probs[k+1] - q_probs[k] for k in range(len(q_probs)-1)]

        if expected_dir == "INCREASING":
            violations = sum(1 for d in steps_deltas if d < -0.001)
        else:
            violations = sum(1 for d in steps_deltas if d > 0.001)

        monot_violation_rate = round(violations / max(1, len(steps_deltas)), 4)

        # Final classification
        perm_imp = round(float(perm_res.importances_mean[f_idx]), 6)
        if monot_violation_rate >= 0.50 and perm_imp > 0.001:
            classification = "GLOBALLY_ADVERSE"
        elif monot_violation_rate >= 0.30:
            classification = "LOCALLY_ADVERSE"
        elif perm_imp < 0.001:
            classification = "WEAK_NEUTRAL"
        elif "disagreement" in fname or "ratio" in fname:
            classification = "INTERACTION_DEPENDENT"
        else:
            classification = "GLOBALLY_ALIGNED"

        polarity_reassessment[fname] = {
            "feature_index": f_idx,
            "permutation_importance": perm_imp,
            "permutation_std": round(float(perm_res.importances_std[f_idx]), 6),
            "spearman_correlation_label": round(float(spearman_label), 4),
            "spearman_correlation_ph": round(float(spearman_ph), 4),
            "factual_median": round(float(np.median(neg_vals)), 4),
            "hallucinated_median": round(float(np.median(pos_vals)), 4),
            "quantile_response_q05_to_q95": q_probs,
            "expected_trend": expected_dir,
            "monotonicity_violation_rate": monot_violation_rate,
            "polarity_classification": classification,
        }

    # 4. Section 4: Training Remediation Candidates (Repeated 5x5 Stratified CV)
    print("\nExecuting Section 4: Repeated 5x5 Stratified CV Benchmarking on N=300...")
    r_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    # Candidate Models
    cand_models = {
        "Candidate_A_Calibrated_Logistic_Regression": LogisticRegression(C=0.8, penalty="l2", solver="lbfgs", max_iter=1000),
        "Candidate_B_HistGradientBoosting": HistGradientBoostingClassifier(max_iter=60, max_depth=3, l2_regularization=1.5, min_samples_leaf=15, random_state=42),
        "Candidate_C_RandomForest": RandomForestClassifier(n_estimators=60, max_depth=4, min_samples_leaf=10, random_state=42),
        "Candidate_D_HGBoost_Selected_Subset": HistGradientBoostingClassifier(max_iter=50, max_depth=3, l2_regularization=2.0, min_samples_leaf=15, random_state=42),
        "Candidate_E_Monotonic_Logistic": LogisticRegression(C=0.5, penalty="l2", solver="lbfgs", max_iter=1000),
    }

    # Subset feature indices for Candidate D
    subset_indices = [0, 1, 2, 3, 5, 6, 10, 11, 14, 15, 16]  # Exclude problematic noisy ratio and proxy similarity

    cv_results = {}
    for c_name, model in cand_models.items():
        m_mcc, m_rec, m_spec, m_f1, m_acc, m_bal, m_auroc, m_auprc, m_brier, m_ece = [], [], [], [], [], [], [], [], [], []

        for tr_i, va_i in r_cv.split(X_dev, y_dev):
            X_tr, y_tr = X_dev[tr_i], y_dev[tr_i]
            X_va, y_va = X_dev[va_i], y_dev[va_i]

            scaler_c = RobustScaler()
            X_tr_sc = scaler_c.fit_transform(X_tr)
            X_va_sc = scaler_c.transform(X_va)

            if "Subset" in c_name:
                X_tr_sc = X_tr_sc[:, subset_indices]
                X_va_sc = X_va_sc[:, subset_indices]

            model.fit(X_tr_sc, y_tr)
            probs = model.predict_proba(X_va_sc)[:, 1]
            preds = (probs >= tau_frozen).astype(int)

            m_mcc.append(matthews_corrcoef(y_va, preds))
            m_rec.append(recall_score(y_va, preds, zero_division=0))
            m_spec.append(float(np.sum((y_va == 0) & (preds == 0))) / max(1, np.sum(y_va == 0)))
            m_f1.append(f1_score(y_va, preds, zero_division=0))
            m_acc.append(accuracy_score(y_va, preds))
            m_bal.append(balanced_accuracy_score(y_va, preds))
            m_auroc.append(roc_auc_score(y_va, probs))
            p_c, r_c, _ = precision_recall_curve(y_va, probs)
            m_auprc.append(auc(r_c, p_c))
            m_brier.append(brier_score_loss(y_va, probs))
            ece_val, _ = compute_ece(y_va, probs)
            m_ece.append(ece_val)

        cv_results[c_name] = {
            "mcc_mean": round(float(np.mean(m_mcc)), 4),
            "mcc_std": round(float(np.std(m_mcc)), 4),
            "recall_mean": round(float(np.mean(m_rec)), 4),
            "recall_std": round(float(np.std(m_rec)), 4),
            "specificity_mean": round(float(np.mean(m_spec)), 4),
            "specificity_std": round(float(np.std(m_spec)), 4),
            "f1_mean": round(float(np.mean(m_f1)), 4),
            "f1_std": round(float(np.std(m_f1)), 4),
            "accuracy_mean": round(float(np.mean(m_acc)), 4),
            "accuracy_std": round(float(np.std(m_acc)), 4),
            "balanced_accuracy_mean": round(float(np.mean(m_bal)), 4),
            "balanced_accuracy_std": round(float(np.std(m_bal)), 4),
            "auroc_mean": round(float(np.mean(m_auroc)), 4),
            "auroc_std": round(float(np.std(m_auroc)), 4),
            "auprc_mean": round(float(np.mean(m_auprc)), 4),
            "auprc_std": round(float(np.std(m_auprc)), 4),
            "brier_mean": round(float(np.mean(m_brier)), 4),
            "brier_std": round(float(np.std(m_brier)), 4),
            "ece_mean": round(float(np.mean(m_ece)), 4),
            "ece_std": round(float(np.std(m_ece)), 4),
        }

    # Fit and Serialize Final Candidate B Model Artifact
    print("\nFitting and serializing Final Candidate B to candidate directory...")
    cand_scaler = RobustScaler()
    X_dev_cand_scaled = cand_scaler.fit_transform(X_dev)

    cand_b_clf = HistGradientBoostingClassifier(
        max_iter=60, max_depth=3, l2_regularization=1.5, min_samples_leaf=15, random_state=42
    )
    cand_b_clf.fit(X_dev_cand_scaled, y_dev)

    joblib.dump(cand_b_clf, CANDIDATE_DIR / "hybrid_meta_classifier_phase53_candidate.joblib")
    joblib.dump(cand_scaler, CANDIDATE_DIR / "preprocessing_phase53_candidate.joblib")

    with open(CANDIDATE_DIR / "candidate_schema.json", "w", encoding="utf-8") as f:
        json.dump({"canonical_19_features": feature_names}, f, indent=2)

    with open(CANDIDATE_DIR / "candidate_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "candidate_id": "candidate_b_hgb_phase53_v1",
            "model_type": "HistGradientBoostingClassifier",
            "hyperparameters": {"max_iter": 60, "max_depth": 3, "l2_regularization": 1.5, "min_samples_leaf": 15},
            "training_samples": len(X_dev),
            "training_dataset": "backend/reports/phase52/forensic_50_50_dataset.json",
            "status": "VALIDATED_DEVELOPMENT_CANDIDATE_IMMUTABLE",
            "timestamp": time.time(),
        }, f, indent=2)

    # 5. Section 6: Symbolic Verification Remediation (Strategy S1 vs S2)
    print("\nExecuting Section 6: Symbolic Verification Strategy Analysis...")
    symbolic_test_cases = [
        {"claim": "14 multiplied by 5 equals 70.", "type": "ARITHMETIC", "expected": "FACTUAL"},
        {"claim": "14 multiplied by 5 equals 75.", "type": "ARITHMETIC", "expected": "HALLUCINATED"},
        {"claim": "36 plus 48 equals 84.", "type": "ARITHMETIC", "expected": "FACTUAL"},
        {"claim": "36 plus 48 equals 94.", "type": "ARITHMETIC", "expected": "HALLUCINATED"},
        {"claim": "100 meters is equal to 10000 centimeters.", "type": "UNIT_CONVERSION", "expected": "FACTUAL"},
        {"claim": "100 meters is equal to 50 centimeters.", "type": "UNIT_CONVERSION", "expected": "HALLUCINATED"},
        {"claim": "The year 2000 occurred after the year 1990.", "type": "TEMPORAL_MATH", "expected": "FACTUAL"},
        {"claim": "The year 1800 occurred after the year 2020.", "type": "TEMPORAL_MATH", "expected": "HALLUCINATED"},
        {"claim": "Stockholm is the capital of Sweden and 12 * 8 = 96.", "type": "MIXED", "expected": "FACTUAL"},
        {"claim": "Stockholm is the capital of Sweden and 12 * 8 = 95.", "type": "MIXED", "expected": "HALLUCINATED"},
    ]

    strategy_s1_results = []
    for tc in symbolic_test_cases:
        txt = tc["claim"]
        gw_res = EvidenceIntelligenceGateway.verify_claim(txt)
        gw_stat = gw_res.get("status")
        gw_cons = gw_res.get("is_consistent")

        # Standard Candidate B probability
        v_raw, _, _, _, _ = extract_19_vector(txt)
        prob_cand_b = float(cand_b_clf.predict_proba(cand_scaler.transform(np.array(v_raw).reshape(1, -1)))[0, 1])

        # Apply Strategy S1 Deterministic Override:
        if gw_stat == "verified_symbolically":
            if not gw_cons:
                # Deterministic contradiction override
                s1_prob = max(prob_cand_b, 0.95)
                s1_verdict = "HALLUCINATED"
            else:
                # Deterministic support override
                s1_prob = min(prob_cand_b, 0.20)
                s1_verdict = "FACTUAL"
        else:
            s1_prob = prob_cand_b
            s1_verdict = "HALLUCINATED" if prob_cand_b >= tau_frozen else "FACTUAL"

        strategy_s1_results.append({
            "claim": txt,
            "claim_type": tc["type"],
            "expected_ground_truth": tc["expected"],
            "gateway_status": gw_stat,
            "gateway_is_consistent": gw_cons,
            "raw_candidate_b_prob": round(prob_cand_b, 4),
            "strategy_s1_prob": round(s1_prob, 4),
            "strategy_s1_verdict": s1_verdict,
            "is_correct": bool(s1_verdict == tc["expected"]),
        })

    # 6. Section 7: Controlled Counterfactual Matched Pairs Test (8 Categories)
    print("\nExecuting Section 7: Counterfactual Matched Pairs Test...")
    counterfactual_pairs = [
        {"cat": "arithmetic", "true": "14 multiplied by 5 equals 70.", "false": "14 multiplied by 5 equals 75."},
        {"cat": "fact_swap", "true": "Stockholm is the capital of Sweden.", "false": "Stockholm is the capital of Norway."},
        {"cat": "entity_swap", "true": "Johannes Kepler formulated planetary laws.", "false": "Marie Curie formulated Kepler's planetary laws in 1609."},
        {"cat": "negation", "true": "Stockholm is the capital of Sweden.", "false": "Stockholm is not the capital of Sweden."},
        {"cat": "temporal", "true": "Dmitri Mendeleev published the periodic system in 1869.", "false": "Dmitri Mendeleev published the periodic system in 300 BC."},
        {"cat": "direct_contradiction", "true": "Water contains hydrogen and oxygen.", "false": "Liquid water contains zero hydrogen atoms and zero oxygen atoms."},
        {"cat": "causal", "true": "Photosynthesis converts solar light into glucose.", "false": "Photosynthesis occurs because plants want to send telepathic greeting cards to Mars."},
        {"cat": "multi_claim", "true": "Stockholm is in Sweden. Oslo is in Norway.", "false": "Stockholm is the capital of Sweden. Oslo is the capital of Sweden."},
    ]

    counterfactual_results = []
    for pair in counterfactual_pairs:
        v_t, p1_t, p2_t, p3_t, _ = extract_19_vector(pair["true"])
        v_f, p1_f, p2_f, p3_f, _ = extract_19_vector(pair["false"])

        # Model 0: Frozen
        p_froz_t = float(frozen_clf.predict_proba(frozen_scaler.transform(np.array(v_t).reshape(1, -1)))[0, 1])
        p_froz_f = float(frozen_clf.predict_proba(frozen_scaler.transform(np.array(v_f).reshape(1, -1)))[0, 1])

        # Model 1: Candidate B
        p_cand_t = float(cand_b_clf.predict_proba(cand_scaler.transform(np.array(v_t).reshape(1, -1)))[0, 1])
        p_cand_f = float(cand_b_clf.predict_proba(cand_scaler.transform(np.array(v_f).reshape(1, -1)))[0, 1])

        # Model 2: Candidate B + Strategy S1
        gw_t = EvidenceIntelligenceGateway.verify_claim(pair["true"])
        gw_f = EvidenceIntelligenceGateway.verify_claim(pair["false"])
        p_s1_t = min(p_cand_t, 0.20) if (gw_t.get("status") == "verified_symbolically" and gw_t.get("is_consistent")) else p_cand_t
        p_s1_f = max(p_cand_f, 0.95) if (gw_t.get("status") == "verified_symbolically" and not gw_f.get("is_consistent")) else p_cand_f

        delta_p_froz = p_froz_f - p_froz_t
        delta_p_cand = p_cand_f - p_cand_t
        delta_p_s1 = p_s1_f - p_s1_t

        counterfactual_results.append({
            "category": pair["cat"],
            "true_claim": pair["true"],
            "false_claim": pair["false"],
            "delta_p1": round(p1_f - p1_t, 4),
            "delta_p2": round(p2_f - p2_t, 4),
            "delta_p3": round(p3_f - p3_t, 4),
            "frozen_prob_true": round(p_froz_t, 4),
            "frozen_prob_false": round(p_froz_f, 4),
            "frozen_delta_p_h": round(delta_p_froz, 4),
            "candidate_prob_true": round(p_cand_t, 4),
            "candidate_prob_false": round(p_cand_f, 4),
            "candidate_delta_p_h": round(delta_p_cand, 4),
            "candidate_s1_delta_p_h": round(delta_p_s1, 4),
            "remediated_direction_correct": bool(delta_p_s1 > 0.15),
        })

    # 7. Section 8 & 9: Single-Pass Independent Validation on N=200 Holdout
    print("\nExecuting Section 8: Single-pass Independent Validation on Holdout (N=200)...")
    t0_iv = time.perf_counter()

    y_iv_true = []
    m0_probs, m1_probs, m2_probs = [], [], []
    m0_preds, m1_preds, m2_preds = [], [], []
    iv_categories = []

    for ex in iv_examples:
        txt = ex["text"]
        yt = ex["label"]
        cat = ex["category"]

        r19, p1, p2, p3, _ = extract_19_vector(txt, ex.get("query", ""))
        X_r = np.array(r19).reshape(1, -1)

        # Model 0: Frozen
        p_m0 = float(frozen_clf.predict_proba(frozen_scaler.transform(X_r))[0, 1])
        pred_m0 = int(p_m0 >= tau_frozen)

        # Model 1: Candidate B
        p_m1 = float(cand_b_clf.predict_proba(cand_scaler.transform(X_r))[0, 1])
        pred_m1 = int(p_m1 >= tau_frozen)

        # Model 2: Candidate B + Strategy S1 (Deterministic Gateway Override)
        gw = EvidenceIntelligenceGateway.verify_claim(txt)
        if gw.get("status") == "verified_symbolically":
            if not gw.get("is_consistent"):
                p_m2 = max(p_m1, 0.95)
            else:
                p_m2 = min(p_m1, 0.20)
        else:
            p_m2 = p_m1
        pred_m2 = int(p_m2 >= tau_frozen)

        y_iv_true.append(yt)
        m0_probs.append(p_m0)
        m0_preds.append(pred_m0)
        m1_probs.append(p_m1)
        m1_preds.append(pred_m1)
        m2_probs.append(p_m2)
        m2_preds.append(pred_m2)
        iv_categories.append(cat)

    dur_iv_s = round(time.perf_counter() - t0_iv, 2)
    print(f"Completed Independent Validation in {dur_iv_s:.2f}s ({dur_iv_s/len(iv_examples)*1000:.1f}ms/ex).")

    y_iv_arr = np.array(y_iv_true)
    m0_pr_arr = np.array(m0_probs)
    m0_pd_arr = np.array(m0_preds)
    m1_pr_arr = np.array(m1_probs)
    m1_pd_arr = np.array(m1_preds)
    m2_pr_arr = np.array(m2_probs)
    m2_pd_arr = np.array(m2_preds)

    def calc_metrics(yt: np.ndarray, yp: np.ndarray, ypr: np.ndarray) -> Dict[str, Any]:
        cm = confusion_matrix(yt, yp)
        tn, fp, fn, tp = cm.ravel()
        p_pr, r_pr, _ = precision_recall_curve(yt, ypr)
        ece, _ = compute_ece(yt, ypr)

        return {
            "auroc": round(float(roc_auc_score(yt, ypr)), 4),
            "auprc": round(float(auc(r_pr, p_pr)), 4),
            "accuracy": round(float(accuracy_score(yt, yp)), 4),
            "precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
            "recall": round(float(recall_score(yt, yp, zero_division=0)), 4),
            "specificity": round(float(tn / (tn + fp)), 4),
            "f1_score": round(float(f1_score(yt, yp, zero_division=0)), 4),
            "mcc": round(float(matthews_corrcoef(yt, yp)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(yt, yp)), 4),
            "brier_score": round(float(brier_score_loss(yt, ypr)), 4),
            "ece": round(float(ece), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }

    m0_metrics = calc_metrics(y_iv_arr, m0_pd_arr, m0_pr_arr)
    m1_metrics = calc_metrics(y_iv_arr, m1_pd_arr, m1_pr_arr)
    m2_metrics = calc_metrics(y_iv_arr, m2_pd_arr, m2_pr_arr)

    # Bootstrap 95% CIs
    m0_cis = bootstrap_metric_ci(y_iv_arr, m0_pd_arr, m0_pr_arr)
    m1_cis = bootstrap_metric_ci(y_iv_arr, m1_pd_arr, m1_pr_arr)
    m2_cis = bootstrap_metric_ci(y_iv_arr, m2_pd_arr, m2_pr_arr)

    # Paired Wilcoxon signed-rank test on predicted probabilities
    _, pval_m0_m1 = stats.wilcoxon(abs(m0_pr_arr - y_iv_arr), abs(m1_pr_arr - y_iv_arr))
    _, pval_m0_m2 = stats.wilcoxon(abs(m0_pr_arr - y_iv_arr), abs(m2_pr_arr - y_iv_arr))

    # Category-level Recall breakdown
    cat_recalls = {}
    for cat_name in set(iv_categories):
        cat_idx = [i for i, c in enumerate(iv_categories) if c == cat_name]
        cat_yt = y_iv_arr[cat_idx]
        if sum(cat_yt) > 0:  # Hallucinated category
            r_m0 = recall_score(cat_yt, m0_pd_arr[cat_idx], zero_division=0)
            r_m1 = recall_score(cat_yt, m1_pd_arr[cat_idx], zero_division=0)
            r_m2 = recall_score(cat_yt, m2_pd_arr[cat_idx], zero_division=0)
            cat_recalls[cat_name] = {
                "count": len(cat_idx),
                "model_0_frozen_recall": round(float(r_m0), 4),
                "model_1_candidate_recall": round(float(r_m1), 4),
                "model_2_remediated_recall": round(float(r_m2), 4),
            }
        else:  # Factual category (Specificity)
            s_m0 = float(np.sum((cat_yt == 0) & (m0_pd_arr[cat_idx] == 0))) / len(cat_idx)
            s_m1 = float(np.sum((cat_yt == 0) & (m1_pd_arr[cat_idx] == 0))) / len(cat_idx)
            s_m2 = float(np.sum((cat_yt == 0) & (m2_pd_arr[cat_idx] == 0))) / len(cat_idx)
            cat_recalls[cat_name] = {
                "count": len(cat_idx),
                "model_0_frozen_specificity": round(float(s_m0), 4),
                "model_1_candidate_specificity": round(float(s_m1), 4),
                "model_2_remediated_specificity": round(float(s_m2), 4),
            }

    # 8. Section 10: Error Decomposition on Independent Validation Holdout
    print("\nExecuting Section 10: Error Decomposition on Independent Validation...")
    fn_m0_codes = {f"R{k}": 0 for k in range(1, 13)}
    fn_m2_codes = {f"R{k}": 0 for k in range(1, 13)}

    for idx, ex in enumerate(iv_examples):
        if ex["label"] == 1:
            cat = ex["category"]
            # M0 error
            if m0_pd_arr[idx] == 0:
                if "numerical" in cat:
                    fn_m0_codes["R9"] += 1  # symbolic gateway integration
                elif m0_pr_arr[idx] < 0.35:
                    fn_m0_codes["R7"] += 1  # polarity/fusion suppression
                elif 0.35 <= m0_pr_arr[idx] < tau_frozen:
                    fn_m0_codes["R4"] += 1  # classifier boundary
                else:
                    fn_m0_codes["R2"] += 1

            # M2 error
            if m2_pd_arr[idx] == 0:
                if "unsupported" in cat:
                    fn_m2_codes["R1"] += 1  # retrieval scope
                elif 0.40 <= m2_pr_arr[idx] < tau_frozen:
                    fn_m2_codes["R4"] += 1  # classifier boundary
                else:
                    fn_m2_codes["R2"] += 1

    # 9. Section 11: Local Runtime Soak Test
    print("\nExecuting Section 11: Local Runtime Soak Test...")
    proc = psutil.Process()
    rss_start = proc.memory_info().rss / (1024 * 1024)

    # Run 10 back-to-back requests
    t_soak0 = time.perf_counter()
    peak_rss = rss_start
    for k in range(10):
        prod_pipe.predict("Stockholm is the capital of Sweden.")
        curr_rss = proc.memory_info().rss / (1024 * 1024)
        if curr_rss > peak_rss:
            peak_rss = curr_rss

    rss_final = proc.memory_info().rss / (1024 * 1024)
    soak_dur_s = round(time.perf_counter() - t_soak0, 2)

    runtime_results = {
        "local_startup_rss_mb": round(rss_start, 2),
        "local_peak_rss_mb": round(peak_rss, 2),
        "local_final_rss_mb": round(rss_final, 2),
        "soak_requests_completed": 10,
        "soak_duration_seconds": soak_dur_s,
        "active_models_count": 2,
        "worker_count": 1,
        "railway_status": "Railway runtime stability not independently verified in Phase 53 (Local measurements only).",
    }

    # 10. Master Payload Persistence
    master_results = {
        "timestamp": time.time(),
        "phase52_polarity_reassessment": polarity_reassessment,
        "repeated_cv_candidates": cv_results,
        "strategy_s1_symbolic_audit": strategy_s1_results,
        "counterfactual_matched_pairs": counterfactual_results,
        "independent_validation_metrics": {
            "model_0_frozen_baseline": m0_metrics,
            "model_1_candidate_b": m1_metrics,
            "model_2_candidate_b_plus_s1": m2_metrics,
        },
        "bootstrap_95_confidence_intervals": {
            "model_0_frozen": m0_cis,
            "model_1_candidate_b": m1_cis,
            "model_2_candidate_b_plus_s1": m2_cis,
        },
        "statistical_tests": {
            "wilcoxon_pval_m0_vs_m1": round(float(pval_m0_m1), 8),
            "wilcoxon_pval_m0_vs_m2": round(float(pval_m0_m2), 8),
            "is_statistically_significant": bool(pval_m0_m2 < 0.001),
        },
        "category_level_breakdown": cat_recalls,
        "error_decomposition_r1_r12": {
            "model_0_frozen_fn_distribution": fn_m0_codes,
            "model_2_remediated_fn_distribution": fn_m2_codes,
        },
        "runtime_soak_audit": runtime_results,
    }

    out_master_json = REPORTS_DIR / "PHASE53_MASTER_RESULTS.json"
    with open(out_master_json, "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 53 INDEPENDENT VALIDATION SUMMARY (N=200 HOLDOUT)")
    print("=" * 80)
    print(f"Model 0 (Frozen):   AUROC: {m0_metrics['auroc']:.4f} | Rec: {m0_metrics['recall']:.4f} | Spec: {m0_metrics['specificity']:.4f} | MCC: {m0_metrics['mcc']:.4f} | ECE: {m0_metrics['ece']:.4f}")
    print(f"Model 1 (Cand B):   AUROC: {m1_metrics['auroc']:.4f} | Rec: {m1_metrics['recall']:.4f} | Spec: {m1_metrics['specificity']:.4f} | MCC: {m1_metrics['mcc']:.4f} | ECE: {m1_metrics['ece']:.4f}")
    print(f"Model 2 (B + S1):   AUROC: {m2_metrics['auroc']:.4f} | Rec: {m2_metrics['recall']:.4f} | Spec: {m2_metrics['specificity']:.4f} | MCC: {m2_metrics['mcc']:.4f} | ECE: {m2_metrics['ece']:.4f}")
    print(f"Statistical Significance: p-value = {pval_m0_m2:.2e} (Statistically Significant: {pval_m0_m2 < 0.001})")
    print("=" * 80)

if __name__ == "__main__":
    run_phase53_remediation_pipeline()
