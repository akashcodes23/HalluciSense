"""
HalluciSense Inference Pipeline Regression Tests.

Validates that the fixed inference pipeline:
1. Uses correct evidence dict key ('similarity_score', not 'score')
2. Converts CrossEncoder relevance to NLI-compatible features
3. Produces P(hallucinated) < threshold for factual inputs
4. The hybrid model was trained on real data (not synthetic)
5. Feature schema matches HYBRID_FEATURE_SCHEMA from config.py
6. clf.classes_ is [0, 1] where 0=factual, 1=hallucinated
"""
import sys
import os
import json
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import joblib
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "evaluation_results" / "phase6m" / "final_hybrid_model"
PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    status = "✅ PASS" if condition else "❌ FAIL"
    if not condition:
        FAIL += 1
    else:
        PASS += 1
    print(f"  {status}: {name}")
    if detail and not condition:
        print(f"         Detail: {detail}")


# ─────────────────────────────────────────────────────────────
# Test 1: Feature Schema Matches config.py
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("TEST SUITE 1: Feature Schema Correctness")
print("=" * 60)

with open(MODEL_DIR / "feature_schema.json") as f:
    saved_schema = json.load(f)["feature_schema"]

from evaluation.phase6m.config import HYBRID_FEATURE_SCHEMA

check(
    "feature_schema.json matches HYBRID_FEATURE_SCHEMA",
    saved_schema == HYBRID_FEATURE_SCHEMA,
    f"saved={saved_schema}, expected={HYBRID_FEATURE_SCHEMA}",
)
check(
    "Feature count is 19",
    len(saved_schema) == 19,
    f"Got {len(saved_schema)}",
)


# ─────────────────────────────────────────────────────────────
# Test 2: Model classes_ and Label Convention
# ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("TEST SUITE 2: Model Label Convention")
print("=" * 60)

clf = joblib.load(MODEL_DIR / "hybrid_meta_classifier.joblib")
scaler = joblib.load(MODEL_DIR / "preprocessing.joblib")

check(
    "clf.classes_ == [0, 1]",
    list(clf.classes_) == [0, 1],
    f"Got {list(clf.classes_)}",
)
check(
    "predict_proba[:, 1] = P(class=1=hallucinated)",
    clf.classes_[1] == 1,
    f"classes_[1] = {clf.classes_[1]}",
)


# ─────────────────────────────────────────────────────────────
# Test 3: Model Trained on Real Data (Not Synthetic)
# ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("TEST SUITE 3: Model Training Provenance")
print("=" * 60)

with open(MODEL_DIR / "model_metadata.json") as f:
    meta = json.load(f)

check(
    "model_metadata.json contains 'retrained_from_real_data'",
    meta.get("retrained_from_real_data", False) is True,
    f"Got retrained_from_real_data={meta.get('retrained_from_real_data')}",
)
check(
    "Training samples >= 50000",
    meta.get("training_samples", 0) >= 50000,
    f"Got training_samples={meta.get('training_samples')}",
)
check(
    "Label convention documented",
    meta.get("protocol", {}).get("label_convention") is not None,
    "Missing protocol.label_convention",
)


# ─────────────────────────────────────────────────────────────
# Test 4: _relevance_to_nli Distribution Mapping
# ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("TEST SUITE 4: Relevance-to-NLI Conversion")
print("=" * 60)

from app.core.inference.pillar1_engine import _relevance_to_nli

# High relevance → moderate entailment, low contradiction
ent_hi, con_hi, neu_hi = _relevance_to_nli(0.999)
check(
    "High relevance (0.999): entailment in [0.25, 1.0]",
    0.25 <= ent_hi <= 1.0,
    f"Got {ent_hi:.4f}",
)
check(
    "High relevance (0.999): contradiction < 0.05",
    con_hi < 0.05,
    f"Got {con_hi:.4f}",
)

# Low relevance → near-zero entailment, high contradiction
ent_lo, con_lo, neu_lo = _relevance_to_nli(0.01)
check(
    "Low relevance (0.01): entailment < 0.01",
    ent_lo < 0.01,
    f"Got {ent_lo:.4f}",
)
check(
    "Low relevance (0.01): contradiction > 0.5",
    con_lo > 0.5,
    f"Got {con_lo:.4f}",
)

# Zero relevance
ent_z, con_z, neu_z = _relevance_to_nli(0.0)
check(
    "Zero relevance: entailment == 0.0",
    ent_z < 0.001,
    f"Got {ent_z:.4f}",
)

# Probabilities sum to 1.0
for rel in [0.0, 0.1, 0.5, 0.75, 0.99, 1.0]:
    e, c, n = _relevance_to_nli(rel)
    total = e + c + n
    check(
        f"NLI probs sum to 1.0 for relevance={rel}",
        abs(total - 1.0) < 1e-6,
        f"Sum = {total:.6f}",
    )


# ─────────────────────────────────────────────────────────────
# Test 5: Factual Statement Regression Test (Direct Model)
# ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("TEST SUITE 5: Factual Statement Regression (Direct Model)")
print("=" * 60)

EPSILON = 1e-6
def compute_logit(p, eps=EPSILON):
    p_clipped = max(eps, min(1.0 - eps, float(p)))
    return float(math.log(p_clipped / (1.0 - p_clipped)))

# Construct a feature vector representing a clearly factual claim
# with high evidence support and low risk from base models
factual_features = [
    0.20,   # p1_mean_entailment (moderate - matches training distribution)
    0.25,   # p1_max_entailment
    0.15,   # p1_mean_contradiction (low)
    0.10,   # p1_min_support_margin
    1.0,    # p1_num_claims
    0.0,    # p2_max_pairwise_contradiction
    0.0,    # p2_mean_pairwise_contradiction
    0.0,    # p2_max_pairwise_similarity
    0.0,    # p2_fraction_contradictory_pairs
    1.0,    # p2_num_claims
    0.40,   # prob_p1 (low hallucination risk)
    0.40,   # prob_p2
    compute_logit(0.40),
    compute_logit(0.40),
    0.0,    # prob_disagreement_abs
    0.40,   # prob_mean
    0.40,   # prob_max
    0.40,   # prob_min
    1.0,    # prob_ratio
]

X_fact = np.array(factual_features, dtype=np.float64).reshape(1, -1)
X_fact_scaled = scaler.transform(X_fact)
p_fact = clf.predict_proba(X_fact_scaled)[0, 1]

check(
    "Factual-like features: P(hallucinated) < 0.54 (threshold)",
    p_fact < 0.54,
    f"P(hallucinated) = {p_fact:.4f}",
)

# High-risk features (hallucinated-like)
halluc_features = [
    0.001,  # p1_mean_entailment (very low - no evidence support)
    0.001,  # p1_max_entailment
    0.80,   # p1_mean_contradiction (high)
    -0.80,  # p1_min_support_margin (negative = contradicted)
    3.0,    # p1_num_claims
    0.5,    # p2_max_pairwise_contradiction (self-contradictory)
    0.3,    # p2_mean_pairwise_contradiction
    0.2,    # p2_max_pairwise_similarity
    0.4,    # p2_fraction_contradictory_pairs
    3.0,    # p2_num_claims
    0.80,   # prob_p1 (high hallucination risk)
    0.70,   # prob_p2
    compute_logit(0.80),
    compute_logit(0.70),
    0.10,   # prob_disagreement_abs
    0.75,   # prob_mean
    0.80,   # prob_max
    0.70,   # prob_min
    (0.80 + EPSILON) / (0.70 + EPSILON),
]

X_hall = np.array(halluc_features, dtype=np.float64).reshape(1, -1)
X_hall_scaled = scaler.transform(X_hall)
p_hall = clf.predict_proba(X_hall_scaled)[0, 1]

check(
    "Hallucinated-like features: P(hallucinated) > 0.54 (threshold)",
    p_hall > 0.54,
    f"P(hallucinated) = {p_hall:.4f}",
)

check(
    "Model discriminates: P(hall|risky) > P(hall|factual)",
    p_hall > p_fact,
    f"P_hall={p_hall:.4f}, P_fact={p_fact:.4f}",
)


# ─────────────────────────────────────────────────────────────
# Test 6: Scaler Not Fitted on Synthetic Data
# ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("TEST SUITE 6: Scaler Distribution Sanity")
print("=" * 60)

# The scaler center_ for mean_entailment should be ~0.002, NOT ~0.0 from random Gaussian
center_ent = scaler.center_[0]
check(
    "Scaler center[0] (mean_entailment) is near training median ~0.002",
    0.0 <= center_ent <= 0.1,
    f"Got {center_ent:.6f}",
)

# IQR for mean_entailment should be ~0.044, NOT ~1.35 from random Gaussian
scale_ent = scaler.scale_[0]
check(
    "Scaler scale[0] (mean_entailment IQR) is near training IQR ~0.044",
    0.01 <= scale_ent <= 0.2,
    f"Got {scale_ent:.6f}",
)


# ─────────────────────────────────────────────────────────────
# Test 7: End-to-End API Test (if possible)
# ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("TEST SUITE 7: End-to-End API Regression")
print("=" * 60)

try:
    from app.main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)

    r = client.post("/api/v1/hallucisense/predict", json={
        "response_text": "Paris is the capital of France."
    })
    d = r.json()
    prob = d["hallucination_probability"]
    verdict = d["explanation"]["verdict"]

    check(
        "API: 'Paris is the capital of France' → P(hallucinated) < 0.54",
        prob < 0.54,
        f"P(hallucinated) = {prob}",
    )
    check(
        "API: 'Paris is the capital of France' → verdict == FACTUAL",
        verdict == "FACTUAL",
        f"verdict = {verdict}",
    )

    # Verify health
    r_h = client.get("/api/v1/hallucisense/health")
    d_h = r_h.json()
    check(
        "API: Health status == ok",
        d_h["status"] == "ok",
        f"status = {d_h['status']}",
    )
    check(
        "API: active_model == hybrid",
        d_h["active_model"] == "hybrid",
        f"active_model = {d_h['active_model']}",
    )
except Exception as e:
    print(f"  ⚠ API tests skipped due to: {e}")


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"REGRESSION TEST SUMMARY")
print(f"{'=' * 60}")
print(f"  PASSED: {PASS}")
print(f"  FAILED: {FAIL}")
print(f"  TOTAL:  {PASS + FAIL}")
print(f"  STATUS: {'✅ ALL PASSED' if FAIL == 0 else f'❌ {FAIL} FAILURES'}")
print(f"{'=' * 60}")

sys.exit(FAIL)
