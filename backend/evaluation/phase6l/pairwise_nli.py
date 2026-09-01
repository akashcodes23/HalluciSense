"""Phase 6L.1A — Pairwise NLI Execution, Embedding Similarity, and Persistent Caching.

Executes bidirectional claim-to-claim Natural Language Inference (NLI) and dense
sentence embedding cosine similarity pre-screening.

Strict Data Firewall Rule:
    * Accesses DEV partition ONLY. Validation partition (N=12,483) is strictly sealed.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

import structlog

from app.core.engine.entailment import EvidenceEntailmentEngine
from evaluation.phase6j.utils import _serializable
from evaluation.phase6k.forensics import categorize_warning, summarize_warning_records, CapturedWarningRecord
from evaluation.phase6l.config import (
    DEFAULT_NLI_MODEL,
    PHASE6L_CACHE_DIR,
    SIMILARITY_MODEL_NAME,
)

logger = structlog.get_logger(__name__)


# Global model cache to avoid reloading weights repeatedly
_GLOBAL_NLI_ENGINE: Optional[EvidenceEntailmentEngine] = None
_GLOBAL_EMBED_MODEL: Any = None


def get_nli_engine(model_name: str = DEFAULT_NLI_MODEL) -> EvidenceEntailmentEngine:
    """Get or initialize singleton NLI engine."""
    global _GLOBAL_NLI_ENGINE
    if _GLOBAL_NLI_ENGINE is None or _GLOBAL_NLI_ENGINE.model_name != model_name:
        _GLOBAL_NLI_ENGINE = EvidenceEntailmentEngine(model_name=model_name)
    return _GLOBAL_NLI_ENGINE


def get_similarity_model(model_name: str = SIMILARITY_MODEL_NAME) -> Any:
    """Get or initialize singleton SentenceTransformer model via ModelRegistry."""
    from app.core.engine.model_registry import ModelRegistry
    return ModelRegistry.get_sentence_transformer(model_name)


def evaluate_bidirectional_nli_and_similarity(
    pairs: List[Dict[str, Any]],
    nli_model_name: str = DEFAULT_NLI_MODEL,
    similarity_model_name: str = SIMILARITY_MODEL_NAME,
    cache_dir: Path = PHASE6L_CACHE_DIR,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Execute bidirectional NLI and embedding similarity for a list of claim pairs with caching.

    Args:
        pairs: List of claim pair dicts from claim_pairs.py.
        nli_model_name: Name of NLI HuggingFace model.
        similarity_model_name: Name of SentenceTransformer model.
        cache_dir: Directory to persist joblib cache files.
        batch_size: Inference batch size.

    Returns:
        Dict containing evaluated pair results, warning records, runtime, and cache stats.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Compute cache key from dataset pairs & model metadata
    pairs_hash_src = "".join(f"{p['example_id']}:{p['claim_i_index']}:{p['claim_j_index']}" for p in pairs[:50])
    cache_key = hashlib.sha256(f"{pairs_hash_src}:{len(pairs)}:{nli_model_name}:{similarity_model_name}".encode("utf-8")).hexdigest()[:16]
    cache_path = cache_dir / f"pairwise_nli_1a_{cache_key}.joblib"

    if cache_path.exists():
        logger.info("loading_pairwise_nli_from_cache", path=str(cache_path))
        cached_payload = joblib.load(cache_path)
        cached_payload["cache_hit"] = True
        return cached_payload

    logger.info("evaluating_pairwise_nli_start", n_pairs=len(pairs), model=nli_model_name)
    t0 = time.time()

    nli_engine = get_nli_engine(model_name=nli_model_name)
    sim_model = get_similarity_model(model_name=similarity_model_name)

    rec_warns: List[CapturedWarningRecord] = []

    # 1. Compute sentence embeddings for all unique claims
    all_claims_set = set()
    for p in pairs:
        if p["claim_i_text"]:
            all_claims_set.add(p["claim_i_text"])
        if p["claim_j_text"]:
            all_claims_set.add(p["claim_j_text"])

    unique_claims = sorted(list(all_claims_set))
    claim2embed: Dict[str, np.ndarray] = {}

    if unique_claims:
        embeddings = sim_model.encode(unique_claims, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        for text, emb in zip(unique_claims, embeddings):
            claim2embed[text] = emb

    # 2. Build forward (i->j) and reverse (j->i) pair batches
    fwd_claims: List[str] = []
    fwd_evidences: List[str] = []

    rev_claims: List[str] = []
    rev_evidences: List[str] = []

    for p in pairs:
        c_i = p["claim_i_text"]
        c_j = p["claim_j_text"]

        # Forward: premise=c_i, hypothesis=c_j
        fwd_evidences.append(c_i)
        fwd_claims.append(c_j)

        # Reverse: premise=c_j, hypothesis=c_i
        rev_evidences.append(c_j)
        rev_claims.append(c_i)

    # 3. Execute NLI inference under warning tracking
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")

        fwd_results = nli_engine.classify_batch(claims=fwd_claims, evidences=fwd_evidences, batch_size=batch_size)
        rev_results = nli_engine.classify_batch(claims=rev_claims, evidences=rev_evidences, batch_size=batch_size)

        for w in recorded:
            rec_warns.append(categorize_warning(w))

    elapsed = time.time() - t0

    # 4. Construct evaluated pair records with symmetric formulations
    evaluated_pairs = []
    for idx, p in enumerate(pairs):
        fwd_nli = fwd_results[idx]
        rev_nli = rev_results[idx]

        c_i = p["claim_i_text"]
        c_j = p["claim_j_text"]

        # Embed similarity
        if c_i in claim2embed and c_j in claim2embed:
            sim = float(np.dot(claim2embed[c_i], claim2embed[c_j]))
        else:
            sim = 0.0

        c_ij = fwd_nli["contradiction"]
        c_ji = rev_nli["contradiction"]

        e_ij = fwd_nli["entailment"]
        e_ji = rev_nli["entailment"]

        n_ij = fwd_nli["neutral"]
        n_ji = rev_nli["neutral"]

        # Symmetric Contradiction Formulations
        c_max = float(max(c_ij, c_ji))
        c_mean = float((c_ij + c_ji) / 2.0)
        c_min = float(min(c_ij, c_ji))
        c_prob_union = float(1.0 - (1.0 - c_ij) * (1.0 - c_ji))

        delta_c = float(abs(c_ij - c_ji))
        delta_e = float(abs(e_ij - e_ji))

        eval_obj = {
            "example_id": p["example_id"],
            "claim_i_index": p["claim_i_index"],
            "claim_j_index": p["claim_j_index"],
            "claim_i_text": c_i,
            "claim_j_text": c_j,
            "ground_truth": p["ground_truth"],
            "embedding_cosine_similarity": sim,
            "forward_nli": fwd_nli,
            "reverse_nli": rev_nli,
            "c_ij": c_ij,
            "c_ji": c_ji,
            "e_ij": e_ij,
            "e_ji": e_ji,
            "n_ij": n_ij,
            "n_ji": n_ji,
            "delta_c": delta_c,
            "delta_e": delta_e,
            "c_max": c_max,
            "c_mean": c_mean,
            "c_min": c_min,
            "c_prob_union": c_prob_union,
        }
        evaluated_pairs.append(eval_obj)

    warn_summary = summarize_warning_records(rec_warns)

    payload = {
        "cache_hit": False,
        "nli_model_name": nli_model_name,
        "similarity_model_name": similarity_model_name,
        "total_pairs_evaluated": len(evaluated_pairs),
        "total_directional_inferences": len(evaluated_pairs) * 2,
        "elapsed_seconds": float(elapsed),
        "inferences_per_second": float((len(evaluated_pairs) * 2) / max(elapsed, 1e-4)),
        "warning_summary": warn_summary,
        "total_warnings": len(rec_warns),
        "evaluated_pairs": evaluated_pairs,
    }

    # Save to atomic persistent cache
    tmp_path = cache_dir / f"pairwise_nli_1a_{cache_key}.tmp"
    joblib.dump(payload, tmp_path)
    os.replace(tmp_path, cache_path)

    logger.info("evaluating_pairwise_nli_complete", elapsed_s=round(elapsed, 2), path=str(cache_path))
    return payload
