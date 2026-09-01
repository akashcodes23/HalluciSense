"""Production Semantic NLI Grounding Adapter for HalluciSense Phase 39.

Provides real transformer cross-encoder Natural Language Inference (NLI) between
claims and retrieved evidence passages using the singleton ModelRegistry.
Strictly reuses the existing cross-encoder/nli-deberta-v3-small instance without
loading duplicate weights.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

from app.core.engine.entailment import EvidenceEntailmentEngine
from app.core.engine.model_registry import ModelRegistry

logger = structlog.get_logger(__name__)

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
MAX_EVIDENCE_PER_CLAIM = 3
MAX_CLAIMS_FOR_NLI = 15


class SemanticNLIAdapter:
    """Production Adapter for Claim ↔ Evidence Semantic NLI Evaluation."""

    def __init__(self, model_name: str = DEFAULT_NLI_MODEL):
        self.model_name = model_name
        self.engine = EvidenceEntailmentEngine(model_name=model_name)

    def evaluate_pair(self, claim: str, evidence: str) -> Dict[str, Any]:
        """Evaluate a single claim-evidence pair with DeBERTa NLI cross-encoder."""
        if not claim or not claim.strip() or not evidence or not evidence.strip():
            return {
                "entailment": 0.0,
                "neutral": 1.0,
                "contradiction": 0.0,
                "label": "neutral",
                "confidence": 1.0,
                "model_name": self.model_name,
                "latency_ms": 0.0,
            }

        t0 = time.perf_counter()
        try:
            scores = self.engine.classify(claim=claim.strip(), evidence=evidence.strip())
            ent = float(scores.get("entailment", 0.0))
            neu = float(scores.get("neutral", 0.0))
            con = float(scores.get("contradiction", 0.0))

            # Determine dominant label
            if ent >= neu and ent >= con:
                label = "entailment"
                confidence = ent
            elif con >= neu and con >= ent:
                label = "contradiction"
                confidence = con
            else:
                label = "neutral"
                confidence = neu

            latency_ms = (time.perf_counter() - t0) * 1000.0

            return {
                "entailment": round(ent, 6),
                "neutral": round(neu, 6),
                "contradiction": round(con, 6),
                "label": label,
                "confidence": round(confidence, 4),
                "model_name": self.model_name,
                "latency_ms": round(latency_ms, 2),
            }
        except Exception as exc:
            logger.warning("semantic_nli_evaluate_pair_failed", error=str(exc), claim=claim[:50])
            return {
                "entailment": 0.0,
                "neutral": 1.0,
                "contradiction": 0.0,
                "label": "neutral",
                "confidence": 1.0,
                "model_name": self.model_name,
                "latency_ms": 0.0,
                "error": str(exc),
            }

    def evaluate_claim_evidence_grounding(
        self,
        claims: List[Dict[str, Any]],
        evidence_by_claim: Optional[Dict[str, List[Any]]] = None,
        max_evidence_per_claim: int = MAX_EVIDENCE_PER_CLAIM,
    ) -> Dict[str, Any]:
        """Batch evaluate all claims against retrieved evidence passages.

        Args:
            claims: List of claim dicts containing claim_id and text.
            evidence_by_claim: Dict mapping claim_text to list of evidence dicts/objects.
            max_evidence_per_claim: Maximum passages evaluated per claim.

        Returns:
            Dictionary containing structured grounding diagnostics and aggregated features.
        """
        t_start = time.perf_counter()
        bounded_claims = claims[:MAX_CLAIMS_FOR_NLI]
        evaluated_claims: List[Dict[str, Any]] = []

        all_pair_claims: List[str] = []
        all_pair_evidences: List[str] = []
        pair_meta: List[Dict[str, Any]] = []

        # 1. Prepare batch pairs
        for c in bounded_claims:
            c_id = c.get("claim_id", 0)
            c_text = c.get("text", "").strip()
            if not c_text:
                continue

            raw_evidences = []
            if evidence_by_claim and c_text in evidence_by_claim:
                raw_evidences = evidence_by_claim[c_text]

            bounded_ev = raw_evidences[:max_evidence_per_claim]
            if not bounded_ev:
                # No evidence retrieved for this claim
                continue

            for ev_idx, ev in enumerate(bounded_ev):
                if isinstance(ev, dict):
                    ev_text = ev.get("snippet", ev.get("text", str(ev)))
                    ev_title = ev.get("title", f"passage_{ev_idx}")
                    ev_url = ev.get("url", "")
                else:
                    ev_text = getattr(ev, "snippet", getattr(ev, "text", str(ev)))
                    ev_title = getattr(ev, "title", f"passage_{ev_idx}")
                    ev_url = getattr(ev, "url", "")

                if ev_text and ev_text.strip():
                    all_pair_claims.append(c_text)
                    all_pair_evidences.append(ev_text.strip())
                    pair_meta.append({
                        "claim_id": c_id,
                        "claim_text": c_text,
                        "title": ev_title,
                        "url": ev_url,
                        "snippet": ev_text.strip(),
                    })

        # 2. Execute batched NLI inference
        nli_results: List[Dict[str, float]] = []
        if all_pair_claims:
            try:
                nli_results = self.engine.classify_batch(
                    claims=all_pair_claims,
                    evidences=all_pair_evidences,
                    batch_size=16,
                )
            except Exception as exc:
                logger.warning("semantic_nli_batch_inference_failed", error=str(exc))
                nli_results = [{"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0} for _ in all_pair_claims]

        # 3. Group results by claim
        results_by_claim: Dict[int, List[Dict[str, Any]]] = {}
        for meta, nli_res in zip(pair_meta, nli_results):
            c_id = meta["claim_id"]
            if c_id not in results_by_claim:
                results_by_claim[c_id] = []

            ent = float(nli_res.get("entailment", 0.0))
            neu = float(nli_res.get("neutral", 0.0))
            con = float(nli_res.get("contradiction", 0.0))

            if ent >= neu and ent >= con:
                label = "entailment"
                conf = ent
            elif con >= neu and con >= ent:
                label = "contradiction"
                conf = con
            else:
                label = "neutral"
                conf = neu

            results_by_claim[c_id].append({
                "title": meta["title"],
                "url": meta["url"],
                "snippet": meta["snippet"],
                "entailment": round(ent, 4),
                "neutral": round(neu, 4),
                "contradiction": round(con, 4),
                "label": label,
                "confidence": round(conf, 4),
            })

        # 4. Compute aggregated features per claim
        claim_max_ents: List[float] = []
        claim_mean_cons: List[float] = []
        claim_margins: List[float] = []

        for c in bounded_claims:
            c_id = c.get("claim_id", 0)
            c_text = c.get("text", "")
            ev_evals = results_by_claim.get(c_id, [])

            if ev_evals:
                ents = [e["entailment"] for e in ev_evals]
                cons = [e["contradiction"] for e in ev_evals]
                max_e = float(max(ents))
                mean_c = float(np.mean(cons))
                margin = max_e - mean_c

                # Best evidence passage is highest entailment if any entailment, else highest contradiction
                best_ev = max(ev_evals, key=lambda x: x["entailment"])
                primary_status = best_ev["label"]
            else:
                max_e = 0.0
                mean_c = 0.5  # Neutral penalty when ungrounded
                margin = -0.5
                best_ev = None
                primary_status = "insufficient_evidence"

            claim_max_ents.append(max_e)
            claim_mean_cons.append(mean_c)
            claim_margins.append(margin)

            evaluated_claims.append({
                "claim_id": c_id,
                "claim_text": c_text,
                "evidence_count": len(ev_evals),
                "primary_status": primary_status,
                "max_entailment": round(max_e, 4),
                "mean_contradiction": round(mean_c, 4),
                "support_margin": round(margin, 4),
                "evidence_details": ev_evals,
            })

        # 5. Dataset-level aggregated features
        total_time_ms = (time.perf_counter() - t_start) * 1000.0
        n_claims_val = float(len(bounded_claims)) if bounded_claims else 1.0

        agg_mean_entailment = float(np.mean(claim_max_ents)) if claim_max_ents else 0.0
        agg_max_entailment = float(max(claim_max_ents)) if claim_max_ents else 0.0
        agg_mean_contradiction = float(np.mean(claim_mean_cons)) if claim_mean_cons else 0.5
        agg_min_margin = float(min(claim_margins)) if claim_margins else -0.5

        return {
            "status": "evaluated",
            "model_name": self.model_name,
            "total_claims_evaluated": len(bounded_claims),
            "total_pairs_evaluated": len(all_pair_claims),
            "latency_ms": round(total_time_ms, 2),
            "aggregated_features": {
                "mean_entailment": round(agg_mean_entailment, 6),
                "max_entailment": round(agg_max_entailment, 6),
                "mean_contradiction": round(agg_mean_contradiction, 6),
                "min_support_margin": round(agg_min_margin, 6),
                "num_claims": n_claims_val,
            },
            "claims": evaluated_claims,
        }


# Singleton instance
_semantic_nli_adapter: Optional[SemanticNLIAdapter] = None


def get_semantic_nli_adapter() -> SemanticNLIAdapter:
    global _semantic_nli_adapter
    if _semantic_nli_adapter is None:
        _semantic_nli_adapter = SemanticNLIAdapter()
    return _semantic_nli_adapter
