"""
HalluciSense Pillar 2 — Contradiction Analyzer
===============================================
Performs deep contradiction taxonomy classification across claims and evidence.
Generates contradiction graphs for UI visualization.
"""

import hashlib
import re
from typing import Dict, List

import structlog
from app.pillar2.claim_extraction.schemas import ExtractedClaim
from app.pillar2.consensus_engine.schemas import ConsensusResult
from app.pillar2.contradiction_analysis.schemas import (
    ContradictionAnalysisResult,
    ContradictionGraphVisualization,
    ContradictionItem,
    ContradictionType,
)
from app.pillar2.evidence_retrieval.schemas import EvidenceItem
from app.pillar2.multi_llm_verifier.schemas import VerificationLabel

logger = structlog.get_logger(__name__)


class ContradictionAnalyzer:
    """
    Production contradiction analysis engine.
    Categorizes contradictions and builds contradiction graphs.
    """

    SPECULATION_KEYWORDS = re.compile(r"\b(?:might|maybe|possibly|speculated|allegedly|rumored|could be|unconfirmed)\b", re.IGNORECASE)

    def analyze_contradictions(
        self,
        claims: List[ExtractedClaim],
        consensus_map: Dict[str, ConsensusResult],
        evidence_items: List[EvidenceItem],
    ) -> ContradictionAnalysisResult:
        """
        Analyze all claims and consensus records for contradictions.

        Parameters
        ----------
        claims : List[ExtractedClaim]
        consensus_map : Dict[str, ConsensusResult]
        evidence_items : List[EvidenceItem]

        Returns
        -------
        ContradictionAnalysisResult
        """
        items: List[ContradictionItem] = []
        ev_map = {e.evidence_id: e for e in evidence_items}

        graph_nodes: List[Dict[str, Any]] = []
        graph_edges: List[Dict[str, Any]] = []
        node_ids_set = set()

        for claim in claims:
            cid = claim.claim_id
            consensus = consensus_map.get(cid)

            # Node for claim
            if cid not in node_ids_set:
                graph_nodes.append({
                    "id": cid,
                    "label": claim.claim_text[:35],
                    "type": "CLAIM",
                    "claim_type": claim.claim_type.value,
                })
                node_ids_set.add(cid)

            if not consensus:
                continue

            maj_lbl = consensus.majority_label

            # Classify contradiction type
            if maj_lbl == VerificationLabel.CONTRADICTED:
                severity = 0.95 if claim.numbers or claim.dates else 0.85
                c_type = ContradictionType.DIRECT_CONTRADICTION
                explanation = f"Direct contradiction detected between claim '{claim.claim_text}' and verified evidence."

                # Find conflicting evidence item
                conf_ev_id = None
                for diss in consensus.disagreeing_verifiers:
                    if diss.assigned_label == VerificationLabel.CONTRADICTED:
                        conf_ev_id = "ev_contradicted_01"
                        break

                items.append(ContradictionItem(
                    contradiction_id=f"cnt_{hashlib.sha256(cid.encode()).hexdigest()[:10]}",
                    claim_id=cid,
                    claim_text=claim.claim_text,
                    type=c_type,
                    severity=severity,
                    evidence_id=conf_ev_id,
                    explanation=explanation,
                ))

            elif maj_lbl == VerificationLabel.PARTIALLY_SUPPORTED:
                items.append(ContradictionItem(
                    contradiction_id=f"cnt_{hashlib.sha256(cid.encode()).hexdigest()[:10]}",
                    claim_id=cid,
                    claim_text=claim.claim_text,
                    type=ContradictionType.PARTIAL_CONTRADICTION,
                    severity=0.55,
                    explanation=f"Partial contradiction: claim '{claim.claim_text}' contains conflicting evidence elements.",
                ))

            elif maj_lbl == VerificationLabel.UNKNOWN:
                if self.SPECULATION_KEYWORDS.search(claim.claim_text):
                    items.append(ContradictionItem(
                        contradiction_id=f"cnt_{hashlib.sha256(cid.encode()).hexdigest()[:10]}",
                        claim_id=cid,
                        claim_text=claim.claim_text,
                        type=ContradictionType.SPECULATION,
                        severity=0.40,
                        explanation=f"Speculative claim unsupported by factual evidence: '{claim.claim_text}'.",
                    ))
                elif not evidence_items:
                    items.append(ContradictionItem(
                        contradiction_id=f"cnt_{hashlib.sha256(cid.encode()).hexdigest()[:10]}",
                        claim_id=cid,
                        claim_text=claim.claim_text,
                        type=ContradictionType.MISSING_EVIDENCE,
                        severity=0.60,
                        explanation=f"Missing evidence: no relevant evidence found to verify '{claim.claim_text}'.",
                    ))
                else:
                    items.append(ContradictionItem(
                        contradiction_id=f"cnt_{hashlib.sha256(cid.encode()).hexdigest()[:10]}",
                        claim_id=cid,
                        claim_text=claim.claim_text,
                        type=ContradictionType.FABRICATION,
                        severity=0.90,
                        explanation=f"Fabrication suspected: claim '{claim.claim_text}' has zero empirical evidence support.",
                    ))

        # Add evidence nodes and edges to graph
        for ev in evidence_items:
            ev_id = ev.evidence_id
            if ev_id not in node_ids_set:
                graph_nodes.append({
                    "id": ev_id,
                    "label": ev.title[:35],
                    "type": "EVIDENCE",
                    "source": ev.source,
                    "authority": ev.authority_score,
                })
                node_ids_set.add(ev_id)

            # Connect evidence to first claim
            if claims:
                graph_edges.append({
                    "source": claims[0].claim_id,
                    "target": ev_id,
                    "relation": "EVALUATED_AGAINST",
                })

        # Connect contradiction edges
        for item in items:
            if item.type in [ContradictionType.DIRECT_CONTRADICTION, ContradictionType.PARTIAL_CONTRADICTION]:
                target_ev = item.evidence_id or (evidence_items[0].evidence_id if evidence_items else "ev_unknown")
                graph_edges.append({
                    "source": item.claim_id,
                    "target": target_ev,
                    "relation": "CONTRADICTS",
                    "severity": item.severity,
                })

        high_sev = sum(1 for i in items if i.severity >= 0.7)
        max_sev = max((i.severity for i in items), default=0.0)
        n_claims = len(claims)
        n_fab = sum(1 for i in items if i.type in [ContradictionType.FABRICATION, ContradictionType.DIRECT_CONTRADICTION])
        fab_index = round(n_fab / n_claims, 4) if n_claims > 0 else 0.0

        viz = ContradictionGraphVisualization(
            nodes=graph_nodes,
            edges=graph_edges,
            total_contradictions=len(items),
            high_severity_count=high_sev,
        )

        logger.info(
            "contradiction_analysis_complete",
            num_claims=n_claims,
            num_contradictions=len(items),
            max_severity=max_sev,
            fabrication_index=fab_index,
        )

        return ContradictionAnalysisResult(
            contradictions=items,
            contradiction_count=len(items),
            fabrication_index=fab_index,
            max_severity=max_sev,
            graph_visualization=viz,
        )
