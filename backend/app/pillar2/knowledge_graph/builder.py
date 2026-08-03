"""
HalluciSense Pillar 2 — Entity + Relation Graph Builder
========================================================
Constructs structured semantic graphs from extracted atomic claims.
Enables graph reasoning, path analysis, and contradiction graph tracing.
"""

import hashlib
import re
from typing import Dict, List, Set

import structlog
from app.pillar2.claim_extraction.schemas import ExtractedClaim
from app.pillar2.knowledge_graph.schemas import (
    EntityCategory,
    GraphEdge,
    GraphNode,
    SemanticGraph,
)

logger = structlog.get_logger(__name__)


class EntityRelationGraphBuilder:
    """
    Constructs semantic knowledge graphs from extracted claims.
    Supports multi-claim entity co-occurrence and relation linking.
    """

    SCIENTIFIC_TERMS = {"dna", "rna", "crispr", "quantum", "physics", "algorithm", "protein", "gene", "molecule"}
    EVENT_TERMS = {"war", "revolution", "battle", "conference", "election", "olympics", "discovery", "launch"}

    def build_graph(self, claims: List[ExtractedClaim], graph_id: str = "graph_default") -> SemanticGraph:
        """
        Build SemanticGraph from list of ExtractedClaims.

        Parameters
        ----------
        claims : List[ExtractedClaim]
        graph_id : str

        Returns
        -------
        SemanticGraph
        """
        nodes_dict: Dict[str, GraphNode] = {}
        edges_list: List[GraphEdge] = []
        adj_map: Dict[str, List[str]] = {}

        for claim in claims:
            extracted_nodes = self._extract_nodes_from_claim(claim)
            for node in extracted_nodes:
                if node.node_id not in nodes_dict:
                    nodes_dict[node.node_id] = node
                if node.node_id not in adj_map:
                    adj_map[node.node_id] = []

            # Connect nodes within the same claim
            node_ids = [n.node_id for n in extracted_nodes]
            predicate = claim.relations[0] if claim.relations else "related_to"

            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    src_id = node_ids[i]
                    tgt_id = node_ids[j]
                    edge_id = f"edge_{hashlib.sha256(f'{claim.claim_id}:{src_id}:{tgt_id}:{predicate}'.encode()).hexdigest()[:10]}"

                    edge = GraphEdge(
                        edge_id=edge_id,
                        source_id=src_id,
                        target_id=tgt_id,
                        predicate=predicate,
                        weight=claim.confidence,
                        claim_id=claim.claim_id,
                    )
                    edges_list.append(edge)
                    if tgt_id not in adj_map[src_id]:
                        adj_map[src_id].append(tgt_id)

        n_nodes = len(nodes_dict)
        n_edges = len(edges_list)
        density = round(2.0 * n_edges / (n_nodes * (n_nodes - 1)), 4) if n_nodes > 1 else 0.0

        graph = SemanticGraph(
            graph_id=graph_id,
            nodes=list(nodes_dict.values()),
            edges=edges_list,
            num_nodes=n_nodes,
            num_edges=n_edges,
            density=density,
            adjacency_list=adj_map,
        )

        logger.info(
            "graph_built_successfully",
            graph_id=graph_id,
            num_nodes=n_nodes,
            num_edges=n_edges,
            density=density,
        )

        return graph

    def _extract_nodes_from_claim(self, claim: ExtractedClaim) -> List[GraphNode]:
        """Extract GraphNodes from claim entities, dates, numbers, and concepts."""
        nodes: List[GraphNode] = []

        # Entities
        for ent in claim.entities:
            category = self._categorize_entity(ent)
            node_id = f"node_{hashlib.sha256(ent.lower().encode()).hexdigest()[:10]}"
            nodes.append(GraphNode(node_id=node_id, label=ent, category=category))

        # Dates
        for dt in claim.dates:
            node_id = f"node_dt_{hashlib.sha256(dt.lower().encode()).hexdigest()[:10]}"
            nodes.append(GraphNode(node_id=node_id, label=dt, category=EntityCategory.DATE))

        # Numbers
        for num in claim.numbers:
            node_id = f"node_num_{hashlib.sha256(str(num).encode()).hexdigest()[:10]}"
            nodes.append(GraphNode(node_id=node_id, label=str(num), category=EntityCategory.QUANTITY))

        # Fallback node if claim text has no explicit entities
        if not nodes:
            fallback_label = claim.claim_text[:30].strip()
            node_id = f"node_txt_{hashlib.sha256(fallback_label.lower().encode()).hexdigest()[:10]}"
            nodes.append(GraphNode(node_id=node_id, label=fallback_label, category=EntityCategory.OTHER))

        return nodes

    def _categorize_entity(self, label: str) -> EntityCategory:
        """Infer EntityCategory for surface label."""
        lbl_lower = label.lower()
        if any(term in lbl_lower for term in self.SCIENTIFIC_TERMS):
            return EntityCategory.SCIENTIFIC_CONCEPT
        if any(term in lbl_lower for term in self.EVENT_TERMS):
            return EntityCategory.EVENT
        if re.search(r"\b(?:Inc|Corp|University|Organization|Institute|Company|Ltd|Group)\b", label, re.I):
            return EntityCategory.ORGANIZATION
        if re.search(r"\b(?:City|Country|Germany|USA|UK|France|Berlin|London|Paris|Tokyo|Mars|Earth)\b", label, re.I):
            return EntityCategory.LOCATION
        return EntityCategory.PERSON
