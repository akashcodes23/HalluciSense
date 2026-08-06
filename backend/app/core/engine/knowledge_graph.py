"""Part 4 — Hallucination Knowledge Graph Engine.

Constructs directed multi-graph representations G = (V, E) of responses.

Nodes:
- Claims
- Entities
- Evidence Passages
- Sources
- Reasoning Steps

Edges:
- supports
- contradicts
- depends_on
- paraphrases
- refers_to

Computes graph consistency index C_G and exports GraphML XML format.
"""

from __future__ import annotations

import re
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class GraphNode:
    id: str
    label: str
    node_type: str  # "claim", "entity", "evidence", "source", "reasoning"
    risk_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    relation: str  # "supports", "contradicts", "depends_on", "paraphrases", "refers_to"
    weight: float = 1.0


class HallucinationKnowledgeGraph:
    """Hallucination Knowledge Graph construction, analysis, and GraphML export."""

    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []

    def add_node(self, node_id: str, label: str, node_type: str, risk_score: float = 0.0, metadata: Optional[Dict[str, Any]] = None):
        self.nodes[node_id] = GraphNode(
            id=node_id,
            label=label,
            node_type=node_type,
            risk_score=risk_score,
            metadata=metadata or {},
        )

    def add_edge(self, source_id: str, target_id: str, relation: str, weight: float = 1.0):
        self.edges.append(GraphEdge(source_id=source_id, target_id=target_id, relation=relation, weight=weight))

    def build_graph_from_claims_and_evidence(
        self,
        claims: List[str],
        evidence_snippets: List[Dict[str, Any]],
        h_scores: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Construct graph G = (V, E) from claims and evidence passages."""
        self.nodes.clear()
        self.edges.clear()

        # Add Claim Nodes
        for i, c in enumerate(claims):
            c_id = f"claim_{i+1}"
            score = h_scores[i] if (h_scores and i < len(h_scores)) else 0.20
            self.add_node(node_id=c_id, label=c, node_type="claim", risk_score=score)

            # Extract Entity Nodes
            entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", c)
            for ent in set(entities):
                ent_id = f"ent_{hash(ent) & 0xFFFFFF:06x}"
                if ent_id not in self.nodes:
                    self.add_node(node_id=ent_id, label=ent, node_type="entity", risk_score=0.0)
                self.add_edge(source_id=c_id, target_id=ent_id, relation="refers_to")

        # Add Evidence & Source Nodes
        for j, ev in enumerate(evidence_snippets):
            ev_id = f"ev_{j+1}"
            snippet = ev.get("snippet", "")
            source_name = ev.get("source_name", "External Source")
            is_supporting = ev.get("is_supporting", True)

            self.add_node(node_id=ev_id, label=snippet[:50] + "...", node_type="evidence", risk_score=0.0)
            src_id = f"src_{hash(source_name) & 0xFFFFFF:06x}"
            if src_id not in self.nodes:
                self.add_node(node_id=src_id, label=source_name, node_type="source", risk_score=0.0)
            self.add_edge(source_id=ev_id, target_id=src_id, relation="depends_on")

            # Connect Evidence to Claims
            for c_id in self.nodes:
                if self.nodes[c_id].node_type == "claim":
                    rel = "supports" if is_supporting else "contradicts"
                    self.add_edge(source_id=ev_id, target_id=c_id, relation=rel, weight=ev.get("similarity_score", 0.85))

        # Compute Graph Consistency Index C_G
        contradict_count = sum(1 for e in self.edges if e.relation == "contradicts")
        total_claims = max(1, len(claims))
        graph_consistency = round(max(0.0, 1.0 - (contradict_count / (total_claims ** 2))), 4)

        return {
            "graph_consistency_index": graph_consistency,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "contradiction_edges": contradict_count,
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges],
        }

    def export_graphml(self) -> str:
        """Export graph into standard GraphML XML format."""
        graphml = ET.Element("graphml", xmlns="http://graphml.graphdrawing.org/xmlns")
        graph = ET.SubElement(graphml, "graph", id="G", edgedefault="directed")

        for n in self.nodes.values():
            node_elem = ET.SubElement(graph, "node", id=n.id)
            d_label = ET.SubElement(node_elem, "data", key="label")
            d_label.text = n.label
            d_type = ET.SubElement(node_elem, "data", key="type")
            d_type.text = n.node_type

        for i, e in enumerate(self.edges):
            edge_elem = ET.SubElement(graph, "edge", id=f"e{i+1}", source=e.source_id, target=e.target_id)
            d_rel = ET.SubElement(edge_elem, "data", key="relation")
            d_rel.text = e.relation

        return ET.tostring(graphml, encoding="unicode")
