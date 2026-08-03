"""
HalluciSense Pillar 2 — Entity + Relation Graph Schemas
========================================================
Pydantic schemas for graph nodes, edges, entity types, and graph representations.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EntityCategory(str, Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    DATE = "DATE"
    SCIENTIFIC_CONCEPT = "SCIENTIFIC_CONCEPT"
    EVENT = "EVENT"
    QUANTITY = "QUANTITY"
    OTHER = "OTHER"


class GraphNode(BaseModel):
    node_id: str = Field(..., description="Unique node identifier")
    label: str = Field(..., description="Entity name or surface text")
    category: EntityCategory = Field(default=EntityCategory.OTHER, description="Entity category")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Additional node metadata")


class GraphEdge(BaseModel):
    edge_id: str = Field(..., description="Unique edge identifier")
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    predicate: str = Field(..., description="Relational verb or relationship description")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Edge confidence weight")
    claim_id: Optional[str] = Field(None, description="Originating claim ID")


class SemanticGraph(BaseModel):
    graph_id: str = Field(..., description="Unique graph identifier")
    nodes: List[GraphNode] = Field(default_factory=list, description="List of graph nodes")
    edges: List[GraphEdge] = Field(default_factory=list, description="List of directed graph edges")
    num_nodes: int = Field(0, description="Node count")
    num_edges: int = Field(0, description="Edge count")
    density: float = Field(0.0, description="Graph structural density")
    adjacency_list: Dict[str, List[str]] = Field(default_factory=dict, description="Adjacency map source -> targets")
