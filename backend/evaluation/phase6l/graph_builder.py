"""Phase 6L.1B — Contradiction Graph Builder & Topological Feature Extractor.

Constructs an undirected contradiction graph G = (V, E_tau) on response claims,
and computes Family G topological features:
    * contradiction_graph_density (2|E| / (n(n-1)))
    * max_contradiction_degree (normalized max_deg / (n-1))
    * largest_contradictory_component_ratio (LCC_size / n if |E| > 0 else 0.0)

Strict Data Firewall Rule:
    * Label-free: No rule or threshold depends on ground truth target y.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple
import numpy as np
import structlog

from evaluation.phase6l.config import TAU_CONTRADICTION

logger = structlog.get_logger(__name__)


def build_contradiction_graph(
    n_claims: int,
    evaluated_pairs: List[Dict[str, Any]],
    tau_contradiction: float = TAU_CONTRADICTION,
) -> Tuple[Dict[int, Set[int]], int]:
    """Build adjacency list for thresholded contradiction graph G = (V, E).

    Args:
        n_claims: Total atomic claims n.
        evaluated_pairs: List of evaluated claim pair dicts containing 'c_max', 'claim_i_index', 'claim_j_index'.
        tau_contradiction: Threshold for edge inclusion (default 0.50).

    Returns:
        Tuple of (adjacency_dict, total_edge_count).
    """
    adj: Dict[int, Set[int]] = {i: set() for i in range(n_claims)}
    edge_count = 0

    for p in evaluated_pairs:
        c_max_v = p.get("c_max", 0.0)
        if c_max_v >= tau_contradiction:
            i = p["claim_i_index"]
            j = p["claim_j_index"]
            if i < n_claims and j < n_claims and i != j:
                if j not in adj[i]:
                    adj[i].add(j)
                    adj[j].add(i)
                    edge_count += 1

    return adj, edge_count


def compute_largest_connected_component_size(adj: Dict[int, Set[int]], edge_count: int) -> int:
    """Compute the size of the largest connected component in graph G."""
    if edge_count == 0:
        return 0

    visited: Set[int] = set()
    max_component_size = 0

    for node in adj:
        if node not in visited:
            # BFS / DFS traversal
            component_size = 0
            queue = [node]
            visited.add(node)

            while queue:
                curr = queue.pop(0)
                component_size += 1
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if component_size > max_component_size:
                max_component_size = component_size

    return max_component_size


def extract_graph_topological_features(
    n_claims: int,
    evaluated_pairs: List[Dict[str, Any]],
    tau_contradiction: float = TAU_CONTRADICTION,
) -> Dict[str, Any]:
    """Compute Family G topological graph features.

    Args:
        n_claims: Total atomic claim count n.
        evaluated_pairs: List of claim pair dicts.
        tau_contradiction: Threshold for contradiction edge (0.50).

    Returns:
        Dict containing 3 graph features and graph metadata.
    """
    if n_claims < 2:
        return {
            "contradiction_graph_density": 0.0,
            "max_contradiction_degree": 0.0,
            "largest_contradictory_component_ratio": 0.0,
            "total_graph_edges": 0,
            "largest_component_size": 0,
        }

    adj, edge_count = build_contradiction_graph(n_claims, evaluated_pairs, tau_contradiction)
    m_possible_edges = (n_claims * (n_claims - 1)) // 2

    # 20. Density
    density = float(edge_count / m_possible_edges) if m_possible_edges > 0 else 0.0

    # 21. Normalized Max Degree: max_deg / (n - 1)
    degrees = [len(neighbors) for neighbors in adj.values()]
    raw_max_deg = max(degrees) if degrees else 0
    norm_max_degree = float(raw_max_deg / (n_claims - 1)) if n_claims > 1 else 0.0

    # 22. LCC Ratio: LCC_size / n if edge_count > 0 else 0.0
    lcc_size = compute_largest_connected_component_size(adj, edge_count)
    lcc_ratio = float(lcc_size / n_claims) if edge_count > 0 else 0.0

    return {
        "contradiction_graph_density": density,
        "max_contradiction_degree": norm_max_degree,
        "largest_contradictory_component_ratio": lcc_ratio,
        "total_graph_edges": edge_count,
        "largest_component_size": lcc_size,
    }
