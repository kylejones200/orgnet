"""Ego network analysis for individual nodes."""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List

from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


class EgoNetworkAnalyzer:
    """Analyzes ego networks (personal networks) for individuals."""

    def __init__(self, graph: nx.Graph):
        """
        Initialize ego network analyzer.

        Args:
            graph: Organizational graph
        """
        self.graph = graph

    def analyze_ego_network(self, node_id: str, radius: int = 1) -> Dict:
        """
        Analyze ego network for a specific node.

        Args:
            node_id: Node identifier
            radius: Radius of ego network (1 = direct connections only)

        Returns:
            Dictionary with ego network metrics
        """
        if node_id not in self.graph:
            raise ValueError(f"Node {node_id} not in graph")

        # Get ego network
        ego_graph = nx.ego_graph(self.graph, node_id, radius=radius, undirected=True)

        # Basic metrics
        ego_size = ego_graph.number_of_nodes() - 1  # Exclude ego itself
        ego_density = nx.density(ego_graph)

        # Direct connections
        direct_neighbors = list(self.graph.neighbors(node_id))
        num_direct = len(direct_neighbors)

        # Weighted degree in ego network
        total_weight = sum(
            self.graph[node_id][neighbor].get("weight", 1.0) for neighbor in direct_neighbors
        )

        # Clustering in ego network
        clustering = nx.clustering(ego_graph, node_id)

        # Structural holes (constraint of ego)
        constraint = self._compute_ego_constraint(node_id, direct_neighbors)

        # Diversity: departments/roles of connections
        diversity = self._compute_diversity(node_id, direct_neighbors)

        return {
            "node_id": node_id,
            "ego_size": ego_size,
            "num_direct_connections": num_direct,
            "total_weight": total_weight,
            "ego_density": ego_density,
            "clustering": clustering,
            "constraint": constraint,
            "diversity_score": diversity.get("total_diversity", 0),
            "department_diversity": diversity.get("departments", 0),
            "role_diversity": diversity.get("roles", 0),
        }

    def _compute_ego_constraint(self, node_id: str, neighbors: List[str]) -> float:
        """Compute constraint for ego network."""
        if len(neighbors) < 2:
            return 1.0  # Fully constrained

        # Get total weight to neighbors
        total_weight = sum(self.graph[node_id][n].get("weight", 1.0) for n in neighbors)

        if total_weight == 0:
            return 1.0

        constraint = 0.0

        for j in neighbors:
            p_ij = self.graph[node_id][j].get("weight", 1.0) / total_weight

            # Indirect constraint through other neighbors
            indirect = 0.0
            for k in neighbors:
                if k == j:
                    continue
                if self.graph.has_edge(k, j):
                    p_ik = self.graph[node_id][k].get("weight", 1.0) / total_weight
                    p_kj = self.graph[k][j].get("weight", 1.0) / sum(
                        self.graph[k][n].get("weight", 1.0) for n in self.graph.neighbors(k)
                    )
                    indirect += p_ik * p_kj

            constraint += (p_ij + indirect) ** 2

        return constraint

    def _compute_diversity(self, node_id: str, neighbors: List[str]) -> Dict:
        """Compute diversity of connections (departments, roles)."""
        departments = set()
        roles = set()

        for neighbor in neighbors:
            if neighbor in self.graph.nodes():
                node_data = self.graph.nodes[neighbor]
                if "department" in node_data and node_data["department"]:
                    departments.add(node_data["department"])
                if "role" in node_data and node_data["role"]:
                    roles.add(node_data["role"])

        # Normalize by number of connections
        num_connections = len(neighbors)
        dept_diversity = len(departments) / num_connections if num_connections > 0 else 0
        role_diversity = len(roles) / num_connections if num_connections > 0 else 0

        total_diversity = (dept_diversity + role_diversity) / 2

        return {
            "departments": len(departments),
            "roles": len(roles),
            "department_diversity": dept_diversity,
            "role_diversity": role_diversity,
            "total_diversity": total_diversity,
        }

    def compare_ego_networks(self, node_ids: List[str], radius: int = 1) -> pd.DataFrame:
        """
        Compare ego networks of multiple nodes.

        Args:
            node_ids: List of node identifiers
            radius: Radius of ego network

        Returns:
            DataFrame with comparison metrics
        """
        results = []

        for node_id in node_ids:
            try:
                ego_metrics = self.analyze_ego_network(node_id, radius)
                results.append(ego_metrics)
            except ValueError:
                continue

        return pd.DataFrame(results)

    def identify_ego_network_roles(self, node_id: str) -> Dict[str, List[str]]:
        """
        Identify structural roles in ego network (broker, hub, isolated, etc.).

        Args:
            node_id: Node identifier

        Returns:
            Dictionary mapping role types to node IDs
        """
        if node_id not in self.graph:
            return {}

        roles = {
            "broker": [],  # Low constraint, high betweenness
            "hub": [],  # High degree
            "periphery": [],  # Low degree, high constraint
        }

        direct_neighbors = list(self.graph.neighbors(node_id))

        if not direct_neighbors:
            return roles

        # Vectorized degree calculation
        all_degrees = [self.graph.degree(n) for n in self.graph.nodes()]
        degree_75th = np.percentile(all_degrees, 75)
        degree_25th = np.percentile(all_degrees, 25)

        # Vectorized neighbor degree calculation
        neighbor_degrees = [self.graph.degree(n) for n in direct_neighbors]

        # Vectorized role assignment
        for neighbor, degree in zip(direct_neighbors, neighbor_degrees):
            if degree > degree_75th:
                roles["hub"].append(neighbor)
            elif degree < degree_25th:
                roles["periphery"].append(neighbor)
            else:
                # Could be broker
                roles["broker"].append(neighbor)

        return roles
