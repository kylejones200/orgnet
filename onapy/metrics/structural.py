"""Structural analysis (holes, core-periphery)."""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from onapy.utils.logging import get_logger

logger = get_logger(__name__)


class StructuralAnalyzer:
    """Analyzes structural properties of organizational networks."""

    def __init__(self, graph: nx.Graph):
        """
        Initialize structural analyzer.

        Args:
            graph: NetworkX graph
        """
        self.graph = graph

    def compute_constraint(self) -> pd.DataFrame:
        """
        Compute Burt's constraint measure (structural holes).

        Lower constraint = more brokerage opportunity.

        Returns:
            DataFrame with node_id and constraint
        """
        constraint_scores = {}

        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            if len(neighbors) < 2:
                constraint_scores[node] = 1.0  # Fully constrained
                continue

            # Compute p_ij (proportion of i's network invested in j)
            total_weight = sum(self.graph[node][n].get("weight", 1.0) for n in neighbors)

            if total_weight == 0:
                constraint_scores[node] = 1.0
                continue

            constraint = 0.0
            for j in neighbors:
                p_ij = self.graph[node][j].get("weight", 1.0) / total_weight

                # Compute indirect constraint through k
                indirect = 0.0
                for k in neighbors:
                    if k == j:
                        continue
                    if self.graph.has_edge(k, j):
                        p_ik = self.graph[node][k].get("weight", 1.0) / total_weight
                        p_kj = self.graph[k][j].get("weight", 1.0) / sum(
                            self.graph[k][n].get("weight", 1.0) for n in self.graph.neighbors(k)
                        )
                        indirect += p_ik * p_kj

                constraint += (p_ij + indirect) ** 2

            constraint_scores[node] = constraint

        df = pd.DataFrame(
            {
                "node_id": list(constraint_scores.keys()),
                "constraint": list(constraint_scores.values()),
            }
        )

        return df.sort_values("constraint")

    def compute_core_periphery(self) -> pd.DataFrame:
        """
        Compute k-core decomposition and core-periphery structure.

        Returns:
            DataFrame with node_id, coreness, and core_periphery_class
        """
        # K-core decomposition
        coreness = {}
        max_k = 0

        # Find maximum k
        for k in range(1, self.graph.number_of_nodes() + 1):
            try:
                k_core = nx.k_core(self.graph, k=k)
                if k_core.number_of_nodes() > 0:
                    for node in k_core.nodes():
                        coreness[node] = k
                    max_k = k
                else:
                    break
            except Exception:
                break

        # Assign coreness to all nodes
        for node in self.graph.nodes():
            if node not in coreness:
                coreness[node] = 0

        # Classify as core, semi-periphery, or periphery using threshold-based mapping
        thresholds = {
            "core": max_k * 0.7 if max_k > 0 else 0,
            "periphery": max_k * 0.3 if max_k > 0 else 0,
        }

        classification = {}
        for node, k in coreness.items():
            if k >= thresholds["core"]:
                classification[node] = "core"
            elif k >= thresholds["periphery"]:
                classification[node] = "semi-periphery"
            else:
                classification[node] = "periphery"

        df = pd.DataFrame(
            {
                "node_id": list(coreness.keys()),
                "coreness": list(coreness.values()),
                "core_periphery_class": [
                    classification.get(n, "periphery") for n in coreness.keys()
                ],
            }
        )

        return df.sort_values("coreness", ascending=False)

    def identify_brokers(
        self, betweenness_df: pd.DataFrame, constraint_df: pd.DataFrame, top_n: int = 10
    ) -> pd.DataFrame:
        """
        Identify effective brokers (low constraint, high betweenness).

        Args:
            betweenness_df: DataFrame with betweenness centrality
            constraint_df: DataFrame with constraint scores
            top_n: Number of top brokers

        Returns:
            DataFrame with broker analysis
        """
        # Merge dataframes
        merged = betweenness_df.merge(constraint_df, on="node_id", how="inner")

        # Broker score: high betweenness, low constraint
        # Normalize both to [0, 1] and combine
        merged["betweenness_norm"] = (
            merged["betweenness_centrality"] - merged["betweenness_centrality"].min()
        ) / (
            merged["betweenness_centrality"].max() - merged["betweenness_centrality"].min() + 1e-10
        )
        merged["constraint_norm"] = (merged["constraint"].max() - merged["constraint"]) / (
            merged["constraint"].max() - merged["constraint"].min() + 1e-10
        )

        merged["broker_score"] = 0.6 * merged["betweenness_norm"] + 0.4 * merged["constraint_norm"]

        return merged.nlargest(top_n, "broker_score")

    def compute_clustering_coefficient(self) -> pd.DataFrame:
        """
        Compute local clustering coefficient for each node.

        Returns:
            DataFrame with node_id and clustering_coefficient
        """
        clustering = nx.clustering(
            self.graph, weight="weight" if nx.is_weighted(self.graph) else None
        )

        df = pd.DataFrame(
            {
                "node_id": list(clustering.keys()),
                "clustering_coefficient": list(clustering.values()),
            }
        )

        return df.sort_values("clustering_coefficient", ascending=False)
