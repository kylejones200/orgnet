"""Structural analysis (holes, core-periphery, bridges)."""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from orgnet.utils.logging import get_logger
from orgnet.utils.performance import NUMBA_AVAILABLE, compute_constraint_numba
from orgnet.metrics.utils import standardize_metric_output

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

    def compute_constraint(
        self, standardize: bool = True, top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compute Burt's constraint measure (structural holes).

        Lower constraint = more brokerage opportunity.

        Args:
            standardize: If True, add ranks and flags
            top_n: Number of top nodes to flag (for low constraint = high brokerage)

        Returns:
            DataFrame with node_id, value, rank, flags
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

            if NUMBA_AVAILABLE and len(neighbors) > 10:
                weights = np.array(
                    [self.graph[node][n].get("weight", 1.0) for n in neighbors], dtype=np.float64
                )
                neighbor_indices = np.arange(len(neighbors))
                constraint = compute_constraint_numba(neighbor_indices, weights, total_weight)
            else:
                constraint = 0.0
                for j in neighbors:
                    p_ij = self.graph[node][j].get("weight", 1.0) / total_weight

                    indirect = 0.0
                    for k in neighbors:
                        if k == j:
                            continue
                        if self.graph.has_edge(k, j):
                            p_ik = self.graph[node][k].get("weight", 1.0) / total_weight
                            k_neighbors = list(self.graph.neighbors(k))
                            k_total = sum(self.graph[k][n].get("weight", 1.0) for n in k_neighbors)
                            if k_total > 0:
                                p_kj = self.graph[k][j].get("weight", 1.0) / k_total
                                indirect += p_ik * p_kj

                    constraint += (p_ij + indirect) ** 2

            constraint_scores[node] = constraint

        df = pd.DataFrame(
            {
                "node_id": list(constraint_scores.keys()),
                "constraint": list(constraint_scores.values()),
            }
        )

        if standardize:
            # For constraint, lower is better (more brokerage opportunity)
            # We'll use inverted constraint (brokerage opportunity) for ranking
            df["brokerage_opportunity"] = 1.0 / (df["constraint"] + 1e-10)
            df = standardize_metric_output(
                df, value_column="brokerage_opportunity", id_column="node_id", top_n=top_n
            )
            # Rename the standardized value column
            df = df.rename(columns={"value": "brokerage_score"})
            # Drop the temporary brokerage_opportunity column used for ranking
            df = df.drop(columns=["brokerage_opportunity"])

        return df.sort_values("constraint", ascending=True)

    def compute_core_periphery(
        self, standardize: bool = True, top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compute k-core decomposition and core-periphery structure.

        Args:
            standardize: If True, add ranks and flags
            top_n: Number of top nodes to flag (highest coreness)

        Returns:
            DataFrame with node_id, value (coreness), rank, flags, and core_periphery_class
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

        if standardize:
            df = standardize_metric_output(
                df, value_column="coreness", id_column="node_id", top_n=top_n
            )

        return df.sort_values("coreness" if not standardize else "value", ascending=False)

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
        from orgnet.utils.performance import polars_join

        if len(betweenness_df) > 10000 or len(constraint_df) > 10000:
            merged = polars_join(betweenness_df, constraint_df, on="node_id", how="inner")
        else:
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

    def compute_clustering_coefficient(
        self, standardize: bool = True, top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compute local clustering coefficient for each node.

        Args:
            standardize: If True, add ranks and flags
            top_n: Number of top nodes to flag

        Returns:
            DataFrame with node_id, value, rank, flags
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

        if standardize:
            df = standardize_metric_output(
                df, value_column="clustering_coefficient", id_column="node_id", top_n=top_n
            )

        return df.sort_values(
            "clustering_coefficient" if not standardize else "value", ascending=False
        )

    def detect_bridge_nodes(
        self, communities: Optional[Dict] = None, min_betweenness: float = 0.1, method: str = "both"
    ) -> List[str]:
        """
        Detect bridge nodes that connect different parts of the network (from Enron project).

        Bridge nodes have high betweenness centrality and connect different communities.

        Args:
            communities: Optional community detection results from CommunityDetector
            min_betweenness: Minimum betweenness centrality threshold
            method: Detection method ('betweenness', 'community_cut', 'both')

        Returns:
            List of bridge node identifiers
        """
        logger.info("Detecting bridge nodes...")

        bridge_nodes = []

        if method in ["betweenness", "both"]:
            # Method 1: High betweenness centrality
            try:
                if len(self.graph) > 500:
                    betweenness = nx.betweenness_centrality(
                        self.graph, k=min(500, len(self.graph)), normalized=True, seed=42
                    )
                else:
                    betweenness = nx.betweenness_centrality(self.graph, normalized=True)

                # Nodes with high betweenness are potential bridges
                max_betweenness = max(betweenness.values()) if betweenness.values() else 0
                threshold = max_betweenness * min_betweenness

                betweenness_bridges = [
                    node for node, score in betweenness.items() if score >= threshold
                ]

                bridge_nodes.extend(betweenness_bridges)
                logger.info(
                    f"Found {len(betweenness_bridges)} bridge nodes by betweenness centrality"
                )
            except Exception as e:
                logger.warning(f"Betweenness-based bridge detection failed: {e}")

        if method in ["community_cut", "both"] and communities:
            # Method 2: Nodes connecting different communities
            community_bridges = self._find_inter_community_bridges(communities)
            bridge_nodes.extend(community_bridges)
            logger.info(f"Found {len(community_bridges)} bridge nodes connecting communities")

        # Remove duplicates
        bridge_nodes = list(set(bridge_nodes))

        logger.info(f"Total unique bridge nodes: {len(bridge_nodes)}")
        return bridge_nodes

    def _find_inter_community_bridges(self, communities: Dict) -> List[str]:
        """
        Find nodes that connect different communities.

        Args:
            communities: Community detection results with 'communities' key

        Returns:
            List of bridge nodes connecting communities
        """
        bridge_nodes = []

        comm_list = communities.get("communities", [])
        if not comm_list:
            return bridge_nodes

        # Build community membership map
        node_to_community = {}
        for comm_id, nodes in enumerate(comm_list):
            for node in nodes:
                node_to_community[node] = comm_id

        # Find nodes with neighbors in different communities
        for node in self.graph.nodes():
            if node not in node_to_community:
                continue

            neighbor_communities = set()

            for neighbor in self.graph.neighbors(node):
                if neighbor in node_to_community:
                    neighbor_communities.add(node_to_community[neighbor])

            # If node has neighbors in different communities, it's a bridge
            if len(neighbor_communities) > 1:
                bridge_nodes.append(node)

        return bridge_nodes

    def analyze_bridge_structure(self, communities: Optional[Dict] = None) -> Dict:
        """
        Comprehensive bridge node analysis (from Enron project).

        Args:
            communities: Optional community detection results

        Returns:
            Dictionary with bridge analysis results
        """
        logger.info("Analyzing bridge structure...")

        # Detect communities if not provided
        if communities is None:
            from orgnet.metrics.community import CommunityDetector

            detector = CommunityDetector(self.graph)
            communities = detector.detect_communities(method="louvain")

        # Detect bridge nodes
        bridge_nodes = self.detect_bridge_nodes(communities, method="both")

        # Compute bridge scores (betweenness centrality)
        try:
            if len(self.graph) > 500:
                betweenness = nx.betweenness_centrality(
                    self.graph, k=min(500, len(self.graph)), normalized=True, seed=42
                )
            else:
                betweenness = nx.betweenness_centrality(self.graph, normalized=True)

            bridge_scores = {node: betweenness.get(node, 0.0) for node in bridge_nodes}
        except Exception:
            bridge_scores = {node: 1.0 for node in bridge_nodes}

        # Find which communities each bridge connects
        comm_list = communities.get("communities", [])
        node_to_community = {}
        for comm_id, comm_nodes in enumerate(comm_list):
            for node in comm_nodes:
                node_to_community[node] = comm_id

        inter_community_bridges = {}
        for bridge in bridge_nodes:
            if bridge not in node_to_community:
                continue

            connected_communities = set([node_to_community[bridge]])

            for neighbor in self.graph.neighbors(bridge):
                if neighbor in node_to_community:
                    connected_communities.add(node_to_community[neighbor])

            if len(connected_communities) > 1:
                inter_community_bridges[bridge] = list(connected_communities)

        # Find isolated nodes
        isolated_nodes = [node for node in self.graph.nodes() if self.graph.degree(node) == 0]

        return {
            "bridge_nodes": bridge_nodes,
            "bridge_scores": bridge_scores,
            "inter_community_bridges": inter_community_bridges,
            "isolated_nodes": isolated_nodes,
        }

    def identify_critical_bridges(
        self,
        bridge_analysis: Optional[Dict] = None,
        communities: Optional[Dict] = None,
        top_n: int = 10,
    ) -> List[tuple]:
        """
        Identify the most critical bridge nodes (from Enron project).

        Critical bridges:
        - High bridge scores (betweenness)
        - Connect many communities
        - High degree centrality

        Args:
            bridge_analysis: Optional bridge analysis results
            communities: Optional community detection results
            top_n: Number of top bridges to return

        Returns:
            List of (node, score, n_communities, communities) tuples
        """
        if bridge_analysis is None:
            bridge_analysis = self.analyze_bridge_structure(communities)

        bridge_info = []

        for bridge in bridge_analysis["bridge_nodes"]:
            score = bridge_analysis["bridge_scores"].get(bridge, 0.0)
            communities_list = bridge_analysis["inter_community_bridges"].get(bridge, [])

            # Composite score: bridge score * number of communities connected
            composite_score = score * (1 + len(communities_list))

            bridge_info.append((bridge, composite_score, len(communities_list), communities_list))

        # Sort by composite score
        bridge_info.sort(key=lambda x: x[1], reverse=True)

        return bridge_info[:top_n]
