"""Bonding and Bridging analysis for organizational networks.

Bonding: Connections within groups (strong ties, cohesion)
Bridging: Connections between groups (weak ties, information flow)
"""

import networkx as nx
import pandas as pd
from typing import Dict, Optional
from collections import defaultdict

from orgnet.metrics.community import CommunityDetector


class BondingBridgingAnalyzer:
    """Analyzes bonding (within-group) and bridging (between-group) connections."""

    def __init__(self, graph: nx.Graph):
        """
        Initialize bonding/bridging analyzer.

        Args:
            graph: Organizational graph
        """
        self.graph = graph

    def analyze_bonding_bridging(
        self,
        communities: Optional[Dict] = None,
        use_formal_structure: bool = False,
        team_attribute: str = "department",
    ) -> pd.DataFrame:
        """
        Analyze bonding and bridging metrics for each node.

        Bonding: Connections within the same community/team
        Bridging: Connections across different communities/teams

        Args:
            communities: Optional community detection result
            use_formal_structure: If True, use formal structure (departments/teams) instead of communities
            team_attribute: Attribute name for formal structure grouping

        Returns:
            DataFrame with bonding/bridging metrics for each node
        """
        # Determine group assignments
        if use_formal_structure:
            node_to_group = {}
            for node in self.graph.nodes():
                node_data = self.graph.nodes[node]
                node_to_group[node] = node_data.get(team_attribute, "Unknown")
        elif communities:
            node_to_group = communities.get("node_to_community", {})
        else:
            # Use community detection
            detector = CommunityDetector(self.graph)
            communities = detector.detect_communities()
            node_to_group = communities["node_to_community"]

        results = []

        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            node_group = node_to_group.get(node, -1)

            bonding_weight = 0.0
            bridging_weight = 0.0
            bonding_count = 0
            bridging_count = 0

            for neighbor in neighbors:
                neighbor_group = node_to_group.get(neighbor, -1)
                edge_weight = self.graph[node][neighbor].get("weight", 1.0)

                if neighbor_group == node_group:
                    # Same group - bonding tie
                    bonding_weight += edge_weight
                    bonding_count += 1
                else:
                    # Different group - bridging tie
                    bridging_weight += edge_weight
                    bridging_count += 1

            total_weight = bonding_weight + bridging_weight
            bonding_ratio = bonding_weight / total_weight if total_weight > 0 else 0
            bridging_ratio = bridging_weight / total_weight if total_weight > 0 else 0

            # Bonding/Bridging score: high = more bridging, low = more bonding
            if bonding_weight + bridging_weight > 0:
                bridging_score = bridging_weight / (bonding_weight + bridging_weight)
            else:
                bridging_score = 0

            results.append(
                {
                    "node_id": node,
                    "bonding_weight": bonding_weight,
                    "bridging_weight": bridging_weight,
                    "bonding_count": bonding_count,
                    "bridging_count": bridging_count,
                    "bonding_ratio": bonding_ratio,
                    "bridging_ratio": bridging_ratio,
                    "bridging_score": bridging_score,  # 0 = pure bonding, 1 = pure bridging
                    "total_connections": len(neighbors),
                }
            )

        return pd.DataFrame(results)

    def identify_bridges(self, top_n: int = 20) -> pd.DataFrame:
        """
        Identify key bridge-builders (high bridging, low bonding).

        Args:
            top_n: Number of top bridges to return

        Returns:
            DataFrame with top bridges
        """
        bb_df = self.analyze_bonding_bridging()

        # Sort by bridging score (high bridging, balanced connections)
        bb_df = bb_df[bb_df["total_connections"] >= 3]  # At least some connections
        bb_df = bb_df.sort_values("bridging_score", ascending=False)

        return bb_df.head(top_n)

    def identify_bonders(self, top_n: int = 20) -> pd.DataFrame:
        """
        Identify key team bonders (high bonding, low bridging).

        Args:
            top_n: Number of top bonders to return

        Returns:
            DataFrame with top bonders
        """
        bb_df = self.analyze_bonding_bridging()

        # Sort by bonding ratio (high bonding, low bridging)
        bb_df = bb_df[bb_df["total_connections"] >= 3]
        bb_df = bb_df.sort_values("bonding_ratio", ascending=False)

        return bb_df.head(top_n)

    def analyze_group_bonding_bridging(self, communities: Optional[Dict] = None) -> pd.DataFrame:
        """
        Analyze bonding and bridging at the group/community level.

        Args:
            communities: Optional community detection result

        Returns:
            DataFrame with group-level metrics
        """
        if communities is None:
            detector = CommunityDetector(self.graph)
            communities = detector.detect_communities()

        node_to_community = communities["node_to_community"]

        # Group nodes by community
        community_nodes = defaultdict(list)
        for node, comm_id in node_to_community.items():
            community_nodes[comm_id].append(node)

        results = []

        for comm_id, nodes in community_nodes.items():
            # Calculate internal density (bonding)
            subgraph = self.graph.subgraph(nodes)
            internal_density = nx.density(subgraph) if len(nodes) > 1 else 0
            internal_edges = subgraph.number_of_edges()

            # Calculate external connections (bridging)
            external_edges = 0
            external_weight = 0.0
            for node in nodes:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in nodes:
                        external_edges += 1
                        external_weight += self.graph[node][neighbor].get("weight", 1.0)

            total_nodes = len(nodes)
            avg_external_per_node = external_edges / total_nodes if total_nodes > 0 else 0

            results.append(
                {
                    "community_id": comm_id,
                    "size": total_nodes,
                    "internal_density": internal_density,
                    "internal_edges": internal_edges,
                    "external_edges": external_edges,
                    "external_weight": external_weight,
                    "avg_external_per_node": avg_external_per_node,
                    "bonding_strength": internal_density,
                    "bridging_strength": avg_external_per_node,
                }
            )

        return pd.DataFrame(results)

    def calculate_network_bonding_bridging_ratio(self) -> Dict:
        """
        Calculate overall network bonding/bridging balance.

        Returns:
            Dictionary with network-level metrics
        """
        bb_df = self.analyze_bonding_bridging()

        total_bonding = bb_df["bonding_weight"].sum()
        total_bridging = bb_df["bridging_weight"].sum()
        total_connections = total_bonding + total_bridging

        bonding_ratio = total_bonding / total_connections if total_connections > 0 else 0
        bridging_ratio = total_bridging / total_connections if total_connections > 0 else 0

        return {
            "total_bonding_weight": total_bonding,
            "total_bridging_weight": total_bridging,
            "bonding_ratio": bonding_ratio,
            "bridging_ratio": bridging_ratio,
            "bridging_index": bridging_ratio,  # Higher = more bridging
            "interpretation": self._interpret_ratio(bonding_ratio, bridging_ratio),
        }

    def _interpret_ratio(self, bonding_ratio: float, bridging_ratio: float) -> str:
        """Interpret bonding/bridging ratio using dictionary dispatch."""
        interpretations = {
            (
                bridging_ratio > 0.4,
            ): "High bridging network - good cross-group information flow, but may lack cohesion",
            (
                bridging_ratio < 0.2,
            ): "High bonding network - strong group cohesion, but may have silos",
        }

        # Check conditions in order
        for condition, message in interpretations.items():
            if condition[0]:
                return message

        return "Balanced network - good mix of bonding and bridging"
