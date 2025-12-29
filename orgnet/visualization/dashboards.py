"""Dashboard generation for organizational insights."""

import networkx as nx
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict

from orgnet.metrics.centrality import CentralityAnalyzer
from orgnet.metrics.structural import StructuralAnalyzer
from orgnet.metrics.community import CommunityDetector


class DashboardGenerator:
    """Generates organizational health dashboards."""

    def __init__(self, graph: nx.Graph):
        """
        Initialize dashboard generator.

        Args:
            graph: Organizational graph
        """
        self.graph = graph
        self.centrality_analyzer = CentralityAnalyzer(graph)
        self.structural_analyzer = StructuralAnalyzer(graph)
        self.community_detector = CommunityDetector(graph)

    def generate_health_dashboard(self) -> Dict:
        """
        Generate organizational health dashboard metrics.

        Returns:
            Dictionary with health metrics
        """
        # Basic network metrics
        density = nx.density(self.graph)
        avg_clustering = nx.average_clustering(self.graph)

        # Path length (for largest component)
        if nx.is_connected(self.graph):
            avg_path_length = nx.average_shortest_path_length(self.graph)
        else:
            largest_cc = max(nx.connected_components(self.graph), key=len)
            subgraph = self.graph.subgraph(largest_cc)
            if len(largest_cc) > 1:
                avg_path_length = nx.average_shortest_path_length(subgraph)
            else:
                avg_path_length = 0

        # Modularity
        communities_result = self.community_detector.detect_communities()
        modularity = communities_result.get("modularity", 0.0)

        # Largest component size
        largest_component_size = len(max(nx.connected_components(self.graph), key=len))
        total_nodes = self.graph.number_of_nodes()
        largest_component_pct = (
            (largest_component_size / total_nodes * 100) if total_nodes > 0 else 0
        )

        # Core size
        core_periphery = self.structural_analyzer.compute_core_periphery()
        core_size = len(core_periphery[core_periphery["core_periphery_class"] == "core"])
        core_pct = (core_size / total_nodes * 100) if total_nodes > 0 else 0

        # Centrality inequality (Gini coefficient)
        betweenness_df = self.centrality_analyzer.compute_betweenness_centrality()
        gini = self._compute_gini(betweenness_df["betweenness_centrality"].values)

        return {
            "network_density": density,
            "avg_clustering": avg_clustering,
            "avg_path_length": avg_path_length,
            "modularity": modularity,
            "largest_component_pct": largest_component_pct,
            "core_size_pct": core_pct,
            "centrality_gini": gini,
            "num_nodes": total_nodes,
            "num_edges": self.graph.number_of_edges(),
            "num_communities": communities_result.get("num_communities", 0),
        }

    def _compute_gini(self, values: pd.Series) -> float:
        """Compute Gini coefficient for inequality measure."""
        values = values[values > 0]  # Remove zeros
        if len(values) == 0:
            return 0.0

        values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n

    def generate_executive_summary(self) -> Dict:
        """
        Generate executive summary report.

        Returns:
            Dictionary with executive summary
        """
        health_metrics = self.generate_health_dashboard()

        # Key findings
        findings = []

        # Check for silos
        if health_metrics["modularity"] > 0.6:
            findings.append(
                {
                    "type": "warning",
                    "title": "High Modularity Detected",
                    "description": "Organization shows high modularity, indicating potential silos.",
                    "recommendation": "Consider cross-team initiatives to improve connectivity.",
                }
            )

        # Check for fragmentation
        if health_metrics["largest_component_pct"] < 95:
            findings.append(
                {
                    "type": "warning",
                    "title": "Network Fragmentation",
                    "description": f'Only {health_metrics["largest_component_pct"]:.1f}% of organization in largest component.',
                    "recommendation": "Investigate isolated groups and improve connectivity.",
                }
            )

        # Check for inequality
        if health_metrics["centrality_gini"] > 0.5:
            findings.append(
                {
                    "type": "info",
                    "title": "High Centrality Inequality",
                    "description": "Network influence is highly concentrated.",
                    "recommendation": "Consider distributing responsibilities more evenly.",
                }
            )

        # Top brokers
        betweenness_df = self.centrality_analyzer.compute_betweenness_centrality()
        constraint_df = self.structural_analyzer.compute_constraint()
        brokers = self.structural_analyzer.identify_brokers(betweenness_df, constraint_df, top_n=5)

        return {
            "timestamp": datetime.now().isoformat(),
            "health_metrics": health_metrics,
            "key_findings": findings,
            "top_brokers": brokers.to_dict("records") if not brokers.empty else [],
            "status": self._compute_overall_status(health_metrics),
        }

    def _compute_overall_status(self, metrics: Dict) -> str:
        """Compute overall organizational health status."""
        warning_conditions = [
            metrics["modularity"] > 0.6,
            metrics["largest_component_pct"] < 95,
            metrics["network_density"] < 0.04,
        ]
        warnings = sum(warning_conditions)

        status_map = {0: "healthy", 1: "moderate"}

        return status_map.get(warnings, "needs_attention")
