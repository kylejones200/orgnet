"""Temporal graph handling."""

import networkx as nx
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from orgnet.data.models import Person, Interaction, Meeting, Document, CodeCommit
from orgnet.graph.builder import GraphBuilder
from orgnet.config import Config
from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


class TemporalGraph:
    """Manages temporal graph snapshots."""

    def __init__(self, config: Config):
        """
        Initialize temporal graph manager.

        Args:
            config: Configuration object
        """
        self.config = config
        self.graph_builder = GraphBuilder(config)
        self.snapshot_interval_days = config.graph_config.get("snapshot_interval_days", 7)

    def build_snapshots(
        self,
        people: List[Person],
        interactions: Optional[List[Interaction]] = None,
        meetings: Optional[List[Meeting]] = None,
        documents: Optional[List[Document]] = None,
        commits: Optional[List[CodeCommit]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict]:
        """
        Build temporal graph snapshots.

        Args:
            people: List of Person objects
            interactions: List of Interaction objects
            meetings: List of Meeting objects
            documents: List of Document objects
            commits: List of CodeCommit objects
            start_date: Start date for snapshots
            end_date: End date for snapshots

        Returns:
            List of dictionaries with 'timestamp' and 'graph' keys
        """
        if end_date is None:
            end_date = datetime.now()

        if start_date is None:
            # Find earliest data point
            earliest = end_date
            for data_list in [interactions, meetings, documents, commits]:
                if data_list:
                    for item in data_list:
                        if hasattr(item, "timestamp"):
                            if item.timestamp < earliest:
                                earliest = item.timestamp
                        elif hasattr(item, "start_time"):
                            if item.start_time < earliest:
                                earliest = item.start_time
                        elif hasattr(item, "created_at"):
                            if item.created_at < earliest:
                                earliest = item.created_at
            start_date = earliest

        logger.info(f"Building temporal snapshots from {start_date} to {end_date}")

        snapshots = []
        current_date = start_date

        while current_date <= end_date:
            snapshot_end = current_date + timedelta(days=self.snapshot_interval_days)

            # Filter data to this time window
            snapshot_interactions = (
                self._filter_by_date(interactions, current_date, snapshot_end)
                if interactions
                else None
            )
            snapshot_meetings = (
                self._filter_by_date(meetings, current_date, snapshot_end) if meetings else None
            )
            snapshot_documents = (
                self._filter_by_date(documents, current_date, snapshot_end) if documents else None
            )
            snapshot_commits = (
                self._filter_by_date(commits, current_date, snapshot_end) if commits else None
            )

            # Build graph for this snapshot
            graph = self.graph_builder.build_graph(
                people=people,
                interactions=snapshot_interactions,
                meetings=snapshot_meetings,
                documents=snapshot_documents,
                commits=snapshot_commits,
            )

            snapshots.append(
                {
                    "timestamp": current_date,
                    "graph": graph,
                    "node_count": graph.number_of_nodes(),
                    "edge_count": graph.number_of_edges(),
                }
            )

            current_date = snapshot_end

        logger.info(f"Built {len(snapshots)} temporal snapshots")
        return snapshots

    def _filter_by_date(self, items: List, start: datetime, end: datetime) -> List:
        """Filter items by date range."""
        filtered = []
        for item in items:
            timestamp = None
            if hasattr(item, "timestamp"):
                timestamp = item.timestamp
            elif hasattr(item, "start_time"):
                timestamp = item.start_time
            elif hasattr(item, "created_at"):
                timestamp = item.created_at

            if timestamp and start <= timestamp <= end:
                filtered.append(item)

        return filtered

    def compute_temporal_metrics(self, snapshots: List[Dict]) -> pd.DataFrame:
        """
        Compute metrics for each snapshot.

        Args:
            snapshots: List of snapshot dictionaries

        Returns:
            DataFrame with temporal metrics
        """
        metrics = []

        for snapshot in snapshots:
            graph = snapshot["graph"]
            timestamp = snapshot["timestamp"]

            if graph.number_of_nodes() == 0:
                continue

            # Basic metrics
            density = nx.density(graph)
            avg_clustering = nx.average_clustering(graph)

            # Path length (only if connected)
            if nx.is_connected(graph):
                avg_path_length = nx.average_shortest_path_length(graph)
            else:
                # Compute for largest component
                largest_cc = max(nx.connected_components(graph), key=len)
                subgraph = graph.subgraph(largest_cc)
                if len(largest_cc) > 1:
                    avg_path_length = nx.average_shortest_path_length(subgraph)
                else:
                    avg_path_length = 0

            metrics.append(
                {
                    "timestamp": timestamp,
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                    "density": density,
                    "avg_clustering": avg_clustering,
                    "avg_path_length": avg_path_length,
                    "connected_components": nx.number_connected_components(graph),
                    "largest_component_size": (
                        len(max(nx.connected_components(graph), key=len))
                        if graph.number_of_nodes() > 0
                        else 0
                    ),
                }
            )

        return pd.DataFrame(metrics)

    def analyze_network_evolution(
        self, snapshots: List[Dict], node_subset: Optional[List[str]] = None
    ) -> Dict:
        """
        Analyze how network structure evolves over time (from Enron project).

        Args:
            snapshots: List of snapshot dictionaries
            node_subset: Optional subset of nodes to track

        Returns:
            Dictionary with evolution analysis results
        """
        logger.info("Analyzing network evolution...")

        if not snapshots:
            return {
                "time_periods": [],
                "network_size_over_time": [],
                "network_density_over_time": [],
                "node_metrics_over_time": {},
                "community_evolution": {},
            }

        time_periods = [s["timestamp"] for s in snapshots]
        networks = [s["graph"] for s in snapshots]

        # Track metrics over time
        network_size_over_time = [len(G.nodes()) for G in networks]
        network_density_over_time = []

        for G in networks:
            if isinstance(G, nx.DiGraph):
                n = len(G)
                if n > 1:
                    max_edges = n * (n - 1)
                    density = G.number_of_edges() / max_edges if max_edges > 0 else 0
                else:
                    density = 0.0
            else:
                density = nx.density(G) if len(G) > 0 else 0.0
            network_density_over_time.append(density)

        # Track node metrics over time
        all_nodes = set()
        for G in networks:
            all_nodes.update(G.nodes())

        if node_subset:
            tracked_nodes = [n for n in node_subset if n in all_nodes]
        else:
            # Track top nodes by average degree
            node_degrees = {}
            for G in networks:
                for node in G.nodes():
                    if node not in node_degrees:
                        node_degrees[node] = []
                    node_degrees[node].append(G.degree(node))

            import numpy as np

            avg_degrees = {node: np.mean(degrees) for node, degrees in node_degrees.items()}
            tracked_nodes = sorted(avg_degrees.items(), key=lambda x: x[1], reverse=True)[:50]
            tracked_nodes = [node for node, _ in tracked_nodes]

        node_metrics_over_time = {}
        for node in tracked_nodes:
            metrics = []
            for G in networks:
                if node in G:
                    metrics.append(G.degree(node))
                else:
                    metrics.append(0.0)
            node_metrics_over_time[node] = metrics

        # Community evolution (would need community detection per period)
        community_evolution = {}

        return {
            "time_periods": time_periods,
            "network_size_over_time": network_size_over_time,
            "network_density_over_time": network_density_over_time,
            "node_metrics_over_time": node_metrics_over_time,
            "community_evolution": community_evolution,
        }

    def detect_network_changes(
        self, evolution_analysis: Dict, threshold: float = 0.2
    ) -> List[Dict]:
        """
        Detect significant changes in network structure over time (from Enron project).

        Args:
            evolution_analysis: Results from analyze_network_evolution
            threshold: Threshold for significant change (relative change)

        Returns:
            List of change events
        """
        changes = []

        networks = evolution_analysis.get("network_size_over_time", [])
        time_periods = evolution_analysis.get("time_periods", [])

        for i in range(1, len(networks)):
            prev_size = networks[i - 1]
            curr_size = networks[i]

            if prev_size > 0:
                size_change = abs(curr_size - prev_size) / prev_size
                if size_change > threshold:
                    changes.append(
                        {
                            "period": time_periods[i] if i < len(time_periods) else None,
                            "type": "size_change",
                            "change": size_change,
                            "prev_size": prev_size,
                            "curr_size": curr_size,
                        }
                    )

        logger.info(f"Detected {len(changes)} significant network changes")
        return changes
