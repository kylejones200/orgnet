"""Anomaly detection in organizational networks."""

import networkx as nx
import numpy as np
import pandas as pd
import re
from collections import Counter
from typing import Dict, List, Optional, Set

from orgnet.data.models import Person, InteractionType
from orgnet.metrics.centrality import CentralityAnalyzer
from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


class AnomalyDetector:
    """Detects anomalies in organizational networks."""

    def __init__(self, graph: nx.Graph, people: List[Person]):
        """
        Initialize anomaly detector.

        Args:
            graph: Organizational graph
            people: List of Person objects with attributes
        """
        self.graph = graph
        self.people = {p.id: p for p in people}
        self.centrality_analyzer = CentralityAnalyzer(graph)

    def detect_all_anomalies(self, include_network_structure: bool = True) -> Dict:
        """
        Detect all types of anomalies.

        Args:
            include_network_structure: If True, include network structure anomalies

        Returns:
            Dictionary with different anomaly types
        """
        logger.info("Detecting all anomaly types")
        result = {"node": self.detect_node_anomalies(), "edge": self.detect_edge_anomalies()}

        if include_network_structure:
            result["network_structure"] = self.detect_network_structure_anomalies()

        return result

    def detect_node_anomalies(self) -> pd.DataFrame:
        """
        Detect node-level anomalies (isolation, overload).

        Returns:
            DataFrame with detected anomalies
        """
        logger.info("Detecting node-level anomalies")

        # Compute centralities
        degree_df = self.centrality_analyzer.compute_degree_centrality()
        betweenness_df = self.centrality_analyzer.compute_betweenness_centrality()

        # Combine anomaly types
        isolation_df = self._detect_isolation_anomalies(degree_df)
        overload_df = self._detect_overload_anomalies(degree_df, betweenness_df)

        return pd.concat([isolation_df, overload_df], ignore_index=True)

    def _detect_isolation_anomalies(self, degree_df: pd.DataFrame) -> pd.DataFrame:
        """Detect isolated nodes using vectorized operations."""
        # Add person data to dataframe for vectorization
        degree_df = degree_df.copy()
        degree_df["person"] = degree_df["node_id"].map(self.people)
        degree_df["role"] = degree_df["person"].apply(
            lambda p: (p.role or "Unknown") if p else "Unknown"
        )

        # Vectorized grouping by role
        from orgnet.utils.performance import polars_groupby

        if len(degree_df) > 10000:
            role_stats = polars_groupby(
                degree_df, by="role", agg={"degree_weighted": ["mean", "std"]}
            )
            role_stats.columns = ["role", "mean", "std"]
            role_stats = role_stats.set_index("role")
        else:
            role_stats = degree_df.groupby("role")["degree_weighted"].agg(["mean", "std"])
        role_stats.columns = ["role_mean", "role_std"]

        # Merge with role statistics
        degree_df = degree_df.merge(role_stats, left_on="role", right_index=True, how="left")

        # Vectorized z-score calculation
        degree_df["z_score"] = (degree_df["degree_weighted"] - degree_df["role_mean"]) / degree_df[
            "role_std"
        ].replace(0, np.nan)

        # Filter anomalies (z-score < -2)
        anomalies = degree_df[(degree_df["z_score"] < -2) & (degree_df["role_std"] > 0)].copy()

        if anomalies.empty:
            return pd.DataFrame()

        # Vectorized severity assignment
        anomalies["severity"] = np.where(anomalies["z_score"] < -3, "high", "medium")
        anomalies["type"] = "isolation"
        anomalies["metric"] = "degree"
        anomalies["expected_mean"] = anomalies["role_mean"]
        anomalies["description"] = anomalies.apply(
            lambda row: (
                f'Isolated node: {row["degree_weighted"]:.1f} connections '
                f'(expected {row["role_mean"]:.1f} for {row["role"]})'
            ),
            axis=1,
        )

        return anomalies[
            [
                "node_id",
                "type",
                "severity",
                "metric",
                "degree_weighted",
                "expected_mean",
                "z_score",
                "description",
            ]
        ].rename(columns={"degree_weighted": "value"})

    def _detect_overload_anomalies(
        self, degree_df: pd.DataFrame, betweenness_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Detect overloaded nodes using vectorized operations."""
        # Merge dataframes
        from orgnet.utils.performance import polars_join

        if len(degree_df) > 10000 or len(betweenness_df) > 10000:
            merged = polars_join(degree_df, betweenness_df, on="node_id", how="inner")
        else:
            merged = degree_df.merge(betweenness_df, on="node_id", how="inner")

        # Vectorized percentile calculation
        merged["betweenness_percentile"] = merged["betweenness_centrality"].rank(pct=True)
        merged["degree_percentile"] = merged["degree_weighted"].rank(pct=True)

        # Add person data
        merged["person"] = merged["node_id"].map(self.people)
        merged["job_level"] = merged["person"].apply(
            lambda p: (p.job_level or "").lower() if p else ""
        )

        # Vectorized senior role detection
        senior_keywords = {"senior", "principal", "lead", "manager"}
        merged["is_senior"] = merged["job_level"].apply(
            lambda level: any(keyword in (level or "") for keyword in senior_keywords)
        )

        # Vectorized anomaly detection
        burnout_anomalies = merged[
            (merged["betweenness_percentile"] > 0.95) & (~merged["is_senior"])
        ].copy()

        degree_anomalies = merged[merged["degree_percentile"] > 0.98].copy()

        anomalies_list = []

        # Build burnout anomalies
        if not burnout_anomalies.empty:
            burnout_anomalies["type"] = "overload"
            burnout_anomalies["severity"] = "high"
            burnout_anomalies["metric"] = "betweenness"
            burnout_anomalies["value"] = burnout_anomalies["betweenness_centrality"]
            burnout_anomalies["percentile"] = burnout_anomalies["betweenness_percentile"]
            burnout_anomalies["description"] = burnout_anomalies.apply(
                lambda row: (
                    f'High betweenness ({row["betweenness_percentile"] * 100:.1f}th percentile) '
                    f"with non-senior role - potential burnout risk"
                ),
                axis=1,
            )
            anomalies_list.append(
                burnout_anomalies[
                    [
                        "node_id",
                        "type",
                        "severity",
                        "metric",
                        "value",
                        "percentile",
                        "degree_percentile",
                        "job_level",
                        "description",
                    ]
                ]
            )

        # Build degree anomalies
        if not degree_anomalies.empty:
            degree_anomalies["type"] = "overload"
            degree_anomalies["severity"] = "high"
            degree_anomalies["metric"] = "degree"
            degree_anomalies["value"] = degree_anomalies["degree_weighted"]
            degree_anomalies["percentile"] = degree_anomalies["degree_percentile"]
            degree_anomalies["description"] = degree_anomalies.apply(
                lambda row: (
                    f'Extremely high connectivity ({row["degree_percentile"] * 100:.1f}th percentile) '
                    f"- potential overload"
                ),
                axis=1,
            )
            anomalies_list.append(
                degree_anomalies[
                    ["node_id", "type", "severity", "metric", "value", "percentile", "description"]
                ]
            )

        if not anomalies_list:
            return pd.DataFrame()

        return pd.concat(anomalies_list, ignore_index=True)

    def detect_edge_anomalies(self, link_predictor=None) -> pd.DataFrame:
        """
        Detect edge-level anomalies (unexpected connections, missing expected connections).

        Args:
            link_predictor: Optional LinkPredictor instance for prediction-based detection

        Returns:
            DataFrame with edge anomalies
        """
        logger.info("Detecting edge-level anomalies")

        if link_predictor is None:
            return self._detect_unexpected_connections()

        # Use link prediction for missing connections
        predicted_links = link_predictor.predict_links(top_k=50)

        # Vectorized filtering
        missing = predicted_links[
            (predicted_links["predicted_score"] > 0.8)
            & predicted_links.apply(
                lambda row: not self.graph.has_edge(row["node1"], row["node2"]), axis=1
            )
        ].copy()

        if missing.empty:
            return pd.DataFrame()

        missing["type"] = "missing_expected_connection"
        missing["description"] = missing.apply(
            lambda row: f'Missing expected connection (predicted score: {row["predicted_score"]:.2f})',
            axis=1,
        )

        return missing[["type", "node1", "node2", "predicted_score", "description"]].rename(
            columns={"predicted_score": "score"}
        )

    def _detect_unexpected_connections(self) -> pd.DataFrame:
        """Detect unexpected connections using vectorized operations."""
        # Build edge list with person data
        edges_data = []
        for u, v, data in self.graph.edges(data=True):
            if u in self.people and v in self.people:
                edges_data.append(
                    {
                        "node1": u,
                        "node2": v,
                        "weight": data.get("weight", 1.0),
                        "dept1": self.people[u].department,
                        "dept2": self.people[v].department,
                    }
                )

        if not edges_data:
            return pd.DataFrame()

        edges_df = pd.DataFrame(edges_data)
        edges_df["different_dept"] = edges_df["dept1"] != edges_df["dept2"]
        edges_df["weak_weight"] = edges_df["weight"] < 0.1

        # Vectorized filtering
        unexpected = edges_df[edges_df["different_dept"] & edges_df["weak_weight"]].copy()

        if unexpected.empty:
            return pd.DataFrame()

        unexpected["type"] = "unexpected_connection"
        unexpected["description"] = unexpected.apply(
            lambda row: f'Weak cross-department connection between {row["dept1"]} and {row["dept2"]}',
            axis=1,
        )

        return unexpected[["type", "node1", "node2", "weight", "description"]]

    def detect_temporal_anomalies(self, snapshots: List[Dict]) -> pd.DataFrame:
        """
        Detect temporal anomalies (sudden connectivity changes, relationship decay).

        Args:
            snapshots: List of temporal snapshots with 'timestamp' and 'graph'

        Returns:
            DataFrame with temporal anomalies
        """
        logger.info("Detecting temporal anomalies")

        if len(snapshots) < 2:
            return pd.DataFrame()

        # Build time series data
        time_series_data = []
        for snapshot in snapshots:
            graph = snapshot["graph"]
            timestamp = snapshot["timestamp"]
            for node in graph.nodes():
                time_series_data.append(
                    {
                        "node_id": node,
                        "timestamp": timestamp,
                        "degree": graph.degree(node, weight="weight"),
                    }
                )

        if not time_series_data:
            return pd.DataFrame()

        ts_df = pd.DataFrame(time_series_data)
        ts_df = ts_df.sort_values(["node_id", "timestamp"])

        # Vectorized change detection
        ts_df["prev_degree"] = ts_df.groupby("node_id")["degree"].shift(1)
        ts_df["change_ratio"] = (ts_df["degree"] - ts_df["prev_degree"]).abs() / ts_df[
            "prev_degree"
        ].replace(0, np.nan)

        # Filter anomalies (change ratio > 0.5)
        anomalies = ts_df[(ts_df["change_ratio"] > 0.5) & ts_df["prev_degree"].notna()].copy()

        if anomalies.empty:
            return pd.DataFrame()

        anomalies["type"] = "sudden_change"
        anomalies["previous_degree"] = anomalies["prev_degree"]
        anomalies["current_degree"] = anomalies["degree"]
        anomalies["description"] = anomalies.apply(
            lambda row: (
                f'Sudden connectivity change: {row["prev_degree"]:.1f} -> {row["degree"]:.1f} '
                f'({row["change_ratio"] * 100:.1f}% change)'
            ),
            axis=1,
        )

        return anomalies[
            [
                "node_id",
                "type",
                "timestamp",
                "previous_degree",
                "current_degree",
                "change_ratio",
                "description",
            ]
        ]

    def detect_network_structure_anomalies(
        self,
        communities: Optional[Dict] = None,
        density_threshold: float = 0.8,
        size_threshold: int = 10,
        connectivity_threshold: float = 3.0,
    ) -> Dict:
        """
        Detect network structure anomalies (from Enron project).

        Detects:
        - Structural anomalies: Unusually dense communities (tight-knit groups)
        - Connectivity anomalies: Nodes with anomalous degree/clustering patterns
        - Community evolution anomalies: Sudden changes in community structure

        Args:
            communities: Optional community detection results from CommunityDetector
            density_threshold: Minimum density for suspicious communities
            size_threshold: Maximum size for suspicious tight-knit groups
            connectivity_threshold: Z-score threshold for connectivity anomalies

        Returns:
            Dictionary with network structure anomaly results
        """
        logger.info("Detecting network structure anomalies")

        # Detect structural anomalies (tight-knit groups)
        anomalous_communities = self._detect_structural_anomalies(
            communities, density_threshold, size_threshold
        )

        # Detect connectivity anomalies
        anomalous_nodes = self._detect_connectivity_anomalies(connectivity_threshold)

        # Calculate anomaly scores
        anomaly_scores = self._calculate_network_anomaly_scores(
            anomalous_nodes, anomalous_communities
        )

        # Categorize anomalies
        degrees = dict(self.graph.degree())
        degree_mean = np.mean(list(degrees.values())) if degrees else 0

        anomaly_types = {
            "high_degree": [n for n in anomalous_nodes if degrees.get(n, 0) > degree_mean],
            "low_degree": [n for n in anomalous_nodes if degrees.get(n, 0) < degree_mean],
            "suspicious_community": (
                list(set().union(*anomalous_communities)) if anomalous_communities else []
            ),
        }

        metrics = {
            "anomalous_nodes": len(anomalous_nodes),
            "anomalous_communities": len(anomalous_communities),
            "avg_anomaly_score": np.mean(list(anomaly_scores.values())) if anomaly_scores else 0.0,
        }

        logger.info(f"Network structure anomalies: {metrics}")

        return {
            "anomalous_nodes": anomalous_nodes,
            "anomalous_communities": anomalous_communities,
            "anomaly_scores": anomaly_scores,
            "anomaly_types": anomaly_types,
            "metrics": metrics,
        }

    def _detect_structural_anomalies(
        self, communities: Optional[Dict], density_threshold: float, size_threshold: int
    ) -> List[Set[str]]:
        """
        Detect anomalous network structures (tight-knit groups).

        Based on Enron paper: "tight balls with spikes"
        - Unusually dense communities
        - Small tight-knit groups

        Args:
            communities: Optional community detection results
            density_threshold: Minimum density for suspicious communities
            size_threshold: Maximum size for suspicious tight-knit groups

        Returns:
            List of anomalous community sets
        """
        anomalous_communities = []

        if communities is None:
            # Run community detection if not provided
            from orgnet.metrics.community import CommunityDetector

            detector = CommunityDetector(self.graph)
            communities = detector.detect_communities(method="louvain")

        comm_list = communities.get("communities", [])

        for comm in comm_list:
            if len(comm) > size_threshold:
                continue

            # Calculate density
            subgraph = self.graph.subgraph(comm)
            if isinstance(subgraph, nx.DiGraph):
                n = len(comm)
                if n < 2:
                    continue
                max_edges = n * (n - 1)
                density = subgraph.number_of_edges() / max_edges if max_edges > 0 else 0
            else:
                density = nx.density(subgraph) if len(subgraph) > 1 else 0

            # High density + small size = suspicious
            if density >= density_threshold:
                anomalous_communities.append(set(comm))

        logger.info(f"Detected {len(anomalous_communities)} structurally anomalous communities")
        return anomalous_communities

    def _detect_connectivity_anomalies(self, threshold: float) -> List[str]:
        """
        Detect nodes with anomalous connectivity patterns.

        Anomalies:
        - Nodes with unusually high/low degree
        - Nodes with unusual clustering coefficient
        - Isolated nodes or hubs

        Args:
            threshold: Z-score threshold for anomaly detection

        Returns:
            List of anomalous node identifiers
        """
        # Calculate node metrics
        degrees = dict(self.graph.degree())
        clustering = nx.clustering(
            self.graph.to_undirected() if isinstance(self.graph, nx.DiGraph) else self.graph
        )

        if not degrees:
            return []

        # Calculate z-scores
        degree_values = list(degrees.values())
        clustering_values = list(clustering.values())

        degree_mean = np.mean(degree_values) if degree_values else 0
        degree_std = np.std(degree_values) if degree_values and len(degree_values) > 1 else 1
        clustering_mean = np.mean(clustering_values) if clustering_values else 0
        clustering_std = (
            np.std(clustering_values) if clustering_values and len(clustering_values) > 1 else 1
        )

        anomalous_nodes = []

        for node in self.graph.nodes():
            degree_z = (
                abs((degrees.get(node, 0) - degree_mean) / degree_std) if degree_std > 0 else 0
            )
            clustering_z = (
                abs((clustering.get(node, 0) - clustering_mean) / clustering_std)
                if clustering_std > 0
                else 0
            )

            # Anomalous if either metric is extreme
            if degree_z > threshold or clustering_z > threshold:
                anomalous_nodes.append(node)

        logger.info(f"Detected {len(anomalous_nodes)} connectivity anomalies")
        return anomalous_nodes

    def _calculate_network_anomaly_scores(
        self, anomalous_nodes: List[str], anomalous_communities: List[Set[str]]
    ) -> Dict[str, float]:
        """
        Calculate anomaly scores for nodes based on network structure.

        Args:
            anomalous_nodes: List of anomalous node IDs
            anomalous_communities: List of anomalous community sets

        Returns:
            Dictionary mapping node ID to anomaly score (0-1)
        """
        anomaly_scores = {}
        degrees = dict(self.graph.degree())
        degree_mean = np.mean(list(degrees.values())) if degrees else 0
        degree_std = np.std(list(degrees.values())) if degrees and len(degrees) > 1 else 1

        for node in self.graph.nodes():
            score = 0.0

            # High/low degree anomaly
            if node in anomalous_nodes:
                degree_z = (
                    abs((degrees.get(node, 0) - degree_mean) / degree_std) if degree_std > 0 else 0
                )
                score += min(degree_z / 3.0, 1.0)

            # Community membership anomaly
            for comm in anomalous_communities:
                if node in comm:
                    score += 0.5

            anomaly_scores[node] = min(score, 1.0)

        return anomaly_scores

    def detect_temporal_anomalies_from_interactions(
        self,
        interactions: List,
        volume_threshold: float = 2.0,
        response_time_threshold: float = 2.0,
        business_hours: tuple = (9, 17),
        weekend_threshold: float = 0.1,
    ) -> Dict:
        """
        Detect temporal anomalies from interaction data (enhanced from Enron project).

        Detects:
        - Volume spikes: Unusual spikes in email volume
        - Off-hours anomalies: Unusual off-hours communication patterns
        - Response time anomalies: Anomalous response times in email threads

        Args:
            interactions: List of Interaction objects
            volume_threshold: Z-score threshold for volume spikes
            response_time_threshold: Z-score threshold for response time anomalies
            business_hours: Tuple of (start_hour, end_hour)
            weekend_threshold: Minimum ratio of weekend emails to flag as anomaly

        Returns:
            Dictionary with temporal anomaly results
        """
        logger.info("Detecting temporal anomalies from interactions")

        if not interactions:
            return {
                "volume_spikes": [],
                "off_hours_anomalies": [],
                "response_time_anomalies": [],
                "metrics": {},
            }

        # Convert interactions to DataFrame for analysis
        data = []
        for interaction in interactions:
            if interaction.interaction_type != InteractionType.EMAIL:
                continue

            data.append(
                {
                    "id": interaction.id,
                    "sender": interaction.source_id,
                    "recipient": interaction.target_id,
                    "timestamp": interaction.timestamp,
                    "thread_id": interaction.thread_id,
                    "response_time": interaction.response_time_seconds,
                }
            )

        if not data:
            return {
                "volume_spikes": [],
                "off_hours_anomalies": [],
                "response_time_anomalies": [],
                "metrics": {},
            }

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Detect volume spikes
        volume_results = self._detect_volume_spikes(df, threshold=volume_threshold)

        # Detect off-hours anomalies
        off_hours_results = self._detect_off_hours_anomalies(
            df, business_hours=business_hours, weekend_threshold=weekend_threshold
        )

        # Detect response time anomalies
        response_time_results = self._detect_response_time_anomalies(
            df, threshold=response_time_threshold
        )

        metrics = {
            "total_volume_spikes": volume_results.get("spike_count", 0),
            "total_off_hours": off_hours_results.get("anomaly_count", 0),
            "total_response_time_anomalies": response_time_results.get("anomaly_count", 0),
        }

        return {
            "volume_spikes": volume_results.get("spike_periods", []),
            "off_hours_anomalies": off_hours_results.get("anomalous_senders", []),
            "response_time_anomalies": response_time_results.get("anomalous_interactions", []),
            "metrics": metrics,
        }

    def _detect_volume_spikes(
        self, df: pd.DataFrame, time_period: str = "hour", threshold: float = 2.0
    ) -> Dict:
        """
        Detect unusual spikes in email volume.

        Args:
            df: DataFrame with email data (must have 'timestamp' column)
            time_period: Time period to analyze ('hour', 'day', 'week')
            threshold: Z-score threshold for anomaly detection

        Returns:
            Dictionary with volume spike results
        """
        if len(df) == 0:
            return {"spike_periods": [], "spike_count": 0}

        # Group by time period
        if time_period == "hour":
            df["period"] = df["timestamp"].dt.floor("H")
        elif time_period == "day":
            df["period"] = df["timestamp"].dt.date
        elif time_period == "week":
            df["period"] = df["timestamp"].dt.to_period("W")
        else:
            raise ValueError(f"Unknown time period: {time_period}")

        # Count emails per period
        period_counts = df.groupby("period").size()

        if len(period_counts) < 2:
            return {"spike_periods": [], "spike_count": 0}

        # Calculate z-scores
        mean_count = period_counts.mean()
        std_count = period_counts.std()

        if std_count > 0:
            z_scores = (period_counts - mean_count) / std_count
        else:
            z_scores = pd.Series(0.0, index=period_counts.index)

        # Flag spikes
        spike_periods = z_scores[z_scores > threshold].index.tolist()

        logger.info(f"Detected {len(spike_periods)} volume spike periods")
        return {"spike_periods": spike_periods, "spike_count": len(spike_periods)}

    def _detect_off_hours_anomalies(
        self, df: pd.DataFrame, business_hours: tuple = (9, 17), weekend_threshold: float = 0.1
    ) -> Dict:
        """
        Detect anomalies in off-hours communication patterns.

        Args:
            df: DataFrame with email data (must have 'timestamp' and 'sender' columns)
            business_hours: Tuple of (start_hour, end_hour)
            weekend_threshold: Minimum ratio of weekend emails to flag as anomaly

        Returns:
            Dictionary with off-hours anomaly results
        """
        if len(df) == 0:
            return {"anomalous_senders": [], "anomaly_count": 0}

        df = df.copy()

        # Extract temporal features
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_business_hours"] = (
            (df["hour"] >= business_hours[0])
            & (df["hour"] < business_hours[1])
            & (df["is_weekend"] == 0)
        ).astype(int)
        df["is_off_hours"] = (df["is_business_hours"] == 0).astype(int)

        # Calculate off-hours ratio per sender
        if "sender" not in df.columns:
            return {"anomalous_senders": [], "anomaly_count": 0}

        sender_stats = df.groupby("sender").agg({"is_off_hours": "mean", "is_weekend": "mean"})

        # Flag senders with high off-hours ratio
        high_off_hours = sender_stats[sender_stats["is_off_hours"] > 0.3].index.tolist()
        high_weekend = sender_stats[sender_stats["is_weekend"] > weekend_threshold].index.tolist()

        anomalous_senders = list(set(high_off_hours + high_weekend))

        logger.info(f"Detected {len(anomalous_senders)} senders with off-hours anomalies")
        return {"anomalous_senders": anomalous_senders, "anomaly_count": len(anomalous_senders)}

    def _detect_response_time_anomalies(self, df: pd.DataFrame, threshold: float = 2.0) -> Dict:
        """
        Detect anomalous response times in email threads.

        Args:
            df: DataFrame with email data (must have 'timestamp' and 'thread_id' columns)
            threshold: Z-score threshold for anomaly detection

        Returns:
            Dictionary with response time anomaly results
        """
        if len(df) == 0 or "thread_id" not in df.columns:
            return {"anomalous_interactions": [], "anomaly_count": 0}

        df = df.copy()
        df = df.dropna(subset=["timestamp", "thread_id"])

        if len(df) == 0:
            return {"anomalous_interactions": [], "anomaly_count": 0}

        # Calculate response times within threads
        response_times = []

        for thread_id in df["thread_id"].unique():
            thread_df = df[df["thread_id"] == thread_id].sort_values("timestamp")

            if len(thread_df) < 2:
                continue

            for i in range(1, len(thread_df)):
                time_diff = (
                    thread_df.iloc[i]["timestamp"] - thread_df.iloc[i - 1]["timestamp"]
                ).total_seconds() / 3600
                if time_diff > 0:
                    response_times.append(time_diff)

        if not response_times:
            return {"anomalous_interactions": [], "anomaly_count": 0}

        # Calculate z-scores
        mean_rt = np.mean(response_times)
        std_rt = np.std(response_times)

        # Flag threads with anomalous response times
        anomalous_interactions = []

        for thread_id in df["thread_id"].unique():
            thread_df = df[df["thread_id"] == thread_id].sort_values("timestamp")

            if len(thread_df) < 2:
                continue

            for i in range(1, len(thread_df)):
                time_diff = (
                    thread_df.iloc[i]["timestamp"] - thread_df.iloc[i - 1]["timestamp"]
                ).total_seconds() / 3600

                if time_diff > 0 and std_rt > 0:
                    z_score = abs((time_diff - mean_rt) / std_rt)
                    if z_score > threshold:
                        anomalous_interactions.append(thread_df.iloc[i]["id"])

        logger.info(f"Detected {len(anomalous_interactions)} response time anomalies")
        return {
            "anomalous_interactions": anomalous_interactions,
            "anomaly_count": len(anomalous_interactions),
        }

    def detect_content_anomalies_from_interactions(
        self, interactions: List, language_threshold: float = 2.0, topic_threshold: float = 0.05
    ) -> Dict:
        """
        Detect content anomalies from interaction data (from Enron project).

        Detects:
        - Unusual language patterns: Rare vocabulary, special characters, sentence structure
        - Topic anomalies: Emails with unusual topics or rare keywords

        Args:
            interactions: List of Interaction objects with content
            language_threshold: Z-score threshold for language anomalies
            topic_threshold: Minimum frequency threshold for rare topics

        Returns:
            Dictionary with content anomaly results
        """
        logger.info("Detecting content anomalies from interactions")

        if not interactions:
            return {
                "anomalous_interactions": [],
                "anomaly_scores": {},
                "anomaly_types": {},
                "metrics": {},
            }

        # Convert interactions to DataFrame for analysis
        data = []
        for interaction in interactions:
            if not interaction.content:
                continue

            data.append(
                {
                    "id": interaction.id,
                    "text": interaction.content,
                    "subject": interaction.channel or "",
                }
            )

        if not data:
            return {
                "anomalous_interactions": [],
                "anomaly_scores": {},
                "anomaly_types": {},
                "metrics": {},
            }

        df = pd.DataFrame(data)

        # Detect unusual language
        df = self._detect_unusual_language(df, threshold=language_threshold)

        # Detect topic anomalies
        df = self._detect_topic_anomalies(df, threshold=topic_threshold)

        # Combined anomaly score
        df["content_anomaly_score"] = (
            df.get("language_anomaly_score", 0) * 0.5 + df.get("topic_anomaly_score", 0) * 0.5
        )
        df["is_content_anomaly"] = (df["content_anomaly_score"] > 0.5).astype(int)

        # Filter anomalies
        anomalous_interactions = df[df["is_content_anomaly"] == 1]["id"].tolist()

        # Categorize anomaly types
        anomaly_types = {
            "language": df[df["is_language_anomaly"] == 1]["id"].tolist(),
            "topic": df[df["is_topic_anomaly"] == 1]["id"].tolist(),
        }

        anomaly_scores = dict(zip(df["id"], df["content_anomaly_score"]))

        metrics = {
            "total_anomalies": len(anomalous_interactions),
            "language_anomalies": df["is_language_anomaly"].sum(),
            "topic_anomalies": df["is_topic_anomaly"].sum(),
            "avg_anomaly_score": df["content_anomaly_score"].mean(),
        }

        logger.info(f"Content anomalies: {metrics}")

        return {
            "anomalous_interactions": anomalous_interactions,
            "anomaly_scores": anomaly_scores,
            "anomaly_types": anomaly_types,
            "metrics": metrics,
        }

    def _detect_unusual_language(
        self, df: pd.DataFrame, text_col: str = "text", threshold: float = 2.0
    ) -> pd.DataFrame:
        """
        Detect emails with unusual language patterns.

        Args:
            df: DataFrame with email data
            text_col: Column name for email text
            threshold: Z-score threshold for anomaly detection

        Returns:
            DataFrame with language anomaly scores
        """
        df = df.copy()

        # Feature extraction
        texts = df[text_col].fillna("").astype(str)

        # Average word length
        df["avg_word_length"] = texts.apply(
            lambda x: np.mean([len(w) for w in re.findall(r"\b\w+\b", x)]) if x else 0
        )

        # Special character ratio
        df["special_char_ratio"] = texts.apply(
            lambda x: (
                len(re.findall(r'[!@#$%^&*()_+\-=\[\]{};\'\\:"|,.<>?/~`]', x)) / max(len(x), 1)
                if x and isinstance(x, str)
                else 0.0
            )
        )

        # Capitalization ratio
        df["caps_ratio"] = texts.apply(
            lambda x: (
                sum(1 for c in x if c.isupper()) / max(len(x), 1)
                if x and isinstance(x, str)
                else 0.0
            )
        )

        # Sentence length variance
        df["sentence_length_variance"] = texts.apply(
            lambda x: (
                np.var([len(s.split()) for s in re.split(r"[.!?]+", x) if s.strip()])
                if x and isinstance(x, str)
                else 0.0
            )
        )

        # Calculate z-scores
        features = [
            "avg_word_length",
            "special_char_ratio",
            "caps_ratio",
            "sentence_length_variance",
        ]
        anomaly_scores = []

        from orgnet.utils.performance import NUMBA_AVAILABLE, z_score_numba

        for feature in features:
            if feature in df.columns and df[feature].std() > 0:
                if NUMBA_AVAILABLE and len(df) > 1000:
                    values = df[feature].values
                    mean = df[feature].mean()
                    std = df[feature].std()
                    z_scores = z_score_numba(values, mean, std)
                else:
                    z_scores = np.abs((df[feature] - df[feature].mean()) / df[feature].std())
                anomaly_scores.append(z_scores)

        if anomaly_scores:
            df["language_anomaly_score"] = np.mean(anomaly_scores, axis=0)
            df["is_language_anomaly"] = (df["language_anomaly_score"] > threshold).astype(int)
        else:
            df["language_anomaly_score"] = 0.0
            df["is_language_anomaly"] = 0

        return df

    def _detect_topic_anomalies(
        self, df: pd.DataFrame, text_col: str = "text", threshold: float = 0.05
    ) -> pd.DataFrame:
        """
        Detect emails with unusual topics.

        Args:
            df: DataFrame with email data
            text_col: Column name for email text
            threshold: Minimum frequency threshold for rare topics

        Returns:
            DataFrame with topic anomaly scores
        """
        df = df.copy()

        # Simple keyword-based topic detection
        common_words = Counter()
        for text in df[text_col].fillna("").astype(str):
            words = re.findall(r"\b\w{4,}\b", text.lower())
            common_words.update(words)

        # Find rare words
        total_words = sum(common_words.values())
        if total_words > 0:
            rare_words = {
                word for word, count in common_words.items() if count / total_words < threshold
            }
        else:
            rare_words = set()

        def has_rare_words(text):
            words = set(re.findall(r"\b\w{4,}\b", text.lower()))
            return len(words.intersection(rare_words)) > 0

        df["is_topic_anomaly"] = (
            df[text_col].fillna("").astype(str).apply(has_rare_words).astype(int)
        )
        df["topic_anomaly_score"] = df["is_topic_anomaly"].astype(float)

        return df

    def detect_communication_pattern_anomalies(
        self,
        interactions: List,
        key_people: Optional[List[str]] = None,
        anomaly_threshold: float = 0.5,
    ) -> Dict:
        """
        Detect communication pattern anomalies (from Enron project).

        Detects:
        - Small tight-knit groups (2-5 people communicating frequently)
        - Unusual communication frequency patterns
        - Off-hours communication clusters

        Args:
            interactions: List of Interaction objects
            key_people: Optional list of key people IDs (high centrality nodes)
            anomaly_threshold: Minimum anomaly score to flag

        Returns:
            Dictionary with communication pattern anomaly results
        """
        logger.info("Detecting communication pattern anomalies")

        if not interactions:
            return {
                "anomalous_interactions": [],
                "anomalous_groups": [],
                "anomaly_scores": {},
                "metrics": {},
            }

        # Convert to DataFrame
        data = []
        for interaction in interactions:
            if interaction.interaction_type != InteractionType.EMAIL:
                continue

            data.append(
                {
                    "id": interaction.id,
                    "sender": interaction.source_id,
                    "recipient": interaction.target_id,
                    "timestamp": interaction.timestamp,
                }
            )

        if not data:
            return {
                "anomalous_interactions": [],
                "anomalous_groups": [],
                "anomaly_scores": {},
                "metrics": {},
            }

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Extract temporal features
        df["hour"] = df["timestamp"].dt.hour
        df["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int)
        df["is_off_hours"] = (
            (df["hour"] < 9) | (df["hour"] >= 17) | (df["is_weekend"] == 1)
        ).astype(int)

        # Identify key people if not provided (use high degree nodes)
        if key_people is None:
            degrees = dict(self.graph.degree())
            if degrees:
                sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
                key_people = [node for node, _ in sorted_nodes[:20]]  # Top 20 by degree

        from collections import defaultdict

        groups = defaultdict(list)
        for idx, (sender, recipient) in enumerate(zip(df["sender"], df["recipient"])):
            participants = frozenset([sender, recipient])
            groups[participants].append(df.iloc[idx]["id"])

        # Identify anomalous groups
        anomalous_groups = []
        anomalous_interactions = []
        anomaly_scores = {}

        for participants, interaction_ids in groups.items():
            if len(participants) < 2 or len(participants) > 5:
                continue

            # Check if group contains key people
            has_key_people = any(p in key_people for p in participants) if key_people else True

            # Get interactions for this group
            group_interactions = df[df["id"].isin(interaction_ids)]

            # Calculate anomaly score
            score = 0.0

            # Small group (2-5 people) communicating frequently
            if 2 <= len(participants) <= 5 and len(group_interactions) >= 3:
                score += 0.3

            # Off-hours communication
            off_hours_ratio = group_interactions["is_off_hours"].mean()
            if off_hours_ratio > 0.3:
                score += 0.2

            # Weekend communication
            weekend_ratio = group_interactions["is_weekend"].mean()
            if weekend_ratio > 0.1:
                score += 0.2

            # Key people in group
            if has_key_people:
                score += 0.3

            if score >= anomaly_threshold:
                anomalous_groups.append(set(participants))
                anomalous_interactions.extend(interaction_ids)
                anomaly_scores[",".join(sorted(participants))] = score

        metrics = {
            "total_anomalies": len(set(anomalous_interactions)),
            "anomalous_groups": len(anomalous_groups),
            "avg_anomaly_score": np.mean(list(anomaly_scores.values())) if anomaly_scores else 0.0,
        }

        logger.info(f"Communication pattern anomalies: {metrics}")

        return {
            "anomalous_interactions": list(set(anomalous_interactions)),
            "anomalous_groups": anomalous_groups,
            "anomaly_scores": anomaly_scores,
            "metrics": metrics,
        }
