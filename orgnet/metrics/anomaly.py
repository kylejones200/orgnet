"""Anomaly detection in organizational networks."""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List

from orgnet.data.models import Person
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

    def detect_all_anomalies(self) -> Dict:
        """
        Detect all types of anomalies.

        Returns:
            Dictionary with different anomaly types
        """
        logger.info("Detecting all anomaly types")
        return {"node": self.detect_node_anomalies(), "edge": self.detect_edge_anomalies()}

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
            lambda level: any(keyword in level for keyword in senior_keywords)
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
