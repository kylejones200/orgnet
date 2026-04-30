"""Metrics for graphs and node tables."""

import logging

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def degree_centrality(graph: nx.Graph, nodes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute degree centrality for all nodes.

    Args:
        graph: NetworkX graph
        nodes: Optional node table to filter/align results

    Returns:
        DataFrame with node_id and degree_centrality columns, indexed by node_id
    """
    centrality = nx.degree_centrality(graph)
    df = pd.DataFrame(list(centrality.items()), columns=["node_id", "degree_centrality"]).set_index(
        "node_id"
    )

    if nodes is not None:
        # Align with node table if provided
        node_ids = set(nodes["node_id"].values) if "node_id" in nodes.columns else set(nodes.index)
        df = df.reindex(node_ids).fillna(0.0)

    return df


def betweenness_centrality(graph: nx.Graph, nodes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute betweenness centrality for all nodes.

    Args:
        graph: NetworkX graph
        nodes: Optional node table to filter/align results

    Returns:
        DataFrame with node_id and betweenness_centrality columns, indexed by node_id
    """
    centrality = nx.betweenness_centrality(
        graph, weight="weight" if nx.is_weighted(graph) else None
    )
    df = pd.DataFrame(
        list(centrality.items()), columns=["node_id", "betweenness_centrality"]
    ).set_index("node_id")

    if nodes is not None:
        # Align with node table if provided
        node_ids = set(nodes["node_id"].values) if "node_id" in nodes.columns else set(nodes.index)
        df = df.reindex(node_ids).fillna(0.0)

    return df


def density(graph: nx.Graph) -> float:
    """Compute graph density.

    Args:
        graph: NetworkX graph

    Returns:
        Graph density (0.0 to 1.0)
    """
    n = graph.number_of_nodes()
    if n <= 1:
        return 0.0

    if isinstance(graph, nx.DiGraph):
        max_edges = n * (n - 1)
    else:
        max_edges = n * (n - 1) / 2

    if max_edges == 0:
        return 0.0

    return graph.number_of_edges() / max_edges


def drift_score(
    metric_table_1: pd.DataFrame,
    metric_table_2: pd.DataFrame,
    metric_col: str = "degree_centrality",
) -> float:
    """Compute drift score between two metric tables.

    Simple drift score: absolute difference in mean metric values.

    Args:
        metric_table_1: First metric table (MetricTableLike)
        metric_table_2: Second metric table (MetricTableLike)
        metric_col: Name of metric column to compare

    Returns:
        Drift score (non-negative float, higher = more drift)
    """
    if metric_col not in metric_table_1.columns or metric_col not in metric_table_2.columns:
        raise ValueError(f"Metric column '{metric_col}' not found in both tables")

    mean1 = metric_table_1[metric_col].mean()
    mean2 = metric_table_2[metric_col].mean()

    return abs(mean1 - mean2)
