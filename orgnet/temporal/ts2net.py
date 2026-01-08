"""ts2net integration for temporal network features.

This module provides a thin adapter layer for ts2net, converting time series
to network features that can be used in orgnet analysis.

See: https://github.com/kylejones200/ts2net
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from orgnet.utils.logging import get_logger

try:
    from ts2net import HVG, NVG, RecurrenceNetwork, TransitionNetwork, build_network

    HAS_TS2NET = True
except ImportError:
    HAS_TS2NET = False
    HVG = None
    NVG = None
    RecurrenceNetwork = None
    TransitionNetwork = None
    build_network = None

logger = get_logger(__name__)


@dataclass
class Ts2NetConfig:
    """Configuration for ts2net based temporal features.

    Attributes:
        method: Name of ts2net method. Options: 'hvg', 'nvg', 'recurrence', 'transition'.
        min_length: Minimum length for a valid series.
        max_length: Optional cap on series length. The adapter can downsample if needed.
        output: Output mode for large series. Options: 'edges', 'degrees', 'stats'.
        limit: Horizon limit for NVG (prevents memory issues with large series).
        max_edges: Maximum edges for NVG.
        max_memory_mb: Maximum memory in MB for NVG.
        # Recurrence network parameters
        m: Embedding dimension for recurrence (None = auto via FNN).
        tau: Time delay for recurrence.
        rule: Recurrence rule ('knn', 'epsilon', 'radius').
        k: Number of neighbors for k-NN rule.
        epsilon: Threshold for epsilon-recurrence.
        # Transition network parameters
        symbolizer: Symbolizer for transition networks ('ordinal', 'equal_width', 'equal_freq', 'kmeans').
        order: Order for ordinal patterns.
    """

    method: str = "hvg"
    min_length: int = 20
    max_length: Optional[int] = None
    output: str = "edges"  # 'edges', 'degrees', 'stats'
    limit: Optional[int] = None  # For NVG
    max_edges: Optional[int] = None  # For NVG
    max_memory_mb: Optional[int] = None  # For NVG
    # Recurrence network parameters
    m: Optional[int] = None
    tau: int = 1
    rule: str = "knn"
    k: int = 5
    epsilon: Optional[float] = None
    # Transition network parameters
    symbolizer: str = "ordinal"
    order: int = 3


def _prepare_series(values: pd.Series, cfg: Ts2NetConfig) -> np.ndarray:
    """Prepare a single time series for ts2net.

    Args:
        values: Series of numeric values in time order.
        cfg: Ts2NetConfig instance.

    Returns:
        Numpy array with optional downsample.

    Raises:
        ValueError: If the series is too short.
    """
    arr = values.to_numpy(dtype=float)

    if arr.size < cfg.min_length:
        raise ValueError(
            f"Series length {arr.size} below min_length {cfg.min_length} for ts2net features."
        )

    if cfg.max_length is not None and arr.size > cfg.max_length:
        # Simple uniform downsample
        idx = np.linspace(0, arr.size - 1, cfg.max_length).round().astype(int)
        arr = arr[idx]

    return arr


def _build_ts2net_graph(series: np.ndarray, cfg: Ts2NetConfig):
    """Build a ts2net graph for one series.

    This wrapper hides the exact ts2net API from the rest of orgnet.
    """
    if not HAS_TS2NET:
        raise ImportError("ts2net is not available. Install with: pip install ts2net")

    method = cfg.method.lower()

    try:
        if method == "hvg":
            builder = HVG(weighted=False, limit=cfg.limit, output=cfg.output)
            builder.build(series)
            return builder
        elif method == "nvg":
            kwargs = {"weighted": False, "output": cfg.output}
            if cfg.limit is not None:
                kwargs["limit"] = cfg.limit
            if cfg.max_edges is not None:
                kwargs["max_edges"] = cfg.max_edges
            if cfg.max_memory_mb is not None:
                kwargs["max_memory_mb"] = cfg.max_memory_mb
            builder = NVG(**kwargs)
            builder.build(series)
            return builder
        elif method == "recurrence":
            kwargs = {"rule": cfg.rule, "tau": cfg.tau}
            if cfg.m is not None:
                kwargs["m"] = cfg.m
            if cfg.rule == "knn":
                kwargs["k"] = cfg.k
            elif cfg.rule == "epsilon" and cfg.epsilon is not None:
                kwargs["epsilon"] = cfg.epsilon
            builder = RecurrenceNetwork(**kwargs)
            builder.build(series)
            return builder
        elif method == "transition":
            builder = TransitionNetwork(symbolizer=cfg.symbolizer, order=cfg.order)
            builder.build(series)
            return builder
        else:
            # Try using build_network convenience function
            builder = build_network(
                series, method, **{k: v for k, v in cfg.__dict__.items() if v is not None}
            )
            return builder
    except Exception as e:
        logger.error(f"Failed to build ts2net graph with method {method}: {e}")
        raise


def _graph_features(builder) -> Dict[str, float]:
    """Compute a compact feature set from a ts2net graph builder.

    Args:
        builder: ts2net graph builder instance (after build() has been called).

    Returns:
        Dictionary of numeric features.
    """
    features = {}

    # Basic graph statistics
    features["ts2net_nodes"] = float(builder.n_nodes)
    features["ts2net_edges"] = float(builder.n_edges)

    if builder.n_nodes == 0:
        # Return zeros for all features
        features["ts2net_avg_degree"] = 0.0
        features["ts2net_clustering"] = 0.0
        features["ts2net_assort"] = 0.0
        features["ts2net_avg_path_length"] = 0.0
        return features

    # Degree statistics
    try:
        degrees = builder.degree_sequence()
        features["ts2net_avg_degree"] = float(np.mean(degrees))
        features["ts2net_std_degree"] = float(np.std(degrees))
        features["ts2net_max_degree"] = float(np.max(degrees))
        features["ts2net_min_degree"] = float(np.min(degrees))
    except Exception:
        features["ts2net_avg_degree"] = 0.0
        features["ts2net_std_degree"] = 0.0
        features["ts2net_max_degree"] = 0.0
        features["ts2net_min_degree"] = 0.0

    # NetworkX features (if conversion is available)
    try:
        graph = builder.as_networkx()
        import networkx as nx

        # Clustering coefficient
        try:
            clustering = nx.average_clustering(graph)
            features["ts2net_clustering"] = float(clustering)
        except Exception:
            features["ts2net_clustering"] = 0.0

        # Degree assortativity
        try:
            assort = nx.degree_assortativity_coefficient(graph)
            features["ts2net_assort"] = float(assort)
        except Exception:
            features["ts2net_assort"] = 0.0

        # Average path length (only if connected)
        try:
            if nx.is_connected(graph):
                avg_path_length = nx.average_shortest_path_length(graph)
            else:
                # Compute for largest component
                largest_cc = max(nx.connected_components(graph), key=len)
                subgraph = graph.subgraph(largest_cc)
                if len(largest_cc) > 1:
                    avg_path_length = nx.average_shortest_path_length(subgraph)
                else:
                    avg_path_length = 0.0
            features["ts2net_avg_path_length"] = float(avg_path_length)
        except Exception:
            features["ts2net_avg_path_length"] = 0.0

        # Density
        try:
            density = nx.density(graph)
            features["ts2net_density"] = float(density)
        except Exception:
            features["ts2net_density"] = 0.0

    except Exception as e:
        # NetworkX conversion not available or failed
        logger.debug(f"NetworkX conversion not available: {e}")
        features["ts2net_clustering"] = 0.0
        features["ts2net_assort"] = 0.0
        features["ts2net_avg_path_length"] = 0.0
        features["ts2net_density"] = 0.0

    return features


def features_for_series(values: pd.Series, cfg: Optional[Ts2NetConfig] = None) -> Dict[str, float]:
    """Compute ts2net based features for a single univariate series.

    Args:
        values: Numeric series in time order.
        cfg: Optional configuration. Uses defaults if None.

    Returns:
        Dictionary with feature names and values.

    Raises:
        ImportError: If ts2net is not installed.
        ValueError: If series is too short.
    """
    if not HAS_TS2NET:
        raise ImportError("ts2net is not available. Install with: pip install ts2net")

    if cfg is None:
        cfg = Ts2NetConfig()

    arr = _prepare_series(values.sort_index(), cfg)
    builder = _build_ts2net_graph(arr, cfg)
    feats = _graph_features(builder)

    # Add method prefix to feature names
    return {f"ts2net_{cfg.method}_{k}": v for k, v in feats.items()}


def edge_temporal_features(
    interactions: pd.DataFrame,
    cfg: Optional[Ts2NetConfig] = None,
    freq: str = "W",
) -> pd.DataFrame:
    """Build ts2net features for each edge from interaction time series.

    Assumes one row per interaction.

    Args:
        interactions: DataFrame with at least columns
            ['sender_id', 'receiver_id', 'timestamp'].
        cfg: Optional Ts2NetConfig instance.
        freq: Resample frequency. For example 'D' or 'W'.

    Returns:
        DataFrame indexed by (source_id, target_id) with ts2net feature columns.
    """
    if not HAS_TS2NET:
        raise ImportError("ts2net is not available. Install with: pip install ts2net")

    if cfg is None:
        cfg = Ts2NetConfig()

    df = interactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Make undirected pair key. Adjust if your graph uses direction.
    pair_cols = ["source_id", "target_id"]
    df["source_id"] = df[["sender_id", "receiver_id"]].min(axis=1)
    df["target_id"] = df[["sender_id", "receiver_id"]].max(axis=1)

    df = df.set_index("timestamp")

    # Build count series per pair at the chosen frequency
    counts = df.groupby(pair_cols).size().rename("count").to_frame().reset_index()
    counts = counts.set_index("timestamp", append=True)
    counts = counts.groupby(pair_cols)["count"].resample(freq).sum()

    # Now counts is a Series with MultiIndex (source_id, target_id, timestamp)
    results: List[Tuple[Tuple[str, str], Dict[str, float]]] = []

    for (src, tgt), group in counts.groupby(level=[0, 1]):
        series = group.droplevel([0, 1])

        try:
            feats = features_for_series(series, cfg)
        except ValueError:
            # Too short for ts2net. Skip this edge.
            logger.debug(f"Skipping edge ({src}, {tgt}): series too short")
            continue
        except Exception as e:
            logger.warning(f"Failed to compute features for edge ({src}, {tgt}): {e}")
            continue

        results.append(((src, tgt), feats))

    if not results:
        return pd.DataFrame(columns=["source_id", "target_id"])

    rows = []
    for (src, tgt), feat_dict in results:
        row = {"source_id": src, "target_id": tgt}
        row.update(feat_dict)
        rows.append(row)

    features_df = pd.DataFrame(rows).set_index(["source_id", "target_id"])
    return features_df


def person_temporal_features(
    interactions: pd.DataFrame,
    person_id: str,
    cfg: Optional[Ts2NetConfig] = None,
    freq: str = "W",
) -> Dict[str, float]:
    """Build ts2net features for a single person from their interaction time series.

    Args:
        interactions: DataFrame with at least columns ['sender_id', 'receiver_id', 'timestamp'].
        person_id: Person identifier to extract features for.
        cfg: Optional Ts2NetConfig instance.
        freq: Resample frequency. For example 'D' or 'W'.

    Returns:
        Dictionary with ts2net feature names and values.
    """
    if not HAS_TS2NET:
        raise ImportError("ts2net is not available. Install with: pip install ts2net")

    if cfg is None:
        cfg = Ts2NetConfig()

    df = interactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Filter to interactions involving this person
    person_interactions = df[(df["sender_id"] == person_id) | (df["receiver_id"] == person_id)]

    if person_interactions.empty:
        raise ValueError(f"No interactions found for person {person_id}")

    # Count interactions per period
    person_interactions = person_interactions.set_index("timestamp")
    counts = person_interactions.resample(freq).size()

    try:
        feats = features_for_series(counts, cfg)
        return feats
    except ValueError as e:
        logger.warning(f"Failed to compute features for person {person_id}: {e}")
        return {}


def merge_edge_temporal_features(
    edge_metrics: pd.DataFrame,
    temporal_features: pd.DataFrame,
) -> pd.DataFrame:
    """Merge ts2net features into existing orgnet edge metrics.

    Args:
        edge_metrics: DataFrame with edge level metrics.
        temporal_features: DataFrame from edge_temporal_features.

    Returns:
        DataFrame with extra ts2net feature columns.
    """
    if not {"source_id", "target_id"}.issubset(edge_metrics.columns):
        msg = "edge_metrics must contain 'source_id' and 'target_id' columns."
        raise ValueError(msg)

    merged = edge_metrics.merge(
        temporal_features.reset_index(),
        on=["source_id", "target_id"],
        how="left",
    )
    return merged
