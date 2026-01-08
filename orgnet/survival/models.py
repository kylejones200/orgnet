"""Survival analysis models for organizational networks.

This module provides the main interface for survival analysis, integrating
orgnet metrics and ts2net features to build person-time tables and fit
survival models.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence

import networkx as nx
import pandas as pd

from orgnet.survival.discrete import (
    DiscreteTimeSurvivalModel,
    build_person_period_table,
    fit_discrete_time_model,
    attach_survival_outputs,
)
from orgnet.utils.logging import get_logger

try:
    from orgnet.temporal.ts2net import features_for_series, Ts2NetConfig

    HAS_TS2NET = True
except ImportError:
    HAS_TS2NET = False
    Ts2NetConfig = None

logger = get_logger(__name__)


def build_person_time_table(
    temporal_snapshots: List[Dict],
    events: pd.DataFrame,
    id_col: str = "person_id",
    period_col: str = "period",
    event_period_col: str = "event_period",
    censored_col: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    freq: str = "W",
    feature_lag: int = 1,
    include_network_features: bool = True,
    include_ts2net_features: bool = False,
    ts2net_config: Optional[Ts2NetConfig] = None,
) -> pd.DataFrame:
    """Build person-time table from temporal graph snapshots and events.

    This function creates one row per person per period with network features
    and optional ts2net features. Features are computed with a lag to avoid
    lookahead bias.

    Args:
        temporal_snapshots: List of dicts with 'timestamp' and 'graph' keys.
            Each dict represents a snapshot of the network at a point in time.
        events: DataFrame with event information. Must contain:
            - id_col: person identifier
            - event_period_col: period when event occurred or last observed
            - censored_col (optional): boolean flag for censoring
        id_col: Name of person identifier column.
        period_col: Name of period index column to create.
        event_period_col: Name of event period column in events.
        censored_col: Optional name of censor flag column.
        start_date: Start date for analysis. If None, uses earliest snapshot.
        end_date: End date for analysis. If None, uses latest snapshot.
        freq: Pandas frequency string (e.g., 'W' for weekly, 'M' for monthly).
        feature_lag: Number of periods to lag features (default 1).
        include_network_features: If True, include orgnet network metrics.
        include_ts2net_features: If True, include ts2net temporal features.
        ts2net_config: Optional Ts2NetConfig for ts2net features.

    Returns:
        DataFrame with one row per person per period, including:
        - id_col, period_col, event indicator
        - Network features (if enabled): degree, betweenness, clustering, etc.
        - ts2net features (if enabled): visibility graph metrics, etc.
    """
    if not temporal_snapshots:
        raise ValueError("temporal_snapshots cannot be empty")

    # Determine date range
    if start_date is None:
        start_date = min(s["timestamp"] for s in temporal_snapshots)
    if end_date is None:
        end_date = max(s["timestamp"] for s in temporal_snapshots)

    # Create period index
    periods = pd.date_range(start=start_date, end=end_date, freq=freq)
    period_to_idx = {p: i + 1 for i, p in enumerate(periods)}

    # Get all unique person IDs from snapshots
    all_person_ids = set()
    for snapshot in temporal_snapshots:
        graph = snapshot["graph"]
        all_person_ids.update(graph.nodes())

    if not all_person_ids:
        raise ValueError("No person IDs found in temporal snapshots")

    # Build person-period feature table
    rows = []
    for person_id in all_person_ids:
        for period_date in periods:
            period_idx = period_to_idx[period_date]

            # Find the snapshot for this period (use snapshot at or before period_date)
            snapshot = _find_snapshot_for_period(temporal_snapshots, period_date)

            if snapshot is None:
                continue

            graph = snapshot["graph"]

            # Skip if person not in this snapshot
            if person_id not in graph:
                continue

            row = {
                id_col: person_id,
                period_col: period_idx,
                "period_date": period_date,
            }

            # Add network features with lag
            if include_network_features:
                if freq == "W":
                    lag_period_date = period_date - timedelta(weeks=feature_lag)
                else:
                    lag_period_date = period_date - timedelta(days=feature_lag * 7)
                lag_snapshot = _find_snapshot_for_period(temporal_snapshots, lag_period_date)

                if lag_snapshot and person_id in lag_snapshot["graph"]:
                    lag_graph = lag_snapshot["graph"]
                    network_feats = _extract_network_features(lag_graph, person_id)
                    row.update(network_feats)

            # Add ts2net features (requires time series of interactions)
            if include_ts2net_features and HAS_TS2NET:
                ts2net_feats = _extract_ts2net_features(
                    temporal_snapshots, person_id, period_date, feature_lag, ts2net_config
                )
                if ts2net_feats:
                    row.update(ts2net_feats)

            rows.append(row)

    features_df = pd.DataFrame(rows)

    if features_df.empty:
        raise ValueError("No features extracted. Check temporal_snapshots and date range.")

    # Merge with events to create person-period table with event indicator
    person_period_df = build_person_period_table(
        features=features_df,
        events=events,
        id_col=id_col,
        period_col=period_col,
        event_period_col=event_period_col,
        censored_col=censored_col,
    )

    return person_period_df


def _find_snapshot_for_period(snapshots: List[Dict], target_date: datetime) -> Optional[Dict]:
    """Find the snapshot closest to but not after target_date."""
    valid_snapshots = [s for s in snapshots if s["timestamp"] <= target_date]
    if not valid_snapshots:
        return None
    return max(valid_snapshots, key=lambda s: s["timestamp"])


def _extract_network_features(graph: nx.Graph, person_id: str) -> Dict[str, float]:
    """Extract network features for a person from a graph snapshot."""
    if person_id not in graph:
        return {}

    features = {}

    # Degree centrality
    degree = graph.degree(person_id)
    features["degree"] = float(degree)

    # Weighted degree
    weighted_degree = sum(
        graph[person_id].get(neighbor, {}).get("weight", 1.0)
        for neighbor in graph.neighbors(person_id)
    )
    features["weighted_degree"] = float(weighted_degree)

    # Betweenness centrality (approximate for large graphs)
    try:
        betweenness = nx.betweenness_centrality(graph, k=min(100, graph.number_of_nodes()))
        features["betweenness"] = float(betweenness.get(person_id, 0.0))
    except Exception:
        features["betweenness"] = 0.0

    # Clustering coefficient
    try:
        clustering = nx.clustering(graph, person_id)
        features["clustering"] = float(clustering)
    except Exception:
        features["clustering"] = 0.0

    # Community ID (if available)
    try:
        communities = nx.community.greedy_modularity_communities(graph)
        for i, comm in enumerate(communities):
            if person_id in comm:
                features["community_id"] = float(i)
                break
        else:
            features["community_id"] = -1.0
    except Exception:
        features["community_id"] = -1.0

    # Cross-team edges (if department/team info available)
    # This is a placeholder - would need team/department metadata
    features["cross_team_edges"] = 0.0  # TODO: implement with team metadata

    return features


def _extract_ts2net_features(
    snapshots: List[Dict],
    person_id: str,
    period_date: datetime,
    lag: int,
    ts2net_config: Optional[Ts2NetConfig],
) -> Dict[str, float]:
    """Extract ts2net features for a person up to a given period."""
    if not HAS_TS2NET:
        return {}

    # Get historical snapshots up to lag period before current
    cutoff_date = period_date - timedelta(weeks=lag)
    historical_snapshots = [s for s in snapshots if s["timestamp"] <= cutoff_date]

    if len(historical_snapshots) < 20:  # Need minimum length for ts2net
        return {}

    # Build time series of degree over time
    degree_series = []
    for snapshot in sorted(historical_snapshots, key=lambda s: s["timestamp"]):
        graph = snapshot["graph"]
        if person_id in graph:
            degree = graph.degree(person_id)
            degree_series.append(degree)
        else:
            degree_series.append(0)

    if len(degree_series) < 20:
        return {}

    # Convert to pandas Series
    dates = [s["timestamp"] for s in sorted(historical_snapshots, key=lambda s: s["timestamp"])]
    series = pd.Series(degree_series, index=dates)

    try:
        if ts2net_config is None:
            ts2net_config = Ts2NetConfig()
        feats = features_for_series(series, ts2net_config)
        # Features already have ts2net_ prefix from features_for_series
        return feats
    except Exception as e:
        logger.warning(f"Failed to extract ts2net features for {person_id}: {e}")
        return {}


def fit_survival_model(
    person_period_df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    id_col: str = "person_id",
    period_col: str = "period",
    event_col: str = "event",
    model_type: str = "discrete",
    **model_kwargs,
) -> DiscreteTimeSurvivalModel:
    """Fit a survival model on person-period data.

    Args:
        person_period_df: Person-period table with features and event indicator.
        feature_cols: Names of feature columns. If None, uses all numeric columns
            except id_col, period_col, event_col.
        id_col: Name of person identifier column.
        period_col: Name of period index column.
        event_col: Name of event indicator column.
        model_type: Type of model ('discrete' for discrete time logistic).
        **model_kwargs: Additional arguments passed to model fitting.

    Returns:
        Fitted survival model.
    """
    if feature_cols is None:
        # Auto-detect feature columns
        exclude_cols = {id_col, period_col, event_col, "period_date", "hazard", "survival"}
        feature_cols = [
            col
            for col in person_period_df.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(person_period_df[col])
        ]

    if not feature_cols:
        raise ValueError(
            "No feature columns found. Specify feature_cols or ensure numeric columns exist."
        )

    logger.info(f"Fitting {model_type} survival model with {len(feature_cols)} features")

    if model_type == "discrete":
        return fit_discrete_time_model(
            person_period_df=person_period_df,
            feature_cols=feature_cols,
            id_col=id_col,
            period_col=period_col,
            event_col=event_col,
            **model_kwargs,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def score_hazard(
    model: DiscreteTimeSurvivalModel,
    person_period_df: pd.DataFrame,
    id_col: str = "person_id",
    period_col: str = "period",
) -> pd.DataFrame:
    """Score hazard and survival for current person-period data.

    Args:
        model: Fitted survival model.
        person_period_df: Person-period table with features.
        id_col: Name of person identifier column.
        period_col: Name of period index column.

    Returns:
        DataFrame with hazard and survival predictions added.
    """
    return attach_survival_outputs(person_period_df, model)


def compute_hazard_flags(
    hazard_df: pd.DataFrame,
    id_col: str = "person_id",
    period_col: str = "period",
    hazard_col: str = "hazard",
    top_decile: bool = True,
    traffic_light: bool = True,
) -> pd.DataFrame:
    """Compute hazard flags for dashboard display.

    Args:
        hazard_df: DataFrame with hazard predictions.
        id_col: Name of person identifier column.
        period_col: Name of period index column.
        hazard_col: Name of hazard column.
        top_decile: If True, add binary flag for top decile hazard.
        traffic_light: If True, add traffic light band (low/medium/high).

    Returns:
        DataFrame with additional flag columns.
    """
    df = hazard_df.copy()

    if top_decile:
        threshold = df[hazard_col].quantile(0.9)
        df["high_hazard_flag"] = (df[hazard_col] >= threshold).astype(int)

    if traffic_light:
        low_threshold = df[hazard_col].quantile(0.33)
        high_threshold = df[hazard_col].quantile(0.67)
        df["hazard_band"] = pd.cut(
            df[hazard_col],
            bins=[0, low_threshold, high_threshold, 1.0],
            labels=["low", "medium", "high"],
        )

    return df
