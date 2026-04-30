"""Canonical organizational health thresholds and predicates.

Single source of truth for rule thresholds used by summaries, dashboards,
and insight generation. Presentation code may format strings; it must not
redefine these cutoffs.
"""

from __future__ import annotations

# Network structure
MODULARITY_HIGH_SILO_THRESHOLD = 0.6
LARGEST_COMPONENT_HEALTHY_MIN_PCT = 95.0
NETWORK_DENSITY_LOW_THRESHOLD = 0.04
CENTRALITY_GINI_INFO_THRESHOLD = 0.5

# Bonding / bridging
BRIDGING_RATIO_LOW_THRESHOLD = 0.3

# Department / communication
DEPARTMENT_ISOLATION_HIGH_RATIO = 0.8
EXTERNAL_COMMUNICATION_RATIO_LOW_THRESHOLD = 0.3

# Innovation
INNOVATION_INDEX_LOW_THRESHOLD = 0.4

# Burnout / anomaly presentation thresholds
OVERLOAD_PERCENTILE_HIGH_THRESHOLD = 0.9

# Metric-to-text translation (executive summaries)
METRIC_TRANSLATION_THRESHOLDS: dict[str, float] = {
    "closeness_centrality": 0.5,
    "modularity": 0.4,
    "network_density": 0.05,
    "betweenness_centrality": 0.1,
}


def is_high_modularity_silo(modularity: float) -> bool:
    return modularity > MODULARITY_HIGH_SILO_THRESHOLD


def is_fragmented_network(largest_component_pct: float) -> bool:
    return largest_component_pct < LARGEST_COMPONENT_HEALTHY_MIN_PCT


def is_low_network_density(density: float) -> bool:
    return density < NETWORK_DENSITY_LOW_THRESHOLD


def is_high_centrality_inequality(gini: float) -> bool:
    return gini > CENTRALITY_GINI_INFO_THRESHOLD


def is_low_bridging_ratio(bridging_ratio: float) -> bool:
    return bridging_ratio < BRIDGING_RATIO_LOW_THRESHOLD


def is_department_isolated(isolation_ratio: float) -> bool:
    return isolation_ratio > DEPARTMENT_ISOLATION_HIGH_RATIO


def is_low_external_communication_ratio(ratio: float) -> bool:
    return ratio < EXTERNAL_COMMUNICATION_RATIO_LOW_THRESHOLD


def is_low_innovation_index(avg_index: float) -> bool:
    return avg_index < INNOVATION_INDEX_LOW_THRESHOLD


def overall_health_warning_count(
    modularity: float,
    largest_component_pct: float,
    network_density: float,
) -> int:
    """Count structural warning signals for coarse status bucketing."""
    conditions = [
        is_high_modularity_silo(modularity),
        is_fragmented_network(largest_component_pct),
        is_low_network_density(network_density),
    ]
    return sum(conditions)
