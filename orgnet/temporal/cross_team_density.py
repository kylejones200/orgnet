"""Cross-team edge density change tracking.

This module tracks changes in cross-team collaboration density before and after
organizational events (reorgs, team changes, etc.) with clear metrics and narratives.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from orgnet.graph.temporal import TemporalGraph
from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


class CrossTeamDensityTracker:
    """Tracks cross-team edge density changes over time."""

    def __init__(
        self,
        temporal_graph: TemporalGraph,
        people: List,
        interactions: Optional[List] = None,
        team_attribute: str = "department",
    ):
        """
        Initialize cross-team density tracker.

        Args:
            temporal_graph: TemporalGraph instance
            people: List of Person objects with team/department attribute
            interactions: List of Interaction objects
            team_attribute: Attribute name for team/department (default: "department")
        """
        self.temporal_graph = temporal_graph
        self.people = {p.id: p for p in people}
        self.interactions = interactions or []
        self.team_attribute = team_attribute

    def compute_cross_team_density(
        self,
        snapshot_date: datetime,
        before_days: int = 30,
        after_days: int = 30,
    ) -> Dict:
        """
        Compute cross-team density before and after a specific date.

        Args:
            snapshot_date: Date of event (e.g., reorg date)
            before_days: Days before event to analyze
            after_days: Days after event to analyze

        Returns:
            Dictionary with before/after metrics and change analysis
        """
        before_start = snapshot_date - timedelta(days=before_days)
        after_end = snapshot_date + timedelta(days=after_days)

        # Build snapshots for before period
        before_snapshots = self.temporal_graph.build_snapshots(
            people=list(self.people.values()),
            interactions=self.interactions,
            start_date=before_start,
            end_date=snapshot_date,
        )

        # Build snapshots for after period
        after_snapshots = self.temporal_graph.build_snapshots(
            people=list(self.people.values()),
            interactions=self.interactions,
            start_date=snapshot_date,
            end_date=after_end,
        )

        # Compute metrics for before period
        before_metrics = self._compute_period_metrics(before_snapshots)

        # Compute metrics for after period
        after_metrics = self._compute_period_metrics(after_snapshots)

        # Calculate changes
        density_change = (
            after_metrics["avg_cross_team_density"] - before_metrics["avg_cross_team_density"]
        )
        density_change_pct = (
            (density_change / before_metrics["avg_cross_team_density"] * 100)
            if before_metrics["avg_cross_team_density"] > 0
            else 0
        )

        edge_count_change = (
            after_metrics["total_cross_team_edges"] - before_metrics["total_cross_team_edges"]
        )
        edge_count_change_pct = (
            (edge_count_change / before_metrics["total_cross_team_edges"] * 100)
            if before_metrics["total_cross_team_edges"] > 0
            else 0
        )

        # Classify change
        change_classification = self._classify_change(density_change_pct, edge_count_change_pct)

        return {
            "event_date": snapshot_date.isoformat(),
            "before_period": {
                "start": before_start.isoformat(),
                "end": snapshot_date.isoformat(),
                "avg_cross_team_density": before_metrics["avg_cross_team_density"],
                "total_cross_team_edges": before_metrics["total_cross_team_edges"],
                "total_edges": before_metrics["total_edges"],
                "cross_team_ratio": before_metrics["cross_team_ratio"],
            },
            "after_period": {
                "start": snapshot_date.isoformat(),
                "end": after_end.isoformat(),
                "avg_cross_team_density": after_metrics["avg_cross_team_density"],
                "total_cross_team_edges": after_metrics["total_cross_team_edges"],
                "total_edges": after_metrics["total_edges"],
                "cross_team_ratio": after_metrics["cross_team_ratio"],
            },
            "change": {
                "density_change": density_change,
                "density_change_pct": density_change_pct,
                "edge_count_change": edge_count_change,
                "edge_count_change_pct": edge_count_change_pct,
                "classification": change_classification,
            },
        }

    def _compute_period_metrics(self, snapshots: List[Dict]) -> Dict:
        """Compute cross-team metrics for a period."""
        if not snapshots:
            return {
                "avg_cross_team_density": 0.0,
                "total_cross_team_edges": 0,
                "total_edges": 0,
                "cross_team_ratio": 0.0,
            }

        cross_team_densities = []
        total_cross_team_edges = 0
        total_edges = 0

        for snapshot in snapshots:
            graph = snapshot["graph"]
            if graph.number_of_nodes() == 0:
                continue

            # Get team assignments
            node_to_team = {}
            for node in graph.nodes():
                person = self.people.get(node)
                if person and hasattr(person, self.team_attribute):
                    team = getattr(person, self.team_attribute)
                    node_to_team[node] = team
                else:
                    node_to_team[node] = "unknown"

            # Count cross-team edges
            cross_team_count = 0
            for u, v in graph.edges():
                team_u = node_to_team.get(u, "unknown")
                team_v = node_to_team.get(v, "unknown")
                if team_u != team_v and team_u != "unknown" and team_v != "unknown":
                    cross_team_count += 1

            total_edges_in_snapshot = graph.number_of_edges()
            total_edges += total_edges_in_snapshot
            total_cross_team_edges += cross_team_count

            # Cross-team density (proportion of edges that are cross-team)
            if total_edges_in_snapshot > 0:
                cross_team_ratio = cross_team_count / total_edges_in_snapshot
                cross_team_densities.append(cross_team_ratio)

        avg_cross_team_density = (
            sum(cross_team_densities) / len(cross_team_densities) if cross_team_densities else 0.0
        )
        cross_team_ratio = total_cross_team_edges / total_edges if total_edges > 0 else 0.0

        return {
            "avg_cross_team_density": avg_cross_team_density,
            "total_cross_team_edges": total_cross_team_edges,
            "total_edges": total_edges,
            "cross_team_ratio": cross_team_ratio,
        }

    def _classify_change(self, density_change_pct: float, edge_change_pct: float) -> str:
        """Classify the type of change with threshold rules."""
        # Significant increase
        if density_change_pct > 20 and edge_change_pct > 10:
            return "significant_increase"
        elif density_change_pct > 10:
            return "moderate_increase"
        # Significant decrease
        elif density_change_pct < -20 and edge_change_pct < -10:
            return "significant_decrease"
        elif density_change_pct < -10:
            return "moderate_decrease"
        # Stable
        else:
            return "stable"

    def generate_change_narrative(self, change_metrics: Dict) -> Dict[str, str]:
        """
        Generate text narrative for cross-team density change.

        Args:
            change_metrics: Dictionary from compute_cross_team_density()

        Returns:
            Dictionary with narrative components
        """
        change = change_metrics["change"]
        before = change_metrics["before_period"]
        after = change_metrics["after_period"]

        classification = change["classification"]
        density_change_pct = change["density_change_pct"]

        # Summary
        summary = (
            f"Cross-team collaboration density changed by {density_change_pct:.1f}% "
            f"after the event on {change_metrics['event_date'][:10]}. "
        )
        summary += (
            f"Before: {before['cross_team_ratio']:.2%} of edges were cross-team. "
            f"After: {after['cross_team_ratio']:.2%} of edges are cross-team."
        )

        # Pattern description
        if classification == "significant_increase":
            pattern = (
                "Strong increase in cross-team collaboration detected. "
                "The reorganization appears to have successfully increased inter-team connectivity."
            )
        elif classification == "moderate_increase":
            pattern = (
                "Moderate increase in cross-team collaboration. "
                "Some improvement in inter-team connections observed."
            )
        elif classification == "significant_decrease":
            pattern = (
                "Significant decrease in cross-team collaboration. "
                "The reorganization may have created silos or reduced inter-team communication."
            )
        elif classification == "moderate_decrease":
            pattern = (
                "Moderate decrease in cross-team collaboration. "
                "Some reduction in inter-team connections observed."
            )
        else:  # stable
            pattern = (
                "Cross-team collaboration density remained relatively stable. "
                "The reorganization did not significantly impact inter-team connectivity patterns."
            )

        # Hypothesis
        if classification in ["significant_decrease", "moderate_decrease"]:
            hypothesis = (
                "The decrease may indicate: (1) Teams are more focused internally after reorganization, "
                "(2) New team boundaries are not yet well-established, "
                "(3) Communication channels need time to rebuild, or "
                "(4) Structural changes created physical or organizational barriers."
            )
        elif classification in ["significant_increase", "moderate_increase"]:
            hypothesis = (
                "The increase may indicate: (1) Reorganization successfully broke down silos, "
                "(2) New team structures facilitate better cross-team collaboration, "
                "(3) Increased need for coordination across teams, or "
                "(4) New communication tools or processes are working effectively."
            )
        else:
            hypothesis = (
                "Stability suggests: (1) Existing collaboration patterns were preserved, "
                "(2) Teams maintained relationships through the transition, or "
                "(3) The reorganization did not significantly alter team boundaries."
            )

        # Intervention suggestions
        if classification in ["significant_decrease", "moderate_decrease"]:
            intervention = (
                "Recommended interventions: (1) Schedule cross-team meetings and workshops, "
                "(2) Create shared projects requiring inter-team collaboration, "
                "(3) Identify and support key bridge individuals, "
                "(4) Review communication tools and processes, "
                "(5) Consider team co-location or virtual collaboration spaces."
            )
        elif classification == "stable":
            intervention = (
                "Consider: (1) Monitor trends over longer period, "
                "(2) Identify opportunities to increase cross-team collaboration, "
                "(3) Support existing bridge individuals, "
                "(4) Celebrate successful cross-team initiatives."
            )
        else:  # increase
            intervention = (
                "Positive trend detected. Recommendations: (1) Sustain current initiatives, "
                "(2) Identify what's working and replicate, "
                "(3) Support key bridge individuals, "
                "(4) Monitor for potential overload of bridge individuals."
            )

        return {
            "summary": summary,
            "pattern": pattern,
            "hypothesis": hypothesis,
            "intervention": intervention,
            "classification": classification,
            "density_change_pct": density_change_pct,
        }

    def track_density_over_time(
        self, start_date: datetime, end_date: datetime, snapshot_interval_days: int = 7
    ) -> pd.DataFrame:
        """
        Track cross-team density over a time period.

        Args:
            start_date: Start of tracking period
            end_date: End of tracking period
            snapshot_interval_days: Interval between snapshots

        Returns:
            DataFrame with density metrics over time
        """
        snapshots = self.temporal_graph.build_snapshots(
            people=list(self.people.values()),
            interactions=self.interactions,
            start_date=start_date,
            end_date=end_date,
        )

        density_data = []
        for snapshot in snapshots:
            metrics = self._compute_period_metrics([snapshot])
            density_data.append(
                {
                    "date": snapshot["timestamp"],
                    "cross_team_density": metrics["avg_cross_team_density"],
                    "cross_team_ratio": metrics["cross_team_ratio"],
                    "total_cross_team_edges": metrics["total_cross_team_edges"],
                    "total_edges": metrics["total_edges"],
                }
            )

        return pd.DataFrame(density_data)
