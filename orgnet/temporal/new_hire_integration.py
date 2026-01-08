"""New hire integration tracking over first 90 days.

This module implements end-to-end tracking of new hire network integration
with weekly snapshots, clear metrics, threshold rules, and text narratives.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import networkx as nx

from orgnet.graph.temporal import TemporalGraph
from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


class NewHireIntegrationTracker:
    """Tracks new hire network integration over their first 90 days."""

    def __init__(
        self,
        temporal_graph: TemporalGraph,
        people: List,
        interactions: Optional[List] = None,
    ):
        """
        Initialize new hire integration tracker.

        Args:
            temporal_graph: TemporalGraph instance for building snapshots
            people: List of Person objects
            interactions: List of Interaction objects
        """
        self.temporal_graph = temporal_graph
        self.people = {p.id: p for p in people}
        self.interactions = interactions or []

    def track_integration(
        self,
        person_id: str,
        window_days: int = 90,
        snapshot_interval_days: int = 7,
    ) -> pd.DataFrame:
        """
        Track network integration for a new hire over time.

        Args:
            person_id: Person identifier
            window_days: Tracking window (default 90 days)
            snapshot_interval_days: Interval between snapshots (default 7 = weekly)

        Returns:
            DataFrame with weekly integration metrics
        """
        person = self.people.get(person_id)
        if not person:
            raise ValueError(f"Person {person_id} not found")

        if not hasattr(person, "start_date") or person.start_date is None:
            raise ValueError(f"Person {person_id} missing start_date")

        start_date = person.start_date
        end_date = start_date + timedelta(days=window_days)

        # Build weekly snapshots
        snapshots = self.temporal_graph.build_snapshots(
            people=list(self.people.values()),
            interactions=self.interactions,
            start_date=start_date,
            end_date=end_date,
        )

        # Track metrics over time
        integration_data = []

        for snapshot in snapshots:
            snapshot_date = snapshot["timestamp"]
            graph = snapshot["graph"]

            if person_id not in graph:
                # Person not yet in network
                integration_data.append(
                    {
                        "week": (snapshot_date - start_date).days // 7 + 1,
                        "date": snapshot_date,
                        "degree": 0,
                        "betweenness": 0.0,
                        "integration_score": 0.0,
                        "status": "not_connected",
                    }
                )
                continue

            # Compute metrics for this person in this snapshot
            degree = graph.degree(person_id)
            weighted_degree = sum(
                graph[person_id].get(neighbor, {}).get("weight", 1.0)
                for neighbor in graph.neighbors(person_id)
            )

            # Betweenness (approximate for large graphs)
            try:
                betweenness = nx.betweenness_centrality(graph, k=min(100, graph.number_of_nodes()))
                betweenness_score = betweenness.get(person_id, 0.0)
            except Exception:
                betweenness_score = 0.0

            # Integration score: weighted combination
            weeks_elapsed = (snapshot_date - start_date).days / 7.0
            expected_degree = self._expected_degree_at_week(weeks_elapsed, person)
            integration_score = min(weighted_degree / max(expected_degree, 1.0), 1.0)

            # Status classification
            status = self._classify_status(integration_score, weeks_elapsed)

            integration_data.append(
                {
                    "week": int(weeks_elapsed) + 1,
                    "date": snapshot_date,
                    "degree": degree,
                    "weighted_degree": weighted_degree,
                    "betweenness": betweenness_score,
                    "integration_score": integration_score,
                    "expected_degree": expected_degree,
                    "status": status,
                }
            )

        return pd.DataFrame(integration_data)

    def _expected_degree_at_week(self, weeks: float, person) -> float:
        """Estimate expected degree at a given week."""
        # Role-based baseline
        role_baselines = {
            "engineer": 15.0,
            "manager": 25.0,
            "product": 20.0,
            "design": 12.0,
            "sales": 18.0,
            "default": 10.0,
        }

        role_lower = (person.role or "").lower() if hasattr(person, "role") else ""
        baseline = role_baselines.get("default")
        for role_key, value in role_baselines.items():
            if role_key in role_lower:
                baseline = value
                break

        # Linear growth over 12 weeks, then plateau
        if weeks <= 12:
            return baseline * (weeks / 12.0)
        else:
            return baseline

    def _classify_status(self, score: float, weeks: float) -> str:
        """Classify integration status with threshold rules."""
        if weeks < 4:
            # Early weeks: lower expectations
            if score >= 0.3:
                return "on_track"
            elif score >= 0.15:
                return "slow_start"
            else:
                return "at_risk"
        elif weeks < 8:
            # Mid weeks: moderate expectations
            if score >= 0.6:
                return "well_integrated"
            elif score >= 0.4:
                return "on_track"
            elif score >= 0.2:
                return "slow_integration"
            else:
                return "at_risk"
        else:
            # Later weeks: higher expectations
            if score >= 0.8:
                return "well_integrated"
            elif score >= 0.6:
                return "on_track"
            elif score >= 0.4:
                return "slow_integration"
            else:
                return "at_risk"

    def generate_integration_narrative(
        self, integration_df: pd.DataFrame, person_id: str
    ) -> Dict[str, str]:
        """
        Generate text narrative for integration report.

        Args:
            integration_df: DataFrame from track_integration()
            person_id: Person identifier

        Returns:
            Dictionary with narrative components
        """
        if integration_df.empty:
            return {
                "summary": "No integration data available.",
                "trend": "Unable to determine trend.",
                "recommendation": "Ensure person has interactions in the network.",
            }

        person = self.people.get(person_id, None)
        person_name = person.name if person and hasattr(person, "name") else person_id

        latest = integration_df.iloc[-1]
        first_week = integration_df.iloc[0] if len(integration_df) > 0 else latest

        # Summary
        summary = f"{person_name} has been tracked for {len(integration_df)} weeks. "
        summary += f"Current integration score: {latest['integration_score']:.2f} "
        summary += f"(Status: {latest['status']})."

        # Trend
        if len(integration_df) >= 2:
            score_change = latest["integration_score"] - first_week["integration_score"]
            if score_change > 0.2:
                trend = "Strong positive trend - integration is accelerating."
            elif score_change > 0.05:
                trend = "Steady improvement in network integration."
            elif score_change > -0.05:
                trend = "Integration progress is flat."
            else:
                trend = "Integration is declining - intervention may be needed."
        else:
            trend = "Insufficient data to determine trend."

        # Recommendation based on status
        status = latest["status"]
        if status == "at_risk":
            recommendation = (
                "Immediate action recommended: Connect with manager, assign mentor, "
                "schedule regular check-ins, and identify key network connections to facilitate."
            )
        elif status == "slow_integration":
            recommendation = (
                "Monitor closely: Consider introducing to key team members, "
                "involve in cross-functional projects, and provide networking opportunities."
            )
        elif status == "slow_start":
            recommendation = (
                "Early stage: Ensure onboarding buddy is active, schedule team introductions, "
                "and provide clear role expectations."
            )
        elif status == "on_track":
            recommendation = (
                "Integration is progressing well. Continue current support and monitor progress."
            )
        else:  # well_integrated
            recommendation = (
                "Excellent integration. Person is well-connected in the network. "
                "Consider leveraging their connections to help other new hires."
            )

        return {
            "summary": summary,
            "trend": trend,
            "recommendation": recommendation,
            "current_status": status,
            "current_score": float(latest["integration_score"]),
            "weeks_tracked": len(integration_df),
        }

    def identify_at_risk_hires(self, window_days: int = 90, threshold: float = 0.3) -> pd.DataFrame:
        """
        Identify all new hires at risk across the organization.

        Args:
            window_days: Tracking window
            threshold: Integration score threshold for at-risk classification

        Returns:
            DataFrame with at-risk hires and their metrics
        """
        at_risk_data = []

        for person_id, person in self.people.items():
            if not hasattr(person, "start_date") or person.start_date is None:
                continue

            tenure_days = (datetime.now() - person.start_date).days
            if tenure_days > window_days or tenure_days < 7:
                continue  # Not in tracking window

            try:
                integration_df = self.track_integration(person_id, window_days=window_days)
                if integration_df.empty:
                    continue

                latest = integration_df.iloc[-1]
                if latest["integration_score"] < threshold:
                    narrative = self.generate_integration_narrative(integration_df, person_id)

                    at_risk_data.append(
                        {
                            "person_id": person_id,
                            "name": person.name if hasattr(person, "name") else person_id,
                            "department": (
                                person.department if hasattr(person, "department") else "Unknown"
                            ),
                            "weeks_elapsed": latest["week"],
                            "integration_score": latest["integration_score"],
                            "current_degree": latest["degree"],
                            "status": latest["status"],
                            "narrative": narrative["summary"],
                            "recommendation": narrative["recommendation"],
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to track integration for {person_id}: {e}")
                continue

        return pd.DataFrame(at_risk_data)
