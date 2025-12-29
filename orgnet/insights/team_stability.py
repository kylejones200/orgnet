"""Team stability analysis based on size and tenure."""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List
from collections import defaultdict

from orgnet.data.models import Person


class TeamStabilityAnalyzer:
    """Analyzes team stability based on size and tenure effects."""

    def __init__(self, graph: nx.Graph, people: List[Person]):
        """
        Initialize team stability analyzer.

        Args:
            graph: Organizational graph
            people: List of Person objects
        """
        self.graph = graph
        self.people = {p.id: p for p in people}

    def analyze_team_stability(self, team_attribute: str = "team") -> pd.DataFrame:
        """
        Analyze team stability metrics including size and tenure effects.

        Implements the "Time-Size Paradox" concept:
        - Larger teams have higher fragmentation/turnover risk
        - Longer-tenured teams show more stability ("long-gets-longer" effect)

        Args:
            team_attribute: Attribute name to group by ('team', 'department', etc.)

        Returns:
            DataFrame with team stability metrics
        """
        # Group people by team
        teams = defaultdict(list)
        for person_id, person in self.people.items():
            team_name = getattr(person, team_attribute, None) or "Unknown"
            teams[team_name].append(person_id)

        results = []

        for team_name, member_ids in teams.items():
            if not member_ids:
                continue

            team_size = len(member_ids)

            # Calculate average tenure
            tenures = []
            for member_id in member_ids:
                if member_id in self.people:
                    person = self.people[member_id]
                    if person.tenure_days:
                        tenures.append(person.tenure_days)

            avg_tenure = np.mean(tenures) if tenures else 0
            median_tenure = np.median(tenures) if tenures else 0

            # Calculate network metrics for team
            team_subgraph = self.graph.subgraph(member_ids)
            team_density = nx.density(team_subgraph) if team_subgraph.number_of_nodes() > 1 else 0

            # Calculate internal vs external connectivity
            internal_edges = team_subgraph.number_of_edges()
            total_edges = sum(
                1 for u, v in self.graph.edges() if u in member_ids or v in member_ids
            )
            external_edges = total_edges - internal_edges

            internal_ratio = internal_edges / total_edges if total_edges > 0 else 0

            # Stability risk score (larger teams = higher risk)
            # Based on research: larger teams have higher turnover probability
            size_risk = min(team_size / 20.0, 1.0)  # Normalize, assume 20+ is high risk

            # Tenure stability score (longer tenure = more stable)
            # "Long-gets-longer" effect
            tenure_stability = min(avg_tenure / 365.0, 1.0)  # Normalize by 1 year

            # Combined stability score (lower is more stable)
            stability_risk = size_risk * (1 - tenure_stability * 0.5)  # Tenure reduces risk

            results.append(
                {
                    "team": team_name,
                    "size": team_size,
                    "avg_tenure_days": avg_tenure,
                    "median_tenure_days": median_tenure,
                    "team_density": team_density,
                    "internal_edges": internal_edges,
                    "external_edges": external_edges,
                    "internal_ratio": internal_ratio,
                    "size_risk": size_risk,
                    "tenure_stability": tenure_stability,
                    "stability_risk": stability_risk,
                    "recommendation": self._get_recommendation(
                        team_size, avg_tenure, stability_risk
                    ),
                }
            )

        df = pd.DataFrame(results)
        return df.sort_values("stability_risk", ascending=False)

    def _get_recommendation(self, size: int, avg_tenure: float, risk: float) -> str:
        """Get recommendation based on team stability analysis using dictionary dispatch."""
        # Define recommendation strings
        recommendations = {
            "high_risk_large": "Consider splitting into smaller teams (high size risk)",
            "high_risk_low_tenure": "Focus on retention and team building (low tenure)",
            "high_risk_other": "High stability risk - investigate team dynamics",
            "moderate_risk": "Monitor team stability - moderate risk",
            "stable_small": "Highly stable small team - good cohesion",
            "stable_default": "Stable team - maintain current structure",
        }

        # Decision logic using dictionary dispatch pattern
        if risk > 0.7:
            return (
                recommendations["high_risk_large"]
                if size > 15
                else (
                    recommendations["high_risk_low_tenure"]
                    if avg_tenure < 180
                    else recommendations["high_risk_other"]
                )
            )
        if risk > 0.5:
            return recommendations["moderate_risk"]
        if size < 5 and avg_tenure > 365:
            return recommendations["stable_small"]

        return recommendations["stable_default"]

    def identify_at_risk_teams(self, threshold: float = 0.6) -> pd.DataFrame:
        """
        Identify teams at risk of fragmentation or turnover.

        Args:
            threshold: Stability risk threshold (higher = more risk)

        Returns:
            DataFrame with at-risk teams
        """
        stability_df = self.analyze_team_stability()
        at_risk = stability_df[stability_df["stability_risk"] >= threshold]
        return at_risk.sort_values("stability_risk", ascending=False)

    def analyze_size_tenure_relationship(self) -> Dict:
        """
        Analyze the relationship between team size and tenure patterns.

        Returns:
            Dictionary with correlation and insights
        """
        stability_df = self.analyze_team_stability()

        # Calculate correlation between size and stability
        correlation = stability_df["size"].corr(stability_df["stability_risk"])

        # Group by size ranges
        size_ranges = {
            "small (2-5)": (2, 5),
            "medium (6-10)": (6, 10),
            "large (11-15)": (11, 15),
            "very_large (16+)": (16, 1000),
        }

        size_analysis = {}
        for range_name, (min_size, max_size) in size_ranges.items():
            subset = stability_df[
                (stability_df["size"] >= min_size) & (stability_df["size"] <= max_size)
            ]
            if len(subset) > 0:
                size_analysis[range_name] = {
                    "count": len(subset),
                    "avg_stability_risk": subset["stability_risk"].mean(),
                    "avg_tenure": subset["avg_tenure_days"].mean(),
                    "avg_density": subset["team_density"].mean(),
                }

        return {
            "size_stability_correlation": correlation,
            "size_range_analysis": size_analysis,
            "insights": self._generate_size_insights(size_analysis),
        }

    def _generate_size_insights(self, size_analysis: Dict) -> List[str]:
        """Generate insights from size analysis."""
        insights = []

        if "small (2-5)" in size_analysis and "very_large (16+)" in size_analysis:
            small_risk = size_analysis["small (2-5)"]["avg_stability_risk"]
            large_risk = size_analysis["very_large (16+)"]["avg_stability_risk"]

            if large_risk > small_risk * 1.5:
                insights.append(
                    f"Large teams ({large_risk:.2f} risk) show significantly higher "
                    f"stability risk than small teams ({small_risk:.2f} risk). "
                    "Consider breaking large teams into smaller units."
                )

        return insights
