"""Onboarding integration analysis."""

import pandas as pd
import networkx as nx
from typing import List

from orgnet.metrics.centrality import CentralityAnalyzer


class OnboardingAnalyzer:
    """Analyzes onboarding and integration of new hires."""

    def __init__(self, graph: nx.Graph, people: List):
        """
        Initialize onboarding analyzer.

        Args:
            graph: Current organizational graph
            people: List of Person objects with start_date
        """
        self.graph = graph
        self.people = {p.id: p for p in people}

    def compute_integration_scores(self, window_days: int = 90) -> pd.DataFrame:
        """
        Compute integration scores for recent hires.

        Args:
            window_days: Time window for analysis

        Returns:
            DataFrame with integration scores
        """
        integration_data = []
        centrality_analyzer = CentralityAnalyzer(self.graph)
        degree_df = centrality_analyzer.compute_degree_centrality()

        for person_id, person in self.people.items():
            if person.tenure_days is None:
                continue

            if person.tenure_days > window_days:
                continue  # Not a recent hire

            # Get current connectivity
            if person_id in degree_df["node_id"].values:
                current_degree = degree_df[degree_df["node_id"] == person_id][
                    "degree_weighted"
                ].values[0]
            else:
                current_degree = 0

            # Expected degree based on tenure and role
            expected_degree = self._estimate_expected_degree(person, window_days)

            # Integration score
            if expected_degree > 0:
                integration_score = min(current_degree / expected_degree, 1.0)
            else:
                integration_score = 0.0

            integration_data.append(
                {
                    "person_id": person_id,
                    "name": person.name,
                    "department": person.department,
                    "role": person.role,
                    "tenure_days": person.tenure_days,
                    "current_degree": current_degree,
                    "expected_degree": expected_degree,
                    "integration_score": integration_score,
                    "status": self._classify_integration(integration_score, person.tenure_days),
                }
            )

        return pd.DataFrame(integration_data)

    def _estimate_expected_degree(self, person, window_days: int) -> float:
        """Estimate expected degree based on role and tenure."""
        # Simple heuristic: role-based baseline
        role_baselines = {
            "engineer": 15.0,
            "manager": 25.0,
            "product": 20.0,
            "design": 12.0,
            "sales": 18.0,
            "default": 10.0,
        }

        role_lower = (person.role or "").lower()
        baseline = role_baselines.get("default")
        for role_key, value in role_baselines.items():
            if role_key in role_lower:
                baseline = value
                break

        # Scale by tenure (linear growth assumption)
        tenure_factor = min(person.tenure_days / window_days, 1.0)

        return baseline * tenure_factor

    def _classify_integration(self, score: float, tenure_days: int) -> str:
        """Classify integration status using threshold-based mapping."""
        thresholds = [
            (0.8, "well_integrated"),
            (0.5, "moderate_integration"),
            (0.3, "slow_integration"),
        ]

        for threshold, status in thresholds:
            if score >= threshold:
                return status

        return "at_risk"

    def identify_at_risk_hires(self, window_days: int = 90, threshold: float = 0.3) -> pd.DataFrame:
        """
        Identify new hires at risk of poor integration.

        Args:
            window_days: Time window
            threshold: Integration score threshold

        Returns:
            DataFrame with at-risk hires
        """
        integration_df = self.compute_integration_scores(window_days)
        at_risk = integration_df[integration_df["integration_score"] < threshold]
        return at_risk.sort_values("integration_score")
