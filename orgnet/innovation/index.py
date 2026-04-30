"""Innovation Index from Bridging vs. Bonding Analysis.

This module productizes the existing bonding/bridging analysis as an "Innovation Index"
to help companies identify teams with high bridging scores (connecting disparate silos)
as sources of innovation.
"""

import networkx as nx
import pandas as pd

from orgnet.metrics.bonding_bridging import BondingBridgingAnalyzer
from orgnet.metrics.community import CommunityDetector
from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


class InnovationIndex:
    """Innovation Index based on Bridging vs. Bonding Analysis.

    Productizes existing bonding/bridging analysis as an "Innovation Index".
    Teams with high bridging scores are often sources of innovation because
    they connect disparate silos and facilitate information flow.
    """

    def __init__(self, graph: nx.Graph, **params):
        """Initialize innovation index analyzer.

        Args:
            graph: Organizational network graph
        """
        self.graph = graph
        self.bonding_bridging_analyzer = BondingBridgingAnalyzer(graph)
        self.community_detector = CommunityDetector(graph)

    def compute(
        self, graph: nx.Graph | None = None, nodes: pd.DataFrame | None = None, **kwargs
    ) -> pd.DataFrame:
        """Compute Innovation Index scores.

        Args:
            graph: Optional graph (uses self.graph if None)
            nodes: Optional node table (not used but required by BaseMetric)

        Returns:
            DataFrame with columns: person_id, innovation_index, bridging_score,
                bonding_score, innovation_potential, innovation_level
        """
        if graph is None:
            graph = self.graph

        # Detect communities if not provided
        communities_result = self.community_detector.detect_communities(method="louvain")
        communities = communities_result.get("communities", [])

        # Analyze bonding/bridging
        bb_df = self.bonding_bridging_analyzer.analyze_bonding_bridging(communities=communities)

        # Compute innovation index from bridging score
        innovation_results = []

        for _, row in bb_df.iterrows():
            person_id = row["node_id"]
            bridging_score = row.get("bridging_weight", 0.0)
            bonding_score = row.get("bonding_weight", 0.0)

            # Innovation Index = normalized bridging score
            # High bridging = connects different groups = innovation potential
            innovation_index = self._compute_innovation_index(bridging_score, bonding_score)

            # Innovation potential classification
            innovation_potential = self._classify_innovation_potential(innovation_index)

            innovation_results.append(
                {
                    "person_id": person_id,
                    "innovation_index": innovation_index,
                    "bridging_score": bridging_score,
                    "bonding_score": bonding_score,
                    "innovation_potential": innovation_potential,
                    "innovation_level": self._classify_innovation_level(innovation_index),
                }
            )

        return pd.DataFrame(innovation_results)

    def _compute_innovation_index(self, bridging_score: float, bonding_score: float) -> float:
        """Compute Innovation Index from bridging and bonding scores.

        Args:
            bridging_score: Bridging weight (connections between groups)
            bonding_score: Bonding weight (connections within groups)

        Returns:
            Innovation Index between 0 and 1 (higher = more innovative)
        """
        total = bridging_score + bonding_score

        if total == 0:
            return 0.0

        # Innovation Index = ratio of bridging to total
        # High bridging ratio = connects different groups = innovation potential
        innovation_index = bridging_score / total

        return innovation_index

    def _classify_innovation_potential(self, innovation_index: float) -> str:
        """Classify innovation potential level.

        Args:
            innovation_index: Innovation Index score between 0 and 1

        Returns:
            Potential level: 'low', 'medium', 'high', 'very_high'
        """
        if innovation_index >= 0.7:
            return "very_high"
        elif innovation_index >= 0.5:
            return "high"
        elif innovation_index >= 0.3:
            return "medium"
        else:
            return "low"

    def _classify_innovation_level(self, innovation_index: float) -> str:
        """Classify innovation level for reporting.

        Args:
            innovation_index: Innovation Index score

        Returns:
            Innovation level: 'connector', 'broker', 'innovator', 'isolated'
        """
        if innovation_index >= 0.7:
            return "innovator"  # High bridging, connects many groups
        elif innovation_index >= 0.5:
            return "broker"  # Moderate bridging, facilitates information flow
        elif innovation_index >= 0.3:
            return "connector"  # Some bridging, limited innovation potential
        else:
            return "isolated"  # Low bridging, mostly internal connections

    def get_top_innovators(self, top_n: int = 20) -> pd.DataFrame:
        """Get top innovators by Innovation Index.

        Args:
            top_n: Number of top innovators to return

        Returns:
            DataFrame with top innovators
        """
        innovation_df = self.compute()

        if innovation_df.empty:
            return pd.DataFrame()

        top_innovators = innovation_df.nlargest(top_n, "innovation_index")[
            ["person_id", "innovation_index", "innovation_potential", "innovation_level"]
        ].reset_index(drop=True)

        return top_innovators

    def get_team_innovation_scores(self, teams: dict[str, list[str]]) -> pd.DataFrame:
        """Compute Innovation Index scores for teams.

        Args:
            teams: Dictionary mapping team_id to list of person_ids

        Returns:
            DataFrame with team innovation metrics
        """
        innovation_df = self.compute()

        if innovation_df.empty:
            return pd.DataFrame()

        team_results = []

        for team_id, person_ids in teams.items():
            team_df = innovation_df[innovation_df["person_id"].isin(person_ids)]

            if team_df.empty:
                continue

            # Aggregate team metrics
            avg_innovation_index = team_df["innovation_index"].mean()
            avg_bridging = team_df["bridging_score"].mean()
            avg_bonding = team_df["bonding_score"].mean()

            team_results.append(
                {
                    "team_id": team_id,
                    "team_size": len(team_df),
                    "avg_innovation_index": avg_innovation_index,
                    "avg_bridging_score": avg_bridging,
                    "avg_bonding_score": avg_bonding,
                    "innovation_potential": self._classify_innovation_potential(
                        avg_innovation_index
                    ),
                }
            )

        return pd.DataFrame(team_results).sort_values("avg_innovation_index", ascending=False)
