"""Innovation performance scoring for HR reviews.

This module generates performance-review-ready innovation behavior scores
by combining all behavioral innovation metrics into actionable insights.
"""

from datetime import datetime

import networkx as nx
import pandas as pd

from orgnet.innovation.behavior import InnovationBehaviorAnalyzer
from orgnet.innovation.collaboration_depth import CollaborationDepthAnalyzer
from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


class InnovationPerformanceScorer:
    """Generates performance-review-ready innovation behavior scores.

    Aggregates all innovation behavior metrics into:
    - Category scores (assumption challenging, influence, etc.)
    - Percentile rankings vs. organization
    - Evidence/examples for each score
    - Recommendations for development
    """

    def __init__(
        self,
        graph: nx.Graph,
        documents: list[dict] | None = None,
        interactions: list[dict] | None = None,
        code_commits: list[dict] | None = None,
        projects: list[dict] | None = None,
        people: list | None = None,
    ):
        """Initialize innovation performance scorer.

        Args:
            graph: Organizational network graph
            documents: List of document dicts
            interactions: List of interaction dicts
            code_commits: List of code commit dicts
            projects: List of project dicts
            people: List of Person objects
        """
        self.graph = graph
        self.documents = documents or []
        self.interactions = interactions or []
        self.code_commits = code_commits or []
        self.projects = projects or []
        self.people = people or []

        # Initialize analyzers
        self.behavior_analyzer = InnovationBehaviorAnalyzer(
            graph=graph,
            documents=documents,
            interactions=interactions,
            code_commits=code_commits,
            people=people,
        )

        self.collaboration_analyzer = CollaborationDepthAnalyzer(
            graph=graph,
            interactions=interactions,
            projects=projects,
            documents=documents,
            code_commits=code_commits,
            people=people,
        )

    def score_person(self, person_id: str, time_period: str | None = None) -> dict:
        """Generate performance-review-ready innovation scores for a person.

        Args:
            person_id: Person identifier
            time_period: Optional time period description (e.g., "Q1 2025")

        Returns:
            Dictionary with:
            - person_id
            - time_period
            - category_scores (dict with scores 0-100)
            - percentile_rankings (dict with percentiles)
            - innovation_performance_score (overall 0-100)
            - evidence (dict with examples for each category)
            - recommendations (list of development recommendations)
            - peer_benchmark (comparison stats)
        """
        # Get behavior scores
        behavior_df = self.behavior_analyzer.compute()
        person_behavior = behavior_df[behavior_df["person_id"] == person_id]

        if person_behavior.empty:
            logger.warning(f"No behavior data found for {person_id}")
            return self._empty_score(person_id, time_period)

        behavior_row = person_behavior.iloc[0]

        # Get collaboration scores
        collaboration_df = self.collaboration_analyzer.compute()
        person_collaboration = collaboration_df[collaboration_df["person_id"] == person_id]

        collaboration_row = person_collaboration.iloc[0] if not person_collaboration.empty else None

        # Calculate category scores (0-100)
        category_scores = {
            "assumption_challenging": int(behavior_row["assumption_challenging_score"] * 100),
            "cross_functional_influence": int(behavior_row["cross_functional_influence"] * 100),
            "process_innovation": int(behavior_row["process_innovation_score"] * 100),
            "value_creation_beyond_scope": int(behavior_row["value_creation_beyond_scope"] * 100),
        }

        if collaboration_row is not None:
            category_scores["collaboration_depth"] = int(
                collaboration_row["collaboration_depth_score"] * 100
            )
            category_scores["experimentation"] = int(
                collaboration_row["experimentation_rate"] * 100
            )
            category_scores["collaboration_velocity"] = int(
                collaboration_row["collaboration_velocity"] * 100
            )

        # Calculate percentile rankings
        percentile_rankings = self._calculate_percentiles(person_id, behavior_df, collaboration_df)

        innovation_performance_score = self._calculate_overall_score(category_scores)

        # Get evidence/examples
        evidence = self.behavior_analyzer.get_evidence_for_person(person_id)

        # Generate recommendations
        recommendations = self._generate_recommendations(category_scores, percentile_rankings)

        # Peer benchmark comparison
        peer_benchmark = self._calculate_peer_benchmark(person_id, behavior_df, collaboration_df)

        return {
            "person_id": person_id,
            "time_period": time_period or "Current Period",
            "generated_at": datetime.now().isoformat(),
            "category_scores": category_scores,
            "percentile_rankings": percentile_rankings,
            "innovation_performance_score": innovation_performance_score,
            "performance_level": self._classify_performance_level(innovation_performance_score),
            "evidence": evidence,
            "recommendations": recommendations,
            "peer_benchmark": peer_benchmark,
        }

    def score_organization(self) -> pd.DataFrame:
        """Score all people in the organization.

        Returns:
            DataFrame with innovation performance scores for all people
        """
        person_ids = list(self.graph.nodes())
        results = []

        for person_id in person_ids:
            score = self.score_person(person_id)
            results.append(
                {
                    "person_id": person_id,
                    "innovation_performance_score": score["innovation_performance_score"],
                    "performance_level": score["performance_level"],
                    "assumption_challenging": score["category_scores"]["assumption_challenging"],
                    "cross_functional_influence": score["category_scores"][
                        "cross_functional_influence"
                    ],
                    "process_innovation": score["category_scores"]["process_innovation"],
                    "value_creation_beyond_scope": score["category_scores"][
                        "value_creation_beyond_scope"
                    ],
                }
            )

        return pd.DataFrame(results).sort_values("innovation_performance_score", ascending=False)

    def _calculate_percentiles(
        self,
        person_id: str,
        behavior_df: pd.DataFrame,
        collaboration_df: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate percentile rankings for person.

        Args:
            person_id: Person identifier
            behavior_df: Behavior scores DataFrame
            collaboration_df: Collaboration scores DataFrame

        Returns:
            Dictionary with percentile rankings (0-100)
        """
        percentiles = {}

        # Get person's scores
        person_behavior = behavior_df[behavior_df["person_id"] == person_id]
        person_collaboration = collaboration_df[collaboration_df["person_id"] == person_id]

        if person_behavior.empty:
            return {}

        behavior_row = person_behavior.iloc[0]

        # Calculate percentiles for behavior metrics
        for metric in [
            "assumption_challenging_score",
            "cross_functional_influence",
            "process_innovation_score",
            "value_creation_beyond_scope",
            "innovation_behavior_score",
        ]:
            if metric in behavior_df.columns:
                person_value = behavior_row[metric]
                percentile = (behavior_df[metric] <= person_value).sum() / len(behavior_df) * 100
                metric_name = metric.replace("_score", "").replace("_", " ")
                percentiles[metric_name] = round(percentile, 1)

        # Calculate percentiles for collaboration metrics
        if not person_collaboration.empty:
            collaboration_row = person_collaboration.iloc[0]

            for metric in [
                "collaboration_depth_score",
                "experimentation_rate",
                "collaboration_velocity",
                "cross_functional_collaboration_score",
            ]:
                if metric in collaboration_df.columns:
                    person_value = collaboration_row[metric]
                    percentile = (
                        (collaboration_df[metric] <= person_value).sum()
                        / len(collaboration_df)
                        * 100
                    )
                    metric_name = (
                        metric.replace("_score", "").replace("_rate", "").replace("_", " ")
                    )
                    percentiles[metric_name] = round(percentile, 1)

        return percentiles

    def _calculate_overall_score(self, category_scores: dict[str, int]) -> int:
        """Calculate overall innovation performance score.

        Args:
            category_scores: Dictionary of category scores (0-100)

        Returns:
            Overall score (0-100)
        """
        # Core categories (from behavior)
        core_categories = [
            "assumption_challenging",
            "cross_functional_influence",
            "process_innovation",
            "value_creation_beyond_scope",
        ]

        # Optional categories (from collaboration)
        optional_categories = [
            "collaboration_depth",
            "experimentation",
            "collaboration_velocity",
        ]

        # Calculate weighted average
        core_scores = [category_scores.get(cat, 0) for cat in core_categories]
        core_avg = sum(core_scores) / len(core_scores) if core_scores else 0

        optional_scores = [
            category_scores.get(cat, 0) for cat in optional_categories if cat in category_scores
        ]
        optional_avg = sum(optional_scores) / len(optional_scores) if optional_scores else 0

        # Weight: 70% core, 30% optional (if available)
        if optional_scores:
            overall = int(core_avg * 0.7 + optional_avg * 0.3)
        else:
            overall = int(core_avg)

        return min(overall, 100)

    def _classify_performance_level(self, score: int) -> str:
        """Classify performance level from score.

        Args:
            score: Innovation performance score (0-100)

        Returns:
            Performance level: 'Top Performer', 'High Performer', 'Solid Performer', 'Developing', 'At Risk'
        """
        if score >= 80:
            return "Top Performer"
        elif score >= 65:
            return "High Performer"
        elif score >= 50:
            return "Solid Performer"
        elif score >= 35:
            return "Developing"
        else:
            return "At Risk"

    def _generate_recommendations(
        self, category_scores: dict[str, int], percentile_rankings: dict[str, float]
    ) -> list[dict]:
        """Generate development recommendations based on scores.

        Args:
            category_scores: Category scores (0-100)
            percentile_rankings: Percentile rankings

        Returns:
            List of recommendation dictionaries with 'category', 'current_score',
            'percentile', 'recommendation', 'priority'
        """
        recommendations = []

        # Define thresholds for recommendations
        LOW_THRESHOLD = 40
        MEDIUM_THRESHOLD = 60

        # Check each category
        category_descriptions = {
            "assumption_challenging": "Assumption Challenging",
            "cross_functional_influence": "Cross-Functional Influence",
            "process_innovation": "Process Innovation",
            "value_creation_beyond_scope": "Value Creation Beyond Scope",
            "collaboration_depth": "Collaboration Depth",
            "experimentation": "Low-Stakes Experimentation",
            "collaboration_velocity": "Collaboration Velocity",
        }

        for category, description in category_descriptions.items():
            score = category_scores.get(category, 0)
            percentile = percentile_rankings.get(description.replace(" ", " ").lower(), 50.0)

            if score < LOW_THRESHOLD:
                priority = "High"
                recommendation = self._get_recommendation_text(category, "low")
            elif score < MEDIUM_THRESHOLD:
                priority = "Medium"
                recommendation = self._get_recommendation_text(category, "medium")
            else:
                priority = "Low"
                recommendation = self._get_recommendation_text(category, "high")

            recommendations.append(
                {
                    "category": description,
                    "current_score": score,
                    "percentile": percentile,
                    "recommendation": recommendation,
                    "priority": priority,
                }
            )

        # Sort by priority and score
        priority_order = {"High": 0, "Medium": 1, "Low": 2}
        recommendations.sort(
            key=lambda x: (priority_order.get(x["priority"], 99), -x["current_score"])
        )

        return recommendations

    def _get_recommendation_text(self, category: str, level: str) -> str:
        """Get recommendation text for category and level.

        Args:
            category: Category name
            level: Level ('low', 'medium', 'high')

        Returns:
            Recommendation text
        """
        recommendations = {
            "assumption_challenging": {
                "low": "Increase questioning behavior. Practice asking 'why' and 'what if' in discussions. Challenge assumptions in team meetings.",
                "medium": "Continue questioning assumptions. Look for opportunities to challenge status quo and explore alternatives.",
                "high": "Excellent assumption challenging. Continue this behavior and mentor others in critical thinking.",
            },
            "cross_functional_influence": {
                "low": "Build cross-functional relationships. Attend cross-team meetings and collaborate with other departments.",
                "medium": "Expand cross-functional impact. Look for opportunities to share knowledge across teams.",
                "high": "Strong cross-functional influence. Continue building bridges and facilitating knowledge transfer.",
            },
            "process_innovation": {
                "low": "Experiment more. Try small pilots and tests. Propose new approaches to existing processes.",
                "medium": "Increase experimentation. Look for opportunities to improve processes through small tests.",
                "high": "Strong process innovation. Continue experimenting and share successful approaches with others.",
            },
            "value_creation_beyond_scope": {
                "low": "Contribute beyond primary role. Volunteer for cross-functional projects. Share expertise with other teams.",
                "medium": "Expand contributions. Look for opportunities to add value outside immediate responsibilities.",
                "high": "Excellent beyond-scope contributions. Continue adding value across the organization.",
            },
            "collaboration_depth": {
                "low": "Deepen collaborations. Build recurring relationships. Participate in long-term projects with diverse teams.",
                "medium": "Strengthen collaboration depth. Focus on building sustained relationships rather than one-off interactions.",
                "high": "Strong collaboration depth. Continue building deep, meaningful partnerships across teams.",
            },
            "experimentation": {
                "low": "Increase low-stakes experimentation. Try more pilots, prototypes, and small tests. Fail fast and learn.",
                "medium": "Experiment more frequently. Look for opportunities to test new approaches with minimal risk.",
                "high": "Excellent experimentation rate. Continue testing and iterating. Share learnings with others.",
            },
            "collaboration_velocity": {
                "low": "Improve collaboration speed. Reduce response times. Focus on faster problem-solving across teams.",
                "medium": "Maintain collaboration velocity. Look for opportunities to accelerate cross-functional problem-solving.",
                "high": "Strong collaboration velocity. Continue maintaining fast, effective cross-team collaboration.",
            },
        }

        category_recs = recommendations.get(category, {})
        return category_recs.get(level, "Continue current performance.")

    def _calculate_peer_benchmark(
        self,
        person_id: str,
        behavior_df: pd.DataFrame,
        collaboration_df: pd.DataFrame,
    ) -> dict:
        """Calculate peer benchmark statistics.

        Args:
            person_id: Person identifier
            behavior_df: Behavior scores DataFrame
            collaboration_df: Collaboration scores DataFrame

        Returns:
            Dictionary with peer benchmark statistics
        """
        if behavior_df.empty:
            return {}

        person_behavior = behavior_df[behavior_df["person_id"] == person_id]

        if person_behavior.empty:
            return {}

        behavior_row = person_behavior.iloc[0]

        # Calculate organization averages
        org_avg_innovation = behavior_df["innovation_behavior_score"].mean()
        person_innovation = behavior_row["innovation_behavior_score"]

        # Calculate percentiles
        innovation_percentile = (
            (behavior_df["innovation_behavior_score"] <= person_innovation).sum()
            / len(behavior_df)
            * 100
        )

        return {
            "organization_average": round(org_avg_innovation * 100, 1),
            "person_score": round(person_innovation * 100, 1),
            "percentile_rank": round(innovation_percentile, 1),
            "comparison": (
                "Above Average" if person_innovation > org_avg_innovation else "Below Average"
            ),
        }

    def _empty_score(self, person_id: str, time_period: str | None) -> dict:
        """Return empty score structure when no data available.

        Args:
            person_id: Person identifier
            time_period: Optional time period

        Returns:
            Empty score dictionary
        """
        return {
            "person_id": person_id,
            "time_period": time_period or "Current Period",
            "generated_at": datetime.now().isoformat(),
            "category_scores": {},
            "percentile_rankings": {},
            "innovation_performance_score": 0,
            "performance_level": "No Data",
            "evidence": {},
            "recommendations": [
                {
                    "category": "Data Collection",
                    "current_score": 0,
                    "percentile": 0,
                    "recommendation": "Insufficient data for innovation behavior analysis. Ensure activity data (documents, interactions, code commits) is available.",
                    "priority": "High",
                }
            ],
            "peer_benchmark": {},
        }
