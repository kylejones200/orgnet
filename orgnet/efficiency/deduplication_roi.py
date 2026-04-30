"""Deduplication ROI Calculator.

This module uses Time-Size Paradox analysis to quantify how much time/labor
is being wasted by redundant groups working in silos. This provides quantitative
evidence for COOs to justify cross-pollination initiatives.
"""

import networkx as nx
import pandas as pd

from orgnet.efficiency.redundancy_detection import RedundancyDetector
from orgnet.insights.team_stability import TeamStabilityAnalyzer
from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


class DeduplicationROICalculator:
    """Deduplication ROI Calculator.

    Uses Time-Size Paradox analysis to quantify waste from redundant work.
    This provides quantitative evidence for COOs to justify cross-pollination.

    Quantifies: How much time/labor is being wasted by redundant groups
    working in silos when they could be sharing solutions.
    """

    def __init__(
        self,
        graph: nx.Graph,
        redundancy_detector: RedundancyDetector,
        people: list | None = None,
        avg_hourly_cost: float = 100.0,
        redundancy_waste_factor: float = 0.5,
        **params,
    ):
        """Initialize deduplication ROI calculator.

        Args:
            graph: Organizational network graph
            redundancy_detector: RedundancyDetector instance
            people: Optional list of Person objects
            avg_hourly_cost: Average hourly cost per employee (default: $100)
            redundancy_waste_factor: Percentage of redundant work that is waste (default: 0.5 = 50%)
        """
        self.graph = graph
        self.redundancy_detector = redundancy_detector
        self.people = people or []
        self.avg_hourly_cost = avg_hourly_cost
        self.redundancy_waste_factor = redundancy_waste_factor

        # Initialize team stability analyzer for Time-Size Paradox analysis
        self.team_stability = TeamStabilityAnalyzer(graph, people) if people else None

    def compute(
        self, graph: nx.Graph | None = None, redundancy_data: pd.DataFrame | None = None, **kwargs
    ) -> pd.DataFrame:
        """Calculate deduplication ROI for redundant work.

        Args:
            graph: Optional graph (uses self.graph if None)
            redundancy_data: Optional redundancy detection results

        Returns:
            DataFrame with deduplication ROI calculations
        """
        if graph is not None:
            self.graph = graph
            self.redundancy_detector.graph = graph

        # Get redundancy detections
        if redundancy_data is None:
            redundancy_data = self.redundancy_detector.compute()

        if redundancy_data.empty:
            logger.warning("No redundancy data available for ROI calculation")
            return pd.DataFrame()

        # Calculate ROI for each redundancy
        roi_calculations = []

        for _, redundancy_row in redundancy_data.iterrows():
            comm_a = int(redundancy_row["community_a"])
            comm_b = int(redundancy_row["community_b"])
            redundancy_score = redundancy_row["redundancy_score"]
            topic_similarity = redundancy_row["topic_similarity"]
            bridging_ratio = redundancy_row["bridging_ratio"]

            # Estimate waste using Time-Size Paradox
            waste_estimate = self._estimate_redundancy_waste(
                comm_a, comm_b, redundancy_score, topic_similarity
            )

            # Calculate cost of redundant work
            redundant_cost = self._calculate_redundant_cost(waste_estimate)

            # Calculate cost of deduplication (knowledge transfer)
            deduplication_cost = self._calculate_deduplication_cost(comm_a, comm_b, bridging_ratio)

            # Calculate ROI
            if deduplication_cost > 0:
                roi_ratio = redundant_cost / deduplication_cost
                net_savings = redundant_cost - deduplication_cost
            else:
                roi_ratio = 0.0
                net_savings = redundant_cost

            # Time-to-value (how quickly can we achieve deduplication)
            time_to_value = self._estimate_time_to_value(bridging_ratio)

            roi_calculations.append(
                {
                    "community_a": comm_a,
                    "community_b": comm_b,
                    "redundancy_score": redundancy_score,
                    "topic_similarity": topic_similarity,
                    "bridging_ratio": bridging_ratio,
                    "estimated_waste_hours": waste_estimate["total_hours"],
                    "estimated_waste_weeks": waste_estimate["weeks"],
                    "redundant_cost": redundant_cost,
                    "deduplication_cost": deduplication_cost,
                    "roi_ratio": roi_ratio,
                    "net_savings": net_savings,
                    "time_to_value_weeks": time_to_value,
                    "priority": (
                        "high" if roi_ratio >= 5.0 else ("medium" if roi_ratio >= 2.0 else "low")
                    ),
                    "recommendation": (
                        f"Communities {comm_a} and {comm_b} are wasting "
                        f"${redundant_cost:,.0f} in redundant work. "
                        f"Deduplication cost: ${deduplication_cost:,.0f}. "
                        f"ROI: {roi_ratio:.2f}x. Estimated time to value: {time_to_value} weeks."
                    ),
                }
            )

        if not roi_calculations:
            return pd.DataFrame()

        roi_df = pd.DataFrame(roi_calculations)
        roi_df = roi_df.sort_values("roi_ratio", ascending=False)

        return roi_df

    def _estimate_redundancy_waste(
        self, comm_a: int, comm_b: int, redundancy_score: float, topic_similarity: float
    ) -> dict:
        """Estimate waste from redundant work using Time-Size Paradox.

        Args:
            comm_a: Community A ID
            comm_b: Community B ID
            redundancy_score: Redundancy score
            topic_similarity: Topic similarity between communities

        Returns:
            Dictionary with waste estimates
        """
        # Get community sizes from actual communities
        comm_a_size = 5  # Default minimum
        comm_b_size = 5

        try:
            if hasattr(self.redundancy_detector, "community_detector"):
                comm_result = self.redundancy_detector.community_detector.detect_communities(
                    method="louvain"
                )
                communities = comm_result.get("communities", [])

                if comm_a < len(communities):
                    comm_a_size = len(communities[comm_a])
                if comm_b < len(communities):
                    comm_b_size = len(communities[comm_b])
        except Exception as e:
            logger.warning(f"Error getting community sizes, using defaults: {e}")
            # Fallback: estimate based on redundancy score
            comm_a_size = max(5, int(20 * redundancy_score))
            comm_b_size = max(5, int(20 * redundancy_score))

        # Time-Size Paradox: Larger teams have higher waste from redundancy
        # Base waste = function of similarity and team sizes
        base_waste_hours_per_person_per_week = 10.0  # Base: 10 hours/week redundant work

        # Adjust for similarity (higher similarity = more waste)
        similarity_factor = topic_similarity

        # Adjust for team size (larger teams = more waste from fragmentation)
        avg_team_size = (comm_a_size + comm_b_size) / 2.0
        size_factor = min(avg_team_size / 10.0, 2.0)  # Max 2x for large teams

        # Total waste per person per week
        waste_per_person_per_week = (
            base_waste_hours_per_person_per_week * similarity_factor * size_factor
        )

        # Total waste across both communities
        total_people = comm_a_size + comm_b_size
        waste_per_week = waste_per_person_per_week * total_people

        # Estimate project duration (assuming 3-6 months for typical projects)
        project_duration_weeks = 12.0  # Average 12 weeks
        total_waste_hours = waste_per_week * project_duration_weeks

        return {
            "comm_a_size": comm_a_size,
            "comm_b_size": comm_b_size,
            "total_people": total_people,
            "waste_per_person_per_week": waste_per_person_per_week,
            "waste_per_week": waste_per_week,
            "weeks": project_duration_weeks,
            "total_hours": total_waste_hours,
        }

    def _calculate_redundant_cost(self, waste_estimate: dict) -> float:
        """Calculate cost of redundant work.

        Args:
            waste_estimate: Waste estimate dictionary

        Returns:
            Cost in dollars
        """
        total_hours = waste_estimate["total_hours"]
        redundant_cost = total_hours * self.avg_hourly_cost * self.redundancy_waste_factor

        return redundant_cost

    def _calculate_deduplication_cost(
        self, comm_a: int, comm_b: int, bridging_ratio: float
    ) -> float:
        """Calculate cost of deduplication (knowledge transfer).

        Args:
            comm_a: Community A ID
            comm_b: Community B ID
            bridging_ratio: Current bridging ratio between communities

        Returns:
            Deduplication cost in dollars
        """
        # Base deduplication activities:
        # 1. Knowledge transfer sessions (4 hours per session, 2 sessions)
        # 2. Documentation creation (8 hours)
        # 3. Follow-up meetings (4 hours)
        # 4. Facilitation overhead (4 hours)

        base_hours = 4 + 8 + 4 + 4  # 20 hours total
        participants_per_community = 3  # 3 people per community

        # Adjust for bridging ratio (lower bridging = more effort needed)
        bridging_adjustment = 1.0 + (1.0 - bridging_ratio) * 0.5  # Max 1.5x for zero bridging

        total_hours = base_hours * bridging_adjustment * participants_per_community
        deduplication_cost = total_hours * self.avg_hourly_cost

        return deduplication_cost

    def _estimate_time_to_value(self, bridging_ratio: float) -> float:
        """Estimate time to value for deduplication.

        Args:
            bridging_ratio: Current bridging ratio between communities

        Returns:
            Time to value in weeks
        """
        # Base time to value: 2-4 weeks
        base_weeks = 3.0

        # Adjust for bridging ratio (lower bridging = longer time)
        bridging_adjustment = 1.0 + (1.0 - bridging_ratio) * 1.0  # Max 2x for zero bridging

        time_to_value = base_weeks * bridging_adjustment

        return time_to_value

    def generate_deduplication_roi_report(self, roi_data: pd.DataFrame | None = None) -> dict:
        """Generate comprehensive deduplication ROI report for COO.

        Args:
            roi_data: Optional pre-computed ROI DataFrame

        Returns:
            Dictionary with deduplication ROI report
        """
        if roi_data is None:
            roi_data = self.compute()

        if roi_data.empty:
            return {
                "summary": "No redundancy detected - no ROI to calculate",
                "total_redundancies": 0,
                "total_waste_cost": 0.0,
            }

        # Summary statistics
        total_redundancies = len(roi_data)
        total_waste_cost = roi_data["redundant_cost"].sum()
        total_deduplication_cost = roi_data["deduplication_cost"].sum()
        total_net_savings = roi_data["net_savings"].sum()
        avg_roi = roi_data["roi_ratio"].mean()

        # High-priority opportunities (ROI >= 5.0)
        high_priority = roi_data[roi_data["roi_ratio"] >= 5.0]
        medium_priority = roi_data[(roi_data["roi_ratio"] >= 2.0) & (roi_data["roi_ratio"] < 5.0)]

        # Top opportunities
        top_opportunities = roi_data.head(10)[
            [
                "community_a",
                "community_b",
                "redundancy_score",
                "redundant_cost",
                "deduplication_cost",
                "roi_ratio",
                "net_savings",
                "time_to_value_weeks",
                "priority",
                "recommendation",
            ]
        ].to_dict("records")

        # Natural language summary
        summary = (
            f"Identified {total_redundancies} redundancy opportunities. "
            f"Total estimated waste: ${total_waste_cost:,.0f}. "
            f"Total deduplication cost: ${total_deduplication_cost:,.0f}. "
            f"Total net savings: ${total_net_savings:,.0f} "
            f"(avg ROI: {avg_roi:.2f}x). {len(high_priority)} high-priority opportunities "
            f"(ROI >= 5.0x) for immediate action."
        )

        # Waste breakdown by category
        waste_breakdown = {
            "total_waste_hours": roi_data["estimated_waste_hours"].sum(),
            "avg_waste_per_redundancy": roi_data["estimated_waste_hours"].mean(),
            "total_waste_cost": total_waste_cost,
            "avg_waste_cost_per_redundancy": roi_data["redundant_cost"].mean(),
            "waste_as_percentage_of_cost": (
                (total_waste_cost / (total_waste_cost + total_deduplication_cost)) * 100
                if (total_waste_cost + total_deduplication_cost) > 0
                else 0.0
            ),
        }

        return {
            "summary": summary,
            "total_redundancies": total_redundancies,
            "total_waste_cost": total_waste_cost,
            "total_deduplication_cost": total_deduplication_cost,
            "total_net_savings": total_net_savings,
            "avg_roi": avg_roi,
            "high_priority_count": len(high_priority),
            "medium_priority_count": len(medium_priority),
            "waste_breakdown": waste_breakdown,
            "top_opportunities": top_opportunities,
            "recommendations": roi_data["recommendation"].tolist()[:10],
            "key_insights": self._generate_roi_insights(
                roi_data, total_waste_cost, total_net_savings, avg_roi
            ),
        }

    def _generate_roi_insights(
        self, roi_data: pd.DataFrame, total_waste: float, net_savings: float, avg_roi: float
    ) -> list[str]:
        """Generate key insights from ROI analysis.

        Args:
            roi_data: ROI DataFrame
            total_waste: Total waste cost
            net_savings: Total net savings
            avg_roi: Average ROI ratio

        Returns:
            List of insight strings
        """
        insights = []

        # Overall waste insight
        insights.append(
            f"The organization is wasting ${total_waste:,.0f} annually on redundant work "
            "across silos. This represents significant inefficiency that could be reduced "
            "through better cross-community collaboration."
        )

        # ROI insight
        if avg_roi >= 5.0:
            insights.append(
                f"Average deduplication ROI is {avg_roi:.2f}x, indicating extremely high "
                "value opportunities. Immediate action recommended."
            )
        elif avg_roi >= 2.0:
            insights.append(
                f"Average deduplication ROI is {avg_roi:.2f}x, indicating good value "
                "opportunities. Prioritize high-ROI redundancies for quick wins."
            )

        # Savings insight
        if net_savings > 0:
            insights.append(
                f"Total net savings potential: ${net_savings:,.0f} after deduplication costs. "
                "This represents a significant efficiency gain for the organization."
            )

        # Time-to-value insight
        avg_time_to_value = roi_data["time_to_value_weeks"].mean()
        insights.append(
            f"Average time to value: {avg_time_to_value:.1f} weeks. Most deduplication "
            "initiatives can achieve results within 1-2 months."
        )

        # Priority distribution
        high_priority_pct = (
            len(roi_data[roi_data["priority"] == "high"]) / len(roi_data) * 100
            if len(roi_data) > 0
            else 0.0
        )
        insights.append(
            f"{high_priority_pct:.1f}% of redundancies are high-priority (ROI >= 5.0x). "
            "Focus on these for maximum impact."
        )

        return insights
