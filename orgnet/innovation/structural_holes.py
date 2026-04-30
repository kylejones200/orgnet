"""Structural Hole Monetization for ROI calculation.

This module uses structural hole metrics (constraint) to identify where information
is getting stuck and provides direct ROI by showing companies where to "plug" gaps
to speed up product development.
"""

import networkx as nx
import pandas as pd

from orgnet.metrics.centrality import CentralityAnalyzer
from orgnet.metrics.community import CommunityDetector
from orgnet.metrics.structural import StructuralAnalyzer
from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


class StructuralHoleMonetizer:
    """Structural Hole Monetization for ROI calculation.

    Uses structural hole metrics (constraint) to identify where information is stuck
    and provides ROI estimates for "plugging" gaps to speed up product development.
    """

    def __init__(
        self,
        graph: nx.Graph,
        avg_engineer_salary: float = 150000.0,
        project_delay_cost_per_day: float = 5000.0,
        **params,
    ):
        """Initialize structural hole monetizer.

        Args:
            graph: Organizational network graph
            avg_engineer_salary: Average engineer salary for ROI calculations (default: $150K)
            project_delay_cost_per_day: Cost per day of project delay (default: $5K)
        """
        self.graph = graph
        self.avg_engineer_salary = avg_engineer_salary
        self.project_delay_cost_per_day = project_delay_cost_per_day

        self.structural_analyzer = StructuralAnalyzer(graph)
        self.centrality_analyzer = CentralityAnalyzer(graph)
        self.community_detector = CommunityDetector(graph)

    def compute(
        self, graph: nx.Graph | None = None, nodes: pd.DataFrame | None = None, **kwargs
    ) -> pd.DataFrame:
        """Compute structural hole monetization metrics.

        Args:
            graph: Optional graph (uses self.graph if None)
            nodes: Optional node table (not used but required by BaseMetric)

        Returns:
            DataFrame with structural hole metrics and ROI estimates
        """
        if graph is not None:
            self.graph = graph
            self.structural_analyzer = StructuralAnalyzer(graph)
            self.centrality_analyzer = CentralityAnalyzer(graph)
            self.community_detector = CommunityDetector(graph)

        # Detect communities to identify information boundaries
        communities_result = self.community_detector.detect_communities(method="louvain")
        communities = communities_result.get("communities", [])

        # Compute constraint (structural holes)
        constraint_df = self.structural_analyzer.compute_constraint(standardize=False)

        # Compute betweenness centrality (information flow)
        betweenness_df = self.centrality_analyzer.compute_betweenness_centrality()

        # Compute betweenness centrality for each node
        try:
            betweenness_centrality = nx.betweenness_centrality(self.graph, normalized=True)
        except Exception:
            betweenness_centrality = {}

        # Build community mapping
        node_to_community = self._build_community_mapping(communities)

        # Compute ROI metrics for each structural hole
        monetization_results = []

        for _, row in constraint_df.iterrows():
            node_id = row["node_id"]
            constraint = row["constraint"]

            if node_id not in self.graph:
                continue

            # Identify structural hole characteristics
            hole_metrics = self._compute_structural_hole_metrics(
                node_id, constraint, node_to_community, betweenness_centrality
            )

            # Compute ROI estimates
            roi_metrics = self._compute_roi_estimates(
                node_id, constraint, hole_metrics, communities
            )

            monetization_results.append(
                {
                    "node_id": node_id,
                    "constraint": constraint,
                    "brokerage_opportunity": 1.0 - constraint,  # Low constraint = high opportunity
                    "structural_hole_size": hole_metrics["hole_size"],
                    "connected_communities": hole_metrics["num_communities"],
                    "information_flow_bottleneck": hole_metrics["is_bottleneck"],
                    "estimated_delay_days": roi_metrics["delay_days"],
                    "estimated_delay_cost": roi_metrics["delay_cost"],
                    "plugging_benefit_days": roi_metrics["benefit_days"],
                    "plugging_benefit_cost": roi_metrics["benefit_cost"],
                    "roi_ratio": roi_metrics["roi_ratio"],
                    "recommended_action": roi_metrics["recommended_action"],
                    "priority": roi_metrics["priority"],
                }
            )

        return pd.DataFrame(monetization_results).sort_values("roi_ratio", ascending=False)

    def _build_community_mapping(self, communities: list[list[str]]) -> dict[str, int]:
        """Build mapping from node to community ID.

        Args:
            communities: List of community node lists

        Returns:
            Dictionary mapping node_id to community_id
        """
        node_to_community = {}
        for comm_id, comm_nodes in enumerate(communities):
            for node in comm_nodes:
                node_to_community[node] = comm_id
        return node_to_community

    def _compute_structural_hole_metrics(
        self,
        node_id: str,
        constraint: float,
        node_to_community: dict[str, int],
        betweenness_centrality: dict[str, float],
    ) -> dict:
        """Compute structural hole characteristics.

        Args:
            node_id: Node identifier
            constraint: Constraint score (higher = more constrained = fewer holes)
            node_to_community: Node to community mapping
            betweenness_centrality: Betweenness centrality scores

        Returns:
            Dictionary with structural hole metrics
        """
        if node_id not in self.graph:
            return {
                "hole_size": 0.0,
                "num_communities": 0,
                "is_bottleneck": False,
            }

        neighbors = list(self.graph.neighbors(node_id))
        if not neighbors:
            return {
                "hole_size": 0.0,
                "num_communities": 1,
                "is_bottleneck": False,
            }

        # Count unique communities connected
        connected_communities = set()
        node_community = node_to_community.get(node_id, -1)

        for neighbor in neighbors:
            neighbor_community = node_to_community.get(neighbor, -1)
            if neighbor_community != node_community:
                connected_communities.add(neighbor_community)

        num_communities = len(connected_communities) + (1 if node_community >= 0 else 0)

        # Structural hole size = brokerage opportunity
        # Low constraint = large structural hole = high brokerage opportunity
        hole_size = 1.0 - constraint  # Inverted constraint = hole size

        # Bottleneck: high constraint + high betweenness = information stuck here
        betweenness = betweenness_centrality.get(node_id, 0.0)
        is_bottleneck = constraint > 0.7 and betweenness > 0.1

        return {
            "hole_size": hole_size,
            "num_communities": num_communities,
            "is_bottleneck": is_bottleneck,
            "betweenness": betweenness,
        }

    def _compute_roi_estimates(
        self,
        node_id: str,
        constraint: float,
        hole_metrics: dict,
        communities: list[list[str]],
    ) -> dict:
        """Compute ROI estimates for plugging structural hole.

        Args:
            node_id: Node identifier
            constraint: Constraint score
            hole_metrics: Structural hole characteristics
            communities: List of communities

        Returns:
            Dictionary with ROI metrics
        """
        # High constraint = information is stuck here = high cost
        # Low constraint = already has brokerage = lower priority for plugging

        hole_size = hole_metrics["hole_size"]
        num_communities = hole_metrics["num_communities"]
        is_bottleneck = hole_metrics["is_bottleneck"]

        # Estimate delay: high constraint + multiple communities = more delay
        # Constraint > 0.7 = significant bottleneck
        # More communities = more coordination needed = more delay
        if constraint > 0.8:
            base_delay_days = 10.0  # High constraint = significant delay
        elif constraint > 0.6:
            base_delay_days = 5.0  # Medium constraint = moderate delay
        else:
            base_delay_days = 1.0  # Low constraint = minimal delay

        # Multiply by number of communities (more communities = more coordination overhead)
        delay_days = base_delay_days * max(num_communities, 1) * (1.0 + constraint)

        # Estimate delay cost
        delay_cost = delay_days * self.project_delay_cost_per_day

        # Estimate benefit from plugging (adding connections to reduce constraint)
        # Benefit = reduction in delay from improved information flow
        # Assumption: reducing constraint by 0.3 (e.g., 0.8 -> 0.5) reduces delay by 60%
        constraint_reduction = 0.3  # Target reduction
        benefit_multiplier = 0.6  # 60% delay reduction
        benefit_days = delay_days * benefit_multiplier
        benefit_cost = benefit_days * self.project_delay_cost_per_day

        # ROI ratio = benefit / cost of intervention
        # Cost: adding 1-2 connections (engineering time)
        intervention_cost = self.avg_engineer_salary / 365.0 * 2.0  # ~2 days of engineer time
        roi_ratio = benefit_cost / intervention_cost if intervention_cost > 0 else 0.0

        # Recommended action based on ROI
        if roi_ratio > 5.0 and is_bottleneck:
            recommended_action = "high_priority_plug"  # High ROI, significant bottleneck
            priority = "critical"
        elif roi_ratio > 3.0:
            recommended_action = "medium_priority_plug"  # Good ROI
            priority = "high"
        elif roi_ratio > 1.5:
            recommended_action = "low_priority_plug"  # Positive ROI but lower priority
            priority = "medium"
        elif is_bottleneck:
            recommended_action = "monitor"  # Bottleneck but low ROI
            priority = "low"
        else:
            recommended_action = "no_action"  # Low ROI, not a bottleneck
            priority = "none"

        return {
            "delay_days": delay_days,
            "delay_cost": delay_cost,
            "benefit_days": benefit_days,
            "benefit_cost": benefit_cost,
            "intervention_cost": intervention_cost,
            "roi_ratio": roi_ratio,
            "recommended_action": recommended_action,
            "priority": priority,
        }

    def get_high_roi_holes(self, min_roi: float = 3.0, top_n: int = 20) -> pd.DataFrame:
        """Get structural holes with high ROI for plugging.

        Args:
            min_roi: Minimum ROI ratio to include (default: 3.0)
            top_n: Number of top holes to return (default: 20)

        Returns:
            DataFrame with high ROI structural holes
        """
        monetization_df = self.compute()

        if monetization_df.empty:
            return pd.DataFrame()

        # Filter by minimum ROI and priority
        high_roi = monetization_df[
            (monetization_df["roi_ratio"] >= min_roi)
            & (monetization_df["priority"].isin(["critical", "high"]))
        ]

        # Sort by ROI ratio
        top_holes = high_roi.nlargest(top_n, "roi_ratio")

        return top_holes[
            [
                "node_id",
                "constraint",
                "brokerage_opportunity",
                "connected_communities",
                "estimated_delay_cost",
                "plugging_benefit_cost",
                "roi_ratio",
                "recommended_action",
                "priority",
            ]
        ]

    def generate_roi_report(self, min_roi: float = 2.0) -> dict:
        """Generate comprehensive ROI report for structural holes.

        Args:
            min_roi: Minimum ROI ratio to include in report

        Returns:
            Dictionary with ROI report statistics and recommendations
        """
        monetization_df = self.compute()

        if monetization_df.empty:
            return {
                "summary": "No structural holes identified",
                "total_holes": 0,
                "high_roi_count": 0,
                "total_estimated_delay_cost": 0.0,
                "total_potential_benefit": 0.0,
                "recommendations": [],
            }

        # Summary statistics
        total_holes = len(monetization_df)
        high_roi = monetization_df[monetization_df["roi_ratio"] >= min_roi]
        high_roi_count = len(high_roi)

        total_delay_cost = monetization_df["estimated_delay_cost"].sum()
        total_potential_benefit = monetization_df["plugging_benefit_cost"].sum()

        critical = monetization_df[
            (monetization_df["priority"] == "critical")
            & (monetization_df["information_flow_bottleneck"])
        ]

        # Top recommendations
        top_recommendations = (
            monetization_df[
                monetization_df["recommended_action"].isin(
                    ["high_priority_plug", "medium_priority_plug"]
                )
            ]
            .nlargest(10, "roi_ratio")[
                ["node_id", "connected_communities", "estimated_delay_cost", "roi_ratio"]
            ]
            .to_dict("records")
        )

        return {
            "summary": f"Identified {total_holes} structural holes. "
            f"{high_roi_count} have ROI >= {min_roi}.",
            "total_holes": total_holes,
            "high_roi_count": high_roi_count,
            "critical_bottlenecks": len(critical),
            "total_estimated_delay_cost": total_delay_cost,
            "total_potential_benefit": total_potential_benefit,
            "avg_roi": monetization_df["roi_ratio"].mean(),
            "top_recommendations": top_recommendations,
            "recommendations": [
                {
                    "action": "Plug critical bottlenecks",
                    "count": len(critical),
                    "estimated_benefit": critical["plugging_benefit_cost"].sum(),
                },
                {
                    "action": "Prioritize high ROI holes",
                    "count": high_roi_count,
                    "estimated_benefit": high_roi["plugging_benefit_cost"].sum(),
                },
            ],
        }

    def identify_key_man_risks(
        self, communities: dict | None = None, min_communities_connected: int = 2
    ) -> pd.DataFrame:
        """Identify "key-man risks" for succession planning.

        By identifying structural holes, this shows companies where information
        is being bottlenecked by single individuals. This is critical for
        Succession Planning—identifying "key-man risks" where a single person's
        departure would break the flow of information between two critical departments.

        Args:
            communities: Optional community detection results
            min_communities_connected: Minimum number of communities connected for key-man risk (default: 2)

        Returns:
            DataFrame with key-man risks identified
        """
        monetization_df = self.compute()

        if monetization_df.empty:
            return pd.DataFrame()

        # Filter to bottlenecks connecting multiple communities
        key_man_risks = monetization_df[
            (monetization_df["information_flow_bottleneck"])
            & (monetization_df["connected_communities"] >= min_communities_connected)
        ].copy()

        if key_man_risks.empty:
            return pd.DataFrame()

        # Detect communities if not provided
        if communities is None:
            communities_result = self.community_detector.detect_communities(method="louvain")
            communities = communities_result.get("communities", [])

        # Build community mapping
        node_to_community = self._build_community_mapping(communities)

        # Add succession planning specific metrics
        key_man_risks["succession_risk"] = self._calculate_succession_risk(
            key_man_risks, node_to_community
        )

        # Classify succession risk level
        key_man_risks["risk_level"] = key_man_risks["succession_risk"].apply(
            lambda x: (
                "critical"
                if x >= 0.8
                else ("high" if x >= 0.6 else ("medium" if x >= 0.4 else "low"))
            )
        )

        # Generate succession planning recommendations
        key_man_risks["succession_recommendation"] = key_man_risks.apply(
            lambda row: self._generate_succession_recommendation(row, node_to_community),
            axis=1,
        )

        # Sort by succession risk
        key_man_risks = key_man_risks.sort_values("succession_risk", ascending=False)

        return key_man_risks[
            [
                "node_id",
                "connected_communities",
                "constraint",
                "brokerage_opportunity",
                "succession_risk",
                "risk_level",
                "estimated_delay_cost",
                "plugging_benefit_cost",
                "succession_recommendation",
            ]
        ]

    def _calculate_succession_risk(
        self, key_man_df: pd.DataFrame, node_to_community: dict[str, int]
    ) -> pd.Series:
        """Calculate succession risk score for key-man risks.

        Args:
            key_man_df: DataFrame with key-man risks
            node_to_community: Node to community mapping

        Returns:
            Series with succession risk scores
        """
        succession_risks = []

        for _, row in key_man_df.iterrows():
            node_id = row["node_id"]
            connected_communities = row["connected_communities"]
            constraint = row["constraint"]
            bottleneck = row["information_flow_bottleneck"]

            # Succession risk = function of:
            # 1. Number of communities connected (more = higher risk)
            # 2. Constraint level (higher = more critical)
            # 3. Bottleneck status (bottleneck = higher risk)

            # Normalize community count (assume max 10 communities)
            community_risk = min(connected_communities / 10.0, 1.0)

            # Constraint risk (higher constraint = more critical)
            constraint_risk = constraint

            # Bottleneck risk (binary: bottleneck = 1.0)
            bottleneck_risk = 1.0 if bottleneck else 0.5

            # Combined succession risk
            succession_risk = (
                community_risk * 0.4  # Communities connected
                + constraint_risk * 0.4  # Constraint level
                + bottleneck_risk * 0.2  # Bottleneck status
            )

            succession_risks.append(succession_risk)

        return pd.Series(succession_risks)

    def _generate_succession_recommendation(
        self, row: pd.Series, node_to_community: dict[str, int]
    ) -> str:
        """Generate succession planning recommendation.

        Args:
            row: Key-man risk row
            node_to_community: Node to community mapping

        Returns:
            Recommendation string
        """
        node_id = row["node_id"]
        connected_communities = row["connected_communities"]
        succession_risk = row["succession_risk"]

        if succession_risk >= 0.8:
            return (
                f"CRITICAL: {node_id} is a critical single point of failure connecting "
                f"{connected_communities} communities. Immediate succession planning required: "
                "identify backup contacts, document knowledge, and build redundancy."
            )
        elif succession_risk >= 0.6:
            return (
                f"HIGH: {node_id} connects {connected_communities} communities. "
                "Succession planning recommended: identify backup contacts and document processes."
            )
        elif succession_risk >= 0.4:
            return (
                f"MEDIUM: {node_id} connects {connected_communities} communities. "
                "Consider succession planning: identify potential backups."
            )
        else:
            return (
                f"LOW: {node_id} connects {connected_communities} communities. "
                "Monitor for succession planning."
            )

    def generate_succession_planning_report(self, min_risk: float = 0.6) -> dict:
        """Generate comprehensive succession planning report.

        This report identifies "key-man risks" where a single person's departure
        would break information flow between critical departments, enabling
        proactive succession planning.

        Args:
            min_risk: Minimum succession risk to include (default: 0.6)

        Returns:
            Dictionary with succession planning report
        """
        key_man_risks = self.identify_key_man_risks()

        if key_man_risks.empty:
            return {
                "summary": "No key-man risks identified",
                "total_key_man_risks": 0,
                "critical_risks": 0,
            }

        # Filter by minimum risk
        high_risk = key_man_risks[key_man_risks["succession_risk"] >= min_risk]

        # Summary statistics
        total_risks = len(key_man_risks)
        critical_risks = len(key_man_risks[key_man_risks["risk_level"] == "critical"])
        high_risks = len(key_man_risks[key_man_risks["risk_level"] == "high"])

        # Top key-man risks
        top_risks = key_man_risks.nlargest(10, "succession_risk")[
            [
                "node_id",
                "connected_communities",
                "succession_risk",
                "risk_level",
                "estimated_delay_cost",
                "succession_recommendation",
            ]
        ].to_dict("records")

        # Risk distribution by level
        risk_distribution = key_man_risks["risk_level"].value_counts().to_dict()

        return {
            "summary": (
                f"Identified {total_risks} key-man risks: "
                f"{critical_risks} critical, {high_risks} high"
            ),
            "total_key_man_risks": total_risks,
            "high_risk_count": len(high_risk),
            "risk_distribution": risk_distribution,
            "critical_risks": critical_risks,
            "high_risks": high_risks,
            "top_key_man_risks": top_risks,
            "recommendations": [
                {
                    "action": "Critical Succession Planning",
                    "count": critical_risks,
                    "description": f"{critical_risks} individuals are critical single points of failure",
                },
                {
                    "action": "High-Priority Succession Planning",
                    "count": high_risks,
                    "description": f"{high_risks} individuals are high-priority succession risks",
                },
            ],
        }
