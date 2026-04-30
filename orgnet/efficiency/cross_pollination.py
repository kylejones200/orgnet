"""Cross-Pollination Engine for Solution Transfer.

This module identifies when Group A has already solved X, and Group B is starting
X from scratch, then recommends bridging interventions to transfer knowledge.
This helps COOs facilitate cross-pollination and reduce redundant work.
"""

import networkx as nx
import pandas as pd

from orgnet.efficiency.redundancy_detection import RedundancyDetector
from orgnet.innovation.expertise import ExpertiseInferenceEngine
from orgnet.innovation.structural_holes import StructuralHoleMonetizer
from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


class CrossPollinationEngine:
    """Cross-Pollination Engine for Solution Transfer.

    Uses expertise inference and structural hole identification to recommend
    knowledge transfer between groups. When Group A has expertise and Group B
    is struggling with a related problem, this engine recommends bridging interventions.

    Example: "Group A has already solved X; Group B is starting X from scratch.
    We recommend connecting User Y (a high-bridging node) to facilitate knowledge transfer."
    """

    def __init__(
        self,
        graph: nx.Graph,
        redundancy_detector: RedundancyDetector,
        expertise_engine: ExpertiseInferenceEngine,
        **params,
    ):
        """Initialize cross-pollination engine.

        Args:
            graph: Organizational network graph
            redundancy_detector: RedundancyDetector instance for identifying redundant work
            expertise_engine: ExpertiseInferenceEngine instance for finding experts
        """
        self.graph = graph
        self.redundancy_detector = redundancy_detector
        self.expertise_engine = expertise_engine
        self.structural_hole_monetizer = StructuralHoleMonetizer(graph)

    def compute(
        self, graph: nx.Graph | None = None, redundancy_data: pd.DataFrame | None = None, **kwargs
    ) -> pd.DataFrame:
        """Generate cross-pollination recommendations.

        Args:
            graph: Optional graph (uses self.graph if None)
            redundancy_data: Optional redundancy detection results

        Returns:
            DataFrame with cross-pollination recommendations
        """
        if graph is not None:
            self.graph = graph
            if hasattr(self.redundancy_detector, "graph"):
                self.redundancy_detector.graph = graph
            if hasattr(self.redundancy_detector, "community_detector"):
                self.redundancy_detector.community_detector.graph = graph

        # Validate inputs
        if self.graph is None or len(self.graph) == 0:
            logger.warning("Graph is empty or None")
            return pd.DataFrame()

        # Get redundancy detections
        if redundancy_data is None:
            try:
                redundancy_data = self.redundancy_detector.compute()
            except Exception as e:
                logger.error(f"Error computing redundancy data: {e}", exc_info=True)
                return pd.DataFrame()

        if redundancy_data.empty:
            logger.warning("No redundancy data available for cross-pollination")
            return pd.DataFrame()

        # Get communities for expert identification
        communities = []
        try:
            if hasattr(self.redundancy_detector, "community_detector"):
                comm_result = self.redundancy_detector.community_detector.detect_communities(
                    method="louvain"
                )
                communities = comm_result.get("communities", [])
        except Exception as e:
            logger.warning(f"Error detecting communities: {e}", exc_info=True)

        # Generate cross-pollination recommendations
        recommendations = []

        for _, redundancy_row in redundancy_data.iterrows():
            try:
                comm_a = int(redundancy_row["community_a"])
                comm_b = int(redundancy_row["community_b"])
                shared_topics = redundancy_row.get("shared_topics", {})

                # Validate shared_topics is a dict
                if not isinstance(shared_topics, dict):
                    try:
                        shared_topics = (
                            eval(shared_topics) if isinstance(shared_topics, str) else {}
                        )
                    except Exception:
                        shared_topics = {}

                if not shared_topics:
                    continue

                # Step 1: Identify experts in Group A for shared topics
                group_a_experts = self._identify_group_experts(comm_a, shared_topics, communities)

                # Step 2: Identify knowledge gaps in Group B
                group_b_gaps = self._identify_knowledge_gaps(comm_b, shared_topics, communities)

                # Step 3: Find structural holes between groups
                structural_holes = self._find_structural_holes_between_groups(comm_a, comm_b)

                # Step 4: Recommend bridging candidates
                bridging_candidates = self._recommend_bridging_candidates(
                    comm_a, comm_b, group_a_experts, structural_holes, communities
                )

                # Step 5: Calculate potential ROI of knowledge transfer
                transfer_roi = self._calculate_transfer_roi(
                    redundancy_row, group_a_experts, group_b_gaps
                )

                # Generate recommendation
                recommendation = self._generate_cross_pollination_recommendation(
                    comm_a,
                    comm_b,
                    shared_topics,
                    group_a_experts,
                    group_b_gaps,
                    bridging_candidates,
                    transfer_roi,
                )

                recommendations.append(recommendation)
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Error processing redundancy row: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing redundancy row: {e}", exc_info=True)
                continue

        if not recommendations:
            return pd.DataFrame()

        recommendations_df = pd.DataFrame(recommendations)
        recommendations_df = recommendations_df.sort_values("transfer_roi", ascending=False)

        return recommendations_df

    def _identify_group_experts(self, community_id: int, topics: dict) -> list[dict]:
        """Identify experts in a community for specific topics.

        Args:
            community_id: Community ID
            topics: Dictionary of topic IDs to scores

        Returns:
            List of expert dictionaries
        """
        # Get community nodes from redundancy detector's communities
        experts = []

        # Community context available from redundancy detector for future filtering
        if hasattr(self.redundancy_detector, "community_detector"):
            self.redundancy_detector.community_detector.detect_communities(method="louvain")

        # Use expertise engine to find experts for each topic
        for topic_id, score in topics.items():
            # Find experts for this topic using the expertise engine
            try:
                # Get expertise DataFrame from engine
                expertise_df = self.expertise_engine.transform(pd.DataFrame())  # Get all expertise
                topic_experts = (
                    expertise_df[expertise_df["topic"] == topic_id]
                    .sort_values("expertise_score", ascending=False)
                    .head(5)
                    if not expertise_df.empty and "topic" in expertise_df.columns
                    else pd.DataFrame()
                )
            except Exception as e:
                logger.warning(f"Error finding experts for topic {topic_id}: {e}", exc_info=True)
                topic_experts = pd.DataFrame()

            if not topic_experts.empty:
                for _, expert_row in topic_experts.iterrows():
                    experts.append(
                        {
                            "person_id": expert_row["person_id"],
                            "topic_id": topic_id,
                            "expertise_score": expert_row["expertise_score"],
                            "topic_relevance": score,
                        }
                    )

        # Sort by combined expertise score
        experts.sort(key=lambda x: x["expertise_score"] * x["topic_relevance"], reverse=True)

        return experts[:10]  # Top 10 experts

    def _identify_knowledge_gaps(
        self, community_id: int, topics: dict, communities: list[list[str]] | None = None
    ) -> dict[int, float]:
        """Identify knowledge gaps in a community for specific topics.

        Args:
            community_id: Community ID
            topics: Dictionary of topic IDs to scores
            communities: Optional list of community node lists

        Returns:
            Dictionary mapping topic IDs to gap scores
        """
        gaps = {}

        # Get communities if not provided
        if communities is None:
            if hasattr(self.redundancy_detector, "community_detector"):
                try:
                    comm_result = self.redundancy_detector.community_detector.detect_communities(
                        method="louvain"
                    )
                    communities = comm_result.get("communities", [])
                except Exception as e:
                    logger.warning(
                        f"Error detecting communities for gap analysis: {e}", exc_info=True
                    )
                    communities = []
            else:
                communities = []

        # Check expertise in community for each topic
        for topic_id, score in topics.items():
            if not isinstance(topic_id, int) or topic_id < 0:
                gaps[topic_id] = 1.0  # Complete gap for invalid topics
                continue

            # Find experts for this topic using graph-based heuristics
            # In production, would use properly fitted expertise engine
            try:
                # Calculate average expertise in community based on network metrics
                community_expertise_scores = []

                if communities and community_id < len(communities):
                    comm_nodes = communities[community_id]

                    for node_id in comm_nodes:
                        if node_id in self.graph:
                            # Use degree centrality as proxy for expertise
                            degree = self.graph.degree(node_id)
                            # Normalize (assuming max degree ~100)
                            expertise_score = min(degree / 100.0, 1.0)
                            community_expertise_scores.append(expertise_score)

                if community_expertise_scores:
                    avg_expertise = sum(community_expertise_scores) / len(
                        community_expertise_scores
                    )
                else:
                    avg_expertise = 0.0

                # Gap score: inverse of expertise (higher = bigger gap)
                gap_score = 1.0 - min(avg_expertise, 1.0)
                gaps[topic_id] = gap_score
            except Exception as e:
                logger.warning(
                    f"Error calculating knowledge gap for topic {topic_id}: {e}", exc_info=True
                )
                gaps[topic_id] = 1.0  # Complete gap on error

        return gaps

    def _find_structural_holes_between_groups(self, comm_a: int, comm_b: int) -> pd.DataFrame:
        """Find structural holes between two communities.

        Args:
            comm_a: Community A ID
            comm_b: Community B ID

        Returns:
            DataFrame with structural hole information
        """
        # Get structural holes from monetizer
        structural_holes_df = self.structural_hole_monetizer.compute()

        if structural_holes_df.empty:
            return pd.DataFrame()

        # Filter to nodes that bridge between communities
        # Get communities to properly identify bridging nodes
        try:
            if hasattr(self.redundancy_detector, "community_detector"):
                comm_result = self.redundancy_detector.community_detector.detect_communities(
                    method="louvain"
                )
                communities = comm_result.get("communities", [])

                if comm_a < len(communities) and comm_b < len(communities):
                    comm_a_nodes = set(communities[comm_a])
                    comm_b_nodes = set(communities[comm_b])

                    # Filter to nodes that actually bridge between communities
                    bridging_nodes = []
                    for _, row in structural_holes_df.iterrows():
                        node_id = row.get("node_id") or row.get("person_id")
                        if node_id and node_id in self.graph:
                            neighbors = set(self.graph.neighbors(node_id))
                            # Check if node connects both communities
                            connects_a = bool(neighbors & comm_a_nodes)
                            connects_b = bool(neighbors & comm_b_nodes)
                            if connects_a and connects_b:
                                bridging_nodes.append(node_id)

                    if bridging_nodes:
                        return structural_holes_df[
                            structural_holes_df["node_id"].isin(bridging_nodes)
                        ].head(10)
        except Exception as e:
            logger.warning(f"Error filtering structural holes by communities: {e}", exc_info=True)

        # Fallback: return nodes marked as bottlenecks
        return structural_holes_df[
            structural_holes_df.get("information_flow_bottleneck", False) is True
        ].head(10)

    def _recommend_bridging_candidates(
        self,
        comm_a: int,
        comm_b: int,
        group_a_experts: list[dict],
        structural_holes: pd.DataFrame,
        communities: list,
    ) -> list[dict]:
        """Recommend bridging candidates for knowledge transfer.

        Args:
            comm_a: Community A ID
            comm_b: Community B ID
            group_a_experts: Experts from Group A
            structural_holes: Structural holes between groups

        Returns:
            List of bridging candidate dictionaries
        """
        candidates = []

        # Candidates from experts in Group A
        for expert in group_a_experts[:5]:  # Top 5 experts
            person_id = expert["person_id"]

            # Check if this person has bridging connections
            if person_id and person_id in self.graph:
                try:
                    neighbors = set(self.graph.neighbors(person_id))

                    # Calculate actual bridging score based on cross-community connections
                    if communities and comm_a < len(communities) and comm_b < len(communities):
                        comm_a_nodes = set(communities[comm_a])
                        comm_b_nodes = set(communities[comm_b])

                        # Count connections to the other community
                        if person_id in comm_a_nodes:
                            cross_comm_connections = len(neighbors & comm_b_nodes)
                        elif person_id in comm_b_nodes:
                            cross_comm_connections = len(neighbors & comm_a_nodes)
                        else:
                            # Person is outside both communities - check if they connect both
                            cross_comm_connections = len(
                                (neighbors & comm_a_nodes) & (neighbors & comm_b_nodes)
                            )

                        # Normalize bridging score (max ~20% of total degree for high bridging)
                        total_degree = len(neighbors)
                        bridging_score = (
                            min(cross_comm_connections / max(total_degree, 1.0), 1.0)
                            if total_degree > 0
                            else 0.0
                        )
                    else:
                        # Fallback: use degree centrality as proxy
                        total_degree = len(neighbors)
                        bridging_score = min(total_degree / 100.0, 1.0)

                    candidates.append(
                        {
                            "person_id": person_id,
                            "type": "expert",
                            "expertise_score": float(expert.get("expertise_score", 0.0)),
                            "topic_id": int(expert.get("topic_id", -1)),
                            "bridging_score": bridging_score,
                            "recommendation": (
                                f"Connect {person_id} (expert in topic {expert.get('topic_id', 'N/A')}) "
                                f"with Group {comm_b} to facilitate knowledge transfer"
                            ),
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f"Error processing expert candidate {person_id}: {e}", exc_info=True
                    )
                    continue

        # Candidates from structural holes (high-bridging nodes)
        if not structural_holes.empty:
            for _, hole_row in structural_holes.head(5).iterrows():
                try:
                    node_id = hole_row.get("node_id") or hole_row.get("person_id")
                    if not node_id or node_id not in self.graph:
                        continue

                    brokerage = float(hole_row.get("brokerage_opportunity", 0.0))

                    # Verify node actually bridges between communities
                    if communities and comm_a < len(communities) and comm_b < len(communities):
                        comm_a_nodes = set(communities[comm_a])
                        comm_b_nodes = set(communities[comm_b])
                        neighbors = set(self.graph.neighbors(node_id))

                        # Count cross-community connections
                        cross_comm_connections = len(
                            (neighbors & comm_a_nodes) | (neighbors & comm_b_nodes)
                        )
                        bridging_score = (
                            min(brokerage * 2.0, 1.0) if cross_comm_connections > 0 else 0.0
                        )
                    else:
                        bridging_score = min(brokerage * 2.0, 1.0)

                    candidates.append(
                        {
                            "person_id": node_id,
                            "type": "bridge",
                            "brokerage_opportunity": brokerage,
                            "bridging_score": bridging_score,
                            "recommendation": (
                                f"Leverage {node_id} (high-bridging node) to connect "
                                f"Groups {comm_a} and {comm_b}"
                            ),
                        }
                    )
                except (KeyError, TypeError, ValueError) as e:
                    logger.warning(f"Error processing structural hole candidate: {e}")
                    continue

        # Sort by bridging score
        candidates.sort(key=lambda x: x.get("bridging_score", 0.0), reverse=True)

        return candidates[:5]  # Top 5 candidates

    def _calculate_transfer_roi(
        self,
        redundancy_row: pd.Series,
        group_a_experts: list[dict],
        group_b_gaps: dict[int, float],
    ) -> float:
        """Calculate ROI of knowledge transfer.

        Args:
            redundancy_row: Redundancy detection row
            group_a_experts: Experts from Group A
            group_b_gaps: Knowledge gaps in Group B

        Returns:
            ROI estimate (ratio)
        """
        # Base ROI from redundancy score
        redundancy_score = redundancy_row.get("redundancy_score", 0.0)

        # Expert availability (more experts = higher ROI)
        expert_count = len(group_a_experts)
        expert_availability = min(expert_count / 5.0, 1.0)  # Normalize

        # Knowledge gap severity (bigger gaps = higher ROI)
        avg_gap = sum(group_b_gaps.values()) / len(group_b_gaps) if group_b_gaps else 0.0

        # Estimated savings: avoid redundant work
        # Assumption: 50% effort saved by reusing solution
        effort_savings = redundancy_score * 0.5

        # Transfer cost: time to facilitate knowledge transfer
        # Assumption: 10% of effort needed for transfer
        transfer_cost = 0.1

        # ROI = (savings / cost) * availability * gap_severity
        if transfer_cost > 0:
            roi = (effort_savings / transfer_cost) * expert_availability * (1.0 + avg_gap)
        else:
            roi = 0.0

        return roi

    def _generate_cross_pollination_recommendation(
        self,
        comm_a: int,
        comm_b: int,
        shared_topics: dict,
        group_a_experts: list[dict],
        group_b_gaps: dict[int, float],
        bridging_candidates: list[dict],
        transfer_roi: float,
    ) -> dict:
        """Generate cross-pollination recommendation.

        Args:
            comm_a: Community A ID
            comm_b: Community B ID
            shared_topics: Shared topics dictionary
            group_a_experts: Experts from Group A
            group_b_gaps: Knowledge gaps in Group B
            bridging_candidates: Bridging candidates
            transfer_roi: Transfer ROI estimate

        Returns:
            Recommendation dictionary
        """
        # Find top shared topic
        top_topic = max(shared_topics.items(), key=lambda x: x[1]) if shared_topics else (0, 0.0)

        # Top expert for this topic
        top_expert = group_a_experts[0] if group_a_experts else None
        top_expert_id = top_expert["person_id"] if top_expert else "N/A"

        # Top bridging candidate
        top_candidate = bridging_candidates[0] if bridging_candidates else None
        top_candidate_id = top_candidate["person_id"] if top_candidate else "N/A"

        # Generate natural language recommendation
        if top_expert and top_candidate:
            recommendation_text = (
                f"Group {comm_a} has already solved Topic {top_topic[0]} (expertise: "
                f"{top_expert['expertise_score']:.2f}). Group {comm_b} is starting "
                f"from scratch (knowledge gap: {group_b_gaps.get(top_topic[0], 0.0):.1%}). "
                f"We recommend connecting User {top_candidate_id} (a high-bridging node) "
                f"to facilitate knowledge transfer. Estimated ROI: {transfer_roi:.2f}x."
            )
        else:
            recommendation_text = (
                f"Groups {comm_a} and {comm_b} are working on similar topics "
                f"(Topic {top_topic[0]}) with low bridging. Consider connecting "
                f"these groups to reduce redundant work. Estimated ROI: {transfer_roi:.2f}x."
            )

        return {
            "from_community": comm_a,
            "to_community": comm_b,
            "shared_topics": list(shared_topics.keys()),
            "top_shared_topic": top_topic[0],
            "top_shared_topic_score": top_topic[1],
            "group_a_expert_count": len(group_a_experts),
            "top_expert": top_expert_id,
            "top_expert_score": top_expert["expertise_score"] if top_expert else 0.0,
            "group_b_avg_gap": (
                sum(group_b_gaps.values()) / len(group_b_gaps) if group_b_gaps else 0.0
            ),
            "top_bridging_candidate": top_candidate_id,
            "top_candidate_score": (
                top_candidate.get("bridging_score", 0.0) if top_candidate else 0.0
            ),
            "transfer_roi": transfer_roi,
            "recommendation": recommendation_text,
            "action_items": [
                f"Schedule knowledge transfer session between Groups {comm_a} and {comm_b}",
                f"Facilitate introduction between {top_expert_id} and Group {comm_b} representatives",
                f"Leverage {top_candidate_id} as bridging contact",
                "Document solution from Group A for Group B reuse",
            ],
            "priority": (
                "high" if transfer_roi >= 5.0 else ("medium" if transfer_roi >= 2.0 else "low")
            ),
        }

    def generate_cross_pollination_report(
        self, recommendations_df: pd.DataFrame | None = None
    ) -> dict:
        """Generate comprehensive cross-pollination report for COO.

        Args:
            recommendations_df: Optional pre-computed recommendations DataFrame

        Returns:
            Dictionary with cross-pollination report
        """
        if recommendations_df is None:
            recommendations_df = self.compute()

        if recommendations_df.empty:
            return {
                "summary": "No cross-pollination opportunities identified",
                "recommendation_count": 0,
            }

        # Summary statistics
        total_recommendations = len(recommendations_df)
        high_priority = recommendations_df[recommendations_df["priority"] == "high"]
        medium_priority = recommendations_df[recommendations_df["priority"] == "medium"]

        # Total potential ROI
        total_roi = recommendations_df["transfer_roi"].sum()
        avg_roi = recommendations_df["transfer_roi"].mean()

        # Top recommendations
        top_recommendations = recommendations_df.head(10)[
            [
                "from_community",
                "to_community",
                "top_shared_topic",
                "transfer_roi",
                "recommendation",
                "priority",
            ]
        ].to_dict("records")

        # Natural language summary
        summary = (
            f"Identified {total_recommendations} cross-pollination opportunities. "
            f"Total estimated ROI: {total_roi:.2f}x (avg: {avg_roi:.2f}x). "
            f"{len(high_priority)} high-priority recommendations for immediate action."
        )

        return {
            "summary": summary,
            "recommendation_count": total_recommendations,
            "high_priority_count": len(high_priority),
            "medium_priority_count": len(medium_priority),
            "total_estimated_roi": total_roi,
            "avg_roi": avg_roi,
            "top_recommendations": top_recommendations,
            "action_items": recommendations_df["action_items"].explode().unique().tolist()[:20],
        }
