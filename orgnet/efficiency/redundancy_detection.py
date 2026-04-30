"""Work Redundancy Detection through Topic-Community Overlap.

This module identifies when disparate groups are working on the same themes
by combining Community Detection with NLP Topic Modeling (BERTopic/LDA).
This is critical for a COO focused on innovation and work deduplication.
"""

from collections import defaultdict

import networkx as nx
import pandas as pd

from orgnet.metrics.bonding_bridging import BondingBridgingAnalyzer
from orgnet.metrics.community import CommunityDetector
from orgnet.nlp.topics import TopicModeler
from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


class RedundancyDetector:
    """Work Redundancy Detection through Topic-Community Overlap.

    Identifies when two or more distinct communities are heavily engaged with
    the same technical or project-based topics, indicating redundant work.
    This helps COOs identify duplicate efforts across silos.

    Example: "Community A and Community B are both 80% focused on 'Cloud
    Migration Strategies,' yet there is zero bridging between these groups."
    """

    def __init__(
        self,
        graph: nx.Graph,
        documents: list | None = None,
        slack_messages: list | None = None,
        code_commits: list | None = None,
        min_topic_similarity: float = 0.7,
        min_bridging_threshold: float = 0.1,
        **params,
    ):
        """Initialize redundancy detector.

        Args:
            graph: Organizational network graph
            documents: Optional list of document texts for topic modeling
            slack_messages: Optional list of Slack messages for topic modeling
            code_commits: Optional list of code commit messages for topic modeling
            min_topic_similarity: Minimum topic similarity to flag redundancy (default: 0.7)
            min_bridging_threshold: Minimum bridging ratio to consider communities connected (default: 0.1)
        """
        self.graph = graph
        self.documents = documents or []
        self.slack_messages = slack_messages or []
        self.code_commits = code_commits or []
        self.min_topic_similarity = min_topic_similarity
        self.min_bridging_threshold = min_bridging_threshold

        # Initialize components
        self.community_detector = CommunityDetector(graph)
        self.bonding_bridging = BondingBridgingAnalyzer(graph)
        self.topic_modeler = TopicModeler()

    def compute(
        self, graph: nx.Graph | None = None, documents: list | None = None, **kwargs
    ) -> pd.DataFrame:
        """Detect redundant work across communities.

        Args:
            graph: Optional graph (uses self.graph if None)
            documents: Optional documents (uses self.documents if None)

        Returns:
            DataFrame with redundancy detections
        """
        if graph is not None:
            self.graph = graph
            self.community_detector.graph = graph
            self.bonding_bridging.graph = graph

        if documents is not None:
            self.documents = documents

        # Step 1: Structural Grouping (Community Detection)
        communities_result = self.community_detector.detect_communities(method="louvain")
        communities = communities_result.get("communities", [])

        if len(communities) < 2:
            logger.warning("Need at least 2 communities to detect redundancy")
            return pd.DataFrame()

        # Step 2: Topic Extraction (BERTopic)
        topic_model = self._extract_topics()

        # Step 3: Map topics to communities
        community_topics = self._map_topics_to_communities(communities, topic_model)

        # Step 4: Detect redundancy (Topic-Community Overlap + Low Bridging)
        redundancies = self._detect_redundancy(communities, community_topics)

        return redundancies

    def _extract_topics(self) -> dict:
        """Extract topics from digital exhaust using topic modeling.

        Returns:
            Dictionary with topic model results
        """
        # Combine all text sources
        all_texts = []

        # Add documents
        if self.documents:
            for doc in self.documents:
                if hasattr(doc, "text"):
                    all_texts.append(doc.text)
                elif isinstance(doc, str):
                    all_texts.append(doc)

        # Add Slack messages
        if self.slack_messages:
            for msg in self.slack_messages:
                if hasattr(msg, "text"):
                    all_texts.append(msg.text)
                elif isinstance(msg, str):
                    all_texts.append(msg)

        # Add code commit messages
        if self.code_commits:
            for commit in self.code_commits:
                if hasattr(commit, "message"):
                    all_texts.append(commit.message)
                elif isinstance(commit, dict) and "message" in commit:
                    all_texts.append(commit["message"])

        if not all_texts:
            logger.warning("No text data available for topic modeling")
            return {"topics": [], "topic_names": []}

        # Fit topic modeler
        try:
            self.topic_modeler.fit(all_texts)
            topics = self.topic_modeler.get_topics()

            return {
                "topics": topics,
                "topic_names": [f"Topic {i}" for i in range(len(topics))],
                "texts": all_texts,
            }
        except Exception as e:
            logger.warning(f"Error fitting topic modeler: {e}")
            return {"topics": [], "topic_names": []}

    def _map_topics_to_communities(
        self, communities: list[list[str]], topic_model: dict
    ) -> dict[int, dict]:
        """Map topics to communities.

        Args:
            communities: List of community node lists
            topic_model: Topic model results

        Returns:
            Dictionary mapping community_id to topic distribution
        """
        community_topics = {}

        if not topic_model.get("topics"):
            return community_topics

        # Get topic assignments for each text
        texts = topic_model.get("texts", [])
        topic_assignments = self.topic_modeler.transform(texts) if texts else []

        # Map nodes to communities
        node_to_community = {}
        for comm_id, comm_nodes in enumerate(communities):
            for node in comm_nodes:
                node_to_community[node] = comm_id

        # Build text-to-node mapping from documents/slack/commits
        text_to_node_map = {}
        text_index = 0

        # Map documents to author nodes
        for doc in self.documents:
            if hasattr(doc, "author_ids") and doc.author_ids:
                for author_id in doc.author_ids:
                    text_to_node_map[text_index] = author_id
                    text_index += 1
                    break  # One document mapped to first author
            elif hasattr(doc, "author_id") and doc.author_id:
                text_to_node_map[text_index] = doc.author_id
                text_index += 1
            elif isinstance(doc, dict) and "author_id" in doc:
                text_to_node_map[text_index] = doc["author_id"]
                text_index += 1
            else:
                text_index += 1  # Skip unmappable documents

        # Map Slack messages to source nodes
        for msg in self.slack_messages:
            if hasattr(msg, "source_id") and msg.source_id:
                text_to_node_map[text_index] = msg.source_id
            elif isinstance(msg, dict) and "source_id" in msg:
                text_to_node_map[text_index] = msg["source_id"]
            text_index += 1

        # Map code commits to author nodes
        for commit in self.code_commits:
            if hasattr(commit, "author_id") and commit.author_id:
                text_to_node_map[text_index] = commit.author_id
            elif isinstance(commit, dict) and "author_id" in commit:
                text_to_node_map[text_index] = commit["author_id"]
            text_index += 1

        # For each community, calculate topic distribution
        for comm_id, comm_nodes in enumerate(communities):
            # Find texts associated with community members
            comm_text_indices = []
            for text_idx, node_id in text_to_node_map.items():
                if node_id in comm_nodes and text_idx < len(texts):
                    comm_text_indices.append(text_idx)

            if not comm_text_indices:
                community_topics[comm_id] = {
                    "topic_distribution": {},
                    "dominant_topics": [],
                    "total_texts": 0,
                }
                continue

            # Count topic assignments for this community
            topic_counts = defaultdict(int)
            for text_idx in comm_text_indices:
                if text_idx < len(topic_assignments):
                    topic_id = topic_assignments[text_idx]
                    if topic_id >= 0:  # Only count valid topics
                        topic_counts[topic_id] += 1

            total = sum(topic_counts.values())
            topic_distribution = {
                topic_id: count / total if total > 0 else 0.0
                for topic_id, count in topic_counts.items()
            }

            # Identify dominant topics (top 3)
            dominant_topics = sorted(topic_distribution.items(), key=lambda x: x[1], reverse=True)[
                :3
            ]

            community_topics[comm_id] = {
                "topic_distribution": topic_distribution,
                "dominant_topics": [t[0] for t in dominant_topics],
                "topic_scores": dict(dominant_topics),
                "total_texts": total,
            }

        return community_topics

    def _detect_redundancy(
        self, communities: list[list[str]], community_topics: dict[int, dict]
    ) -> pd.DataFrame:
        """Detect redundancy: Topic-Community Overlap + Low Bridging.

        Args:
            communities: List of community node lists
            community_topics: Community topic mappings

        Returns:
            DataFrame with redundancy detections
        """
        redundancies = []

        # Analyze bonding/bridging between communities
        bonding_bridging_result = self.bonding_bridging.analyze_group_bonding_bridging(
            communities=communities
        )

        # For each pair of communities, check for redundancy
        for i, comm_a_nodes in enumerate(communities):
            comm_a_topics = community_topics.get(i, {})
            comm_a_dist = comm_a_topics.get("topic_distribution", {})

            for j, comm_b_nodes in enumerate(communities[i + 1 :], start=i + 1):
                comm_b_topics = community_topics.get(j, {})
                comm_b_dist = comm_b_topics.get("topic_distribution", {})

                # Calculate topic similarity
                similarity = self._calculate_topic_similarity(comm_a_dist, comm_b_dist)

                # Check bridging between communities
                bridging_ratio = self._calculate_community_bridging(comm_a_nodes, comm_b_nodes)

                # Flag redundancy if: high similarity + low bridging
                if (
                    similarity >= self.min_topic_similarity
                    and bridging_ratio < self.min_bridging_threshold
                ):
                    # Identify shared topics
                    shared_topics = self._find_shared_topics(
                        comm_a_topics.get("dominant_topics", []),
                        comm_b_topics.get("dominant_topics", []),
                        comm_a_dist,
                        comm_b_dist,
                    )

                    redundancies.append(
                        {
                            "community_a": i,
                            "community_b": j,
                            "community_a_size": len(comm_a_nodes),
                            "community_b_size": len(comm_b_nodes),
                            "topic_similarity": similarity,
                            "bridging_ratio": bridging_ratio,
                            "redundancy_score": similarity * (1.0 - bridging_ratio),
                            "shared_topics": shared_topics,
                            "shared_topic_names": [f"Topic {t}" for t in shared_topics.keys()],
                            "recommendation": (
                                f"Communities {i} and {j} are working on similar topics "
                                f"(similarity: {similarity:.1%}) with low bridging "
                                f"({bridging_ratio:.1%}). Consider connecting these groups "
                                "to reduce redundant work."
                            ),
                            "severity": (
                                "high"
                                if similarity >= 0.8 and bridging_ratio < 0.05
                                else ("medium" if similarity >= 0.7 else "low")
                            ),
                        }
                    )

        if not redundancies:
            return pd.DataFrame()

        redundancies_df = pd.DataFrame(redundancies)
        redundancies_df = redundancies_df.sort_values("redundancy_score", ascending=False)

        return redundancies_df

    def _calculate_topic_similarity(
        self, dist_a: dict[int, float], dist_b: dict[int, float]
    ) -> float:
        """Calculate topic similarity between two communities.

        Args:
            dist_a: Topic distribution for community A
            dist_b: Topic distribution for community B

        Returns:
            Similarity score between 0 and 1
        """
        if not dist_a or not dist_b:
            return 0.0

        # Cosine similarity between topic distributions
        all_topics = set(dist_a.keys()) | set(dist_b.keys())

        if not all_topics:
            return 0.0

        # Build vectors
        vec_a = [dist_a.get(topic, 0.0) for topic in all_topics]
        vec_b = [dist_b.get(topic, 0.0) for topic in all_topics]

        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = dot_product / (norm_a * norm_b)

        return similarity

    def _calculate_community_bridging(
        self, comm_a_nodes: list[str], comm_b_nodes: list[str]
    ) -> float:
        """Calculate bridging ratio between two communities.

        Args:
            comm_a_nodes: Nodes in community A
            comm_b_nodes: Nodes in community B

        Returns:
            Bridging ratio between 0 and 1
        """
        # Count cross-community edges
        cross_edges = 0
        total_edges_a = 0

        for node_a in comm_a_nodes:
            if node_a not in self.graph:
                continue

            neighbors = list(self.graph.neighbors(node_a))
            total_edges_a += len(neighbors)

            for neighbor in neighbors:
                if neighbor in comm_b_nodes:
                    cross_edges += 1

        # Bridging ratio: cross-community edges / total edges
        if total_edges_a == 0:
            return 0.0

        bridging_ratio = cross_edges / total_edges_a

        return bridging_ratio

    def _find_shared_topics(
        self,
        dominant_a: list[int],
        dominant_b: list[int],
        dist_a: dict[int, float],
        dist_b: dict[int, float],
    ) -> dict[int, float]:
        """Find topics shared between communities.

        Args:
            dominant_a: Dominant topics in community A
            dominant_b: Dominant topics in community B
            dist_a: Topic distribution for community A
            dist_b: Topic distribution for community B

        Returns:
            Dictionary mapping shared topic IDs to average scores
        """
        shared_topics = {}

        # Find topics in both dominant lists
        shared_dominant = set(dominant_a) & set(dominant_b)

        # For each shared topic, calculate average score
        for topic_id in shared_dominant:
            score_a = dist_a.get(topic_id, 0.0)
            score_b = dist_b.get(topic_id, 0.0)
            shared_topics[topic_id] = (score_a + score_b) / 2.0

        # Also check topics with high scores in both distributions
        for topic_id in set(dist_a.keys()) | set(dist_b.keys()):
            score_a = dist_a.get(topic_id, 0.0)
            score_b = dist_b.get(topic_id, 0.0)

            # If both have significant scores (>= 0.2), include it
            if score_a >= 0.2 and score_b >= 0.2 and topic_id not in shared_topics:
                shared_topics[topic_id] = (score_a + score_b) / 2.0

        return shared_topics

    def generate_redundancy_alert(self, redundancy_df: pd.DataFrame | None = None) -> dict:
        """Generate redundancy alert for COO.

        Args:
            redundancy_df: Optional pre-computed redundancy DataFrame

        Returns:
            Dictionary with redundancy alert
        """
        if redundancy_df is None:
            redundancy_df = self.compute()

        if redundancy_df.empty:
            return {
                "alert": "No redundancy detected",
                "redundancy_count": 0,
                "high_risk_count": 0,
            }

        # Summary statistics
        total_redundancies = len(redundancy_df)
        high_risk = redundancy_df[redundancy_df["severity"] == "high"]
        medium_risk = redundancy_df[redundancy_df["severity"] == "medium"]

        # Top redundancies
        top_redundancies = redundancy_df.head(10)[
            [
                "community_a",
                "community_b",
                "topic_similarity",
                "bridging_ratio",
                "redundancy_score",
                "shared_topic_names",
                "recommendation",
                "severity",
            ]
        ].to_dict("records")

        # Natural language alert
        if not high_risk.empty:
            top_high = high_risk.iloc[0]
            alert_message = (
                f"🚨 HIGH RISK: Communities {top_high['community_a']} and {top_high['community_b']} "
                f"are both {top_high['topic_similarity']:.1%} focused on the same topics "
                f"({', '.join(top_high['shared_topic_names'][:3])}), yet they have only "
                f"{top_high['bridging_ratio']:.1%} bridging between them. This indicates "
                "significant redundant work."
            )
        elif not medium_risk.empty:
            top_medium = medium_risk.iloc[0]
            alert_message = (
                f"⚠️  MEDIUM RISK: Communities {top_medium['community_a']} and {top_medium['community_b']} "
                f"show {top_medium['topic_similarity']:.1%} topic similarity with low bridging. "
                "Potential redundant work detected."
            )
        else:
            alert_message = (
                f"ℹ️  {total_redundancies} potential redundancies detected. "
                "Review recommendations for details."
            )

        return {
            "alert": alert_message,
            "redundancy_count": total_redundancies,
            "high_risk_count": len(high_risk),
            "medium_risk_count": len(medium_risk),
            "avg_topic_similarity": redundancy_df["topic_similarity"].mean(),
            "avg_bridging_ratio": redundancy_df["bridging_ratio"].mean(),
            "top_redundancies": top_redundancies,
            "recommendations": redundancy_df["recommendation"].tolist()[:10],
        }
