"""Executive network analysis (from Enron project)."""

import networkx as nx
from typing import List, Set, Dict, Optional
from dataclasses import dataclass

from orgnet.utils.logging import get_logger
from orgnet.data.models import Interaction
from orgnet.metrics.community import CommunityDetector

logger = get_logger(__name__)


@dataclass
class ExecutiveNetworkAnalysis:
    """Container for executive network analysis results."""

    executive_interactions: List[Interaction]
    executive_graph: nx.Graph
    communities: Dict
    network_metrics: Dict
    key_executives: Set[str]


class ExecutiveAnalyzer:
    """Analyzes executive communication networks."""

    def __init__(self, graph: nx.Graph, interactions: List[Interaction]):
        """
        Initialize executive analyzer.

        Args:
            graph: Network graph
            interactions: List of Interaction objects
        """
        self.graph = graph
        self.interactions = interactions

    def identify_key_executives(self, method: str = "centrality", top_n: int = 20) -> Set[str]:
        """
        Identify key executives based on communication patterns.

        Args:
            method: Method for identification ('centrality' or 'volume')
            top_n: Number of executives to identify

        Returns:
            Set of key executive identifiers
        """
        logger.info(f"Identifying key executives using method: {method}")

        if method == "centrality":
            # Calculate degree centrality
            if len(self.graph) == 0:
                return set()

            degree_centrality = nx.degree_centrality(self.graph)

            # Get top nodes by centrality
            sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
            key_people = {node.lower().strip() for node, _ in sorted_nodes[:top_n]}

            logger.info(f"Identified {len(key_people)} key executives by centrality")
            return key_people

        elif method == "volume":
            from collections import Counter

            senders = [str(i.source_id).lower().strip() for i in self.interactions]
            recipients = [str(i.target_id).lower().strip() for i in self.interactions]

            sender_counts = Counter(senders)
            recipient_counts = Counter(recipients)
            combined = sender_counts + recipient_counts

            sorted_people = combined.most_common(top_n)
            key_people = {person for person, _ in sorted_people}

            logger.info(f"Identified {len(key_people)} key executives by volume")
            return key_people

        else:
            raise ValueError(f"Unknown method: {method}")

    def filter_executive_communications(self, key_executives: Set[str]) -> List[Interaction]:
        """
        Filter interactions to only communications between key executives.

        Args:
            key_executives: Set of key executive identifiers

        Returns:
            Filtered list of interactions
        """
        logger.info(
            f"Filtering executive communications from {len(self.interactions)} interactions"
        )

        # Normalize key executives set
        key_executives_lower = {str(e).lower().strip() for e in key_executives}

        executive_interactions = [
            i
            for i in self.interactions
            if str(i.source_id).lower().strip() in key_executives_lower
            and str(i.target_id).lower().strip() in key_executives_lower
        ]

        logger.info(f"Executive-to-executive communications: {len(executive_interactions)}")
        return executive_interactions

    def analyze_executive_network(
        self, key_executives: Optional[Set[str]] = None, auto_identify: bool = True, top_n: int = 20
    ) -> ExecutiveNetworkAnalysis:
        """
        Comprehensive executive network analysis.

        Args:
            key_executives: Optional set of known key executives
            auto_identify: Auto-identify executives if not provided
            top_n: Number of executives to identify (if auto_identify)

        Returns:
            ExecutiveNetworkAnalysis with all results
        """
        logger.info("Analyzing executive network")

        # Identify executives if not provided
        if key_executives is None:
            if auto_identify:
                key_executives = self.identify_key_executives(method="centrality", top_n=top_n)
            else:
                raise ValueError("Must provide key_executives or set auto_identify=True")

        # Filter executive communications
        executive_interactions = self.filter_executive_communications(key_executives)

        if not executive_interactions:
            logger.warning("No executive communications found")
            return ExecutiveNetworkAnalysis(
                executive_interactions=[],
                executive_graph=nx.Graph(),
                communities={},
                network_metrics={},
                key_executives=key_executives,
            )

        executive_nodes = {str(i.source_id).lower().strip() for i in executive_interactions} | {
            str(i.target_id).lower().strip() for i in executive_interactions
        }
        executive_subgraph = self.graph.subgraph(list(executive_nodes))

        # Detect communities
        community_detector = CommunityDetector(executive_subgraph, cache=None)
        communities = community_detector.detect_communities(method="louvain")

        # Calculate network metrics
        if len(executive_subgraph) > 0:
            network_metrics = {
                "nodes": executive_subgraph.number_of_nodes(),
                "edges": executive_subgraph.number_of_edges(),
                "density": nx.density(executive_subgraph) if len(executive_subgraph) > 1 else 0.0,
                "avg_clustering": (
                    nx.average_clustering(executive_subgraph)
                    if len(executive_subgraph) > 0
                    else 0.0
                ),
            }
        else:
            network_metrics = {}

        return ExecutiveNetworkAnalysis(
            executive_interactions=executive_interactions,
            executive_graph=executive_subgraph,
            communities=communities,
            network_metrics=network_metrics,
            key_executives=key_executives,
        )
