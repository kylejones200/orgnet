"""Graph construction from organizational data."""

import networkx as nx
import pandas as pd
from typing import List, Dict, Optional

from orgnet.data.models import Person, Interaction, Meeting, Document, CodeCommit
from orgnet.graph.weights import EdgeWeightCalculator
from orgnet.config import Config


class GraphBuilder:
    """Builds organizational network graphs from data."""

    def __init__(self, config: Config):
        """
        Initialize graph builder.

        Args:
            config: Configuration object
        """
        self.config = config
        graph_config = config.graph_config
        self.weight_calculator = EdgeWeightCalculator(
            weight_scheme=graph_config.get("weight_scheme", "composite"),
            recency_decay_lambda=graph_config.get("recency_decay_lambda", 0.1),
            reciprocity_weight=graph_config.get("reciprocity_weight", 0.5),
            responsiveness_weight=graph_config.get("responsiveness_weight", 0.3),
            role_weights=graph_config.get("role_weights", {}),
        )
        self.layer_weights = graph_config.get("layer_weights", {})
        self.min_edge_weight = graph_config.get("min_edge_weight", 0.01)

    def build_graph(
        self,
        people: List[Person],
        interactions: Optional[List[Interaction]] = None,
        meetings: Optional[List[Meeting]] = None,
        documents: Optional[List[Document]] = None,
        commits: Optional[List[CodeCommit]] = None,
    ) -> nx.Graph:
        """
        Build organizational graph from all data sources.

        Args:
            people: List of Person objects
            interactions: List of Interaction objects
            meetings: List of Meeting objects
            documents: List of Document objects
            commits: List of CodeCommit objects

        Returns:
            NetworkX Graph with nodes and weighted edges
        """
        # Create graph
        G = nx.Graph()

        # Add nodes with attributes
        for person in people:
            G.add_node(
                person.id,
                **{
                    "name": person.name,
                    "email": person.email,
                    "department": person.department or "Unknown",
                    "role": person.role or "Unknown",
                    "team": person.team or "Unknown",
                    "location": person.location,
                    "tenure_days": person.tenure_days,
                    "job_level": person.job_level,
                },
            )

        # Build layers
        layers = {}

        if interactions:
            comm_weights = self.weight_calculator.compute_communication_weights(
                interactions, people=people
            )
            layers["communication"] = comm_weights

        if meetings:
            meeting_weights = self.weight_calculator.compute_meeting_weights(meetings)
            layers["meeting"] = meeting_weights

        if documents:
            doc_weights = self.weight_calculator.compute_document_weights(documents)
            layers["collaboration"] = doc_weights

        if commits:
            code_weights = self.weight_calculator.compute_code_weights(commits)
            layers["code"] = code_weights

        # Fuse layers
        if layers:
            fused_weights = self.weight_calculator.fuse_layers(layers, weights=self.layer_weights)

            # Add edges to graph
            for i in fused_weights.index:
                for j in fused_weights.columns:
                    if i == j:
                        continue

                    weight = fused_weights.loc[i, j]
                    if weight >= self.min_edge_weight:
                        if G.has_edge(i, j):
                            G[i][j]["weight"] += weight
                        else:
                            G.add_edge(i, j, weight=weight)

        return G

    def build_directed_graph(
        self, people: List[Person], interactions: Optional[List[Interaction]] = None
    ) -> nx.DiGraph:
        """
        Build directed graph (for communication flows).

        Args:
            people: List of Person objects
            interactions: List of Interaction objects

        Returns:
            NetworkX DiGraph
        """
        G = nx.DiGraph()

        # Add nodes
        for person in people:
            G.add_node(
                person.id,
                **{
                    "name": person.name,
                    "department": person.department or "Unknown",
                    "role": person.role or "Unknown",
                },
            )

        if interactions:
            comm_weights = self.weight_calculator.compute_communication_weights(
                interactions, people=people
            )

            for i in comm_weights.index:
                for j in comm_weights.columns:
                    if i == j:
                        continue

                    weight = comm_weights.loc[i, j]
                    if weight >= self.min_edge_weight:
                        G.add_edge(i, j, weight=weight)

        return G

    def build_multi_layer_graph(
        self,
        people: List[Person],
        interactions: Optional[List[Interaction]] = None,
        meetings: Optional[List[Meeting]] = None,
        documents: Optional[List[Document]] = None,
        commits: Optional[List[CodeCommit]] = None,
    ) -> Dict[str, nx.Graph]:
        """
        Build separate graphs for each layer (preserves layer information).

        Args:
            people: List of Person objects
            interactions: List of Interaction objects
            meetings: List of Meeting objects
            documents: List of Document objects
            commits: List of CodeCommit objects

        Returns:
            Dictionary mapping layer name to Graph
        """
        graphs = {}

        # Create base graph with nodes
        base_G = nx.Graph()
        for person in people:
            base_G.add_node(
                person.id,
                **{
                    "name": person.name,
                    "department": person.department or "Unknown",
                    "role": person.role or "Unknown",
                },
            )

        if interactions:
            G_comm = base_G.copy()
            comm_weights = self.weight_calculator.compute_communication_weights(
                interactions, people=people
            )
            self._add_edges_from_matrix(G_comm, comm_weights)
            graphs["communication"] = G_comm

        if meetings:
            G_meeting = base_G.copy()
            meeting_weights = self.weight_calculator.compute_meeting_weights(meetings)
            self._add_edges_from_matrix(G_meeting, meeting_weights)
            graphs["meeting"] = G_meeting

        if documents:
            G_doc = base_G.copy()
            doc_weights = self.weight_calculator.compute_document_weights(documents)
            self._add_edges_from_matrix(G_doc, doc_weights)
            graphs["collaboration"] = G_doc

        if commits:
            G_code = base_G.copy()
            code_weights = self.weight_calculator.compute_code_weights(commits)
            self._add_edges_from_matrix(G_code, code_weights)
            graphs["code"] = G_code

        return graphs

    def _add_edges_from_matrix(self, G: nx.Graph, matrix: pd.DataFrame):
        """Add edges from weight matrix to graph."""
        for i in matrix.index:
            for j in matrix.columns:
                if i == j:
                    continue

                weight = matrix.loc[i, j]
                if weight >= self.min_edge_weight:
                    if G.has_edge(i, j):
                        G[i][j]["weight"] += weight
                    else:
                        G.add_edge(i, j, weight=weight)
