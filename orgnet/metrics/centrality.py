"""Centrality measures for organizational networks."""

import networkx as nx
import pandas as pd
from typing import Dict

from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


class CentralityAnalyzer:
    """Computes various centrality measures for organizational networks."""

    def __init__(self, graph: nx.Graph):
        """
        Initialize centrality analyzer.

        Args:
            graph: NetworkX graph
        """
        self.graph = graph

    def compute_all_centralities(self, top_n: int = 20) -> Dict[str, pd.DataFrame]:
        """
        Compute all centrality measures.

        Args:
            top_n: Number of top nodes to return in summary

        Returns:
            Dictionary mapping metric name to DataFrame with results
        """
        results = {}

        # Degree centrality
        degree = self.compute_degree_centrality()
        results["degree"] = degree

        # Betweenness centrality
        betweenness = self.compute_betweenness_centrality()
        results["betweenness"] = betweenness

        # Eigenvector centrality
        eigenvector = self.compute_eigenvector_centrality()
        results["eigenvector"] = eigenvector

        # Closeness centrality
        closeness = self.compute_closeness_centrality()
        results["closeness"] = closeness

        # PageRank
        pagerank = self.compute_pagerank()
        results["pagerank"] = pagerank

        # HITS (if directed graph)
        if isinstance(self.graph, nx.DiGraph):
            hits = self.compute_hits()
            results["hits_authority"] = hits["authority"]
            results["hits_hub"] = hits["hub"]

        return results

    def compute_degree_centrality(self) -> pd.DataFrame:
        """
        Compute degree centrality (weighted and unweighted).

        Returns:
            DataFrame with node_id, degree_unweighted, degree_weighted, in_degree, out_degree
        """
        if isinstance(self.graph, nx.DiGraph):
            in_degree = dict(self.graph.in_degree())
            out_degree = dict(self.graph.out_degree())
            degree = {node: in_degree[node] + out_degree[node] for node in self.graph.nodes()}
        else:
            degree = dict(self.graph.degree())
            in_degree = degree
            out_degree = degree

        # Weighted degree
        weighted_degree = {}
        for node in self.graph.nodes():
            weight_sum = sum(
                self.graph[node][neighbor].get("weight", 1.0)
                for neighbor in self.graph.neighbors(node)
            )
            weighted_degree[node] = weight_sum

        df = pd.DataFrame(
            {
                "node_id": list(self.graph.nodes()),
                "degree_unweighted": [degree.get(n, 0) for n in self.graph.nodes()],
                "degree_weighted": [weighted_degree.get(n, 0) for n in self.graph.nodes()],
                "in_degree": [in_degree.get(n, 0) for n in self.graph.nodes()],
                "out_degree": [out_degree.get(n, 0) for n in self.graph.nodes()],
            }
        )

        return df.sort_values("degree_weighted", ascending=False)

    def compute_betweenness_centrality(self, normalized: bool = True) -> pd.DataFrame:
        """
        Compute betweenness centrality.

        Args:
            normalized: Whether to normalize by number of pairs

        Returns:
            DataFrame with node_id and betweenness_centrality
        """
        if self.graph.number_of_nodes() == 0:
            return pd.DataFrame(columns=["node_id", "betweenness_centrality"])

        betweenness = nx.betweenness_centrality(
            self.graph,
            weight="weight" if nx.is_weighted(self.graph) else None,
            normalized=normalized,
        )

        df = pd.DataFrame(
            {
                "node_id": list(betweenness.keys()),
                "betweenness_centrality": list(betweenness.values()),
            }
        )

        return df.sort_values("betweenness_centrality", ascending=False)

    def compute_eigenvector_centrality(self, max_iter: int = 100) -> pd.DataFrame:
        """
        Compute eigenvector centrality.

        Args:
            max_iter: Maximum iterations

        Returns:
            DataFrame with node_id and eigenvector_centrality
        """
        if self.graph.number_of_nodes() == 0:
            return pd.DataFrame(columns=["node_id", "eigenvector_centrality"])

        try:
            eigenvector = nx.eigenvector_centrality(
                self.graph,
                weight="weight" if nx.is_weighted(self.graph) else None,
                max_iter=max_iter,
            )
        except nx.PowerIterationFailedConvergence:
            logger.warning("Eigenvector centrality failed to converge, falling back to PageRank")
            # Fallback to PageRank if eigenvector fails
            eigenvector = nx.pagerank(
                self.graph, weight="weight" if nx.is_weighted(self.graph) else None
            )

        df = pd.DataFrame(
            {
                "node_id": list(eigenvector.keys()),
                "eigenvector_centrality": list(eigenvector.values()),
            }
        )

        return df.sort_values("eigenvector_centrality", ascending=False)

    def compute_closeness_centrality(self) -> pd.DataFrame:
        """
        Compute closeness centrality.

        Returns:
            DataFrame with node_id and closeness_centrality
        """
        if self.graph.number_of_nodes() == 0:
            return pd.DataFrame(columns=["node_id", "closeness_centrality"])

        # Only compute for connected components
        closeness = {}
        for component in nx.connected_components(self.graph):
            if len(component) < 2:
                continue

            subgraph = self.graph.subgraph(component)
            component_closeness = nx.closeness_centrality(
                subgraph, distance="weight" if nx.is_weighted(self.graph) else None
            )
            closeness.update(component_closeness)

        # Set disconnected nodes to 0
        for node in self.graph.nodes():
            if node not in closeness:
                closeness[node] = 0.0

        df = pd.DataFrame(
            {"node_id": list(closeness.keys()), "closeness_centrality": list(closeness.values())}
        )

        return df.sort_values("closeness_centrality", ascending=False)

    def compute_pagerank(self, damping: float = 0.85) -> pd.DataFrame:
        """
        Compute PageRank centrality.

        Args:
            damping: Damping factor

        Returns:
            DataFrame with node_id and pagerank
        """
        if self.graph.number_of_nodes() == 0:
            return pd.DataFrame(columns=["node_id", "pagerank"])

        pagerank = nx.pagerank(
            self.graph, weight="weight" if nx.is_weighted(self.graph) else None, alpha=damping
        )

        df = pd.DataFrame({"node_id": list(pagerank.keys()), "pagerank": list(pagerank.values())})

        return df.sort_values("pagerank", ascending=False)

    def compute_hits(self, max_iter: int = 100, normalized: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Compute HITS (Hyperlink-Induced Topic Search) scores (from Enron project).

        HITS identifies:
        - Authorities: Nodes with high in-degree (many links to them)
        - Hubs: Nodes with high out-degree (links to many authorities)

        Args:
            max_iter: Maximum iterations
            normalized: Whether to normalize scores

        Returns:
            Dictionary with 'authority' and 'hub' DataFrames
        """
        if self.graph.number_of_nodes() == 0:
            return {
                "authority": pd.DataFrame(columns=["node_id", "authority_score"]),
                "hub": pd.DataFrame(columns=["node_id", "hub_score"]),
            }

        # HITS requires directed graph
        if not isinstance(self.graph, nx.DiGraph):
            G = self.graph.to_directed()
        else:
            G = self.graph

        try:
            hits = nx.hits(G, max_iter=max_iter, normalized=normalized)
            authority_scores, hub_scores = hits

            logger.info(f"Computed HITS for {len(authority_scores)} nodes")
        except Exception as e:
            logger.warning(f"HITS computation failed: {e}, using in/out degree as fallback")
            # Fallback: use in/out degree as proxies
            in_degree = dict(G.in_degree())
            out_degree = dict(G.out_degree())
            max_in = max(in_degree.values()) if in_degree.values() else 1
            max_out = max(out_degree.values()) if out_degree.values() else 1

            authority_scores = {n: in_degree.get(n, 0) / max_in for n in G.nodes()}
            hub_scores = {n: out_degree.get(n, 0) / max_out for n in G.nodes()}

        authority_df = pd.DataFrame(
            {
                "node_id": list(authority_scores.keys()),
                "authority_score": list(authority_scores.values()),
            }
        ).sort_values("authority_score", ascending=False)

        hub_df = pd.DataFrame(
            {"node_id": list(hub_scores.keys()), "hub_score": list(hub_scores.values())}
        ).sort_values("hub_score", ascending=False)

        return {"authority": authority_df, "hub": hub_df}

    def get_top_central_nodes(self, metric: str = "betweenness", top_n: int = 10) -> pd.DataFrame:
        """
        Get top central nodes by specified metric.

        Args:
            metric: Centrality metric name
            top_n: Number of top nodes

        Returns:
            DataFrame with top nodes
        """
        metric_config = {
            "degree": (self.compute_degree_centrality, "degree_weighted"),
            "betweenness": (self.compute_betweenness_centrality, "betweenness_centrality"),
            "eigenvector": (self.compute_eigenvector_centrality, "eigenvector_centrality"),
            "closeness": (self.compute_closeness_centrality, "closeness_centrality"),
            "pagerank": (self.compute_pagerank, "pagerank"),
            "hits_authority": (lambda: self.compute_hits()["authority"], "authority_score"),
            "hits_hub": (lambda: self.compute_hits()["hub"], "hub_score"),
        }

        compute_func, sort_column = metric_config.get(metric, (None, None))
        if compute_func is None:
            raise ValueError(f"Unknown metric: {metric}. Choose from {list(metric_config.keys())}")

        df = compute_func()
        return df.nlargest(top_n, sort_column)
