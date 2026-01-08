"""Centrality measures for organizational networks."""

import networkx as nx
import pandas as pd
from typing import Dict, Optional

from orgnet.utils.logging import get_logger
from orgnet.metrics.utils import standardize_metric_output

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

    def compute_all_centralities(
        self, top_n: int = 20, standardize: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute all centrality measures.

        Args:
            top_n: Number of top nodes to flag
            standardize: If True, add ranks and flags to outputs

        Returns:
            Dictionary mapping metric name to DataFrame with standardized results
        """
        results = {}

        # Degree centrality (use weighted degree as primary value)
        degree = self.compute_degree_centrality(standardize=standardize, top_n=top_n)
        results["degree"] = degree

        # Betweenness centrality
        betweenness = self.compute_betweenness_centrality(standardize=standardize, top_n=top_n)
        results["betweenness"] = betweenness

        # Eigenvector centrality
        eigenvector = self.compute_eigenvector_centrality(standardize=standardize, top_n=top_n)
        results["eigenvector"] = eigenvector

        # Closeness centrality
        closeness = self.compute_closeness_centrality(standardize=standardize, top_n=top_n)
        results["closeness"] = closeness

        # PageRank
        pagerank = self.compute_pagerank(standardize=standardize, top_n=top_n)
        results["pagerank"] = pagerank

        # HITS (if directed graph)
        if isinstance(self.graph, nx.DiGraph):
            hits = self.compute_hits(standardize=standardize, top_n=top_n)
            results["hits_authority"] = hits["authority"]
            results["hits_hub"] = hits["hub"]

        return results

    def compute_degree_centrality(
        self, standardize: bool = True, top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compute degree centrality (weighted and unweighted).

        Args:
            standardize: If True, add ranks and flags
            top_n: Number of top nodes to flag (uses default if None)

        Returns:
            DataFrame with node_id, value (degree_weighted), rank, flags, and additional columns
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

        if standardize:
            df = standardize_metric_output(
                df, value_column="degree_weighted", id_column="node_id", top_n=top_n
            )

        return df.sort_values("degree_weighted" if not standardize else "value", ascending=False)

    def compute_betweenness_centrality(
        self, normalized: bool = True, standardize: bool = True, top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compute betweenness centrality.

        Args:
            normalized: Whether to normalize by number of pairs
            standardize: If True, add ranks and flags
            top_n: Number of top nodes to flag

        Returns:
            DataFrame with node_id, value, rank, flags
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

        if standardize:
            df = standardize_metric_output(
                df, value_column="betweenness_centrality", id_column="node_id", top_n=top_n
            )

        return df.sort_values(
            "betweenness_centrality" if not standardize else "value", ascending=False
        )

    def compute_eigenvector_centrality(
        self, max_iter: int = 100, standardize: bool = True, top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compute eigenvector centrality.

        Args:
            max_iter: Maximum iterations
            standardize: If True, add ranks and flags
            top_n: Number of top nodes to flag

        Returns:
            DataFrame with node_id, value, rank, flags
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

        if standardize:
            df = standardize_metric_output(
                df, value_column="eigenvector_centrality", id_column="node_id", top_n=top_n
            )

        return df.sort_values(
            "eigenvector_centrality" if not standardize else "value", ascending=False
        )

    def compute_closeness_centrality(
        self, standardize: bool = True, top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compute closeness centrality.

        Args:
            standardize: If True, add ranks and flags
            top_n: Number of top nodes to flag

        Returns:
            DataFrame with node_id, value, rank, flags
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

        if standardize:
            df = standardize_metric_output(
                df, value_column="closeness_centrality", id_column="node_id", top_n=top_n
            )

        return df.sort_values(
            "closeness_centrality" if not standardize else "value", ascending=False
        )

    def compute_pagerank(
        self, damping: float = 0.85, standardize: bool = True, top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compute PageRank centrality.

        Args:
            damping: Damping factor
            standardize: If True, add ranks and flags
            top_n: Number of top nodes to flag

        Returns:
            DataFrame with node_id, value, rank, flags
        """
        if self.graph.number_of_nodes() == 0:
            return pd.DataFrame(columns=["node_id", "pagerank"])

        pagerank = nx.pagerank(
            self.graph, weight="weight" if nx.is_weighted(self.graph) else None, alpha=damping
        )

        df = pd.DataFrame({"node_id": list(pagerank.keys()), "pagerank": list(pagerank.values())})

        if standardize:
            df = standardize_metric_output(
                df, value_column="pagerank", id_column="node_id", top_n=top_n
            )

        return df.sort_values("pagerank" if not standardize else "value", ascending=False)

    def compute_hits(
        self,
        max_iter: int = 100,
        normalized: bool = True,
        standardize: bool = True,
        top_n: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute HITS (Hyperlink-Induced Topic Search) scores (from Enron project).

        HITS identifies:
        - Authorities: Nodes with high in-degree (many links to them)
        - Hubs: Nodes with high out-degree (links to many authorities)

        Args:
            max_iter: Maximum iterations
            normalized: Whether to normalize scores
            standardize: If True, add ranks and flags
            top_n: Number of top nodes to flag

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
        )

        hub_df = pd.DataFrame(
            {"node_id": list(hub_scores.keys()), "hub_score": list(hub_scores.values())}
        )

        if standardize:
            authority_df = standardize_metric_output(
                authority_df, value_column="authority_score", id_column="node_id", top_n=top_n
            )
            hub_df = standardize_metric_output(
                hub_df, value_column="hub_score", id_column="node_id", top_n=top_n
            )

        authority_df = authority_df.sort_values(
            "authority_score" if not standardize else "value", ascending=False
        )
        hub_df = hub_df.sort_values("hub_score" if not standardize else "value", ascending=False)

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
