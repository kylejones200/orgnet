"""Community detection algorithms."""

import networkx as nx
import pandas as pd
from typing import Dict, Optional

try:
    import igraph as ig

    HAS_IGRAPH = True
except ImportError:
    HAS_IGRAPH = False

try:
    from networkx.algorithms import community as nx_community
except ImportError:
    nx_community = None

from orgnet.utils.performance import ParquetCache


class CommunityDetector:
    """Detects communities in organizational networks."""

    def __init__(self, graph: nx.Graph, cache: Optional[ParquetCache] = None):
        """
        Initialize community detector.

        Args:
            graph: NetworkX graph
            cache: Optional cache for expensive computations
        """
        self.graph = graph
        self.cache = cache

    def detect_communities(
        self, method: str = "louvain", resolution: float = 1.0, random_state: Optional[int] = None
    ) -> Dict:
        """
        Detect communities using specified method.

        Args:
            method: Detection method ('louvain', 'infomap', 'label_propagation', 'sbm')
            resolution: Resolution parameter (for Louvain)
            random_state: Random seed

        Returns:
            Dictionary with community assignments and metrics
        """
        cache_key = None
        if self.cache:
            import hashlib

            graph_hash = hashlib.md5(str(sorted(self.graph.nodes())).encode()).hexdigest()
            cache_key = f"communities_{method}_{resolution}_{random_state}_{graph_hash}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                try:
                    if isinstance(cached, pd.DataFrame) and len(cached) > 0:
                        result_dict = cached.iloc[0].to_dict()
                        if "communities" in result_dict and isinstance(
                            result_dict["communities"], str
                        ):
                            import ast

                            result_dict["communities"] = ast.literal_eval(
                                result_dict["communities"]
                            )
                        return result_dict
                except Exception:
                    pass

        method_map = {
            "louvain": lambda: self._louvain(resolution),
            "infomap": self._infomap,
            "label_propagation": self._label_propagation,
            "sbm": self._sbm,
        }

        detector = method_map.get(method)
        if detector is None:
            raise ValueError(f"Unknown method: {method}. Choose from {list(method_map.keys())}")

        result = detector()

        if self.cache and cache_key:
            try:
                result_copy = result.copy()
                if "communities" in result_copy:
                    result_copy["communities"] = str(result_copy["communities"])
                result_df = pd.DataFrame([result_copy])
                self.cache.set(cache_key, result_df)
            except Exception:
                pass

        return result

    def _louvain(self, resolution: float = 1.0) -> Dict:
        """
        Louvain community detection.

        Args:
            resolution: Resolution parameter

        Returns:
            Dictionary with communities and metrics
        """
        if nx_community is None:
            raise ImportError("networkx community module not available")

        communities = nx_community.louvain_communities(
            self.graph, weight="weight", resolution=resolution, seed=42
        )

        # Create node to community mapping
        node_to_community = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_to_community[node] = i

        # Compute modularity
        modularity = nx_community.modularity(self.graph, communities, weight="weight")

        return {
            "method": "louvain",
            "communities": communities,
            "node_to_community": node_to_community,
            "num_communities": len(communities),
            "modularity": modularity,
            "resolution": resolution,
        }

    def _infomap(self) -> Dict:
        """
        Infomap community detection (requires igraph).

        Returns:
            Dictionary with communities and metrics
        """
        if not HAS_IGRAPH:
            raise ImportError("igraph not available. Install with: pip install python-igraph")

        # Convert to igraph
        ig_graph = self._nx_to_igraph()

        # Run Infomap
        communities_ig = ig_graph.community_infomap(edge_weights="weight")

        # Convert back to NetworkX format
        communities = []
        node_to_community = {}

        for i, membership in enumerate(communities_ig.membership):
            node_id = ig_graph.vs[i]["name"]
            comm_id = membership

            if comm_id >= len(communities):
                communities.extend([set()] * (comm_id - len(communities) + 1))

            communities[comm_id].add(node_id)
            node_to_community[node_id] = comm_id

        # Compute modularity
        modularity = communities_ig.modularity

        return {
            "method": "infomap",
            "communities": communities,
            "node_to_community": node_to_community,
            "num_communities": len(communities),
            "modularity": modularity,
        }

    def _label_propagation(self) -> Dict:
        """
        Label propagation community detection.

        Returns:
            Dictionary with communities and metrics
        """
        if nx_community is None:
            raise ImportError("networkx community module not available")

        communities = nx_community.asyn_lpa_communities(self.graph, weight="weight", seed=42)

        communities = list(communities)

        # Create node to community mapping
        node_to_community = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_to_community[node] = i

        # Compute modularity
        modularity = nx_community.modularity(self.graph, communities, weight="weight")

        return {
            "method": "label_propagation",
            "communities": communities,
            "node_to_community": node_to_community,
            "num_communities": len(communities),
            "modularity": modularity,
        }

    def _sbm(self) -> Dict:
        """
        Stochastic Block Model (simplified version).
        Uses Louvain as approximation.

        Returns:
            Dictionary with communities and metrics
        """
        # For a full SBM implementation, would need graph-tool or similar
        # Using Louvain as approximation
        return self._louvain(resolution=1.0)

    def _nx_to_igraph(self):
        """Convert NetworkX graph to igraph."""
        if not HAS_IGRAPH:
            raise ImportError("igraph not available")

        # Create igraph graph
        ig_graph = ig.Graph()

        # Add nodes
        nodes = list(self.graph.nodes())
        ig_graph.add_vertices(len(nodes))
        for i, node in enumerate(nodes):
            ig_graph.vs[i]["name"] = node

        # Add edges
        edges = []
        weights = []
        node_to_index = {node: i for i, node in enumerate(nodes)}

        for u, v, data in self.graph.edges(data=True):
            edges.append((node_to_index[u], node_to_index[v]))
            weights.append(data.get("weight", 1.0))

        ig_graph.add_edges(edges)
        ig_graph.es["weight"] = weights

        return ig_graph

    def compare_with_formal_structure(self, formal_communities: Dict[str, str]) -> Dict:
        """
        Compare detected communities with formal organizational structure.

        Args:
            formal_communities: Dictionary mapping node_id to formal department/team

        Returns:
            Dictionary with comparison metrics (NMI, ARI, alignment analysis)
        """
        # Get detected communities
        result = self.detect_communities()
        detected = result["node_to_community"]

        # Compute Normalized Mutual Information (NMI)
        nmi = self._compute_nmi(detected, formal_communities)

        # Compute Adjusted Rand Index (ARI)
        ari = self._compute_ari(detected, formal_communities)

        # Alignment analysis
        alignment = self._analyze_alignment(detected, formal_communities)

        return {
            "nmi": nmi,
            "ari": ari,
            "alignment_analysis": alignment,
            "detected_communities": result["num_communities"],
            "formal_communities": len(set(formal_communities.values())),
        }

    def _compute_nmi(self, detected: Dict, formal: Dict) -> float:
        """Compute Normalized Mutual Information."""
        from sklearn.metrics import normalized_mutual_info_score

        nodes = set(detected.keys()) & set(formal.keys())
        if len(nodes) == 0:
            return 0.0

        detected_labels = [detected[n] for n in nodes]
        formal_labels = [formal[n] for n in nodes]

        return normalized_mutual_info_score(detected_labels, formal_labels)

    def _compute_ari(self, detected: Dict, formal: Dict) -> float:
        """Compute Adjusted Rand Index."""
        from sklearn.metrics import adjusted_rand_score

        nodes = set(detected.keys()) & set(formal.keys())
        if len(nodes) == 0:
            return 0.0

        detected_labels = [detected[n] for n in nodes]
        formal_labels = [formal[n] for n in nodes]

        return adjusted_rand_score(detected_labels, formal_labels)

    def _analyze_alignment(self, detected: Dict, formal: Dict) -> Dict:
        """Analyze alignment between detected and formal communities."""
        # Group nodes by detected community
        detected_groups = {}
        for node, comm_id in detected.items():
            if comm_id not in detected_groups:
                detected_groups[comm_id] = []
            detected_groups[comm_id].append(node)

        # For each detected community, see formal structure
        alignment = {}
        for comm_id, nodes in detected_groups.items():
            formal_depts = {}
            for node in nodes:
                if node in formal:
                    dept = formal[node]
                    formal_depts[dept] = formal_depts.get(dept, 0) + 1

            if formal_depts:
                dominant_dept = max(formal_depts.items(), key=lambda x: x[1])
                alignment[comm_id] = {
                    "dominant_department": dominant_dept[0],
                    "department_purity": dominant_dept[1] / len(nodes),
                    "departments_represented": len(formal_depts),
                }

        return alignment
