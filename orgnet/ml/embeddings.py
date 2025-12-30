"""Node embedding methods (Node2Vec, etc.)."""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from orgnet.utils.logging import get_logger

try:
    from node2vec import Node2Vec

    HAS_NODE2VEC = True
except ImportError:
    HAS_NODE2VEC = False

logger = get_logger(__name__)


class NodeEmbedder:
    """Generates node embeddings for organizational networks."""

    def __init__(self, graph: nx.Graph):
        """
        Initialize node embedder.

        Args:
            graph: NetworkX graph
        """
        self.graph = graph
        self.embeddings: Optional[np.ndarray] = None
        self.node_to_index: Optional[Dict] = None

    def fit_node2vec(
        self,
        dimensions: int = 64,
        walk_length: int = 30,
        num_walks: int = 200,
        p: float = 1.0,
        q: float = 2.0,
        window: int = 10,
        min_count: int = 1,
        workers: int = 4,
    ) -> np.ndarray:
        """
        Fit Node2Vec embeddings.

        Args:
            dimensions: Embedding dimensions
            walk_length: Length of random walks
            num_walks: Number of walks per node
            p: Return parameter (BFS-like if low)
            q: In-out parameter (DFS-like if low)
            window: Context window size
            min_count: Minimum word count
            workers: Number of workers

        Returns:
            Embedding matrix
        """
        if not HAS_NODE2VEC:
            raise ImportError("node2vec not available. Install with: pip install node2vec")

        # Create Node2Vec model
        node2vec = Node2Vec(
            self.graph,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            p=p,
            q=q,
            workers=workers,
            weight_key="weight",
        )

        # Fit model
        model = node2vec.fit(window=window, min_count=min_count, batch_words=4)

        # Extract embeddings using vectorized approach
        nodes = list(self.graph.nodes())
        self.node_to_index = {node: i for i, node in enumerate(nodes)}

        logger.info(f"Extracting Node2Vec embeddings for {len(nodes)} nodes")

        embeddings = np.zeros((len(nodes), dimensions))
        for i, node in enumerate(nodes):
            embeddings[i] = model.wv[node]

        self.embeddings = embeddings

        return embeddings

    def get_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """
        Get embedding for a specific node.

        Args:
            node_id: Node identifier

        Returns:
            Embedding vector or None if not found
        """
        if self.embeddings is None or self.node_to_index is None:
            return None

        if node_id not in self.node_to_index:
            return None

        idx = self.node_to_index[node_id]
        return self.embeddings[idx]

    def find_similar_nodes(self, node_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find most similar nodes based on embeddings.

        Args:
            node_id: Node identifier
            top_k: Number of similar nodes to return

        Returns:
            List of (node_id, similarity_score) tuples
        """
        if self.embeddings is None or self.node_to_index is None:
            return []

        if node_id not in self.node_to_index:
            return []

        node_embedding = self.get_embedding(node_id)
        if node_embedding is None:
            return []

        # Vectorized cosine similarity computation
        other_nodes = [n for n in self.node_to_index.keys() if n != node_id]
        other_indices = [self.node_to_index[n] for n in other_nodes]
        other_embeddings = self.embeddings[other_indices]

        from orgnet.utils.performance import NUMBA_AVAILABLE, cosine_similarity_batch

        if NUMBA_AVAILABLE and len(other_embeddings) > 100:
            similarities_vec = cosine_similarity_batch(other_embeddings, node_embedding)
        else:
            node_norm = np.linalg.norm(node_embedding)
            other_norms = np.linalg.norm(other_embeddings, axis=1)
            similarities_vec = np.dot(other_embeddings, node_embedding) / (
                node_norm * other_norms + 1e-10
            )

        # Get top k using argpartition for efficiency
        top_k_indices = np.argpartition(similarities_vec, -top_k)[-top_k:]
        top_k_indices = top_k_indices[np.argsort(similarities_vec[top_k_indices])[::-1]]

        return [(other_nodes[i], float(similarities_vec[i])) for i in top_k_indices]

    def get_embeddings_dataframe(self) -> pd.DataFrame:
        """
        Get embeddings as DataFrame.

        Returns:
            DataFrame with node_id and embedding columns
        """
        if self.embeddings is None or self.node_to_index is None:
            return pd.DataFrame()

        index_to_node = {v: k for k, v in self.node_to_index.items()}

        data = {"node_id": [index_to_node[i] for i in range(len(self.embeddings))]}

        for dim in range(self.embeddings.shape[1]):
            data[f"dim_{dim}"] = self.embeddings[:, dim]

        return pd.DataFrame(data)
