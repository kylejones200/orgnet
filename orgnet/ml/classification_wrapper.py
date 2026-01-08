"""Clean wrapper for ML classification tasks.

This module provides flagship ML models with simple fit/predict interfaces
for organizational network classification tasks (e.g., role prediction, attrition).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from orgnet.utils.logging import get_logger

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Create dummy classes to avoid NameError when parsing the file
    nn = None
    torch = None
    optim = None
    Data = None
    GCNConv = None

try:
    from node2vec import Node2Vec

    HAS_NODE2VEC = True
except ImportError:
    HAS_NODE2VEC = False

logger = get_logger(__name__)


class SimpleGCNClassifier:
    """Simple GCN for node classification tasks (flagship model)."""

    def __init__(self, hidden_dim: int = 64, num_classes: int = 2):
        """
        Initialize simple GCN classifier.

        Args:
            hidden_dim: Hidden dimension
            num_classes: Number of classes
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required. Install with: pip install torch torch-geometric")

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.model: Optional[nn.Module] = None
        self.node_to_index: Optional[Dict] = None
        self.label_encoder: Optional[Dict] = None

    def _graph_to_pyg_data(self, graph: nx.Graph, node_features: np.ndarray) -> Data:
        """Convert NetworkX graph to PyTorch Geometric Data."""
        from torch_geometric.utils import from_networkx

        # Add node features
        for i, node in enumerate(graph.nodes()):
            graph.nodes[node]["features"] = node_features[i]

        data = from_networkx(graph)
        data.x = torch.FloatTensor(node_features)

        return data

    def fit(
        self,
        graph: nx.Graph,
        node_features: np.ndarray,
        labels: Dict[str, int],
        train_nodes: Optional[List[str]] = None,
        epochs: int = 100,
        lr: float = 0.01,
    ):
        """
        Fit GCN classifier.

        Args:
            graph: NetworkX graph
            node_features: Node feature matrix [num_nodes, num_features]
            labels: Dictionary mapping node_id to class label
            train_nodes: List of node IDs to use for training (if None, uses all)
            epochs: Number of training epochs
            lr: Learning rate
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required")

        nodes = list(graph.nodes())
        self.node_to_index = {node: i for i, node in enumerate(nodes)}

        # Encode labels
        unique_labels = sorted(set(labels.values()))
        self.label_encoder = {label: i for i, label in enumerate(unique_labels)}
        num_classes = len(unique_labels)

        # Create model
        num_features = node_features.shape[1]
        self.model = SimpleGCN(num_features, self.hidden_dim, num_classes)

        # Prepare data
        data = self._graph_to_pyg_data(graph, node_features)
        y = torch.zeros(len(nodes), dtype=torch.long)
        for node_id, label in labels.items():
            if node_id in self.node_to_index:
                idx = self.node_to_index[node_id]
                y[idx] = self.label_encoder[label]

        # Train/test split
        if train_nodes is None:
            train_nodes = nodes

        train_mask = torch.zeros(len(nodes), dtype=torch.bool)
        for node_id in train_nodes:
            if node_id in self.node_to_index:
                train_mask[self.node_to_index[node_id]] = True

        # Training
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            loss = criterion(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

        logger.info("GCN training completed")

    def predict(self, graph: nx.Graph, node_features: np.ndarray) -> Dict[str, int]:
        """
        Predict labels for nodes.

        Args:
            graph: NetworkX graph
            node_features: Node feature matrix

        Returns:
            Dictionary mapping node_id to predicted class
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        self.model.eval()
        data = self._graph_to_pyg_data(graph, node_features)

        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            predictions = out.argmax(dim=1)

        # Decode labels
        reverse_encoder = {v: k for k, v in self.label_encoder.items()}
        node_predictions = {}
        for i, node_id in enumerate(graph.nodes()):
            node_predictions[node_id] = reverse_encoder[predictions[i].item()]

        return node_predictions

    def predict_proba(self, graph: nx.Graph, node_features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            graph: NetworkX graph
            node_features: Node feature matrix

        Returns:
            Probability matrix [num_nodes, num_classes]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        self.model.eval()
        data = self._graph_to_pyg_data(graph, node_features)

        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            probs = torch.softmax(out, dim=1)

        return probs.numpy()


if HAS_TORCH:

    class SimpleGCN(nn.Module):
        """Simple 2-layer GCN for classification."""

        def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
            super(SimpleGCN, self).__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = torch.relu(x)
            x = self.conv2(x, edge_index)
            return x


class Node2VecEmbedder:
    """Node2Vec embedder with clean fit interface (flagship model)."""

    def __init__(self, graph: nx.Graph):
        """
        Initialize Node2Vec embedder.

        Args:
            graph: NetworkX graph
        """
        if not HAS_NODE2VEC:
            raise ImportError("node2vec not available. Install with: pip install node2vec")

        self.graph = graph
        self.model = None
        self.embeddings: Optional[np.ndarray] = None
        self.node_to_index: Optional[Dict] = None

    def fit(
        self,
        dimensions: int = 64,
        walk_length: int = 30,
        num_walks: int = 200,
        p: float = 1.0,
        q: float = 2.0,
        window: int = 10,
        workers: int = 4,
    ) -> np.ndarray:
        """
        Fit Node2Vec embeddings.

        Args:
            dimensions: Embedding dimensions
            walk_length: Length of random walks
            num_walks: Number of walks per node
            p: Return parameter
            q: In-out parameter
            window: Context window size
            workers: Number of workers

        Returns:
            Embedding matrix [num_nodes, dimensions]
        """
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

        self.model = node2vec.fit(window=window, min_count=1, batch_words=4)

        # Extract embeddings
        nodes = list(self.graph.nodes())
        self.node_to_index = {node: i for i, node in enumerate(nodes)}

        embeddings = np.zeros((len(nodes), dimensions))
        for i, node in enumerate(nodes):
            embeddings[i] = self.model.wv[node]

        self.embeddings = embeddings
        logger.info(f"Fitted Node2Vec embeddings: {embeddings.shape}")

        return embeddings

    def get_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific node."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if node_id in self.model.wv:
            return self.model.wv[node_id]
        return None

    def find_similar_nodes(self, node_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find most similar nodes.

        Args:
            node_id: Query node
            top_k: Number of similar nodes to return

        Returns:
            List of (node_id, similarity) tuples
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if node_id not in self.model.wv:
            return []

        similar = self.model.wv.most_similar(node_id, topn=top_k)
        return [(node, score) for node, score in similar]
