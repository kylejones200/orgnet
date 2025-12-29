"""Graph Neural Networks for organizational networks."""

import networkx as nx
from typing import Optional, Dict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, GATConv
    from torch_geometric.data import Data

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Create dummy classes for type hints when torch isn't available
    nn = None
    torch = None
    F = None
    GCNConv = None
    GATConv = None
    Data = None


if HAS_TORCH:

    class OrgGCN(nn.Module):
        """Graph Convolutional Network for organizational networks."""

        def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
            """
            Initialize GCN.

            Args:
                in_channels: Input feature dimension
                hidden_channels: Hidden layer dimension
                out_channels: Output embedding dimension
            """
            if not HAS_TORCH:
                raise ImportError(
                    "PyTorch and PyTorch Geometric required. Install with: pip install torch torch-geometric"
                )

            super(OrgGCN, self).__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)

        def forward(self, x, edge_index, edge_weight=None):
            """
            Forward pass.

            Args:
                x: Node features [num_nodes, in_channels]
                edge_index: Edge connectivity [2, num_edges]
                edge_weight: Edge weights [num_edges]

            Returns:
                Node embeddings [num_nodes, out_channels]
            """
            x = self.conv1(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.conv2(x, edge_index, edge_weight)
            return x

    class OrgGAT(nn.Module):
        """Graph Attention Network for organizational networks."""

        def __init__(
            self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 4
        ):
            """
            Initialize GAT.

            Args:
                in_channels: Input feature dimension
                hidden_channels: Hidden layer dimension
                out_channels: Output embedding dimension
                heads: Number of attention heads
            """
            if not HAS_TORCH:
                raise ImportError(
                    "PyTorch and PyTorch Geometric required. Install with: pip install torch torch-geometric"
                )

            super(OrgGAT, self).__init__()
            self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.1)
            self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, dropout=0.1)

        def forward(self, x, edge_index, edge_weight=None):
            """
            Forward pass.

            Args:
                x: Node features [num_nodes, in_channels]
                edge_index: Edge connectivity [2, num_edges]
                edge_weight: Edge weights [num_edges]

            Returns:
                Node embeddings [num_nodes, out_channels]
            """
            x = self.conv1(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.conv2(x, edge_index, edge_weight)
            return x

else:
    # Dummy classes when PyTorch is not available
    class OrgGCN:
        """Graph Convolutional Network for organizational networks (requires PyTorch)."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch and PyTorch Geometric required. Install with: pip install torch torch-geometric"
            )

    class OrgGAT:
        """Graph Attention Network for organizational networks (requires PyTorch)."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch and PyTorch Geometric required. Install with: pip install torch torch-geometric"
            )


def graph_to_pyg_data(graph: nx.Graph, node_features: Optional[Dict] = None):
    """
    Convert NetworkX graph to PyTorch Geometric Data object.

    Args:
        graph: NetworkX graph
        node_features: Dictionary mapping node_id to feature vector

    Returns:
        PyTorch Geometric Data object
    """
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch and PyTorch Geometric required. Install with: pip install torch torch-geometric"
        )

    # Create node mapping
    nodes = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Create edge index
    edge_list = []
    edge_weights = []
    for u, v, data in graph.edges(data=True):
        edge_list.append([node_to_idx[u], node_to_idx[v]])
        weight = data.get("weight", 1.0)
        edge_weights.append(weight)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)

    # Create node features
    if node_features is None:
        # Default: one-hot encoding of node attributes
        num_nodes = len(nodes)
        x = torch.eye(num_nodes)
    else:
        # Use provided features
        feature_dim = len(next(iter(node_features.values())))
        x = torch.zeros(len(nodes), feature_dim)
        for node, features in node_features.items():
            if node in node_to_idx:
                x[node_to_idx[node]] = torch.tensor(features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
