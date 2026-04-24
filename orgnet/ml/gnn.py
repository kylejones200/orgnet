"""Graph Neural Networks for organizational networks."""

from __future__ import annotations

import networkx as nx
from typing import Dict, Literal, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, SAGEConv

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Create dummy classes for type hints when torch isn't available
    nn = None
    torch = None
    F = None
    GCNConv = None
    GATConv = None
    GATv2Conv = None
    SAGEConv = None
    Data = None

GNNArchitecture = Literal["gcn", "gat", "graphsage", "gatv2"]


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

    class OrgGraphSAGE(nn.Module):
        """GraphSAGE-style model (mean aggregation) for organizational networks."""

        def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            normalize: bool = True,
        ):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean", normalize=normalize)
            self.conv2 = SAGEConv(hidden_channels, out_channels, aggr="mean", normalize=normalize)

        def forward(self, x, edge_index, edge_weight=None):
            # SAGEConv supports edge_weight on recent PyG; ignore if caller passes None only
            if edge_weight is not None:
                x = self.conv1(x, edge_index, edge_weight)
            else:
                x = self.conv1(x, edge_index)
            x = F.relu(x)
            if edge_weight is not None:
                x = self.conv2(x, edge_index, edge_weight)
            else:
                x = self.conv2(x, edge_index)
            return x

    class OrgGATv2(nn.Module):
        """Graph Attention Network v2 (dynamic attention) for organizational networks."""

        def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            heads: int = 4,
        ):
            super().__init__()
            self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=0.1)
            self.conv2 = GATv2Conv(
                hidden_channels * heads,
                out_channels,
                heads=1,
                concat=False,
                dropout=0.1,
            )

        def forward(self, x, edge_index, edge_weight=None):
            ea = edge_weight
            if ea is not None and ea.dim() == 1:
                ea = ea.unsqueeze(-1)
            x = self.conv1(x, edge_index, edge_attr=ea)
            x = F.relu(x)
            x = self.conv2(x, edge_index, edge_attr=ea)
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

    class OrgGraphSAGE:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch and PyTorch Geometric required. Install with: pip install torch torch-geometric"
            )

    class OrgGATv2:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch and PyTorch Geometric required. Install with: pip install torch torch-geometric"
            )


def build_org_gnn(
    architecture: GNNArchitecture,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    *,
    heads: int = 4,
    sage_normalize: bool = True,
) -> "nn.Module":
    """
    Construct a graph neural network by name.

    Args:
        architecture: One of ``gcn``, ``gat``, ``graphsage``, ``gatv2``.
        in_channels: Input feature size per node.
        hidden_channels: Hidden width.
        out_channels: Output embedding size per node.
        heads: Attention heads for GAT / GATv2.
        sage_normalize: L2-normalize after GraphSAGE layers (PyG ``normalize`` flag).

    Returns:
        A ``torch.nn.Module`` accepting ``(x, edge_index, edge_weight=None)``.
    """
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch and PyTorch Geometric required. Install with: pip install torch torch-geometric"
        )
    arch = architecture.lower()
    if arch == "gcn":
        return OrgGCN(in_channels, hidden_channels, out_channels)
    if arch == "gat":
        return OrgGAT(in_channels, hidden_channels, out_channels, heads=heads)
    if arch == "graphsage":
        return OrgGraphSAGE(in_channels, hidden_channels, out_channels, normalize=sage_normalize)
    if arch == "gatv2":
        return OrgGATv2(in_channels, hidden_channels, out_channels, heads=heads)
    raise ValueError(f"Unknown architecture: {architecture!r}. Use gcn, gat, graphsage, gatv2.")


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
