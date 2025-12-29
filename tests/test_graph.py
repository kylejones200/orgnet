"""Tests for graph building and weights."""

import networkx as nx
import pandas as pd

from orgnet.graph.builder import GraphBuilder
from orgnet.graph.weights import EdgeWeightCalculator
from orgnet.config import Config


def test_graph_builder(sample_people, sample_interactions):
    """Test graph building."""
    config = Config()
    builder = GraphBuilder(config)

    graph = builder.build_graph(people=sample_people, interactions=sample_interactions)

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == len(sample_people)
    assert graph.number_of_edges() > 0


def test_edge_weight_calculator(sample_interactions):
    """Test edge weight calculations."""
    config = Config()
    calculator = EdgeWeightCalculator(config)

    # Test layer fusion (vectorized operation) - simpler test that doesn't require complex config
    layers = {
        "layer1": pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["p1", "p2"], columns=["p1", "p2"]),
        "layer2": pd.DataFrame([[0.8, 0.3], [0.3, 0.8]], index=["p1", "p2"], columns=["p1", "p2"]),
    }

    fused = calculator.fuse_layers(layers)
    assert isinstance(fused, pd.DataFrame)
    assert not fused.empty

    # Test layer fusion (vectorized operation)
    layers = {
        "layer1": pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["p1", "p2"], columns=["p1", "p2"]),
        "layer2": pd.DataFrame([[0.8, 0.3], [0.3, 0.8]], index=["p1", "p2"], columns=["p1", "p2"]),
    }

    fused = calculator.fuse_layers(layers)
    assert isinstance(fused, pd.DataFrame)
    assert not fused.empty
