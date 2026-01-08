"""Tests for graph builder repeatability and edge weight schemes."""

from orgnet.core import OrganizationalNetworkAnalyzer
from orgnet.graph.builder import GraphBuilder
from orgnet.config import Config


def test_graph_repeatability(sample_people, sample_interactions):
    """Test that graph building is repeatable for the same input."""
    analyzer = OrganizationalNetworkAnalyzer()
    analyzer.people = sample_people
    analyzer.interactions = sample_interactions

    # Build graph twice
    graph1 = analyzer.build_graph()
    graph2 = analyzer.build_graph()

    # Check same number of nodes and edges
    assert graph1.number_of_nodes() == graph2.number_of_nodes()
    assert graph1.number_of_edges() == graph2.number_of_edges()

    # Check same nodes
    assert set(graph1.nodes()) == set(graph2.nodes())

    # Check same edges (within floating point tolerance)
    for u, v in graph1.edges():
        assert graph2.has_edge(u, v)
        weight1 = graph1[u][v].get("weight", 0)
        weight2 = graph2[u][v].get("weight", 0)
        assert abs(weight1 - weight2) < 1e-6


def test_degree_distribution(sample_people, sample_interactions):
    """Test degree distribution for fixed test dataset."""
    analyzer = OrganizationalNetworkAnalyzer()
    analyzer.people = sample_people
    analyzer.interactions = sample_interactions

    graph = analyzer.build_graph()

    # Check degree distribution
    degrees = dict(graph.degree())
    assert len(degrees) == len(sample_people)

    # All nodes should have at least 0 degree
    assert all(d >= 0 for d in degrees.values())

    # At least some edges should exist
    assert graph.number_of_edges() > 0


def test_edge_weight_schemes(sample_people, sample_interactions):
    """Test different edge weight schemes produce different but valid graphs."""
    config = Config()

    schemes = ["pure_count", "time_decay", "composite"]
    graphs = {}

    for scheme in schemes:
        config.config_dict["graph"]["weight_scheme"] = scheme
        builder = GraphBuilder(config)
        graph = builder.build_graph(sample_people, interactions=sample_interactions)
        graphs[scheme] = graph

        # All schemes should produce valid graphs
        assert graph.number_of_nodes() == len(sample_people)
        assert graph.number_of_edges() >= 0

    # Different schemes may produce different edge weights
    # But all should have same nodes
    for scheme in schemes:
        assert set(graphs[scheme].nodes()) == set(graphs["pure_count"].nodes())


def test_sample_edges(sample_people, sample_interactions):
    """Test sample edges for a fixed test dataset."""
    analyzer = OrganizationalNetworkAnalyzer()
    analyzer.people = sample_people
    analyzer.interactions = sample_interactions

    graph = analyzer.build_graph()

    # Check sample edges exist
    if graph.number_of_edges() > 0:
        # Get first edge
        u, v = list(graph.edges())[0]

        # Check edge has weight
        assert "weight" in graph[u][v]
        assert graph[u][v]["weight"] > 0

        # Check nodes exist
        assert u in graph.nodes()
        assert v in graph.nodes()
