"""Tests for metrics modules."""

from orgnet.metrics.centrality import CentralityAnalyzer
from orgnet.metrics.community import CommunityDetector
from orgnet.metrics.structural import StructuralAnalyzer


def test_centrality_analyzer(sample_graph):
    """Test centrality calculations."""
    analyzer = CentralityAnalyzer(sample_graph)

    # Test degree centrality
    degree_df = analyzer.compute_degree_centrality()
    assert not degree_df.empty
    assert "node_id" in degree_df.columns
    assert "degree_weighted" in degree_df.columns

    # Test betweenness centrality
    betweenness_df = analyzer.compute_betweenness_centrality()
    assert not betweenness_df.empty
    assert "betweenness_centrality" in betweenness_df.columns

    # Test get_top_central_nodes with dictionary dispatch
    top_nodes = analyzer.get_top_central_nodes(metric="degree", top_n=2)
    assert len(top_nodes) <= 2
    assert "node_id" in top_nodes.columns


def test_community_detection(sample_graph):
    """Test community detection."""
    detector = CommunityDetector(sample_graph)

    # Test Louvain method (should work with dictionary dispatch)
    result = detector.detect_communities(method="louvain")
    assert "method" in result
    assert "modularity" in result
    assert "node_to_community" in result or "communities" in result


def test_structural_analyzer(sample_graph):
    """Test structural analysis."""
    analyzer = StructuralAnalyzer(sample_graph)

    # Test constraint calculation
    constraint_df = analyzer.compute_constraint()
    assert not constraint_df.empty
    assert "node_id" in constraint_df.columns
    assert "constraint" in constraint_df.columns

    # Test core-periphery (should use threshold-based classification)
    cp_df = analyzer.compute_core_periphery()
    assert not cp_df.empty
    assert "coreness" in cp_df.columns
    assert "core_periphery_class" in cp_df.columns
    assert all(
        cls in ["core", "semi-periphery", "periphery"] for cls in cp_df["core_periphery_class"]
    )
