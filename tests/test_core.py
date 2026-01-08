"""Tests for core analyzer functionality."""

import os
from orgnet.core import OrganizationalNetworkAnalyzer


def test_analyzer_initialization():
    """Test analyzer can be initialized."""
    analyzer = OrganizationalNetworkAnalyzer()
    assert analyzer is not None
    assert analyzer.graph is None
    assert analyzer.people == []
    assert analyzer.privacy_manager is not None
    assert analyzer.audit_logger is not None


def test_analyzer_with_config():
    """Test analyzer with config file."""
    # Should work with default config
    analyzer = OrganizationalNetworkAnalyzer(config_path="config.yaml")
    assert analyzer is not None


def test_build_graph(sample_people, sample_interactions):
    """Test graph building through analyzer."""
    analyzer = OrganizationalNetworkAnalyzer()
    analyzer.people = sample_people
    analyzer.interactions = sample_interactions

    graph = analyzer.build_graph()
    assert graph is not None
    assert graph.number_of_nodes() == len(sample_people)
    assert graph.number_of_edges() > 0


def test_analyze(sample_people, sample_interactions):
    """Test full analysis workflow."""
    analyzer = OrganizationalNetworkAnalyzer()
    analyzer.people = sample_people
    analyzer.interactions = sample_interactions

    # Build graph first
    analyzer.build_graph()

    # Run analysis
    results = analyzer.analyze()

    assert "centrality" in results
    assert "communities" in results
    assert "constraint" in results or "brokers" in results

    # Check standardized outputs
    if "betweenness" in results["centrality"]:
        betweenness_df = results["centrality"]["betweenness"]
        assert "rank" in betweenness_df.columns or "value" in betweenness_df.columns


def test_generate_report(sample_people, sample_interactions):
    """Test report generation end-to-end."""
    analyzer = OrganizationalNetworkAnalyzer()
    analyzer.people = sample_people
    analyzer.interactions = sample_interactions

    analyzer.build_graph()
    analyzer.analyze()

    report_path = "test_report_output.html"
    try:
        path = analyzer.generate_report(report_path)
        assert os.path.exists(path)

        # Check report content
        with open(path, "r") as f:
            content = f.read()
            assert "Network" in content or "network" in content.lower()
            assert "Centrality" in content or "centrality" in content.lower()
    finally:
        if os.path.exists(report_path):
            os.remove(report_path)


def test_load_data():
    """Test load_data method structure."""
    analyzer = OrganizationalNetworkAnalyzer()

    # Test with empty data paths
    analyzer.load_data({})
    assert analyzer.people == []

    # Test structure (won't actually load files, but tests method exists)
    assert hasattr(analyzer, "load_data")


def test_standardized_metrics(sample_people, sample_interactions):
    """Test that metrics are standardized with ranks and flags."""
    analyzer = OrganizationalNetworkAnalyzer()
    analyzer.people = sample_people
    analyzer.interactions = sample_interactions

    analyzer.build_graph()
    results = analyzer.analyze(standardize_outputs=True)

    # Check centrality metrics have standardized format
    if "betweenness" in results["centrality"]:
        df = results["centrality"]["betweenness"]
        assert "node_id" in df.columns
        assert "value" in df.columns or "betweenness_centrality" in df.columns
        assert "rank" in df.columns
        assert "top_percentile_flag" in df.columns
