"""Tests for core analyzer functionality."""

import pytest
import tempfile
import os

from onapy.core import OrganizationalNetworkAnalyzer
from onapy.config import Config


def test_analyzer_initialization():
    """Test analyzer can be initialized."""
    analyzer = OrganizationalNetworkAnalyzer()
    assert analyzer is not None
    assert analyzer.graph is None
    assert analyzer.people == []


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
