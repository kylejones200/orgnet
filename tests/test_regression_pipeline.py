"""Regression test for full pipeline output.

This test compares full pipeline output for a tiny example to a stored baseline,
ensuring that changes don't break the end-to-end workflow.
"""

import pytest
import os
import json
import hashlib
import pandas as pd
from datetime import datetime, timedelta

from orgnet.core import OrganizationalNetworkAnalyzer
from orgnet.data.models import Person, Interaction, InteractionType


@pytest.fixture
def tiny_org_data():
    """Create tiny organization data for regression testing."""
    base_time = datetime(2024, 1, 1)

    people = [
        Person(
            id="p1",
            name="Alice",
            email="alice@example.com",
            department="Engineering",
            role="Engineer",
            job_level="Individual Contributor",
            tenure_days=365,
        ),
        Person(
            id="p2",
            name="Bob",
            email="bob@example.com",
            department="Engineering",
            role="Manager",
            job_level="Manager",
            tenure_days=730,
        ),
        Person(
            id="p3",
            name="Charlie",
            email="charlie@example.com",
            department="Product",
            role="Product Manager",
            job_level="Individual Contributor",
            tenure_days=180,
        ),
        Person(
            id="p4",
            name="Diana",
            email="diana@example.com",
            department="Product",
            role="Designer",
            job_level="Individual Contributor",
            tenure_days=90,
        ),
    ]

    interactions = [
        Interaction(
            id="i1",
            source_id="p1",
            target_id="p2",
            interaction_type=InteractionType.EMAIL,
            timestamp=base_time,
            channel="email",
        ),
        Interaction(
            id="i2",
            source_id="p1",
            target_id="p3",
            interaction_type=InteractionType.EMAIL,
            timestamp=base_time + timedelta(hours=1),
            channel="email",
        ),
        Interaction(
            id="i3",
            source_id="p2",
            target_id="p3",
            interaction_type=InteractionType.EMAIL,
            timestamp=base_time + timedelta(hours=2),
            channel="email",
        ),
        Interaction(
            id="i4",
            source_id="p3",
            target_id="p4",
            interaction_type=InteractionType.EMAIL,
            timestamp=base_time + timedelta(hours=3),
            channel="email",
        ),
    ]

    return people, interactions


def compute_pipeline_hash(results: dict) -> str:
    """Compute hash of pipeline results for comparison."""
    # Create deterministic representation
    summary = {
        "num_nodes": results.get("graph", {}).get("num_nodes", 0),
        "num_edges": results.get("graph", {}).get("num_edges", 0),
        "num_communities": results.get("communities", {}).get("num_communities", 0),
        "modularity": round(results.get("communities", {}).get("modularity", 0), 4),
        "top_betweenness": (
            [
                {
                    "node_id": row.get("node_id"),
                    "value": round(row.get("value", row.get("betweenness_centrality", 0)), 4),
                }
                for _, row in results.get("centrality", {})
                .get("betweenness", pd.DataFrame())
                .head(3)
                .iterrows()
            ]
            if "betweenness" in results.get("centrality", {})
            else []
        ),
    }

    summary_str = json.dumps(summary, sort_keys=True)
    return hashlib.md5(summary_str.encode()).hexdigest()


def test_regression_pipeline(tiny_org_data):
    """Test that full pipeline produces consistent results."""
    people, interactions = tiny_org_data

    analyzer = OrganizationalNetworkAnalyzer()
    analyzer.people = people
    analyzer.interactions = interactions

    # Run full pipeline
    graph = analyzer.build_graph()
    results = analyzer.analyze()

    # Create results summary
    pipeline_results = {
        "graph": {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
        },
        "communities": results.get("communities", {}),
        "centrality": results.get("centrality", {}),
    }

    # Compute hash
    result_hash = compute_pipeline_hash(pipeline_results)

    # Check basic invariants
    assert graph.number_of_nodes() == len(people)
    assert graph.number_of_edges() >= 0
    assert "communities" in results
    assert "centrality" in results

    # Store baseline hash (first run) or compare to stored
    baseline_path = "tests/baseline_pipeline_hash.txt"
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            baseline_hash = f.read().strip()
        assert result_hash == baseline_hash, "Pipeline output changed from baseline"
    else:
        # Create baseline
        os.makedirs("tests", exist_ok=True)
        with open(baseline_path, "w") as f:
            f.write(result_hash)
        pytest.skip("Baseline created - run test again to verify")


def test_pipeline_metrics_structure(tiny_org_data):
    """Test that pipeline produces expected metric structures."""
    people, interactions = tiny_org_data

    analyzer = OrganizationalNetworkAnalyzer()
    analyzer.people = people
    analyzer.interactions = interactions

    graph = analyzer.build_graph()
    results = analyzer.analyze()

    # Check graph structure
    assert graph.number_of_nodes() == len(people)
    assert all(node in graph.nodes() for person in people for node in [person.id])

    # Check communities structure
    communities = results.get("communities", {})
    assert "num_communities" in communities
    assert "modularity" in communities

    # Check centrality structure
    centrality = results.get("centrality", {})
    if "betweenness" in centrality:
        betweenness_df = centrality["betweenness"]
        assert "node_id" in betweenness_df.columns
        assert (
            "value" in betweenness_df.columns or "betweenness_centrality" in betweenness_df.columns
        )


def test_report_generation_consistency(tiny_org_data):
    """Test that report generation is consistent."""
    people, interactions = tiny_org_data

    analyzer = OrganizationalNetworkAnalyzer()
    analyzer.people = people
    analyzer.interactions = interactions

    analyzer.build_graph()
    analyzer.analyze()

    report_path = "test_regression_report.html"
    try:
        path1 = analyzer.generate_report(report_path)
        assert os.path.exists(path1)

        # Generate again
        path2 = analyzer.generate_report(report_path + ".2")
        assert os.path.exists(path2)

        # Both reports should exist and have content
        with open(path1, "r") as f:
            content1 = f.read()
        with open(path2, "r") as f:
            content2 = f.read()

        # Both should have required sections
        required_sections = ["Network", "Centrality", "Community"]
        for section in required_sections:
            assert section in content1 or section.lower() in content1.lower()
            assert section in content2 or section.lower() in content2.lower()

    finally:
        for path in [report_path, report_path + ".2"]:
            if os.path.exists(path):
                os.remove(path)
