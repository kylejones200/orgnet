"""Pytest configuration and fixtures."""

import pytest
import networkx as nx
import pandas as pd
from datetime import datetime, timedelta

from onapy.data.models import Person, Interaction, InteractionType


@pytest.fixture
def sample_people():
    """Create sample Person objects for testing."""
    return [
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
    ]


@pytest.fixture
def sample_interactions():
    """Create sample Interaction objects for testing."""
    base_time = datetime.now() - timedelta(days=1)
    return [
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
    ]


@pytest.fixture
def sample_graph(sample_people, sample_interactions):
    """Create a sample graph for testing."""
    G = nx.Graph()

    # Add nodes
    for person in sample_people:
        G.add_node(
            person.id, **{"name": person.name, "department": person.department, "role": person.role}
        )

    # Add edges with weights
    edge_weights = {}
    for interaction in sample_interactions:
        pair = tuple(sorted([interaction.source_id, interaction.target_id]))
        edge_weights[pair] = edge_weights.get(pair, 0) + 1.0

    for (u, v), weight in edge_weights.items():
        G.add_edge(u, v, weight=weight)

    return G
