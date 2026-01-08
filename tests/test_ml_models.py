"""Tests for ML flagship models."""

import pytest
import networkx as nx
import numpy as np

try:
    from orgnet.ml.classification_wrapper import SimpleGCNClassifier, Node2VecEmbedder

    HAS_ML = True
except ImportError:
    HAS_ML = False


@pytest.mark.skipif(not HAS_ML, reason="ML dependencies not available")
def test_node2vec_embedder():
    """Test Node2Vec embedder."""
    # Create small test graph
    graph = nx.karate_club_graph()

    try:
        embedder = Node2VecEmbedder(graph)
        embeddings = embedder.fit(dimensions=32, num_walks=10, walk_length=10)

        assert embeddings.shape[0] == graph.number_of_nodes()
        assert embeddings.shape[1] == 32

        # Test getting embedding for a node
        node_id = list(graph.nodes())[0]
        emb = embedder.get_embedding(node_id)
        assert emb is not None
        assert len(emb) == 32

        # Test finding similar nodes
        similar = embedder.find_similar_nodes(node_id, top_k=5)
        assert len(similar) <= 5
    except ImportError:
        pytest.skip("node2vec not installed")


@pytest.mark.skipif(not HAS_ML, reason="ML dependencies not available")
def test_gcn_classifier():
    """Test GCN classifier."""
    # Create small test graph
    graph = nx.karate_club_graph()
    num_nodes = graph.number_of_nodes()

    # Create node features
    node_features = np.random.randn(num_nodes, 10)

    # Create labels
    labels = {node: (i % 2) for i, node in enumerate(graph.nodes())}

    try:
        classifier = SimpleGCNClassifier(hidden_dim=32, num_classes=2)

        # Fit model
        train_nodes = list(graph.nodes())[: num_nodes // 2]
        classifier.fit(graph, node_features, labels, train_nodes=train_nodes, epochs=10)

        # Predict
        predictions = classifier.predict(graph, node_features)
        assert len(predictions) == num_nodes

        # Predict probabilities
        proba = classifier.predict_proba(graph, node_features)
        assert proba.shape[0] == num_nodes
        assert proba.shape[1] == 2
    except ImportError:
        pytest.skip("PyTorch not installed")
