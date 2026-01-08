"""Benchmark scripts for ML models.

This module provides benchmark scripts for evaluating flagship ML models
on synthetic or public datasets, reporting accuracy, ROC AUC, and interpretability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Tuple, Optional

from orgnet.utils.logging import get_logger

try:
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from orgnet.ml.classification_wrapper import SimpleGCNClassifier, Node2VecEmbedder

    HAS_ML = True
except ImportError:
    HAS_ML = False

logger = get_logger(__name__)


def generate_synthetic_org_network(
    num_people: int = 100,
    num_edges: int = 500,
    num_roles: int = 5,
    seed: int = 42,
) -> Tuple[nx.Graph, pd.DataFrame, Dict[str, int]]:
    """
    Generate synthetic organizational network for benchmarking.

    Args:
        num_people: Number of people in organization
        num_edges: Number of edges in graph
        num_roles: Number of different roles
        seed: Random seed

    Returns:
        Tuple of (graph, node_features_df, labels_dict)
    """
    np.random.seed(seed)

    # Create graph using preferential attachment
    graph = nx.barabasi_albert_graph(num_people, num_edges // num_people, seed=seed)

    # Create node features (role, department, tenure)
    node_data = []
    labels = {}

    for i, node in enumerate(graph.nodes()):
        role = f"role_{i % num_roles}"
        department = f"dept_{i % 3}"
        tenure = np.random.randint(0, 1000)  # Days
        job_level = np.random.choice(["IC", "Manager", "Director"], p=[0.6, 0.3, 0.1])

        node_data.append(
            {
                "node_id": node,
                "role": role,
                "department": department,
                "tenure_days": tenure,
                "job_level": job_level,
            }
        )

        # Create labels (e.g., attrition risk: 0 = low, 1 = high)
        # Higher risk for isolated nodes or low tenure
        degree = graph.degree(node)
        risk = 1 if (degree < 3 or tenure < 100) else 0
        labels[node] = risk

    features_df = pd.DataFrame(node_data)

    # Create feature matrix (one-hot encoded)
    feature_matrix = np.zeros((num_people, num_roles + 3 + 3))  # roles + depts + job_levels

    for i, row in features_df.iterrows():
        # Role encoding
        role_idx = int(row["role"].split("_")[1]) % num_roles
        feature_matrix[i, role_idx] = 1.0

        # Department encoding
        dept_idx = int(row["department"].split("_")[1]) % 3
        feature_matrix[i, num_roles + dept_idx] = 1.0

        # Job level encoding
        job_level_map = {"IC": 0, "Manager": 1, "Director": 2}
        job_idx = job_level_map.get(row["job_level"], 0)
        feature_matrix[i, num_roles + 3 + job_idx] = 1.0

        # Add tenure (normalized)
        feature_matrix[i, -1] = row["tenure_days"] / 1000.0

    return graph, feature_matrix, labels


def benchmark_node2vec(
    graph: nx.Graph,
    labels: Dict[str, int],
    dimensions: int = 64,
    num_walks: int = 200,
    walk_length: int = 30,
) -> Dict:
    """
    Benchmark Node2Vec embeddings.

    Args:
        graph: NetworkX graph
        labels: Dictionary mapping node_id to label
        dimensions: Embedding dimensions
        num_walks: Number of walks per node
        walk_length: Length of random walks

    Returns:
        Dictionary with benchmark results
    """
    if not HAS_ML or not HAS_SKLEARN:
        return {"error": "ML dependencies not available"}

    try:
        # Fit embeddings
        embedder = Node2VecEmbedder(graph)
        embeddings = embedder.fit(
            dimensions=dimensions, num_walks=num_walks, walk_length=walk_length
        )

        # Use embeddings for simple classification (k-NN style)
        from sklearn.neighbors import KNeighborsClassifier

        nodes = list(graph.nodes())
        X = embeddings
        y = np.array([labels[node] for node in nodes])

        # Train/test split
        X_train, X_test, y_train, y_test, nodes_train, nodes_test = train_test_split(
            X, y, nodes, test_size=0.2, random_state=42
        )

        # Train classifier
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            roc_auc = 0.0

        # Find example high-risk nodes
        high_risk_nodes = []
        for i, node in enumerate(nodes_test):
            if y_pred_proba[i] > 0.7:
                high_risk_nodes.append((node, float(y_pred_proba[i])))

        return {
            "model": "Node2Vec",
            "accuracy": float(accuracy),
            "roc_auc": float(roc_auc),
            "embedding_dim": dimensions,
            "num_walks": num_walks,
            "walk_length": walk_length,
            "high_risk_examples": high_risk_nodes[:5],
        }

    except Exception as e:
        logger.error(f"Node2Vec benchmark failed: {e}")
        return {"error": str(e)}


def benchmark_gcn_classifier(
    graph: nx.Graph,
    node_features: np.ndarray,
    labels: Dict[str, int],
    hidden_dim: int = 64,
    epochs: int = 100,
) -> Dict:
    """
    Benchmark GCN classifier.

    Args:
        graph: NetworkX graph
        node_features: Node feature matrix [num_nodes, num_features]
        labels: Dictionary mapping node_id to label
        hidden_dim: Hidden dimension
        epochs: Training epochs

    Returns:
        Dictionary with benchmark results
    """
    if not HAS_ML or not HAS_SKLEARN:
        return {"error": "ML dependencies not available"}

    try:
        # Prepare data
        nodes = list(graph.nodes())

        # Train/test split (respecting graph structure)
        train_size = int(0.8 * len(nodes))
        train_nodes = nodes[:train_size]
        test_nodes = nodes[train_size:]

        # Fit model
        classifier = SimpleGCNClassifier(hidden_dim=hidden_dim, num_classes=2)
        classifier.fit(graph, node_features, labels, train_nodes=train_nodes, epochs=epochs)

        # Predict
        predictions = classifier.predict(graph, node_features)
        proba = classifier.predict_proba(graph, node_features)

        # Get test predictions
        y_test = np.array([labels[node] for node in test_nodes])
        y_pred = np.array([predictions[node] for node in test_nodes])
        y_pred_proba = proba[[nodes.index(n) for n in test_nodes], 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            roc_auc = 0.0

        # Feature importance (using GCN embeddings)
        # Find high-risk nodes
        high_risk_nodes = []
        for i, node in enumerate(test_nodes):
            if y_pred_proba[i] > 0.7:
                high_risk_nodes.append((node, float(y_pred_proba[i])))

        return {
            "model": "SimpleGCN",
            "accuracy": float(accuracy),
            "roc_auc": float(roc_auc),
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "high_risk_examples": high_risk_nodes[:5],
        }

    except Exception as e:
        logger.error(f"GCN benchmark failed: {e}")
        return {"error": str(e)}


def run_all_benchmarks(
    num_people: int = 100,
    num_edges: int = 500,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run all ML benchmarks and return results.

    Args:
        num_people: Number of people in synthetic network
        num_edges: Number of edges
        output_path: Optional path to save results CSV

    Returns:
        DataFrame with benchmark results
    """
    logger.info("Generating synthetic organizational network...")
    graph, node_features, labels = generate_synthetic_org_network(
        num_people=num_people, num_edges=num_edges
    )

    results = []

    # Benchmark Node2Vec
    logger.info("Benchmarking Node2Vec...")
    node2vec_results = benchmark_node2vec(graph, labels)
    if "error" not in node2vec_results:
        results.append(node2vec_results)

    # Benchmark GCN
    logger.info("Benchmarking GCN Classifier...")
    gcn_results = benchmark_gcn_classifier(graph, node_features, labels)
    if "error" not in gcn_results:
        results.append(gcn_results)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    if output_path:
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

    return results_df


if __name__ == "__main__":
    # Run benchmarks
    results = run_all_benchmarks(num_people=100, num_edges=500, output_path="ml_benchmarks.csv")
    print("\nBenchmark Results:")
    print(results.to_string())
