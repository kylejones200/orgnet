"""Link prediction for organizational networks."""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class LinkPredictor:
    """Predicts missing or future links in organizational networks."""

    def __init__(self, graph: nx.Graph):
        """
        Initialize link predictor.

        Args:
            graph: NetworkX graph
        """
        self.graph = graph
        self.model = None

    def compute_heuristic_scores(self, node1: str, node2: str) -> Dict[str, float]:
        """
        Compute heuristic link prediction scores.

        Args:
            node1: First node
            node2: Second node

        Returns:
            Dictionary with various heuristic scores
        """
        scores = {}

        # Common neighbors
        neighbors1 = set(self.graph.neighbors(node1)) if node1 in self.graph else set()
        neighbors2 = set(self.graph.neighbors(node2)) if node2 in self.graph else set()
        common_neighbors = neighbors1 & neighbors2
        scores["common_neighbors"] = len(common_neighbors)

        # Jaccard coefficient
        union_neighbors = neighbors1 | neighbors2
        if len(union_neighbors) > 0:
            scores["jaccard"] = len(common_neighbors) / len(union_neighbors)
        else:
            scores["jaccard"] = 0.0

        # Adamic-Adar
        adamic_adar = 0.0
        for neighbor in common_neighbors:
            degree = self.graph.degree(neighbor)
            if degree > 1:
                adamic_adar += 1.0 / np.log(degree)
        scores["adamic_adar"] = adamic_adar

        # Preferential attachment
        degree1 = self.graph.degree(node1) if node1 in self.graph else 0
        degree2 = self.graph.degree(node2) if node2 in self.graph else 0
        scores["preferential_attachment"] = degree1 * degree2

        # Resource allocation
        resource_allocation = 0.0
        for neighbor in common_neighbors:
            degree = self.graph.degree(neighbor)
            if degree > 0:
                resource_allocation += 1.0 / degree
        scores["resource_allocation"] = resource_allocation

        return scores

    def predict_links(
        self, candidates: Optional[List[Tuple[str, str]]] = None, top_k: int = 10
    ) -> pd.DataFrame:
        """
        Predict likely links using heuristic scores.

        Args:
            candidates: List of candidate pairs to evaluate (if None, evaluates all non-edges)
            top_k: Number of top predictions to return

        Returns:
            DataFrame with predictions
        """
        if candidates is None:
            # Generate all non-edges
            nodes = list(self.graph.nodes())
            candidates = []
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i + 1 :]:
                    if not self.graph.has_edge(node1, node2):
                        candidates.append((node1, node2))

        predictions = []

        for node1, node2 in candidates:
            scores = self.compute_heuristic_scores(node1, node2)

            # Combined score (weighted average)
            combined_score = (
                0.3 * scores["common_neighbors"]
                + 0.3 * scores["jaccard"]
                + 0.2 * scores["adamic_adar"]
                + 0.2 * scores["preferential_attachment"]
            )

            predictions.append(
                {
                    "node1": node1,
                    "node2": node2,
                    "predicted_score": combined_score,
                    "common_neighbors": scores["common_neighbors"],
                    "jaccard": scores["jaccard"],
                    "adamic_adar": scores["adamic_adar"],
                    "preferential_attachment": scores["preferential_attachment"],
                }
            )

        df = pd.DataFrame(predictions)
        df = df.sort_values("predicted_score", ascending=False)

        return df.head(top_k)

    def train_ml_model(
        self, embeddings: Optional[np.ndarray] = None, node_to_index: Optional[dict] = None
    ):
        """
        Train ML model for link prediction (using embeddings if available).

        Args:
            embeddings: Node embeddings matrix
            node_to_index: Mapping from node_id to embedding index
        """
        # Generate positive and negative examples
        positive_pairs = list(self.graph.edges())
        nodes = list(self.graph.nodes())

        # Generate negative examples (non-edges)
        negative_pairs = []
        for edge in positive_pairs[: len(positive_pairs)]:  # Same number as positive
            node1, node2 = edge
            # Find a random node that's not connected to node1
            candidates = [n for n in nodes if n != node1 and not self.graph.has_edge(node1, n)]
            if candidates:
                negative_pairs.append((node1, np.random.choice(candidates)))

        # Create features
        X = []
        y = []

        for node1, node2 in positive_pairs + negative_pairs:
            features = self._extract_features(node1, node2, embeddings, node_to_index)
            X.append(features)
            y.append(
                1 if (node1, node2) in positive_pairs or (node2, node1) in positive_pairs else 0
            )

        X = np.array(X)
        y = np.array(y)

        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Return accuracy
        accuracy = self.model.score(X_test, y_test)
        return accuracy

    def _extract_features(
        self,
        node1: str,
        node2: str,
        embeddings: Optional[np.ndarray] = None,
        node_to_index: Optional[dict] = None,
    ) -> np.ndarray:
        """Extract features for link prediction."""
        features = []

        # Heuristic scores
        scores = self.compute_heuristic_scores(node1, node2)
        features.extend(
            [
                scores["common_neighbors"],
                scores["jaccard"],
                scores["adamic_adar"],
                scores["preferential_attachment"],
                scores["resource_allocation"],
            ]
        )

        # Node degrees
        degree1 = self.graph.degree(node1) if node1 in self.graph else 0
        degree2 = self.graph.degree(node2) if node2 in self.graph else 0
        features.extend([degree1, degree2])

        # Embedding features (if available)
        if embeddings is not None and node_to_index is not None:
            if node1 in node_to_index and node2 in node_to_index:
                idx1 = node_to_index[node1]
                idx2 = node_to_index[node2]
                emb1 = embeddings[idx1]
                emb2 = embeddings[idx2]

                # Concatenate, element-wise product, and difference
                features.extend(emb1)
                features.extend(emb2)
                features.extend(emb1 * emb2)
                features.extend(np.abs(emb1 - emb2))

        return np.array(features)

    def predict_with_model(
        self,
        node1: str,
        node2: str,
        embeddings: Optional[np.ndarray] = None,
        node_to_index: Optional[dict] = None,
    ) -> float:
        """
        Predict link probability using trained model.

        Args:
            node1: First node
            node2: Second node
            embeddings: Node embeddings
            node_to_index: Node to index mapping

        Returns:
            Predicted probability
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_ml_model() first.")

        features = self._extract_features(node1, node2, embeddings, node_to_index)
        proba = self.model.predict_proba([features])[0]
        return proba[1]  # Probability of positive class
