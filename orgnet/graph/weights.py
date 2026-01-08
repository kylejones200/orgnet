"""Edge weight calculation for organizational graphs."""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, List

from orgnet.data.models import Interaction, Meeting, Document, CodeCommit
from orgnet.data.processors import DataProcessor


class EdgeWeightCalculator:
    """Calculates edge weights for organizational graphs."""

    def __init__(
        self,
        weight_scheme: str = "composite",
        recency_decay_lambda: float = 0.1,
        reciprocity_weight: float = 0.5,
        responsiveness_weight: float = 0.3,
        role_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize edge weight calculator.

        Args:
            weight_scheme: Weight scheme to use ("pure_count", "time_decay", "role_based", "composite")
            recency_decay_lambda: Decay rate for recency weighting
            reciprocity_weight: Weight for reciprocity adjustment
            responsiveness_weight: Weight for responsiveness adjustment
            role_weights: Dictionary mapping role names to weights (for role_based scheme)
        """
        self.weight_scheme = weight_scheme
        self.recency_decay_lambda = recency_decay_lambda
        self.reciprocity_weight = reciprocity_weight
        self.responsiveness_weight = responsiveness_weight
        self.role_weights = role_weights or {}
        self.processor = DataProcessor()

    def compute_communication_weights(
        self,
        interactions: list[Interaction],
        now: Optional[datetime] = None,
        people: Optional[List] = None,
    ) -> pd.DataFrame:
        """
        Compute communication edge weights using the configured weight scheme.

        Args:
            interactions: List of Interaction objects
            now: Current time (defaults to now)
            people: List of Person objects (for role_based scheme)

        Returns:
            DataFrame with weighted communication matrix
        """
        if now is None:
            now = datetime.now()

        # Basic frequency matrix
        freq_matrix = self.processor.compute_interaction_matrix(interactions)

        # Apply weight scheme
        if self.weight_scheme == "pure_count":
            return freq_matrix
        elif self.weight_scheme == "time_decay":
            return self._apply_recency_decay(interactions, freq_matrix, now)
        elif self.weight_scheme == "role_based":
            return self._apply_role_based_weights(interactions, freq_matrix, people)
        else:  # composite (default)
            return self._compute_composite_weights(interactions, freq_matrix, now)

    def _compute_composite_weights(
        self, interactions: list[Interaction], freq_matrix: pd.DataFrame, now: datetime
    ) -> pd.DataFrame:
        """Compute composite weights with recency, reciprocity, and responsiveness."""
        # Apply recency decay
        recency_matrix = self._apply_recency_decay(interactions, freq_matrix, now)

        # Get reciprocity scores
        reciprocity = self.processor.compute_reciprocity(interactions)

        # Get response times
        response_times = self.processor.compute_response_times(interactions)

        # Apply adjustments
        weighted_matrix = recency_matrix.copy()

        for i in weighted_matrix.index:
            for j in weighted_matrix.columns:
                if i == j:
                    continue

                pair = (i, j)
                reverse_pair = (j, i)

                # Reciprocity adjustment
                recip_score = reciprocity.get(pair, 0.0)
                if pair not in reciprocity and reverse_pair in reciprocity:
                    recip_score = reciprocity[reverse_pair]

                recip_adjustment = 0.5 + 0.5 * recip_score
                weighted_matrix.loc[i, j] *= recip_adjustment

                # Responsiveness adjustment
                resp_time = response_times.get(pair) or response_times.get(reverse_pair)
                if resp_time is not None:
                    responsiveness = 1 / (1 + resp_time / 3600)  # Convert to hours
                    weighted_matrix.loc[i, j] *= 0.7 + 0.3 * responsiveness

        return weighted_matrix

    def _apply_role_based_weights(
        self,
        interactions: list[Interaction],
        freq_matrix: pd.DataFrame,
        people: Optional[List] = None,
    ) -> pd.DataFrame:
        """Apply role-based weights to interactions."""
        if people is None:
            return freq_matrix

        # Create role lookup
        role_lookup = {}
        for person in people:
            role_lookup[person.id] = person.role or "default"

        # Apply role weights
        weighted_matrix = freq_matrix.copy()
        for i in weighted_matrix.index:
            for j in weighted_matrix.columns:
                if i == j:
                    continue

                role_i = role_lookup.get(i, "default")
                role_j = role_lookup.get(j, "default")

                weight_i = self.role_weights.get(
                    role_i.lower(), self.role_weights.get("default", 1.0)
                )
                weight_j = self.role_weights.get(
                    role_j.lower(), self.role_weights.get("default", 1.0)
                )

                # Average role weights
                role_weight = (weight_i + weight_j) / 2.0
                weighted_matrix.loc[i, j] *= role_weight

        return weighted_matrix

    def _apply_recency_decay(
        self, interactions: list[Interaction], freq_matrix: pd.DataFrame, now: datetime
    ) -> pd.DataFrame:
        """Apply exponential recency decay to interactions."""
        decay_matrix = pd.DataFrame(0.0, index=freq_matrix.index, columns=freq_matrix.columns)

        for interaction in interactions:
            time_diff = (now - interaction.timestamp).total_seconds() / 86400  # Days
            decay = np.exp(-self.recency_decay_lambda * time_diff)

            i = interaction.source_id
            j = interaction.target_id
            if i in decay_matrix.index and j in decay_matrix.columns:
                decay_matrix.loc[i, j] += decay

        return decay_matrix

    def compute_meeting_weights(self, meetings: list[Meeting]) -> pd.DataFrame:
        """
        Compute meeting co-attendance weights.

        Args:
            meetings: List of Meeting objects

        Returns:
            DataFrame with weighted meeting matrix
        """
        coattendance = self.processor.compute_meeting_coattendance(meetings)
        weighted = coattendance.copy().astype(float)

        # Weight by meeting size and duration
        for meeting in meetings:
            if len(meeting.attendee_ids) < 2:
                continue

            # Smaller meetings = stronger signal
            size_weight = 1.0 / np.sqrt(len(meeting.attendee_ids))
            duration_weight = meeting.duration_minutes / 60.0  # Normalize to hours

            weight = size_weight * duration_weight

            attendees = meeting.attendee_ids
            for i, person1 in enumerate(attendees):
                for person2 in attendees[i + 1 :]:
                    if person1 in weighted.index and person2 in weighted.columns:
                        weighted.loc[person1, person2] += weight
                        weighted.loc[person2, person1] += weight

        return weighted

    def compute_document_weights(self, documents: list[Document]) -> pd.DataFrame:
        """
        Compute document collaboration weights.

        Args:
            documents: List of Document objects

        Returns:
            DataFrame with weighted document collaboration matrix
        """
        return self.processor.compute_document_collaboration(documents)

    def compute_code_weights(self, commits: list[CodeCommit]) -> pd.DataFrame:
        """
        Compute code collaboration weights.

        Args:
            commits: List of CodeCommit objects

        Returns:
            DataFrame with weighted code collaboration matrix
        """
        return self.processor.compute_code_collaboration(commits)

    def fuse_layers(
        self, layers: Dict[str, pd.DataFrame], weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Fuse multiple graph layers into a single weighted graph.

        Args:
            layers: Dictionary mapping layer name to weight matrix
            weights: Dictionary mapping layer name to fusion weight
                    (defaults to equal weights)

        Returns:
            Fused weight matrix
        """
        if not layers:
            raise ValueError("At least one layer must be provided")

        if weights is None:
            # Equal weights
            weight = 1.0 / len(layers)
            weights = {name: weight for name in layers.keys()}

        # Normalize each layer first
        normalized_layers = {}
        for name, matrix in layers.items():
            if matrix.sum().sum() > 0:
                # Z-score normalization within layer
                mean = matrix.values[matrix.values > 0].mean()
                std = matrix.values[matrix.values > 0].std()
                if std > 0:
                    normalized = (matrix - mean) / std
                    normalized = normalized.clip(lower=0)  # No negative weights
                    normalized_layers[name] = normalized
                else:
                    normalized_layers[name] = matrix
            else:
                normalized_layers[name] = matrix

        # Get all node IDs (vectorized)
        all_nodes = sorted(
            set().union(
                *(set(matrix.index) | set(matrix.columns) for matrix in normalized_layers.values())
            )
        )

        # Create fused matrix
        fused = pd.DataFrame(0.0, index=all_nodes, columns=all_nodes)

        # Vectorized fusion using pandas operations
        for name, matrix in normalized_layers.items():
            layer_weight = weights.get(name, 0.0)
            # Align and add weighted matrix (vectorized operation)
            matrix_aligned = matrix.reindex(index=all_nodes, columns=all_nodes, fill_value=0.0)
            fused += layer_weight * matrix_aligned

        return fused
