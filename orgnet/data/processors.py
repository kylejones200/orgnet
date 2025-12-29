"""Data processing utilities."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from collections import defaultdict

from orgnet.data.models import Interaction, Meeting, Document, CodeCommit


class DataProcessor:
    """Processes raw data into analysis-ready formats."""

    @staticmethod
    def compute_interaction_matrix(interactions: List[Interaction]) -> pd.DataFrame:
        """
        Compute interaction frequency matrix.

        Args:
            interactions: List of Interaction objects

        Returns:
            DataFrame with interaction counts (source_id x target_id)
        """
        interaction_dict = defaultdict(int)

        for interaction in interactions:
            key = (interaction.source_id, interaction.target_id)
            interaction_dict[key] += 1

        # Create DataFrame
        all_ids = set()
        for source, target in interaction_dict.keys():
            # Filter out None and NaN values
            if source is not None and not (isinstance(source, float) and pd.isna(source)):
                all_ids.add(source)
            if target is not None and not (isinstance(target, float) and pd.isna(target)):
                all_ids.add(target)

        # Sort IDs, handling mixed types by converting to strings for sorting
        sorted_ids = sorted(all_ids, key=lambda x: str(x))
        matrix = pd.DataFrame(0, index=sorted_ids, columns=sorted_ids)

        for (source, target), count in interaction_dict.items():
            matrix.loc[source, target] = count

        return matrix

    @staticmethod
    def compute_meeting_coattendance(meetings: List[Meeting]) -> pd.DataFrame:
        """
        Compute meeting co-attendance matrix.

        Args:
            meetings: List of Meeting objects

        Returns:
            DataFrame with co-attendance counts
        """
        coattendance = defaultdict(int)

        for meeting in meetings:
            attendees = meeting.attendee_ids
            # Count pairs
            for i, person1 in enumerate(attendees):
                for person2 in attendees[i + 1 :]:
                    pair = tuple(sorted([person1, person2]))
                    coattendance[pair] += 1

        # Create DataFrame
        all_ids = set()
        for p1, p2 in coattendance.keys():
            all_ids.add(p1)
            all_ids.add(p2)

        matrix = pd.DataFrame(0, index=sorted(all_ids), columns=sorted(all_ids))

        for (p1, p2), count in coattendance.items():
            matrix.loc[p1, p2] = count
            matrix.loc[p2, p1] = count  # Symmetric

        return matrix

    @staticmethod
    def compute_document_collaboration(documents: List[Document]) -> pd.DataFrame:
        """
        Compute document collaboration matrix.

        Args:
            documents: List of Document objects

        Returns:
            DataFrame with collaboration counts
        """
        collaboration = defaultdict(float)

        for doc in documents:
            # Combine authors and editors
            all_contributors = set(doc.author_ids + doc.editor_ids)
            n_contributors = len(all_contributors)

            if n_contributors < 2:
                continue

            # Weight by 1/n_contributors (fewer authors = stronger signal)
            weight = 1.0 / n_contributors

            contributors_list = list(all_contributors)
            for i, person1 in enumerate(contributors_list):
                for person2 in contributors_list[i + 1 :]:
                    pair = tuple(sorted([person1, person2]))
                    collaboration[pair] += weight

        # Create DataFrame using vectorized approach
        all_ids = sorted(set().union(*(set(pair) for pair in collaboration.keys())))
        matrix = pd.DataFrame(0.0, index=all_ids, columns=all_ids)

        # Vectorized assignment
        for (p1, p2), weight in collaboration.items():
            matrix.loc[p1, p2] = weight
            matrix.loc[p2, p1] = weight  # Symmetric

        return matrix

    @staticmethod
    def compute_code_collaboration(commits: List[CodeCommit]) -> pd.DataFrame:
        """
        Compute code collaboration matrix.

        Args:
            commits: List of CodeCommit objects

        Returns:
            DataFrame with code collaboration counts
        """
        collaboration = defaultdict(float)

        # Group commits by file
        file_to_committers = defaultdict(set)
        review_relationships = defaultdict(int)

        for commit in commits:
            # Track file-level collaboration
            for file_path in commit.file_paths:
                file_to_committers[file_path].add(commit.author_id)

            # Track review relationships (higher weight)
            for reviewer_id in commit.reviewer_ids:
                pair = tuple(sorted([commit.author_id, reviewer_id]))
                review_relationships[pair] += 2  # Reviews are higher signal

        # Count file co-occurrences
        for file_path, committers in file_to_committers.items():
            committers_list = list(committers)
            for i, person1 in enumerate(committers_list):
                for person2 in committers_list[i + 1 :]:
                    pair = tuple(sorted([person1, person2]))
                    collaboration[pair] += 1

        # Add review relationships
        for pair, count in review_relationships.items():
            collaboration[pair] += count

        # Create DataFrame using vectorized approach
        all_ids = sorted(set().union(*(set(pair) for pair in collaboration.keys())))
        matrix = pd.DataFrame(0.0, index=all_ids, columns=all_ids)

        # Vectorized assignment
        for (p1, p2), weight in collaboration.items():
            matrix.loc[p1, p2] = weight
            matrix.loc[p2, p1] = weight  # Symmetric

        return matrix

    @staticmethod
    def compute_response_times(interactions: List[Interaction]) -> Dict[Tuple[str, str], float]:
        """
        Compute median response times between pairs.

        Args:
            interactions: List of Interaction objects

        Returns:
            Dictionary mapping (source_id, target_id) -> median response time in seconds
        """
        response_times = defaultdict(list)

        for interaction in interactions:
            if interaction.response_time_seconds is not None:
                key = (interaction.source_id, interaction.target_id)
                response_times[key].append(interaction.response_time_seconds)

        # Compute medians
        medians = {}
        for key, times in response_times.items():
            medians[key] = np.median(times)

        return medians

    @staticmethod
    def compute_reciprocity(interactions: List[Interaction]) -> Dict[Tuple[str, str], float]:
        """
        Compute reciprocity scores for pairs.

        Args:
            interactions: List of Interaction objects

        Returns:
            Dictionary mapping (person1, person2) -> reciprocity score [0, 1]
        """
        out_counts = defaultdict(int)
        in_counts = defaultdict(int)

        for interaction in interactions:
            out_counts[(interaction.source_id, interaction.target_id)] += 1
            in_counts[(interaction.target_id, interaction.source_id)] += 1

        reciprocity = {}
        all_pairs = set(list(out_counts.keys()) + list(in_counts.keys()))

        for pair in all_pairs:
            out_ij = out_counts.get(pair, 0)
            in_ij = in_counts.get(pair, 0)

            if max(out_ij, in_ij) == 0:
                reciprocity[pair] = 0.0
            else:
                reciprocity[pair] = min(out_ij, in_ij) / max(out_ij, in_ij)

        return reciprocity

    @staticmethod
    def filter_recent_interactions(
        interactions: List[Interaction], days: int = 90
    ) -> List[Interaction]:
        """
        Filter interactions to recent time window.

        Args:
            interactions: List of Interaction objects
            days: Number of days to look back

        Returns:
            Filtered list of interactions
        """
        cutoff = datetime.now() - timedelta(days=days)
        return [i for i in interactions if i.timestamp >= cutoff]
