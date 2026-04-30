"""Learning velocity and bias for learning analysis.

This module measures learning behavior and knowledge growth velocity:
- Learning velocity (rate of skill/knowledge acquisition)
- Knowledge breadth growth (increase in diverse expertise over time)
- Curiosity score (exploratory questions, learning-seeking behavior)
- Teaching/sharing score (frequency of knowledge transfer to others)
"""

import re
from collections import defaultdict
from datetime import datetime, timedelta

import networkx as nx
import pandas as pd

from orgnet.utils.logging import get_logger

logger = get_logger(__name__)

# Learning-seeking keywords
LEARNING_KEYWORDS = [
    r"\b(learn|learning|study|research|explore|understand|investigate)\b",
    r"\b(how|why|what|when|where|which|explain|clarify)\b",
    r"\b(tutorial|guide|documentation|resource|example|reference)\b",
    r"\b(question|ask|wondering|curious|interested)\b",
]

# Teaching/sharing keywords
TEACHING_KEYWORDS = [
    r"\b(teach|teaching|share|sharing|explain|explaining|guide|guiding)\b",
    r"\b(help|helping|mentor|mentoring|tutor|tutoring)\b",
    r"\b(document|documentation|tutorial|guide|example|reference)\b",
    r"\b(show|demonstrate|walk through|walkthrough)\b",
]


class LearningVelocityAnalyzer:
    """Measures learning behavior and knowledge growth velocity.

    Tracks over time:
    - New topics engaged
    - New collaborators connected
    - Expertise breadth growth
    - Learning-seeking vs. teaching behavior
    """

    def __init__(
        self,
        graph: nx.Graph,
        documents: list[dict] | None = None,
        interactions: list[dict] | None = None,
        code_commits: list[dict] | None = None,
        expertise_engine: object | None = None,
        **params,
    ):
        """Initialize learning velocity analyzer.

        Args:
            graph: Organizational network graph
            documents: List of document dicts with 'author_id', 'text', 'timestamp'
            interactions: List of interaction dicts with 'sender_id', 'receiver_id',
                         'text', 'timestamp'
            code_commits: List of code commit dicts with 'author_id', 'repo',
                         'message', 'timestamp', 'files_changed'
            expertise_engine: Optional ExpertiseInferenceEngine for topic analysis
        """
        self.graph = graph
        self.documents = documents or []
        self.interactions = interactions or []
        self.code_commits = code_commits or []
        self.expertise_engine = expertise_engine

    def compute(
        self, graph: nx.Graph | None = None, nodes: pd.DataFrame | None = None, **kwargs
    ) -> pd.DataFrame:
        """Compute learning velocity metrics (current snapshot).

        Args:
            graph: Optional graph (uses self.graph if None)
            nodes: Optional node table (not used but required by BaseMetric)
            **kwargs: Additional arguments

        Returns:
            DataFrame with columns: person_id, learning_velocity, knowledge_breadth,
                curiosity_score, teaching_score, learning_behavior_score
        """
        if graph is None:
            graph = self.graph

        person_ids = list(graph.nodes())
        results = []

        for person_id in person_ids:
            # Curiosity score (learning-seeking behavior)
            curiosity = self._compute_curiosity_score(person_id)

            # Teaching/sharing score
            teaching = self._compute_teaching_score(person_id)

            # Knowledge breadth (diverse topics/expertise)
            knowledge_breadth = self._compute_knowledge_breadth(person_id)

            # Learning velocity (rate of new topics/collaborators)
            learning_velocity = self._compute_learning_velocity(person_id)

            # Combined learning behavior score
            learning_score = self._compute_combined_score(
                curiosity, teaching, knowledge_breadth, learning_velocity
            )

            results.append(
                {
                    "person_id": person_id,
                    "learning_velocity": learning_velocity,
                    "knowledge_breadth": knowledge_breadth,
                    "curiosity_score": curiosity,
                    "teaching_score": teaching,
                    "learning_behavior_score": learning_score,
                }
            )

        return pd.DataFrame(results)

    def compute_temporal(
        self,
        graph_snapshots: list[dict],
        documents: list[dict],
        time_windows: list[timedelta] | None = None,
    ) -> pd.DataFrame:
        """Compute learning velocity over time windows.

        Args:
            graph_snapshots: List of dicts with 'timestamp' and 'graph'
            documents: List of document dicts with 'author_id', 'text', 'timestamp'
            time_windows: Optional list of time windows (default: weekly)

        Returns:
            DataFrame with time series of learning metrics
        """
        if not graph_snapshots:
            return pd.DataFrame()

        # Sort snapshots by timestamp
        graph_snapshots = sorted(graph_snapshots, key=lambda x: x.get("timestamp", datetime.min))

        # Default time windows: weekly
        if time_windows is None:
            start_time = graph_snapshots[0]["timestamp"]
            end_time = graph_snapshots[-1]["timestamp"]
            time_windows = []
            current = start_time
            while current < end_time:
                time_windows.append(current)
                current += timedelta(days=7)

        results = []

        # Track topics and collaborators over time
        person_topics = defaultdict(set)
        person_collaborators = defaultdict(set)

        for window_start in time_windows:
            window_end = window_start + timedelta(days=7)

            # Get documents in this window
            window_docs = [
                d
                for d in documents
                if window_start <= d.get("timestamp", datetime.min) < window_end
            ]

            # Process each person
            for person_id in self.graph.nodes():
                # Get person's documents in this window
                person_docs = [d for d in window_docs if d.get("author_id") == person_id]

                # Extract topics (if expertise engine available)
                if self.expertise_engine and person_docs:
                    for doc in person_docs:
                        # Use expertise engine to infer topics
                        pass

                # Track new topics
                current_topics = person_topics[person_id]
                # Add new topics from this window (simplified - would use topic modeling)
                new_topics_count = 0  # Placeholder

                # Track new collaborators
                current_collaborators = person_collaborators[person_id]
                # Get collaborators from interactions in this window
                new_collaborators_count = 0  # Placeholder

                # Calculate learning velocity for this window
                learning_velocity = (new_topics_count + new_collaborators_count) / 7.0  # Per day

                results.append(
                    {
                        "person_id": person_id,
                        "timestamp": window_start,
                        "learning_velocity": learning_velocity,
                        "total_topics": len(current_topics),
                        "total_collaborators": len(current_collaborators),
                    }
                )

        return pd.DataFrame(results)

    def _compute_curiosity_score(self, person_id: str) -> float:
        """Compute curiosity score from learning-seeking behavior.

        Args:
            person_id: Person identifier

        Returns:
            Score between 0 and 1 (higher = more curious/learning-seeking)
        """
        # Get all text from this person
        texts = []

        for interaction in self.interactions:
            if interaction.get("sender_id") == person_id:
                texts.append(interaction.get("text", ""))

        for doc in self.documents:
            if doc.get("author_id") == person_id:
                texts.append(doc.get("text", ""))

        if not texts:
            return 0.0

        total_text = " ".join(texts).lower()

        # Count learning-seeking keywords
        learning_count = 0
        for pattern in LEARNING_KEYWORDS:
            matches = len(re.findall(pattern, total_text, re.IGNORECASE))
            learning_count += matches

        # Normalize by text volume
        text_length = len(total_text)
        if text_length == 0:
            return 0.0

        # Score: learning keywords per 1000 characters
        keywords_per_1k = (learning_count / text_length) * 1000
        score = min(keywords_per_1k / 5.0, 1.0)  # Cap at 5 keywords per 1k chars

        return min(score, 1.0)

    def _compute_teaching_score(self, person_id: str) -> float:
        """Compute teaching/sharing score from knowledge transfer behavior.

        Args:
            person_id: Person identifier

        Returns:
            Score between 0 and 1 (higher = more teaching/sharing)
        """
        # Get all text from this person
        texts = []

        for interaction in self.interactions:
            if interaction.get("sender_id") == person_id:
                texts.append(interaction.get("text", ""))

        for doc in self.documents:
            if doc.get("author_id") == person_id:
                texts.append(doc.get("text", ""))

        if not texts:
            return 0.0

        total_text = " ".join(texts).lower()

        # Count teaching/sharing keywords
        teaching_count = 0
        for pattern in TEACHING_KEYWORDS:
            matches = len(re.findall(pattern, total_text, re.IGNORECASE))
            teaching_count += matches

        # Normalize by text volume
        text_length = len(total_text)
        if text_length == 0:
            return 0.0

        # Score: teaching keywords per 1000 characters
        keywords_per_1k = (teaching_count / text_length) * 1000
        score = min(keywords_per_1k / 3.0, 1.0)  # Cap at 3 keywords per 1k chars

        # Also count document creation (knowledge sharing)
        doc_count = len([d for d in self.documents if d.get("author_id") == person_id])
        doc_score = min(doc_count / 10.0, 1.0)  # Cap at 10 documents

        # Combined score
        combined = score * 0.7 + doc_score * 0.3

        return min(combined, 1.0)

    def _compute_knowledge_breadth(self, person_id: str) -> float:
        """Compute knowledge breadth (diversity of topics/expertise).

        Args:
            person_id: Person identifier

        Returns:
            Score between 0 and 1 (higher = broader knowledge)
        """
        # Get unique topics from documents and interactions
        topics = set()

        # Extract topics from documents (simplified - would use topic modeling)
        for doc in self.documents:
            if doc.get("author_id") == person_id:
                # Extract topics from text (simplified)
                text = doc.get("text", "").lower()
                # Would use expertise_engine or topic modeler here
                pass

        # Get unique repositories (different technologies/domains)
        repos = set()
        for commit in self.code_commits:
            if commit.get("author_id") == person_id:
                repo = commit.get("repo")
                if repo:
                    repos.add(repo)

        # Get unique collaborators (diverse connections)
        collaborators = set()
        for interaction in self.interactions:
            if interaction.get("sender_id") == person_id:
                receiver_id = interaction.get("receiver_id")
                if receiver_id:
                    collaborators.add(receiver_id)

        # Calculate breadth scores
        repo_diversity = min(len(repos) / 5.0, 1.0)  # Cap at 5 repos
        collaborator_diversity = min(len(collaborators) / 20.0, 1.0)  # Cap at 20 collaborators
        topic_diversity = min(len(topics) / 10.0, 1.0) if topics else 0.0  # Cap at 10 topics

        # Combined breadth score
        breadth_score = repo_diversity * 0.4 + collaborator_diversity * 0.4 + topic_diversity * 0.2

        return min(breadth_score, 1.0)

    def _compute_learning_velocity(self, person_id: str) -> float:
        """Compute learning velocity (rate of new topics/collaborators).

        Args:
            person_id: Person identifier

        Returns:
            Score between 0 and 1 (higher = faster learning velocity)
        """
        # we'll use proxies: diversity of recent work

        # Count recent unique repositories (last 30 days)
        recent_commits = [
            c
            for c in self.code_commits
            if c.get("author_id") == person_id
            and (
                not c.get("timestamp")
                or (
                    isinstance(c.get("timestamp"), datetime)
                    and c.get("timestamp") > datetime.now() - timedelta(days=30)
                )
            )
        ]
        recent_repos = len(set(c.get("repo") for c in recent_commits if c.get("repo")))

        # Count recent unique collaborators (last 30 days)
        recent_interactions = [
            i
            for i in self.interactions
            if i.get("sender_id") == person_id
            and (
                not i.get("timestamp")
                or (
                    isinstance(i.get("timestamp"), datetime)
                    and i.get("timestamp") > datetime.now() - timedelta(days=30)
                )
            )
        ]
        recent_collaborators = len(
            set(i.get("receiver_id") for i in recent_interactions if i.get("receiver_id"))
        )

        # Calculate velocity proxies
        repo_velocity = min(recent_repos / 3.0, 1.0)  # Cap at 3 new repos in 30 days
        collaborator_velocity = min(
            recent_collaborators / 10.0, 1.0
        )  # Cap at 10 new collaborators in 30 days

        # Combined velocity score
        velocity_score = repo_velocity * 0.5 + collaborator_velocity * 0.5

        return min(velocity_score, 1.0)

    def _compute_combined_score(
        self,
        curiosity: float,
        teaching: float,
        knowledge_breadth: float,
        learning_velocity: float,
    ) -> float:
        """Compute combined learning behavior score.

        Args:
            curiosity: Curiosity score
            teaching: Teaching score
            knowledge_breadth: Knowledge breadth score
            learning_velocity: Learning velocity score

        Returns:
            Combined score between 0 and 1
        """
        # Weighted combination
        combined = (
            curiosity * 0.25 + teaching * 0.25 + knowledge_breadth * 0.25 + learning_velocity * 0.25
        )

        return min(combined, 1.0)

    def get_top_learners(self, top_n: int = 20) -> pd.DataFrame:
        """Get top learners by learning behavior score.

        Args:
            top_n: Number of top learners to return

        Returns:
            DataFrame with top learners sorted by learning_behavior_score
        """
        results_df = self.compute()

        if results_df.empty:
            return pd.DataFrame()

        top_learners = results_df.nlargest(top_n, "learning_behavior_score").reset_index(drop=True)

        return top_learners
