"""Cross-functional collaboration depth analysis.

This module measures depth and velocity of cross-functional collaboration:
- Collaboration depth (repeated interactions, project co-membership)
- Low-stakes experimentation rate (pilots, tests, prototypes)
- Collaboration velocity (speed of cross-functional problem-solving)
"""

import re
from collections import Counter, defaultdict
from datetime import datetime

import networkx as nx
import pandas as pd

from orgnet.utils.logging import get_logger

logger = get_logger(__name__)

# Experimentation keywords
EXPERIMENTATION_KEYWORDS = [
    r"\b(pilot|test|trial|experiment|prototype|beta|proof of concept|poc|mvp)\b",
    r"\b(try|testing|tryout|experimental|pilot program)\b",
    r"\b(small test|quick test|low stakes|rapid test)\b",
]


class CollaborationDepthAnalyzer:
    """Measures depth and velocity of cross-functional collaboration.

    Goes beyond simple connection counts to measure:
    - Collaboration depth (recurrence, project overlap)
    - Low-stakes experimentation rate
    - Cross-functional velocity (time to solution across teams)
    """

    def __init__(
        self,
        graph: nx.Graph,
        interactions: list[dict] | None = None,
        projects: list[dict] | None = None,
        documents: list[dict] | None = None,
        code_commits: list[dict] | None = None,
        people: list | None = None,
        **params,
    ):
        """Initialize collaboration depth analyzer.

        Args:
            graph: Organizational network graph
            interactions: List of interaction dicts with 'sender_id', 'receiver_id',
                         'timestamp', 'text'
            projects: List of project dicts with 'project_id', 'members' (list of person_ids),
                     'start_date', 'end_date'
            documents: List of document dicts with 'author_id', 'collaborators' (list),
                      'timestamp', 'text'
            code_commits: List of code commit dicts with 'author_id', 'repo', 'timestamp'
            people: List of Person objects with 'id', 'department'
        """
        self.graph = graph
        self.interactions = interactions or []
        self.projects = projects or []
        self.documents = documents or []
        self.code_commits = code_commits or []
        self.people = {p.id: p for p in (people or [])} if people else {}

    def compute(
        self, graph: nx.Graph | None = None, nodes: pd.DataFrame | None = None, **kwargs
    ) -> pd.DataFrame:
        """Compute collaboration depth metrics.

        Args:
            graph: Optional graph (uses self.graph if None)
            nodes: Optional node table (not used but required by BaseMetric)
            **kwargs: Additional arguments

        Returns:
            DataFrame with columns: person_id, collaboration_depth_score,
                experimentation_rate, collaboration_velocity,
                cross_functional_collaboration_score
        """
        if graph is None:
            graph = self.graph

        person_ids = list(graph.nodes())
        results = []

        for person_id in person_ids:
            # Collaboration depth score
            depth_score = self._compute_collaboration_depth(person_id)

            # Low-stakes experimentation rate
            experimentation_rate = self._compute_experimentation_rate(person_id)

            # Collaboration velocity
            velocity = self._compute_collaboration_velocity(person_id)

            # Combined cross-functional collaboration score
            collaboration_score = self._compute_combined_score(
                depth_score, experimentation_rate, velocity
            )

            results.append(
                {
                    "person_id": person_id,
                    "collaboration_depth_score": depth_score,
                    "experimentation_rate": experimentation_rate,
                    "collaboration_velocity": velocity,
                    "cross_functional_collaboration_score": collaboration_score,
                }
            )

        return pd.DataFrame(results)

    def _compute_collaboration_depth(self, person_id: str) -> float:
        """Compute collaboration depth score.

        Measures depth of collaboration through:
        - Repeated interactions (not just one-off)
        - Project co-membership
        - Document collaboration

        Args:
            person_id: Person identifier

        Returns:
            Score between 0 and 1 (higher = deeper collaboration)
        """
        # Track collaborators and interaction frequency
        collaborator_frequency = Counter()

        # From interactions (count repeated interactions)
        for interaction in self.interactions:
            sender_id = interaction.get("sender_id")
            receiver_id = interaction.get("receiver_id")

            if sender_id == person_id and receiver_id:
                collaborator_frequency[receiver_id] += 1
            elif receiver_id == person_id and sender_id:
                collaborator_frequency[sender_id] += 1

        # Count unique collaborators
        unique_collaborators = len(collaborator_frequency)

        # Count recurring collaborators (2+ interactions)
        recurring_collaborators = sum(1 for count in collaborator_frequency.values() if count >= 2)

        # Project co-membership (deep collaboration indicator)
        project_collaborators = set()
        for project in self.projects:
            members = project.get("members", [])
            if person_id in members:
                for member in members:
                    if member != person_id:
                        project_collaborators.add(member)

        # Document collaboration
        doc_collaborators = set()
        for doc in self.documents:
            author_id = doc.get("author_id")
            collaborators = doc.get("collaborators", [])

            if author_id == person_id:
                doc_collaborators.update(collaborators)
            elif person_id in collaborators:
                doc_collaborators.add(author_id)

        # Calculate depth metrics
        total_unique_collaborators = len(
            set(list(collaborator_frequency.keys())) | project_collaborators | doc_collaborators
        )

        if total_unique_collaborators == 0:
            return 0.0

        # Recurrence ratio (how many relationships are repeated)
        recurrence_ratio = recurring_collaborators / max(unique_collaborators, 1)

        # Project collaboration ratio (deep vs. shallow)
        project_ratio = len(project_collaborators) / max(total_unique_collaborators, 1)

        # Document collaboration ratio
        doc_ratio = len(doc_collaborators) / max(total_unique_collaborators, 1)

        # Combined depth score
        depth_score = recurrence_ratio * 0.4 + project_ratio * 0.4 + doc_ratio * 0.2

        return min(depth_score, 1.0)

    def _compute_experimentation_rate(self, person_id: str) -> float:
        """Compute low-stakes experimentation rate.

        Measures frequency of small tests, pilots, proof-of-concepts
        in communications and work.

        Args:
            person_id: Person identifier

        Returns:
            Score between 0 and 1 (higher = more experimentation)
        """
        # Get all text from this person
        texts = []

        for interaction in self.interactions:
            if interaction.get("sender_id") == person_id:
                texts.append(interaction.get("text", ""))

        for doc in self.documents:
            if doc.get("author_id") == person_id:
                texts.append(doc.get("text", ""))

        for commit in self.code_commits:
            if commit.get("author_id") == person_id:
                texts.append(commit.get("message", ""))

        if not texts:
            return 0.0

        total_text = " ".join(texts).lower()

        # Count experimentation keywords
        experimentation_count = 0
        for pattern in EXPERIMENTATION_KEYWORDS:
            matches = len(re.findall(pattern, total_text, re.IGNORECASE))
            experimentation_count += matches

        # Normalize by text volume
        text_length = len(total_text)
        if text_length == 0:
            return 0.0

        # Score: experimentation mentions per 1000 characters
        mentions_per_1k = (experimentation_count / text_length) * 1000
        rate = min(mentions_per_1k / 3.0, 1.0)  # Cap at 3 mentions per 1k chars

        # Also check project patterns (pilot projects)
        pilot_projects = 0
        for project in self.projects:
            members = project.get("members", [])
            project_name = project.get("name", "").lower()

            if person_id in members and any(
                keyword in project_name for keyword in ["pilot", "test", "trial", "beta"]
            ):
                pilot_projects += 1

        # Normalize by total projects
        person_projects = [p for p in self.projects if person_id in p.get("members", [])]
        if person_projects:
            pilot_ratio = pilot_projects / len(person_projects)
            rate = rate * 0.7 + pilot_ratio * 0.3

        return min(rate, 1.0)

    def _compute_collaboration_velocity(self, person_id: str) -> float:
        """Compute collaboration velocity.

        Measures speed of cross-functional problem-solving:
        - Time from problem identification to solution
        - Cross-team response time
        - Project completion speed (if cross-functional)

        Args:
            person_id: Person identifier

        Returns:
            Score between 0 and 1 (higher = faster collaboration velocity)
        """
        # Get person's department
        person = self.people.get(person_id)
        person_dept = getattr(person, "department", None) if person else None

        # Track cross-functional interaction response times
        response_times = []

        # Calculate response times from interactions
        # Group interactions by thread/conversation
        threads = defaultdict(list)
        for interaction in self.interactions:
            thread_id = interaction.get("thread_id") or interaction.get("conversation_id")
            if thread_id:
                threads[thread_id].append(interaction)

        # For each thread where person is involved, calculate cross-functional response time
        for thread_id, thread_interactions in threads.items():
            thread_interactions.sort(key=lambda x: x.get("timestamp") or datetime.min)

            # Check if thread involves cross-functional collaboration
            depts_in_thread = set()
            for interaction in thread_interactions:
                sender_id = interaction.get("sender_id")
                receiver_id = interaction.get("receiver_id")

                sender = self.people.get(sender_id) if sender_id else None
                receiver = self.people.get(receiver_id) if receiver_id else None

                if sender:
                    dept = getattr(sender, "department", None)
                    if dept:
                        depts_in_thread.add(dept)
                if receiver:
                    dept = getattr(receiver, "department", None)
                    if dept:
                        depts_in_thread.add(dept)

            # If multiple departments, it's cross-functional
            if len(depts_in_thread) > 1 and person_id in [
                i.get("sender_id") or i.get("receiver_id") for i in thread_interactions
            ]:
                # Find person's interactions
                person_interactions = [
                    i
                    for i in thread_interactions
                    if i.get("sender_id") == person_id or i.get("receiver_id") == person_id
                ]

                if len(person_interactions) >= 2:
                    # Calculate time between first and last interaction
                    first_time = person_interactions[0].get("timestamp")
                    last_time = person_interactions[-1].get("timestamp")

                    if (
                        first_time
                        and last_time
                        and isinstance(first_time, datetime)
                        and isinstance(last_time, datetime)
                    ):
                        duration = (last_time - first_time).total_seconds() / 3600  # Hours
                        if duration > 0:
                            response_times.append(duration)

        # Calculate velocity from response times
        if not response_times:
            return 0.0

        avg_response_time = sum(response_times) / len(response_times)

        # Score = 1 - (avg_time / 24), capped at 1.0
        velocity_score = max(0.0, 1.0 - (avg_response_time / 24.0))

        # Also check project completion speed for cross-functional projects
        cross_functional_projects = []
        for project in self.projects:
            members = project.get("members", [])
            if person_id in members:
                # Check if project is cross-functional
                project_depts = set()
                for member_id in members:
                    member = self.people.get(member_id)
                    if member:
                        dept = getattr(member, "department", None)
                        if dept:
                            project_depts.add(dept)

                if len(project_depts) > 1:  # Cross-functional
                    start_date = project.get("start_date")
                    end_date = project.get("end_date")

                    if (
                        start_date
                        and end_date
                        and isinstance(start_date, datetime)
                        and isinstance(end_date, datetime)
                    ):
                        duration_days = (end_date - start_date).days
                        if duration_days > 0:
                            cross_functional_projects.append(duration_days)

        if cross_functional_projects:
            avg_project_duration = sum(cross_functional_projects) / len(cross_functional_projects)
            project_velocity = max(0.0, 1.0 - (avg_project_duration / 30.0))
            velocity_score = velocity_score * 0.6 + project_velocity * 0.4

        return min(velocity_score, 1.0)

    def _compute_combined_score(
        self, depth_score: float, experimentation_rate: float, velocity: float
    ) -> float:
        """Compute combined cross-functional collaboration score.

        Args:
            depth_score: Collaboration depth score
            experimentation_rate: Low-stakes experimentation rate
            velocity: Collaboration velocity score

        Returns:
            Combined score between 0 and 1
        """
        # Weighted combination
        combined = depth_score * 0.4 + experimentation_rate * 0.3 + velocity * 0.3

        return min(combined, 1.0)

    def get_top_collaborators(self, top_n: int = 20) -> pd.DataFrame:
        """Get top collaborators by collaboration score.

        Args:
            top_n: Number of top collaborators to return

        Returns:
            DataFrame with top collaborators sorted by cross_functional_collaboration_score
        """
        results_df = self.compute()

        if results_df.empty:
            return pd.DataFrame()

        top_collaborators = results_df.nlargest(
            top_n, "cross_functional_collaboration_score"
        ).reset_index(drop=True)

        return top_collaborators
