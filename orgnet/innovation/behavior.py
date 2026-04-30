"""Innovation behavior analysis beyond network structure.

This module measures behavioral indicators of innovation:
- Assumption challenging (questioning vs. confirming language)
- Cross-functional influence (impact across departments)
- Process innovation (experiments, pilots, new processes)
- Value creation beyond scope (contributions outside primary role)
"""

import re

import networkx as nx
import pandas as pd

from orgnet.utils.logging import get_logger

logger = get_logger(__name__)

# Assumption challenging keywords
QUESTIONING_KEYWORDS = [
    r"\b(why|what if|how about|instead of|alternative|different approach)\b",
    r"\b(assumption|assume|hypothesis|belief|conventional wisdom)\b",
    r"\b(challenge|question|disagree|doubt|skeptical|concern)\b",
    r"\b(but|however|although|whereas|unlike|contrary)\b",
]

CONFIRMING_KEYWORDS = [
    r"\b(agree|correct|right|yes|exactly|precisely|absolutely)\b",
    r"\b(same|similar|consistent|matches|aligns)\b",
    r"\b(follow|implement|execute|adopt)\b",
]

# Process innovation keywords
INNOVATION_KEYWORDS = [
    r"\b(experiment|pilot|prototype|test|try|trial|beta)\b",
    r"\b(new process|new way|new approach|new method)\b",
    r"\b(improve|optimize|streamline|automate|refactor)\b",
    r"\b(innovation|innovative|novel|creative|different)\b",
    r"\b(poc|proof of concept|mvp|minimum viable)\b",
]


class InnovationBehaviorAnalyzer:
    """Measures behavioral indicators of innovation beyond network structure.

    Analyzes actual behavior (what people do and say) rather than just network
    position. Focuses on assumption challenging, cross-functional influence,
    process innovation, and value creation beyond scope.
    """

    def __init__(
        self,
        graph: nx.Graph,
        documents: list[dict] | None = None,
        interactions: list[dict] | None = None,
        code_commits: list[dict] | None = None,
        people: list | None = None,
        **params,
    ):
        """Initialize innovation behavior analyzer.

        Args:
            graph: Organizational network graph
            documents: List of document dicts with 'author_id', 'text', 'timestamp'
            interactions: List of interaction dicts (Slack, email) with 'sender_id',
                         'receiver_id', 'text', 'timestamp', 'channel' or 'department'
            code_commits: List of code commit dicts with 'author_id', 'repo',
                         'message', 'timestamp', 'files_changed'
            people: List of Person objects with 'id', 'department', 'role'
        """
        self.graph = graph
        self.documents = documents or []
        self.interactions = interactions or []
        self.code_commits = code_commits or []
        self.people = {p.id: p for p in (people or [])} if people else {}

    def compute(
        self, graph: nx.Graph | None = None, nodes: pd.DataFrame | None = None, **kwargs
    ) -> pd.DataFrame:
        """Compute innovation behavior metrics.

        Args:
            graph: Optional graph (uses self.graph if None)
            nodes: Optional node table (not used but required by BaseMetric)
            **kwargs: Additional arguments

        Returns:
            DataFrame with columns: person_id, assumption_challenging_score,
                cross_functional_influence, process_innovation_score,
                value_creation_beyond_scope, innovation_behavior_score
        """
        if graph is None:
            graph = self.graph

        # Extract behavioral signals for all people in graph
        person_ids = list(graph.nodes())
        results = []

        for person_id in person_ids:
            # Assumption challenging score
            assumption_score = self._compute_assumption_challenging_score(person_id)

            # Cross-functional influence
            cross_functional_influence = self._compute_cross_functional_influence(person_id)

            # Process innovation score
            process_innovation = self._compute_process_innovation_score(person_id)

            # Value creation beyond scope
            beyond_scope = self._compute_value_creation_beyond_scope(person_id)

            # Combined innovation behavior score
            innovation_behavior_score = self._compute_combined_score(
                assumption_score,
                cross_functional_influence,
                process_innovation,
                beyond_scope,
            )

            results.append(
                {
                    "person_id": person_id,
                    "assumption_challenging_score": assumption_score,
                    "cross_functional_influence": cross_functional_influence,
                    "process_innovation_score": process_innovation,
                    "value_creation_beyond_scope": beyond_scope,
                    "innovation_behavior_score": innovation_behavior_score,
                }
            )

        return pd.DataFrame(results)

    def _compute_assumption_challenging_score(self, person_id: str) -> float:
        """Compute assumption challenging score from text analysis.

        Measures frequency of questioning vs. confirming language in
        documents and interactions.

        Args:
            person_id: Person identifier

        Returns:
            Score between 0 and 1 (higher = more assumption challenging)
        """
        # Get all text from this person
        texts = []

        # From documents
        for doc in self.documents:
            if doc.get("author_id") == person_id:
                texts.append(doc.get("text", ""))

        # From interactions
        for interaction in self.interactions:
            if interaction.get("sender_id") == person_id:
                texts.append(interaction.get("text", ""))

        if not texts:
            return 0.0

        total_text = " ".join(texts).lower()
        if not total_text:
            return 0.0

        # Count questioning patterns
        questioning_count = 0
        for pattern in QUESTIONING_KEYWORDS:
            matches = len(re.findall(pattern, total_text, re.IGNORECASE))
            questioning_count += matches

        # Count confirming patterns
        confirming_count = 0
        for pattern in CONFIRMING_KEYWORDS:
            matches = len(re.findall(pattern, total_text, re.IGNORECASE))
            confirming_count += matches

        # Questioning ratio
        total_indicators = questioning_count + confirming_count
        if total_indicators == 0:
            return 0.0

        questioning_ratio = questioning_count / total_indicators

        # Normalize by text volume (more text = more opportunity)
        text_length_factor = min(len(total_text) / 1000.0, 1.0)  # Cap at 1000 chars

        # Combined score
        score = questioning_ratio * 0.7 + text_length_factor * 0.3

        return min(score, 1.0)

    def _compute_cross_functional_influence(self, person_id: str) -> float:
        """Compute cross-functional influence score.

        Measures number of different departments/teams influenced by
        person's work based on interaction patterns.

        Args:
            person_id: Person identifier

        Returns:
            Score between 0 and 1 (higher = more cross-functional influence)
        """
        # Get person's department
        person = self.people.get(person_id)
        person_dept = getattr(person, "department", None) if person else None

        # Track departments this person interacts with
        influenced_departments = set()

        # From interactions (outbound)
        for interaction in self.interactions:
            if interaction.get("sender_id") == person_id:
                receiver_id = interaction.get("receiver_id")
                if receiver_id:
                    receiver = self.people.get(receiver_id)
                    if receiver:
                        dept = getattr(receiver, "department", None)
                        if dept and dept != person_dept:
                            influenced_departments.add(dept)

                # Also check channel/department from interaction metadata
                channel = interaction.get("channel") or interaction.get("department")
                if channel and channel != person_dept:
                    influenced_departments.add(channel)

        # From code commits (cross-repository suggests cross-functional work)
        repos = set()
        for commit in self.code_commits:
            if commit.get("author_id") == person_id:
                repo = commit.get("repo")
                if repo:
                    repos.add(repo)

        # From documents (shared documents suggest influence)
        shared_docs = 0
        for doc in self.documents:
            if doc.get("author_id") == person_id:
                # Check if document has shared/accessed metadata
                shared_count = doc.get("shared_count", 0) or doc.get("access_count", 0) or 0
                if shared_count > 0:
                    shared_docs += 1

        # Calculate influence score
        # Number of departments influenced (normalized)
        num_departments = len(influenced_departments)
        dept_score = min(num_departments / 5.0, 1.0)  # Cap at 5 departments

        # Repository diversity (cross-functional work)
        repo_diversity = min(len(repos) / 3.0, 1.0)  # Cap at 3 repos

        # Document sharing (influence indicator)
        doc_sharing_score = min(shared_docs / 10.0, 1.0)  # Cap at 10 shared docs

        # Combined score
        cross_functional_score = dept_score * 0.5 + repo_diversity * 0.3 + doc_sharing_score * 0.2

        return min(cross_functional_score, 1.0)

    def _compute_process_innovation_score(self, person_id: str) -> float:
        """Compute process innovation score.

        Measures frequency of "new ways of working" signals like
        experiments, pilots, new processes.

        Args:
            person_id: Person identifier

        Returns:
            Score between 0 and 1 (higher = more process innovation)
        """
        # Get all text from this person
        texts = []

        for doc in self.documents:
            if doc.get("author_id") == person_id:
                texts.append(doc.get("text", ""))

        for interaction in self.interactions:
            if interaction.get("sender_id") == person_id:
                texts.append(interaction.get("text", ""))

        for commit in self.code_commits:
            if commit.get("author_id") == person_id:
                commit_msg = commit.get("message", "")
                texts.append(commit_msg)

        if not texts:
            return 0.0

        total_text = " ".join(texts).lower()

        # Count innovation keywords
        innovation_count = 0
        for pattern in INNOVATION_KEYWORDS:
            matches = len(re.findall(pattern, total_text, re.IGNORECASE))
            innovation_count += matches

        # Normalize by text volume
        text_length = len(total_text)
        if text_length == 0:
            return 0.0

        # Score: innovation mentions per 1000 characters
        mentions_per_1k = (innovation_count / text_length) * 1000
        score = min(mentions_per_1k / 5.0, 1.0)  # Cap at 5 mentions per 1k chars

        # Also check code commit patterns (refactoring, new features)
        refactor_commits = 0
        for commit in self.code_commits:
            if commit.get("author_id") == person_id:
                msg = commit.get("message", "").lower()
                if any(
                    keyword in msg for keyword in ["refactor", "improve", "optimize", "new", "add"]
                ):
                    refactor_commits += 1

        if len(self.code_commits) > 0:
            person_commits = [c for c in self.code_commits if c.get("author_id") == person_id]
            if person_commits:
                refactor_ratio = refactor_commits / len(person_commits)
                score = score * 0.7 + refactor_ratio * 0.3

        return min(score, 1.0)

    def _compute_value_creation_beyond_scope(self, person_id: str) -> float:
        """Compute value creation beyond scope score.

        Measures contributions outside primary role/team based on
        cross-functional work, diverse contributions, and impact.

        Args:
            person_id: Person identifier

        Returns:
            Score between 0 and 1 (higher = more beyond scope contributions)
        """
        person = self.people.get(person_id)
        person_dept = getattr(person, "department", None) if person else None
        person_role = getattr(person, "role", None) if person else None

        # Count contributions outside primary department
        outside_dept_contributions = 0

        # Code commits to repos outside department
        for commit in self.code_commits:
            if commit.get("author_id") == person_id:
                repo = commit.get("repo", "")
                # If repo doesn't match department pattern, it's cross-functional
                if person_dept and person_dept.lower() not in repo.lower():
                    outside_dept_contributions += 1

        # Interactions with different departments
        cross_dept_interactions = 0
        for interaction in self.interactions:
            if interaction.get("sender_id") == person_id:
                receiver_id = interaction.get("receiver_id")
                if receiver_id:
                    receiver = self.people.get(receiver_id)
                    if receiver:
                        dept = getattr(receiver, "department", None)
                        if dept and dept != person_dept:
                            cross_dept_interactions += 1

        # Documents shared across departments
        cross_dept_docs = 0
        for doc in self.documents:
            if doc.get("author_id") == person_id:
                shared_count = doc.get("shared_count", 0) or doc.get("access_count", 0) or 0
                if shared_count > 0:
                    cross_dept_docs += 1

        # Calculate beyond-scope score
        total_contributions = (
            len([c for c in self.code_commits if c.get("author_id") == person_id])
            + len([i for i in self.interactions if i.get("sender_id") == person_id])
            + len([d for d in self.documents if d.get("author_id") == person_id])
        )

        if total_contributions == 0:
            return 0.0

        beyond_scope_count = outside_dept_contributions + cross_dept_interactions + cross_dept_docs
        beyond_scope_ratio = beyond_scope_count / max(total_contributions, 1)

        return min(beyond_scope_ratio, 1.0)

    def _compute_combined_score(
        self,
        assumption_score: float,
        cross_functional: float,
        process_innovation: float,
        beyond_scope: float,
    ) -> float:
        """Compute combined innovation behavior score.

        Args:
            assumption_score: Assumption challenging score
            cross_functional: Cross-functional influence score
            process_innovation: Process innovation score
            beyond_scope: Value creation beyond scope score

        Returns:
            Combined score between 0 and 1
        """
        # Weighted combination
        combined = (
            assumption_score * 0.25
            + cross_functional * 0.30
            + process_innovation * 0.25
            + beyond_scope * 0.20
        )

        return min(combined, 1.0)

    def get_top_innovators(self, top_n: int = 20) -> pd.DataFrame:
        """Get top innovators by behavior score.

        Args:
            top_n: Number of top innovators to return

        Returns:
            DataFrame with top innovators sorted by innovation_behavior_score
        """
        results_df = self.compute()

        if results_df.empty:
            return pd.DataFrame()

        top_innovators = results_df.nlargest(top_n, "innovation_behavior_score").reset_index(
            drop=True
        )

        return top_innovators

    def get_evidence_for_person(self, person_id: str) -> dict:
        """Get evidence/examples supporting innovation behavior scores.

        Args:
            person_id: Person identifier

        Returns:
            Dictionary with evidence for each score category
        """
        evidence = {
            "assumption_challenging": [],
            "cross_functional_influence": [],
            "process_innovation": [],
            "value_creation_beyond_scope": [],
        }

        # Collect assumption challenging examples
        for interaction in self.interactions:
            if interaction.get("sender_id") == person_id:
                text = interaction.get("text", "")
                if any(re.search(pattern, text, re.IGNORECASE) for pattern in QUESTIONING_KEYWORDS):
                    evidence["assumption_challenging"].append(
                        {
                            "type": "interaction",
                            "text": text[:200],  # First 200 chars
                            "timestamp": interaction.get("timestamp"),
                        }
                    )

        # Collect process innovation examples
        for commit in self.code_commits:
            if commit.get("author_id") == person_id:
                msg = commit.get("message", "")
                if any(
                    keyword in msg.lower() for keyword in ["refactor", "improve", "optimize", "new"]
                ):
                    evidence["process_innovation"].append(
                        {
                            "type": "code_commit",
                            "message": msg,
                            "repo": commit.get("repo"),
                            "timestamp": commit.get("timestamp"),
                        }
                    )

        return evidence
