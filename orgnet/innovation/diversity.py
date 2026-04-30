"""Perspective diversity and inclusion in decision-making analysis.

This module measures diversity of perspectives and inclusion:
- Perspective diversity score (diversity of viewpoints)
- Inclusion in decision-making (meeting participation, influence)
- Dissent encouragement score (frequency of dissenting opinions)
- Power balance index (distribution of influence)
"""

import re
from collections import defaultdict

import networkx as nx
import pandas as pd

from orgnet.utils.logging import get_logger

logger = get_logger(__name__)

# Dissent keywords
DISSENT_KEYWORDS = [
    r"\b(disagree|disagreement|differ|different|alternative|opposing)\b",
    r"\b(concern|concerned|worried|skeptical|skepticism)\b",
    r"\b(however|but|although|though|instead|rather)\b",
    r"\b(question|challenge|critique|criticism|critical)\b",
]

# Consensus-seeking keywords
CONSENSUS_KEYWORDS = [
    r"\b(agree|agreement|consensus|unanimous|unified)\b",
    r"\b(same|similar|consistent|align|aligned|match)\b",
    r"\b(approve|approval|support|endorse|back)\b",
]


class PerspectiveDiversityAnalyzer:
    """Measures diversity of perspectives and inclusion in decision-making.

    Analyzes:
    - Perspective diversity in discussions
    - Inclusion in decision-making (meeting participation, influence)
    - Dissent frequency and response (positive/negative sentiment)
    - Power balance across demographic/functional groups
    """

    def __init__(
        self,
        graph: nx.Graph,
        meetings: list[dict] | None = None,
        documents: list[dict] | None = None,
        interactions: list[dict] | None = None,
        people: list | None = None,
        **params,
    ):
        """Initialize perspective diversity analyzer.

        Args:
            graph: Organizational network graph
            meetings: List of meeting dicts with 'meeting_id', 'participants' (list),
                     'timestamp', 'type' (decision-making, informational, etc.),
                     'decisions' (list of dicts with 'decision', 'influencer_ids')
            documents: List of document dicts with 'author_id', 'text', 'timestamp',
                      'decision_document' (boolean)
            interactions: List of interaction dicts with 'sender_id', 'receiver_id',
                         'text', 'timestamp', 'thread_id'
            people: List of Person objects with 'id', 'department', 'role'
        """
        self.graph = graph
        self.meetings = meetings or []
        self.documents = documents or []
        self.interactions = interactions or []
        self.people = {p.id: p for p in (people or [])} if people else {}

    def compute(
        self, graph: nx.Graph | None = None, nodes: pd.DataFrame | None = None, **kwargs
    ) -> pd.DataFrame:
        """Compute perspective diversity metrics.

        Args:
            graph: Optional graph (uses self.graph if None)
            nodes: Optional node table (not used but required by BaseMetric)
            **kwargs: Additional arguments

        Returns:
            DataFrame with columns: person_id, perspective_diversity_score,
                inclusion_in_decision_making, dissent_encouragement_score,
                power_balance_index, diversity_inclusion_score
        """
        if graph is None:
            graph = self.graph

        person_ids = list(graph.nodes())
        results = []

        for person_id in person_ids:
            # Perspective diversity score
            perspective_diversity = self._compute_perspective_diversity(person_id)

            # Inclusion in decision-making
            inclusion = self._compute_inclusion_in_decision_making(person_id)

            # Dissent encouragement score
            dissent_score = self._compute_dissent_encouragement_score(person_id)

            # Power balance index (organization-level, same for all)
            power_balance = self._compute_power_balance_index()

            # Combined diversity and inclusion score
            diversity_score = self._compute_combined_score(
                perspective_diversity, inclusion, dissent_score, power_balance
            )

            results.append(
                {
                    "person_id": person_id,
                    "perspective_diversity_score": perspective_diversity,
                    "inclusion_in_decision_making": inclusion,
                    "dissent_encouragement_score": dissent_score,
                    "power_balance_index": power_balance,
                    "diversity_inclusion_score": diversity_score,
                }
            )

        return pd.DataFrame(results)

    def _compute_perspective_diversity(self, person_id: str) -> float:
        """Compute perspective diversity score.

        Measures diversity of viewpoints in discussions/projects person is involved in.

        Args:
            person_id: Person identifier

        Returns:
            Score between 0 and 1 (higher = more diverse perspectives)
        """
        # Get person's department/role
        person = self.people.get(person_id)
        person_dept = getattr(person, "department", None) if person else None
        person_role = getattr(person, "role", None) if person else None

        # Track departments/roles in person's discussions/projects
        departments_in_discussions = set()
        roles_in_discussions = set()

        # From meetings
        for meeting in self.meetings:
            participants = meeting.get("participants", [])
            if person_id in participants:
                for participant_id in participants:
                    participant = self.people.get(participant_id)
                    if participant:
                        dept = getattr(participant, "department", None)
                        role = getattr(participant, "role", None)
                        if dept:
                            departments_in_discussions.add(dept)
                        if role:
                            roles_in_discussions.add(role)

        # From interactions (discussions)
        for interaction in self.interactions:
            sender_id = interaction.get("sender_id")
            receiver_id = interaction.get("receiver_id")

            if sender_id == person_id or receiver_id == person_id:
                # Get other party
                other_id = receiver_id if sender_id == person_id else sender_id
                if other_id:
                    other = self.people.get(other_id)
                    if other:
                        dept = getattr(other, "department", None)
                        role = getattr(other, "role", None)
                        if dept:
                            departments_in_discussions.add(dept)
                        if role:
                            roles_in_discussions.add(role)

        # From documents (collaboration)
        for doc in self.documents:
            author_id = doc.get("author_id")
            collaborators = doc.get("collaborators", [])

            if person_id == author_id or person_id in collaborators:
                # Get all collaborators
                all_collaborators = [author_id] + collaborators
                for collab_id in all_collaborators:
                    collab = self.people.get(collab_id)
                    if collab:
                        dept = getattr(collab, "department", None)
                        role = getattr(collab, "role", None)
                        if dept:
                            departments_in_discussions.add(dept)
                        if role:
                            roles_in_discussions.add(role)

        # Calculate diversity metrics
        # Number of unique departments (excluding person's own)
        if person_dept and person_dept in departments_in_discussions:
            departments_in_discussions.remove(person_dept)

        dept_diversity = len(departments_in_discussions)
        role_diversity = len(roles_in_discussions)

        # Normalize diversity scores
        dept_diversity_score = min(dept_diversity / 5.0, 1.0)  # Cap at 5 departments
        role_diversity_score = min(role_diversity / 10.0, 1.0)  # Cap at 10 roles

        # Combined perspective diversity score
        perspective_diversity = dept_diversity_score * 0.6 + role_diversity_score * 0.4

        return min(perspective_diversity, 1.0)

    def _compute_inclusion_in_decision_making(self, person_id: str) -> float:
        """Compute inclusion in decision-making score.

        Measures participation in key decisions (meeting attendance, influence on outcomes).

        Args:
            person_id: Person identifier

        Returns:
            Score between 0 and 1 (higher = more inclusion in decisions)
        """
        # Count decision-making meetings person participated in
        decision_meetings = [
            m
            for m in self.meetings
            if m.get("type") == "decision-making" or m.get("decision_document", False)
        ]

        person_decision_meetings = [
            m for m in decision_meetings if person_id in m.get("participants", [])
        ]

        # Count times person was an influencer in decisions
        influence_count = 0
        for meeting in person_decision_meetings:
            decisions = meeting.get("decisions", [])
            for decision in decisions:
                influencer_ids = decision.get("influencer_ids", [])
                if person_id in influencer_ids:
                    influence_count += 1

        # Calculate inclusion scores
        total_decision_meetings = len(decision_meetings)
        participation_score = (
            len(person_decision_meetings) / max(total_decision_meetings, 1)
            if total_decision_meetings > 0
            else 0.0
        )

        total_decisions = sum(len(m.get("decisions", [])) for m in decision_meetings)
        influence_score = influence_count / max(total_decisions, 1) if total_decisions > 0 else 0.0

        # Also check document authoring (decision documents)
        decision_docs = [
            d
            for d in self.documents
            if d.get("decision_document", False) and d.get("author_id") == person_id
        ]
        doc_authoring_score = (
            len(decision_docs)
            / max(len([d for d in self.documents if d.get("decision_document", False)]), 1)
            if any(d.get("decision_document", False) for d in self.documents)
            else 0.0
        )

        # Combined inclusion score
        inclusion_score = (
            participation_score * 0.4 + influence_score * 0.4 + doc_authoring_score * 0.2
        )

        return min(inclusion_score, 1.0)

    def _compute_dissent_encouragement_score(self, person_id: str) -> float:
        """Compute dissent encouragement score.

        Measures frequency of dissenting opinions and positive response to dissent.

        Args:
            person_id: Person identifier

        Returns:
            Score between 0 and 1 (higher = more dissent encouraged)
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

        # Count dissent keywords
        dissent_count = 0
        for pattern in DISSENT_KEYWORDS:
            matches = len(re.findall(pattern, total_text, re.IGNORECASE))
            dissent_count += matches

        # Count consensus-seeking keywords
        consensus_count = 0
        for pattern in CONSENSUS_KEYWORDS:
            matches = len(re.findall(pattern, total_text, re.IGNORECASE))
            consensus_count += matches

        # Calculate dissent ratio
        total_indicators = dissent_count + consensus_count
        if total_indicators == 0:
            return 0.0

        dissent_ratio = dissent_count / total_indicators

        # Higher dissent ratio = more dissent expression
        # But we also want to measure if dissent is encouraged (positive responses)
        # For now, we'll use dissent ratio as proxy
        # Could be enhanced with sentiment analysis of responses to dissent

        return min(dissent_ratio, 1.0)

    def _compute_power_balance_index(self) -> float:
        """Compute power balance index (organization-level).

        Measures distribution of influence across different perspectives/groups.

        Returns:
            Score between 0 and 1 (higher = more balanced power distribution)
        """
        # Calculate centrality for all people
        centrality_scores = {}
        try:
            centrality = nx.betweenness_centrality(self.graph)
            centrality_scores = centrality
        except Exception as e:
            logger.warning(f"Failed to compute centrality: {e}")
            return 0.5  # Default middle score

        if not centrality_scores:
            return 0.5

        # Group people by department
        dept_centrality = defaultdict(list)
        for person_id, score in centrality_scores.items():
            person = self.people.get(person_id)
            if person:
                dept = getattr(person, "department", None)
                if dept:
                    dept_centrality[dept].append(score)

        if not dept_centrality:
            return 0.5

        # Calculate average centrality per department
        dept_avg_centrality = {
            dept: sum(scores) / len(scores) for dept, scores in dept_centrality.items()
        }

        # Calculate Gini coefficient (measure of inequality)
        # Higher Gini = more inequality = lower power balance
        values = list(dept_avg_centrality.values())
        if not values:
            return 0.5

        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = 0
        for i, value in enumerate(sorted_values, 1):
            cumsum += value * (2 * i - n - 1)

        gini = cumsum / (n * sum(sorted_values)) if sum(sorted_values) > 0 else 0.0

        # Power balance = 1 - Gini (inverse relationship)
        power_balance = max(0.0, 1.0 - gini)

        return min(power_balance, 1.0)

    def _compute_combined_score(
        self,
        perspective_diversity: float,
        inclusion: float,
        dissent_score: float,
        power_balance: float,
    ) -> float:
        """Compute combined diversity and inclusion score.

        Args:
            perspective_diversity: Perspective diversity score
            inclusion: Inclusion in decision-making score
            dissent_score: Dissent encouragement score
            power_balance: Power balance index

        Returns:
            Combined score between 0 and 1
        """
        # Weighted combination
        combined = (
            perspective_diversity * 0.3
            + inclusion * 0.3
            + dissent_score * 0.25
            + power_balance * 0.15
        )

        return min(combined, 1.0)

    def get_top_inclusive_leaders(self, top_n: int = 20) -> pd.DataFrame:
        """Get top inclusive leaders by diversity and inclusion score.

        Args:
            top_n: Number of top leaders to return

        Returns:
            DataFrame with top leaders sorted by diversity_inclusion_score
        """
        results_df = self.compute()

        if results_df.empty:
            return pd.DataFrame()

        top_leaders = results_df.nlargest(top_n, "diversity_inclusion_score").reset_index(drop=True)

        return top_leaders
