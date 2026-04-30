"""Talent activation through stretch assignments and cross-functional exposure.

This module measures talent activation:
- Cross-functional exposure (percentage outside primary function)
- Stretch assignment frequency (projects outside comfort zone)
- Rotation participation (rotation programs)
- Functional diversity index (diversity of departments/teams)
"""

import networkx as nx
import pandas as pd

from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


class TalentActivationAnalyzer:
    """Measures talent activation through stretch assignments and cross-functional exposure.

    Tracks:
    - Cross-functional exposure (by time, by projects)
    - Stretch assignment participation
    - Functional diversity
    - Career velocity (internal transfers, promotions)
    """

    def __init__(
        self,
        graph: nx.Graph,
        projects: list[dict] | None = None,
        interactions: list[dict] | None = None,
        code_commits: list[dict] | None = None,
        people: list | None = None,
        hr_data: dict | None = None,
        **params,
    ):
        """Initialize talent activation analyzer.

        Args:
            graph: Organizational network graph
            projects: List of project dicts with 'project_id', 'members', 'department',
                     'start_date', 'end_date'
            interactions: List of interaction dicts with 'sender_id', 'receiver_id',
                         'timestamp', 'department' or 'channel'
            code_commits: List of code commit dicts with 'author_id', 'repo', 'timestamp'
            people: List of Person objects with 'id', 'department', 'role'
            hr_data: Optional HR data dict with 'rotations', 'transfers', 'promotions'
                     (each is a list of dicts with 'person_id', 'from', 'to', 'date')
            **params: Additional parameters for BaseMetric
        """
        self.graph = graph
        self.projects = projects or []
        self.interactions = interactions or []
        self.code_commits = code_commits or []
        self.people = {p.id: p for p in (people or [])} if people else {}
        self.hr_data = hr_data or {}

    def compute(
        self, graph: nx.Graph | None = None, nodes: pd.DataFrame | None = None, **kwargs
    ) -> pd.DataFrame:
        """Compute talent activation metrics.

        Args:
            graph: Optional graph (uses self.graph if None)
            nodes: Optional node table (not used but required by BaseMetric)
            **kwargs: Additional arguments

        Returns:
            DataFrame with columns: person_id, cross_functional_exposure_score,
                stretch_assignment_frequency, rotation_participation,
                functional_diversity_index, career_velocity, talent_activation_score
        """
        if graph is None:
            graph = self.graph

        person_ids = list(graph.nodes())
        results = []

        for person_id in person_ids:
            # Cross-functional exposure score
            exposure_score = self._compute_cross_functional_exposure(person_id)

            # Stretch assignment frequency
            stretch_frequency = self._compute_stretch_assignment_frequency(person_id)

            # Rotation participation
            rotation_participation = self._compute_rotation_participation(person_id)

            # Functional diversity index
            functional_diversity = self._compute_functional_diversity(person_id)

            # Career velocity
            career_velocity = self._compute_career_velocity(person_id)

            # Combined talent activation score
            activation_score = self._compute_combined_score(
                exposure_score,
                stretch_frequency,
                rotation_participation,
                functional_diversity,
                career_velocity,
            )

            results.append(
                {
                    "person_id": person_id,
                    "cross_functional_exposure_score": exposure_score,
                    "stretch_assignment_frequency": stretch_frequency,
                    "rotation_participation": rotation_participation,
                    "functional_diversity_index": functional_diversity,
                    "career_velocity": career_velocity,
                    "talent_activation_score": activation_score,
                }
            )

        return pd.DataFrame(results)

    def _compute_cross_functional_exposure(self, person_id: str) -> float:
        """Compute cross-functional exposure score.

        Measures percentage of time/interactions outside primary function.

        Args:
            person_id: Person identifier

        Returns:
            Score between 0 and 1 (higher = more cross-functional exposure)
        """
        person = self.people.get(person_id)
        person_dept = getattr(person, "department", None) if person else None

        if not person_dept:
            return 0.0

        # Count interactions outside primary department
        total_interactions = 0
        cross_functional_interactions = 0

        for interaction in self.interactions:
            sender_id = interaction.get("sender_id")
            receiver_id = interaction.get("receiver_id")

            if sender_id == person_id or receiver_id == person_id:
                total_interactions += 1

                # Check if interaction is cross-functional
                if sender_id == person_id:
                    receiver = self.people.get(receiver_id) if receiver_id else None
                    if receiver:
                        dept = getattr(receiver, "department", None)
                        if dept and dept != person_dept:
                            cross_functional_interactions += 1

                # Also check channel/department metadata
                channel = interaction.get("channel") or interaction.get("department")
                if channel and channel != person_dept:
                    cross_functional_interactions += 1

        # Count projects outside primary department
        total_projects = 0
        cross_functional_projects = 0

        for project in self.projects:
            members = project.get("members", [])
            if person_id in members:
                total_projects += 1

                project_dept = project.get("department")
                if project_dept and project_dept != person_dept:
                    cross_functional_projects += 1

                # Also check if project members are from different departments
                project_depts = set()
                for member_id in members:
                    member = self.people.get(member_id)
                    if member:
                        dept = getattr(member, "department", None)
                        if dept:
                            project_depts.add(dept)

                if len(project_depts) > 1:  # Cross-functional project
                    cross_functional_projects += 1

        # Calculate exposure ratios
        interaction_ratio = (
            cross_functional_interactions / max(total_interactions, 1)
            if total_interactions > 0
            else 0.0
        )
        project_ratio = (
            cross_functional_projects / max(total_projects, 1) if total_projects > 0 else 0.0
        )

        # Combined exposure score
        exposure_score = interaction_ratio * 0.6 + project_ratio * 0.4

        return min(exposure_score, 1.0)

    def _compute_stretch_assignment_frequency(self, person_id: str) -> float:
        """Compute stretch assignment frequency.

        Measures number of projects outside comfort zone.

        Args:
            person_id: Person identifier

        Returns:
            Score between 0 and 1 (higher = more stretch assignments)
        """
        person = self.people.get(person_id)
        person_role = getattr(person, "role", None) if person else None
        person_dept = getattr(person, "department", None) if person else None

        # Count projects person is involved in
        person_projects = [p for p in self.projects if person_id in p.get("members", [])]

        if not person_projects:
            return 0.0

        # Count stretch projects (outside role/dept or high complexity)
        stretch_projects = 0

        for project in person_projects:
            is_stretch = False

            # Check if project is outside person's department
            project_dept = project.get("department")
            if project_dept and project_dept != person_dept:
                is_stretch = True

            # Check if project requires different skills (cross-functional)
            project_depts = set()
            for member_id in project.get("members", []):
                member = self.people.get(member_id)
                if member:
                    dept = getattr(member, "department", None)
                    if dept:
                        project_depts.add(dept)

            if len(project_depts) > 1:  # Cross-functional = stretch
                is_stretch = True

            # Check project complexity/type (if available)
            project_type = project.get("type") or project.get("category", "").lower()
            if any(
                keyword in project_type
                for keyword in ["innovation", "research", "explore", "experiment", "pilot"]
            ):
                is_stretch = True

            if is_stretch:
                stretch_projects += 1

        # Stretch frequency = ratio of stretch projects
        stretch_frequency = stretch_projects / len(person_projects)

        return min(stretch_frequency, 1.0)

    def _compute_rotation_participation(self, person_id: str) -> float:
        """Compute rotation participation.

        Measures participation in rotation programs from HR data.

        Args:
            person_id: Person identifier

        Returns:
            Score between 0 and 1 (higher = more rotation participation)
        """
        rotations = self.hr_data.get("rotations", [])

        # Count rotations for this person
        person_rotations = [r for r in rotations if r.get("person_id") == person_id]

        # Participation score: 1.0 if any rotations, 0.0 otherwise
        # Could be enhanced with rotation duration/frequency
        participation = 1.0 if person_rotations else 0.0

        # If multiple rotations, normalize by time (if dates available)
        if len(person_rotations) > 1:
            # Could calculate rotation frequency/density here
            pass

        return participation

    def _compute_functional_diversity(self, person_id: str) -> float:
        """Compute functional diversity index.

        Measures diversity of departments/teams worked with.

        Args:
            person_id: Person identifier

        Returns:
            Score between 0 and 1 (higher = more functional diversity)
        """
        person = self.people.get(person_id)
        person_dept = getattr(person, "department", None) if person else None

        # Collect unique departments worked with
        departments_worked_with = set()

        # From projects
        for project in self.projects:
            members = project.get("members", [])
            if person_id in members:
                # Get all departments in project
                for member_id in members:
                    member = self.people.get(member_id)
                    if member:
                        dept = getattr(member, "department", None)
                        if dept:
                            departments_worked_with.add(dept)

                # Also check project department
                project_dept = project.get("department")
                if project_dept:
                    departments_worked_with.add(project_dept)

        # From interactions
        for interaction in self.interactions:
            sender_id = interaction.get("sender_id")
            receiver_id = interaction.get("receiver_id")

            if sender_id == person_id and receiver_id:
                receiver = self.people.get(receiver_id)
                if receiver:
                    dept = getattr(receiver, "department", None)
                    if dept:
                        departments_worked_with.add(dept)

            # Also check channel metadata
            channel = interaction.get("channel") or interaction.get("department")
            if channel:
                departments_worked_with.add(channel)

        # Remove person's own department
        if person_dept and person_dept in departments_worked_with:
            departments_worked_with.remove(person_dept)

        # Calculate diversity score
        num_departments = len(departments_worked_with)
        # Normalize: assume 5+ departments is "high diversity"
        diversity_score = min(num_departments / 5.0, 1.0)

        return diversity_score

    def _compute_career_velocity(self, person_id: str) -> float:
        """Compute career velocity.

        Measures internal transfers, promotions, role changes.

        Args:
            person_id: Person identifier

        Returns:
            Score between 0 and 1 (higher = higher career velocity)
        """
        transfers = self.hr_data.get("transfers", [])
        promotions = self.hr_data.get("promotions", [])

        # Count transfers for this person
        person_transfers = [t for t in transfers if t.get("person_id") == person_id]

        # Count promotions for this person
        person_promotions = [p for p in promotions if p.get("person_id") == person_id]

        # Calculate velocity score
        # Transfers + promotions indicate career movement
        total_movements = len(person_transfers) + len(person_promotions)

        # Normalize: assume 2+ movements is "high velocity"
        velocity_score = min(total_movements / 2.0, 1.0)

        # Could enhance with time-based velocity (movements per year)
        # if dates are available in hr_data

        return min(velocity_score, 1.0)

    def _compute_combined_score(
        self,
        exposure: float,
        stretch: float,
        rotation: float,
        diversity: float,
        velocity: float,
    ) -> float:
        """Compute combined talent activation score.

        Args:
            exposure: Cross-functional exposure score
            stretch: Stretch assignment frequency
            rotation: Rotation participation
            diversity: Functional diversity index
            velocity: Career velocity

        Returns:
            Combined score between 0 and 1
        """
        # Weighted combination
        combined = (
            exposure * 0.3 + stretch * 0.25 + rotation * 0.15 + diversity * 0.2 + velocity * 0.1
        )

        return min(combined, 1.0)

    def get_top_activated_talent(self, top_n: int = 20) -> pd.DataFrame:
        """Get top activated talent by activation score.

        Args:
            top_n: Number of top talents to return

        Returns:
            DataFrame with top talents sorted by talent_activation_score
        """
        results_df = self.compute()

        if results_df.empty:
            return pd.DataFrame()

        top_talents = results_df.nlargest(top_n, "talent_activation_score").reset_index(drop=True)

        return top_talents
