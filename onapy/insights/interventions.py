"""Intervention framework for organizational improvements."""

import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class InterventionPriority(Enum):
    """Intervention priority levels."""

    QUICK_WIN = "quick_win"
    STRATEGIC = "strategic"
    FILL_IN = "fill_in"
    COMPLEX = "complex"


class InterventionType(Enum):
    """Types of interventions."""

    CONNECTION_RECOMMENDATION = "connection_recommendation"
    MEETING_OPTIMIZATION = "meeting_optimization"
    RECOGNITION = "recognition"
    REORGANIZATION = "reorganization"
    PROCESS_CHANGE = "process_change"
    CROSS_TEAM_INITIATIVE = "cross_team_initiative"
    ONBOARDING_SUPPORT = "onboarding_support"


class Intervention:
    """Represents an organizational intervention."""

    def __init__(
        self,
        finding: str,
        hypothesis: str,
        intervention_type: InterventionType,
        description: str,
        priority: InterventionPriority,
        success_metrics: Optional[Dict] = None,
    ):
        """
        Initialize intervention.

        Args:
            finding: The organizational finding that triggered this intervention
            hypothesis: Hypothesis about why this intervention will work
            intervention_type: Type of intervention
            description: Detailed description of the intervention
            priority: Priority level
            success_metrics: Dictionary of metrics to track success
        """
        self.finding = finding
        self.hypothesis = hypothesis
        self.intervention_type = intervention_type
        self.description = description
        self.priority = priority
        self.success_metrics = success_metrics or {}
        self.created_at = datetime.now()
        self.status = "proposed"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "finding": self.finding,
            "hypothesis": self.hypothesis,
            "type": self.intervention_type.value,
            "description": self.description,
            "priority": self.priority.value,
            "success_metrics": self.success_metrics,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
        }


class InterventionFramework:
    """Framework for creating and managing organizational interventions."""

    def __init__(self):
        """Initialize intervention framework."""
        self.interventions: List[Intervention] = []

    def create_intervention(
        self,
        finding: str,
        hypothesis: str,
        intervention_type: InterventionType,
        description: str,
        impact: str = "medium",
        effort: str = "medium",
        success_metrics: Optional[Dict] = None,
    ) -> Intervention:
        """
        Create a new intervention following FINDING → HYPOTHESIS → INTERVENTION → MEASUREMENT.

        Args:
            finding: Organizational finding
            hypothesis: Hypothesis about the cause
            intervention_type: Type of intervention
            description: Intervention description
            impact: Expected impact level (high, medium, low)
            effort: Required effort level (high, medium, low)
            success_metrics: Metrics to measure success

        Returns:
            Intervention object
        """
        # Determine priority based on impact and effort
        priority = self._determine_priority(impact, effort)

        intervention = Intervention(
            finding=finding,
            hypothesis=hypothesis,
            intervention_type=intervention_type,
            description=description,
            priority=priority,
            success_metrics=success_metrics,
        )

        self.interventions.append(intervention)
        return intervention

    def _determine_priority(self, impact: str, effort: str) -> InterventionPriority:
        """
        Determine priority using impact-effort matrix.

        High Impact + Low Effort = Quick Win
        High Impact + High Effort = Strategic
        Low Impact + Low Effort = Fill-In
        Low Impact + High Effort = Complex
        """
        impact_high = impact.lower() in ["high", "h"]
        effort_low = effort.lower() in ["low", "l"]

        # Dictionary dispatch for priority determination
        priority_map = {
            (True, True): InterventionPriority.QUICK_WIN,
            (True, False): InterventionPriority.STRATEGIC,
            (False, True): InterventionPriority.FILL_IN,
            (False, False): InterventionPriority.COMPLEX,
        }

        return priority_map[(impact_high, effort_low)]

    def suggest_interventions_from_findings(self, findings: List[Dict]) -> List[Intervention]:
        """
        Automatically suggest interventions based on findings.

        Args:
            findings: List of finding dictionaries with 'type', 'description', etc.

        Returns:
            List of suggested interventions
        """
        suggestions = []

        # Dictionary dispatch for intervention suggestions
        intervention_configs = {
            "silo": {
                "hypothesis": "Lack of structured touchpoints causes disconnect",
                "intervention_type": InterventionType.CROSS_TEAM_INITIATIVE,
                "description": "Establish regular cross-team sync meetings and shared communication channels",
                "impact": "high",
                "effort": "medium",
                "success_metrics": {
                    "cross_boundary_edges": {"target": "+50%", "timeframe": "6 months"},
                    "meeting_co_attendance": {"target": "+25%", "timeframe": "6 months"},
                },
            },
            "isolation": {
                "hypothesis": "Individual needs support or recognition",
                "intervention_type": InterventionType.RECOGNITION,
                "description": "Provide recognition and support, consider workload redistribution",
                "impact": "medium",
                "effort": "low",
                "success_metrics": {
                    "connectivity_change": {"target": "normalize", "timeframe": "3 months"}
                },
            },
            "overload": {
                "hypothesis": "Individual needs support or recognition",
                "intervention_type": InterventionType.RECOGNITION,
                "description": "Provide recognition and support, consider workload redistribution",
                "impact": "medium",
                "effort": "low",
                "success_metrics": {
                    "connectivity_change": {"target": "normalize", "timeframe": "3 months"}
                },
            },
            "missing_connection": {
                "hypothesis": "Missing connection limits collaboration potential",
                "intervention_type": InterventionType.CONNECTION_RECOMMENDATION,
                "description": "Facilitate introduction and initial collaboration opportunity",
                "impact": "medium",
                "effort": "low",
                "success_metrics": {
                    "edge_formation": {"target": "new edge created", "timeframe": "1 month"}
                },
            },
        }

        for finding in findings:
            finding_type = finding.get("type", "")
            description = finding.get("description", "")

            config = intervention_configs.get(finding_type)
            if config:
                intervention = self.create_intervention(finding=description, **config)
                suggestions.append(intervention)

        return suggestions

    def get_interventions_by_priority(
        self, priority: Optional[InterventionPriority] = None
    ) -> List[Intervention]:
        """
        Get interventions filtered by priority.

        Args:
            priority: Optional priority filter

        Returns:
            List of interventions
        """
        if priority is None:
            return self.interventions

        return [i for i in self.interventions if i.priority == priority]

    def get_interventions_dataframe(self) -> pd.DataFrame:
        """
        Get all interventions as DataFrame.

        Returns:
            DataFrame with interventions
        """
        if not self.interventions:
            return pd.DataFrame()

        return pd.DataFrame([i.to_dict() for i in self.interventions])

    def prioritize_interventions(self) -> pd.DataFrame:
        """
        Prioritize interventions using impact-effort matrix.

        Returns:
            DataFrame with prioritized interventions
        """
        df = self.get_interventions_dataframe()

        if df.empty:
            return df

        # Add priority scores for sorting
        priority_order = {
            InterventionPriority.QUICK_WIN.value: 1,
            InterventionPriority.STRATEGIC.value: 2,
            InterventionPriority.FILL_IN.value: 3,
            InterventionPriority.COMPLEX.value: 4,
        }

        df["priority_score"] = df["priority"].map(priority_order)
        df = df.sort_values("priority_score")

        return df
