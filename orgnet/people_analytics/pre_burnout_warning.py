"""Pre-Burnout Warning System.

This module transforms anomaly detection (overload/isolation) into a commercial
Pre-Burnout Warning System that flags individuals or teams experiencing early
warning signs before burnout occurs.
"""

from datetime import datetime

import networkx as nx
import pandas as pd

from orgnet.data.models import Person
from orgnet.metrics.anomaly import AnomalyDetector
from orgnet.people_analytics.burnout import BurnoutPredictor
from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


class PreBurnoutWarningSystem:
    """Pre-Burnout Warning System for early intervention.

    Combines anomaly detection (overload/isolation) with burnout prediction
    to flag individuals or teams experiencing early warning signs before
    burnout occurs. Marketed as a "Pre-Burnout Warning System" for HR teams.
    """

    def __init__(
        self,
        graph: nx.Graph,
        events: pd.DataFrame,
        people: list[Person] | None = None,
        warning_threshold: float = 0.6,
        **params,
    ):
        """Initialize pre-burnout warning system.

        Args:
            graph: Organizational network graph
            events: Event table with temporal data
            people: Optional list of Person objects
            warning_threshold: Burnout risk threshold for warnings (default: 0.6)
        """
        self.graph = graph
        self.events = events
        self.people = {p.id: p for p in (people or [])} if people else {}
        self.warning_threshold = warning_threshold

        # Initialize components
        self.anomaly_detector = AnomalyDetector(graph, people=list(self.people.values()))
        self.burnout_predictor = BurnoutPredictor()

    def compute(
        self, graph: nx.Graph | None = None, events: pd.DataFrame | None = None, **kwargs
    ) -> pd.DataFrame:
        """Compute pre-burnout warnings for all individuals.

        Args:
            graph: Optional graph (uses self.graph if None)
            events: Optional events (uses self.events if None)

        Returns:
            DataFrame with pre-burnout warnings
        """
        if graph is not None:
            self.graph = graph
            self.anomaly_detector.graph = graph

        if events is not None:
            self.events = events

        # Fit burnout predictor
        self.burnout_predictor.fit(self.events)

        # Detect anomalies (overload/isolation)
        anomalies = self.anomaly_detector.detect_node_anomalies()

        # Predict burnout risk
        burnout_df = self.burnout_predictor.predict(self.events, self.graph)

        warnings = self._generate_warnings(anomalies, burnout_df)

        return warnings

    def _generate_warnings(self, anomalies: pd.DataFrame, burnout_df: pd.DataFrame) -> pd.DataFrame:
        """Generate pre-burnout warnings from anomalies and burnout predictions.

        Args:
            anomalies: DataFrame with anomaly detections
            burnout_df: DataFrame with burnout predictions

        Returns:
            DataFrame with pre-burnout warnings
        """
        if anomalies.empty and burnout_df.empty:
            return pd.DataFrame()

        # Start with burnout predictions
        warnings_list = []

        # Process burnout predictions
        if not burnout_df.empty:
            for _, row in burnout_df.iterrows():
                person_id = row["person_id"]
                burnout_score = row["burnout_score"]
                risk_level = row["risk_level"]

                if burnout_score >= self.warning_threshold:
                    # Get anomaly info for this person
                    person_anomalies = (
                        anomalies[anomalies["node_id"] == person_id]
                        if not anomalies.empty and "node_id" in anomalies.columns
                        else pd.DataFrame()
                    )

                    warning_type = self._determine_warning_type(person_anomalies, burnout_score)

                    severity = self._calculate_warning_severity(burnout_score, person_anomalies)

                    description = self._generate_warning_description(
                        person_id, burnout_score, risk_level, person_anomalies, warning_type
                    )

                    # Generate recommended actions
                    recommended_actions = self._generate_recommended_actions(
                        warning_type, severity, person_anomalies
                    )

                    warnings_list.append(
                        {
                            "person_id": person_id,
                            "warning_type": warning_type,
                            "severity": severity,
                            "burnout_score": burnout_score,
                            "risk_level": risk_level,
                            "anomaly_types": (
                                person_anomalies["type"].unique().tolist()
                                if not person_anomalies.empty and "type" in person_anomalies.columns
                                else []
                            ),
                            "overload_detected": (
                                "overload" in person_anomalies["type"].values
                                if not person_anomalies.empty and "type" in person_anomalies.columns
                                else False
                            ),
                            "isolation_detected": (
                                "isolation" in person_anomalies["type"].values
                                if not person_anomalies.empty and "type" in person_anomalies.columns
                                else False
                            ),
                            "description": description,
                            "recommended_actions": recommended_actions,
                            "urgency": (
                                "high"
                                if severity == "critical"
                                else ("medium" if severity == "high" else "low")
                            ),
                            "detection_date": datetime.now().isoformat(),
                        }
                    )

        return pd.DataFrame(warnings_list)

    def _determine_warning_type(self, anomalies: pd.DataFrame, burnout_score: float) -> str:
        """Determine warning type based on anomalies and burnout score.

        Args:
            anomalies: Person's anomalies
            burnout_score: Burnout risk score

        Returns:
            Warning type: 'overload_warning', 'isolation_warning', 'early_burnout', 'combined'
        """
        if anomalies.empty:
            if burnout_score >= 0.8:
                return "early_burnout"
            return "burnout_risk"

        anomaly_types = anomalies["type"].unique().tolist() if "type" in anomalies.columns else []

        if "overload" in anomaly_types and "isolation" in anomaly_types:
            return "combined"
        elif "overload" in anomaly_types:
            return "overload_warning"
        elif "isolation" in anomaly_types:
            return "isolation_warning"
        else:
            return "early_burnout"

    def _calculate_warning_severity(self, burnout_score: float, anomalies: pd.DataFrame) -> str:
        """Calculate warning severity.

        Args:
            burnout_score: Burnout risk score
            anomalies: Person's anomalies

        Returns:
            Severity: 'critical', 'high', 'medium', 'low'
        """
        # Base severity from burnout score
        if burnout_score >= 0.8:
            base_severity = "critical"
        elif burnout_score >= 0.7:
            base_severity = "high"
        elif burnout_score >= 0.6:
            base_severity = "medium"
        else:
            base_severity = "low"

        # Boost severity if multiple anomaly types
        if not anomalies.empty:
            anomaly_count = len(anomalies)
            if anomaly_count > 1:
                if base_severity == "low":
                    return "medium"
                elif base_severity == "medium":
                    return "high"
                elif base_severity == "high":
                    return "critical"

        return base_severity

    def _generate_warning_description(
        self,
        person_id: str,
        burnout_score: float,
        risk_level: str,
        anomalies: pd.DataFrame,
        warning_type: str,
    ) -> str:
        """Generate natural language warning description.

        Args:
            person_id: Person identifier
            burnout_score: Burnout risk score
            risk_level: Risk level classification
            anomalies: Person's anomalies
            warning_type: Warning type

        Returns:
            Natural language description
        """
        person_name = (
            self.people[person_id].name
            if person_id in self.people and hasattr(self.people[person_id], "name")
            else person_id
        )

        # Base description
        descriptions = {
            "overload_warning": (
                f"{person_name} is showing signs of overload (burnout risk: {burnout_score:.1%}). "
                "High connectivity and message volume detected. Immediate attention recommended."
            ),
            "isolation_warning": (
                f"{person_name} is showing signs of isolation (burnout risk: {burnout_score:.1%}). "
                "Network connectivity has dropped significantly. Support and connection-building recommended."
            ),
            "early_burnout": (
                f"{person_name} is at {risk_level} risk of burnout (score: {burnout_score:.1%}). "
                "Early intervention recommended to prevent turnover."
            ),
            "combined": (
                f"{person_name} is showing both overload and isolation patterns "
                f"(burnout risk: {burnout_score:.1%}). This combination is particularly concerning. "
                "Immediate intervention strongly recommended."
            ),
            "burnout_risk": (
                f"{person_name} is at {risk_level} burnout risk (score: {burnout_score:.1%}). "
                "Preventive measures recommended."
            ),
        }

        description = descriptions.get(warning_type, descriptions["burnout_risk"])

        # Add specific anomaly details
        if not anomalies.empty and "description" in anomalies.columns:
            anomaly_descriptions = anomalies["description"].unique().tolist()[:2]  # Max 2
            if anomaly_descriptions:
                description += " " + " ".join(anomaly_descriptions)

        return description

    def _generate_recommended_actions(
        self, warning_type: str, severity: str, anomalies: pd.DataFrame
    ) -> list[str]:
        """Generate recommended actions based on warning type and severity.

        Args:
            warning_type: Warning type
            severity: Warning severity
            anomalies: Person's anomalies

        Returns:
            List of recommended action strings
        """
        actions_by_type = {
            "overload_warning": [
                "Reduce workload or redistribute responsibilities",
                "Schedule regular 1-on-1s to assess capacity",
                "Consider temporary work reduction or time off",
                "Review and prioritize tasks to eliminate non-essential work",
            ],
            "isolation_warning": [
                "Facilitate connections with key team members",
                "Assign onboarding buddy or mentor",
                "Schedule team-building activities",
                "Check in regularly to assess social integration",
            ],
            "early_burnout": [
                "Schedule wellness check-in with manager",
                "Review work-life balance metrics",
                "Consider workload adjustment",
                "Provide access to employee assistance programs",
            ],
            "combined": [
                "IMMEDIATE: Schedule urgent manager meeting",
                "IMMEDIATE: Assess current workload and redistribute",
                "IMMEDIATE: Facilitate team connections",
                "Consider temporary work reduction",
                "Provide mental health resources",
                "Schedule regular check-ins",
            ],
            "burnout_risk": [
                "Monitor work-life balance metrics",
                "Schedule regular check-ins",
                "Review workload distribution",
            ],
        }

        base_actions = actions_by_type.get(warning_type, actions_by_type["burnout_risk"])

        # Filter by severity
        if severity == "critical":
            return base_actions[:4]  # Top 4 for critical
        elif severity == "high":
            return base_actions[:3]  # Top 3 for high
        else:
            return base_actions[:2]  # Top 2 for medium/low

    def generate_warning_report(self, top_n: int = 20) -> dict:
        """Generate comprehensive pre-burnout warning report.

        Args:
            top_n: Number of top warnings to include

        Returns:
            Dictionary with warning report
        """
        warnings_df = self.compute()

        if warnings_df.empty:
            return {
                "summary": "No pre-burnout warnings detected",
                "total_warnings": 0,
                "critical_count": 0,
                "high_count": 0,
            }

        # Summary statistics
        total_warnings = len(warnings_df)
        critical = warnings_df[warnings_df["severity"] == "critical"]
        high = warnings_df[warnings_df["severity"] == "high"]
        medium = warnings_df[warnings_df["severity"] == "medium"]

        warning_type_dist = (
            warnings_df["warning_type"].value_counts().to_dict()
            if "warning_type" in warnings_df.columns
            else {}
        )

        top_warnings = warnings_df.nlargest(top_n, "burnout_score")[
            [
                "person_id",
                "warning_type",
                "severity",
                "burnout_score",
                "risk_level",
                "description",
                "recommended_actions",
                "urgency",
            ]
        ].to_dict("records")

        return {
            "summary": (
                f"Detected {total_warnings} pre-burnout warnings: "
                f"{len(critical)} critical, {len(high)} high, {len(medium)} medium"
            ),
            "total_warnings": total_warnings,
            "severity_distribution": {
                "critical": len(critical),
                "high": len(high),
                "medium": len(medium),
                "low": len(warnings_df[warnings_df["severity"] == "low"]),
            },
            "warning_type_distribution": warning_type_dist,
            "top_warnings": top_warnings,
            "urgent_actions_required": len(warnings_df[warnings_df["urgency"] == "high"]),
        }

    def get_team_warnings(self, team_attribute: str = "team") -> pd.DataFrame:
        """Get pre-burnout warnings grouped by team.

        Args:
            team_attribute: Attribute name to group by

        Returns:
            DataFrame with team-level warnings
        """
        warnings_df = self.compute()

        if warnings_df.empty or not self.people:
            return pd.DataFrame()

        # Add team information
        warnings_df["team"] = warnings_df["person_id"].apply(
            lambda pid: (
                getattr(self.people[pid], team_attribute, "Unknown")
                if pid in self.people
                else "Unknown"
            )
        )

        # Aggregate by team
        team_warnings = (
            warnings_df.groupby("team")
            .agg(
                {
                    "person_id": "count",
                    "burnout_score": "mean",
                    "severity": lambda x: (x == "critical").sum(),
                }
            )
            .rename(
                columns={
                    "person_id": "warning_count",
                    "burnout_score": "avg_burnout_score",
                    "severity": "critical_count",
                }
            )
            .reset_index()
        )

        team_warnings = team_warnings.sort_values("warning_count", ascending=False)

        return team_warnings
