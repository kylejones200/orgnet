"""Management Playbook for organizational change tracking.

This module transforms the Intervention Framework into a "Management Playbook"
where leaders track if specific changes (e.g., "No-Meeting Wednesdays") actually
improved network density or reduced overload over time. This allows companies
to treat organizational change like a scientific experiment.
"""

from datetime import datetime
from enum import Enum

import pandas as pd

from orgnet.insights.interventions import (
    Intervention,
    InterventionFramework,
    InterventionType,
)
from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


class InterventionTemplate(Enum):
    """Templates for common organizational interventions."""

    NO_MEETING_WEDNESDAYS = "no_meeting_wednesdays"
    CROSS_TEAM_INITIATIVE = "cross_team_initiative"
    MENTORSHIP_PROGRAM = "mentorship_program"
    WORKLOAD_REDISTRIBUTION = "workload_redistribution"
    FOCUS_TIME_BLOCKS = "focus_time_blocks"
    ASYNC_COMMUNICATION = "async_communication"
    TEAM_BUILDING = "team_building"
    RECOGNITION_PROGRAM = "recognition_program"


class ManagementPlaybook(InterventionFramework):
    """Management Playbook for organizational change tracking.

    Transforms the Intervention Framework into a "Management Playbook" where
    leaders track if specific changes actually improved metrics over time.
    This allows companies to treat organizational change like a scientific experiment.
    """

    def __init__(self):
        """Initialize management playbook."""
        super().__init__()
        self.intervention_templates: dict[str, dict] = self._initialize_templates()

    def _initialize_templates(self) -> dict[str, dict]:
        """Initialize intervention templates for common changes.

        Returns:
            Dictionary mapping template names to intervention configurations
        """
        return {
            InterventionTemplate.NO_MEETING_WEDNESDAYS.value: {
                "hypothesis": (
                    "Reducing meeting overhead on Wednesdays will improve code productivity "
                    "and reduce coordination overhead, as measured by network density and code commits"
                ),
                "intervention_type": InterventionType.PROCESS_CHANGE,
                "description": (
                    "Implement 'No-Meeting Wednesdays' policy to protect focus time and "
                    "reduce meeting overhead. Track impact on network density, code commits, "
                    "and coordination overhead."
                ),
                "impact": "high",
                "effort": "low",
                "success_metrics": {
                    "network_density": {"target": "+10%", "timeframe": "3 months"},
                    "code_commits": {"target": "+20%", "timeframe": "3 months"},
                    "coordination_overhead": {"target": "-15%", "timeframe": "3 months"},
                    "meeting_hours_per_week": {"target": "-20%", "timeframe": "3 months"},
                },
            },
            InterventionTemplate.CROSS_TEAM_INITIATIVE.value: {
                "hypothesis": (
                    "Regular cross-team sync meetings will increase network density and "
                    "reduce silos, as measured by cross-boundary edges and modularity"
                ),
                "intervention_type": InterventionType.CROSS_TEAM_INITIATIVE,
                "description": (
                    "Establish regular cross-team sync meetings and shared communication "
                    "channels. Track impact on cross-boundary connections and modularity."
                ),
                "impact": "high",
                "effort": "medium",
                "success_metrics": {
                    "cross_boundary_edges": {"target": "+50%", "timeframe": "6 months"},
                    "modularity": {"target": "-10%", "timeframe": "6 months"},
                    "meeting_co_attendance": {"target": "+25%", "timeframe": "6 months"},
                },
            },
            InterventionTemplate.MENTORSHIP_PROGRAM.value: {
                "hypothesis": (
                    "Structured mentorship will improve onboarding success and network "
                    "integration, as measured by integration scores and tenure retention"
                ),
                "intervention_type": InterventionType.ONBOARDING_SUPPORT,
                "description": (
                    "Establish structured mentorship program pairing new hires with "
                    "experienced team members. Track impact on onboarding integration and retention."
                ),
                "impact": "high",
                "effort": "medium",
                "success_metrics": {
                    "onboarding_integration_score": {"target": "+30%", "timeframe": "6 months"},
                    "retention_rate": {"target": "+15%", "timeframe": "12 months"},
                    "network_connections": {"target": "+25%", "timeframe": "6 months"},
                },
            },
            InterventionTemplate.WORKLOAD_REDISTRIBUTION.value: {
                "hypothesis": (
                    "Redistributing workload from overloaded individuals will reduce "
                    "burnout risk and improve network balance, as measured by overload metrics"
                ),
                "intervention_type": InterventionType.PROCESS_CHANGE,
                "description": (
                    "Redistribute workload from overloaded team members to balance "
                    "responsibilities. Track impact on burnout risk and network balance."
                ),
                "impact": "high",
                "effort": "high",
                "success_metrics": {
                    "burnout_risk": {"target": "-20%", "timeframe": "3 months"},
                    "overload_count": {"target": "-30%", "timeframe": "3 months"},
                    "network_balance": {"target": "+15%", "timeframe": "6 months"},
                },
            },
            InterventionTemplate.FOCUS_TIME_BLOCKS.value: {
                "hypothesis": (
                    "Protected focus time blocks will improve code productivity and "
                    "reduce meeting overhead, as measured by code commits and calendar hours"
                ),
                "intervention_type": InterventionType.PROCESS_CHANGE,
                "description": (
                    "Implement protected focus time blocks (e.g., 2-4 PM daily) for "
                    "deep work. Track impact on code productivity and meeting overhead."
                ),
                "impact": "medium",
                "effort": "low",
                "success_metrics": {
                    "code_commits": {"target": "+15%", "timeframe": "3 months"},
                    "meeting_hours_per_week": {"target": "-10%", "timeframe": "3 months"},
                    "focus_time_hours": {"target": "+5 hours/week", "timeframe": "1 month"},
                },
            },
            InterventionTemplate.ASYNC_COMMUNICATION.value: {
                "hypothesis": (
                    "Shifting to asynchronous communication will reduce meeting overhead "
                    "while maintaining collaboration, as measured by communication patterns"
                ),
                "intervention_type": InterventionType.PROCESS_CHANGE,
                "description": (
                    "Promote asynchronous communication (documentation, async updates) "
                    "over synchronous meetings. Track impact on meeting overhead and collaboration."
                ),
                "impact": "medium",
                "effort": "medium",
                "success_metrics": {
                    "meeting_hours_per_week": {"target": "-25%", "timeframe": "3 months"},
                    "async_documentation": {"target": "+50%", "timeframe": "3 months"},
                    "collaboration_quality": {"target": "maintain", "timeframe": "3 months"},
                },
            },
        }

    def create_intervention_from_template(
        self,
        template: InterventionTemplate,
        finding: str,
        custom_metrics: dict | None = None,
    ) -> Intervention:
        """Create intervention from template (Management Playbook feature).

        This allows leaders to quickly create interventions from proven templates,
        such as "No-Meeting Wednesdays", and track their success over time.

        Args:
            template: Intervention template to use
            finding: Organizational finding that triggered this intervention
            custom_metrics: Optional custom success metrics to override template

        Returns:
            Intervention object
        """
        template_config = self.intervention_templates.get(template.value)

        if not template_config:
            raise ValueError(f"Unknown intervention template: {template.value}")

        # Merge custom metrics if provided
        success_metrics = template_config.get("success_metrics", {})
        if custom_metrics:
            success_metrics.update(custom_metrics)

        # Create intervention from template
        intervention = self.create_intervention(
            finding=finding,
            hypothesis=template_config["hypothesis"],
            intervention_type=template_config["intervention_type"],
            description=template_config["description"],
            impact=template_config["impact"],
            effort=template_config["effort"],
            success_metrics=success_metrics,
        )

        # Add template metadata
        intervention.template_name = template.value
        intervention.template_config = template_config

        logger.info(f"Created intervention from template: {template.value}")

        return intervention

    def track_intervention_experiment(
        self,
        intervention: Intervention,
        baseline_metrics: dict,
        period_metrics: dict,
        measurement_date: datetime | None = None,
    ) -> dict:
        """Track intervention as a scientific experiment over time.

        This allows companies to treat organizational change like a scientific
        experiment, measuring before/after metrics and tracking success over time.

        Args:
            intervention: Intervention to track
            baseline_metrics: Metrics before intervention (baseline)
            period_metrics: Metrics after intervention period
            measurement_date: Optional measurement date (defaults to now)

        Returns:
            Dictionary with experiment results
        """
        if measurement_date is None:
            measurement_date = datetime.now()

        # Record baseline if not already recorded
        if not intervention.before_metrics:
            intervention.record_before_metrics(baseline_metrics)

        # Record period metrics
        intervention.record_after_metrics(period_metrics)

        # Record periodic measurement
        intervention.record_measurement(period_metrics, measurement_type="periodic")

        # Evaluate outcome
        intervention._evaluate_outcome()

        # Generate experiment summary
        experiment_summary = self._generate_experiment_summary(
            intervention, baseline_metrics, period_metrics
        )

        logger.info(
            f"Tracked intervention experiment: {intervention.finding[:50]}... "
            f"Outcome: {intervention.outcome}"
        )

        return experiment_summary

    def _generate_experiment_summary(
        self, intervention: Intervention, baseline: dict, period: dict
    ) -> dict:
        """Generate experiment summary with scientific interpretation.

        Args:
            intervention: Intervention object
            baseline: Baseline metrics
            period: Period metrics

        Returns:
            Dictionary with experiment summary
        """
        # Compare metrics
        metric_changes = {}
        for metric_name, target_spec in intervention.success_metrics.items():
            baseline_value = baseline.get(metric_name, 0.0)
            period_value = period.get(metric_name, 0.0)

            if baseline_value == 0:
                change_pct = 1.0 if period_value > 0 else 0.0
            else:
                change_pct = ((period_value - baseline_value) / abs(baseline_value)) * 100

            # Parse target (e.g., "+50%", "-15%")
            target = target_spec.get("target", "")
            target_met = self._check_target_achieved_change(baseline_value, period_value, target)

            metric_changes[metric_name] = {
                "baseline": baseline_value,
                "period": period_value,
                "change": period_value - baseline_value,
                "change_pct": change_pct,
                "target": target,
                "target_met": target_met,
            }

        # Overall experiment result
        targets_met = sum(1 for m in metric_changes.values() if m["target_met"])
        total_targets = len(metric_changes)
        success_rate = targets_met / total_targets if total_targets > 0 else 0.0

        # Scientific interpretation
        if success_rate >= 0.8:
            interpretation = (
                f"STRONG EVIDENCE: Intervention shows significant improvement. "
                f"{targets_met}/{total_targets} success metrics achieved."
            )
        elif success_rate >= 0.5:
            interpretation = (
                f"MODERATE EVIDENCE: Intervention shows partial improvement. "
                f"{targets_met}/{total_targets} success metrics achieved."
            )
        else:
            interpretation = (
                f"WEAK EVIDENCE: Intervention shows limited improvement. "
                f"{targets_met}/{total_targets} success metrics achieved. "
                "Consider adjusting intervention or extending measurement period."
            )

        return {
            "intervention_id": id(intervention),
            "intervention_finding": intervention.finding,
            "intervention_description": intervention.description,
            "template": getattr(intervention, "template_name", None),
            "baseline_metrics": baseline,
            "period_metrics": period,
            "metric_changes": metric_changes,
            "success_rate": success_rate,
            "targets_met": targets_met,
            "total_targets": total_targets,
            "outcome": intervention.outcome,
            "interpretation": interpretation,
            "recommendation": (
                "Continue intervention"
                if success_rate >= 0.7
                else ("Modify intervention" if success_rate >= 0.5 else "Discontinue intervention")
            ),
        }

    def _check_target_achieved_change(
        self, before_value: float, after_value: float, target: str
    ) -> bool:
        """Check if target change was achieved.

        Args:
            before_value: Metric value before intervention
            after_value: Metric value after intervention
            target: Target specification (e.g., "+50%", "-15%")

        Returns:
            True if target was achieved, False otherwise
        """
        if before_value == 0:
            return after_value > 0

        change_pct = ((after_value - before_value) / abs(before_value)) * 100

        # Parse percentage targets (e.g., "+50%", "-15%")
        if "+" in target and "%" in target:
            target_pct = float(target.replace("+", "").replace("%", ""))
            return change_pct >= target_pct
        elif "-" in target and "%" in target:
            target_pct = float(target.replace("-", "").replace("%", ""))
            return change_pct <= -target_pct  # Negative change expected
        elif "maintain" in target.lower():
            # Maintain means change should be small
            return abs(change_pct) <= 10.0

        # Default: any improvement
        return after_value > before_value

    def generate_playbook_dashboard(self, include_templates: bool = True) -> dict:
        """Generate comprehensive Management Playbook dashboard.

        This creates a dashboard view of all interventions, experiments, and
        their outcomes, allowing leaders to track organizational change over time.

        Args:
            include_templates: If True, include intervention templates in dashboard

        Returns:
            Dictionary with playbook dashboard data
        """
        # Get all interventions
        interventions_df = self.get_interventions_dataframe()

        if interventions_df.empty:
            return {
                "summary": "No interventions in playbook",
                "total_interventions": 0,
                "active_interventions": 0,
                "completed_interventions": 0,
            }

        # Summary statistics
        total_interventions = len(interventions_df)
        completed = interventions_df[interventions_df["status"] == "completed"]
        active = interventions_df[interventions_df["status"].isin(["proposed", "active"])]

        # Outcomes
        if not completed.empty and "outcome" in completed.columns:
            outcomes = completed["outcome"].value_counts().to_dict()
            success_count = outcomes.get("success", 0)
            partial_count = outcomes.get("partial", 0)
            failure_count = outcomes.get("failure", 0)
            success_rate = success_count / len(completed) if len(completed) > 0 else 0.0
        else:
            outcomes = {}
            success_count = 0
            partial_count = 0
            failure_count = 0
            success_rate = 0.0

        # Template usage
        template_usage = {}
        if include_templates and "template_name" in interventions_df.columns:
            template_usage = interventions_df["template_name"].value_counts().to_dict()

        # Top performing interventions (by ROI if available)
        top_performers = []
        if "roi" in completed.columns:
            top_performers = (
                completed.nlargest(5, "roi")[["finding", "type", "outcome", "roi"]].to_dict(
                    "records"
                )
                if not completed.empty
                else []
            )

        return {
            "summary": (
                f"Management Playbook: {total_interventions} interventions, "
                f"{len(completed)} completed, {success_rate:.1%} success rate"
            ),
            "total_interventions": total_interventions,
            "active_interventions": len(active),
            "completed_interventions": len(completed),
            "outcomes": outcomes,
            "success_rate": success_rate,
            "success_count": success_count,
            "partial_count": partial_count,
            "failure_count": failure_count,
            "template_usage": template_usage if include_templates else {},
            "available_templates": list(self.intervention_templates.keys()),
            "top_performers": top_performers,
            "recommendations": self._generate_playbook_recommendations(
                interventions_df, outcomes, success_rate
            ),
        }

    def _generate_playbook_recommendations(
        self,
        interventions_df: pd.DataFrame,
        outcomes: dict,
        success_rate: float,
    ) -> list[dict]:
        """Generate recommendations based on playbook analysis.

        Args:
            interventions_df: All interventions DataFrame
            outcomes: Outcome distribution
            success_rate: Overall success rate

        Returns:
            List of recommendation dictionaries
        """
        recommendations = []

        # Success rate recommendations
        if success_rate < 0.6:
            recommendations.append(
                {
                    "priority": "high",
                    "category": "playbook_effectiveness",
                    "title": "Improve Intervention Success Rate",
                    "description": (
                        f"Current success rate ({success_rate:.1%}) is below target (60%). "
                        "Review failed interventions and adjust templates or measurement criteria."
                    ),
                    "action": "Analyze failed interventions and update templates",
                }
            )

        # Template recommendations
        if "template_name" in interventions_df.columns:
            template_success = {}
            completed = interventions_df[interventions_df["status"] == "completed"]
            if not completed.empty and "outcome" in completed.columns:
                for template in completed["template_name"].unique():
                    template_interventions = completed[completed["template_name"] == template]
                    template_success_rate = (
                        len(template_interventions[template_interventions["outcome"] == "success"])
                        / len(template_interventions)
                        if len(template_interventions) > 0
                        else 0.0
                    )
                    template_success[template] = template_success_rate

                # Recommend high-performing templates
                best_template = (
                    max(template_success.items(), key=lambda x: x[1]) if template_success else None
                )
                if best_template and best_template[1] >= 0.8:
                    recommendations.append(
                        {
                            "priority": "medium",
                            "category": "best_practices",
                            "title": f"Replicate Success: {best_template[0]}",
                            "description": (
                                f"Template '{best_template[0]}' has {best_template[1]:.1%} success rate. "
                                "Consider applying this template to similar findings."
                            ),
                            "action": f"Use {best_template[0]} template for similar interventions",
                        }
                    )

        return recommendations

    def get_template_catalog(self) -> dict:
        """Get catalog of available intervention templates.

        Returns:
            Dictionary with template catalog
        """
        catalog = {}
        for template_name, config in self.intervention_templates.items():
            catalog[template_name] = {
                "hypothesis": config["hypothesis"],
                "description": config["description"],
                "intervention_type": (
                    config["intervention_type"].value
                    if isinstance(config["intervention_type"], Enum)
                    else config["intervention_type"]
                ),
                "impact": config["impact"],
                "effort": config["effort"],
                "success_metrics": config["success_metrics"],
                "use_cases": self._get_template_use_cases(template_name),
            }

        return {
            "summary": f"Management Playbook: {len(self.intervention_templates)} intervention templates",
            "templates": catalog,
            "total_templates": len(self.intervention_templates),
        }

    def _get_template_use_cases(self, template_name: str) -> list[str]:
        """Get use cases for a template.

        Args:
            template_name: Template name

        Returns:
            List of use case strings
        """
        use_cases_map = {
            InterventionTemplate.NO_MEETING_WEDNESDAYS.value: [
                "High coordination overhead detected",
                "Low code productivity with high meeting time",
                "Teams reporting meeting fatigue",
            ],
            InterventionTemplate.CROSS_TEAM_INITIATIVE.value: [
                "Silo detection between departments",
                "Low cross-boundary collaboration",
                "High modularity indicating isolation",
            ],
            InterventionTemplate.MENTORSHIP_PROGRAM.value: [
                "Low onboarding integration scores",
                "High new hire turnover",
                "Slow time-to-productivity for new hires",
            ],
            InterventionTemplate.WORKLOAD_REDISTRIBUTION.value: [
                "Overload anomalies detected",
                "High burnout risk identified",
                "Network imbalance with central overload",
            ],
            InterventionTemplate.FOCUS_TIME_BLOCKS.value: [
                "High meeting overhead",
                "Low deep work time",
                "Interruptions affecting productivity",
            ],
            InterventionTemplate.ASYNC_COMMUNICATION.value: [
                "High meeting overhead",
                "Communication fragmentation",
                "Need for better documentation",
            ],
        }

        return use_cases_map.get(template_name, [])
