"""Onboarding ROI Dashboard.

This module creates an Onboarding ROI Dashboard that shows leadership which
departments successfully "plug in" new talent the fastest and quantifies the
ROI of onboarding programs.
"""

from datetime import datetime, timedelta

import pandas as pd

from orgnet.data.models import Person
from orgnet.graph.temporal import TemporalGraph
from orgnet.temporal.new_hire_integration import NewHireIntegrationTracker
from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


class OnboardingROIDashboard:
    """Onboarding ROI Dashboard for leadership insights.

    Transforms onboarding integration tracking into a comprehensive dashboard
    that shows which departments successfully "plug in" new talent the fastest
    and quantifies the ROI of onboarding programs.
    """

    def __init__(
        self,
        temporal_graph: TemporalGraph,
        people: list[Person],
        interactions: list | None = None,
    ):
        """Initialize onboarding ROI dashboard.

        Args:
            temporal_graph: TemporalGraph instance for building snapshots
            people: List of Person objects
            interactions: List of Interaction objects
        """
        self.temporal_graph = temporal_graph
        self.people = people
        self.interactions = interactions or []
        self.tracker = NewHireIntegrationTracker(temporal_graph, people, interactions)

    def generate_dashboard(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        group_by: str = "department",
    ) -> dict:
        """Generate comprehensive onboarding ROI dashboard.

        Args:
            start_date: Optional start date for cohort analysis
            end_date: Optional end date for cohort analysis
            group_by: Attribute to group by ('department', 'team', etc.)

        Returns:
            Dictionary with onboarding dashboard data
        """
        logger.info("Generating Onboarding ROI Dashboard...")

        # Find all new hires in timeframe
        if start_date is None:
            # Default: last 90 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
        elif end_date is None:
            end_date = datetime.now()

        # Get new hires in timeframe
        cohort_hires = self._get_cohort_hires(start_date, end_date)

        if not cohort_hires:
            return {
                "summary": f"No new hires found in period {start_date.date()} to {end_date.date()}",
                "cohort_size": 0,
                "department_performance": {},
            }

        # Track integration for all cohort hires
        cohort_metrics = []
        for person_id in cohort_hires:
            try:
                integration_df = self.tracker.track_integration(person_id, window_days=90)
                if integration_df.empty:
                    continue

                latest = integration_df.iloc[-1]
                person = self.people.get(person_id) if isinstance(self.people, dict) else None
                if not person:
                    # Try to find in list
                    person = next((p for p in self.people if p.id == person_id), None)

                department = getattr(person, group_by, "Unknown") if person else "Unknown"

                cohort_metrics.append(
                    {
                        "person_id": person_id,
                        group_by: department,
                        "integration_score": latest["integration_score"],
                        "status": latest["status"],
                        "weeks_elapsed": latest["week"],
                        "current_degree": latest["degree"],
                        "integration_timeline": integration_df.to_dict("records"),
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to track integration for {person_id}: {e}")
                continue

        if not cohort_metrics:
            return {
                "summary": "No integration data available for cohort",
                "cohort_size": len(cohort_hires),
            }

        metrics_df = pd.DataFrame(cohort_metrics)

        department_performance = self._analyze_department_performance(metrics_df, group_by)

        # ROI calculation
        roi_analysis = self._calculate_onboarding_roi(metrics_df, group_by)

        speed_analysis = self._analyze_integration_speed(metrics_df, group_by)

        # Success metrics
        success_metrics = self._calculate_success_metrics(metrics_df)

        return {
            "summary": f"Onboarding ROI Dashboard: {len(cohort_hires)} new hires analyzed",
            "cohort_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "cohort_size": len(cohort_hires),
            "success_metrics": success_metrics,
            "department_performance": department_performance,
            "roi_analysis": roi_analysis,
            "speed_analysis": speed_analysis,
            "top_performers": department_performance["top_departments"][:10],
            "bottom_performers": department_performance["bottom_departments"][:10],
            "recommendations": self._generate_dashboard_recommendations(
                department_performance, roi_analysis, speed_analysis
            ),
        }

    def _get_cohort_hires(self, start_date: datetime, end_date: datetime) -> list[str]:
        """Get new hires in timeframe.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of person IDs
        """
        cohort_hires = []

        people_list = self.people.values() if isinstance(self.people, dict) else self.people

        for person in people_list:
            if not hasattr(person, "start_date") or person.start_date is None:
                continue

            if start_date <= person.start_date <= end_date:
                person_id = person.id if hasattr(person, "id") else str(person)
                cohort_hires.append(person_id)

        return cohort_hires

    def _analyze_department_performance(self, metrics_df: pd.DataFrame, group_by: str) -> dict:
        """Analyze performance by department/team.

        Args:
            metrics_df: DataFrame with cohort metrics
            group_by: Grouping attribute

        Returns:
            Dictionary with department performance metrics
        """
        if metrics_df.empty or group_by not in metrics_df.columns:
            return {
                "top_departments": [],
                "bottom_departments": [],
                "department_stats": {},
            }

        # Aggregate by department
        dept_stats = (
            metrics_df.groupby(group_by)
            .agg(
                {
                    "person_id": "count",
                    "integration_score": ["mean", "std", "min", "max"],
                    "weeks_elapsed": "mean",
                }
            )
            .reset_index()
        )

        # Flatten column names
        dept_stats.columns = [
            group_by,
            "hire_count",
            "avg_integration_score",
            "std_integration_score",
            "min_integration_score",
            "max_integration_score",
            "avg_weeks_elapsed",
        ]

        # Calculate success rate (integration score >= 0.6)
        dept_success = (
            metrics_df[metrics_df["integration_score"] >= 0.6]
            .groupby(group_by)
            .size()
            .reset_index(name="success_count")
        )

        dept_stats = dept_stats.merge(dept_success, on=group_by, how="left")
        dept_stats["success_rate"] = (
            dept_stats["success_count"] / dept_stats["hire_count"]
        ).fillna(0.0)

        dept_stats["performance_score"] = (
            dept_stats["avg_integration_score"] * dept_stats["success_rate"]
        )
        dept_stats = dept_stats.sort_values("performance_score", ascending=False)

        # Top and bottom performers
        top_departments = dept_stats.head(10)[
            [
                group_by,
                "hire_count",
                "avg_integration_score",
                "success_rate",
                "avg_weeks_elapsed",
                "performance_score",
            ]
        ].to_dict("records")

        bottom_departments = dept_stats.tail(10)[
            [
                group_by,
                "hire_count",
                "avg_integration_score",
                "success_rate",
                "avg_weeks_elapsed",
                "performance_score",
            ]
        ].to_dict("records")

        # Department stats dictionary
        dept_stats_dict = dept_stats.set_index(group_by).to_dict("index")

        return {
            "top_departments": top_departments,
            "bottom_departments": bottom_departments,
            "department_stats": dept_stats_dict,
        }

    def _calculate_onboarding_roi(self, metrics_df: pd.DataFrame, group_by: str) -> dict:
        """Calculate ROI for onboarding programs by department.

        Args:
            metrics_df: DataFrame with cohort metrics
            group_by: Grouping attribute

        Returns:
            Dictionary with ROI analysis
        """
        if metrics_df.empty:
            return {"total_roi": 0.0, "department_roi": {}}

        # Calculate overall ROI using the tracker's method
        cohort_start = metrics_df["weeks_elapsed"].max() * 7  # Approximate
        cohort_end = datetime.now()

        # Get cohort person IDs
        cohort_person_ids = metrics_df["person_id"].unique().tolist()

        # Use tracker's ROI calculation
        try:
            # Group by department for department-level ROI
            dept_roi = {}
            if group_by in metrics_df.columns:
                for department in metrics_df[group_by].unique():
                    dept_hires = metrics_df[metrics_df[group_by] == department][
                        "person_id"
                    ].tolist()

                    if not dept_hires:
                        continue

                    # Calculate department-level success rate
                    dept_success_rate = (
                        len(
                            metrics_df[
                                (metrics_df[group_by] == department)
                                & (metrics_df["integration_score"] >= 0.6)
                            ]
                        )
                        / len(dept_hires)
                        if len(dept_hires) > 0
                        else 0.0
                    )

                    # ROI calculation (same as tracker)
                    avg_turnover_cost = 50000.0
                    turnover_reduction = 0.5
                    onboarding_cost_per_hire = 10000.0

                    total_onboarding_cost = len(dept_hires) * onboarding_cost_per_hire
                    estimated_savings = (
                        avg_turnover_cost * turnover_reduction * dept_success_rate * len(dept_hires)
                    )
                    roi_ratio = (
                        estimated_savings / total_onboarding_cost
                        if total_onboarding_cost > 0
                        else 0.0
                    )

                    dept_roi[department] = {
                        "hire_count": len(dept_hires),
                        "success_rate": dept_success_rate,
                        "total_onboarding_cost": total_onboarding_cost,
                        "estimated_savings": estimated_savings,
                        "roi_ratio": roi_ratio,
                    }
        except Exception as e:
            logger.warning(f"Error calculating department ROI: {e}")
            dept_roi = {}

        # Overall ROI
        overall_success_rate = (
            len(metrics_df[metrics_df["integration_score"] >= 0.6]) / len(metrics_df)
            if not metrics_df.empty
            else 0.0
        )

        total_hires = len(metrics_df)
        total_onboarding_cost = total_hires * 10000.0
        avg_turnover_cost = 50000.0
        turnover_reduction = 0.5
        estimated_savings = (
            avg_turnover_cost * turnover_reduction * overall_success_rate * total_hires
        )
        total_roi = estimated_savings / total_onboarding_cost if total_onboarding_cost > 0 else 0.0

        return {
            "total_roi": total_roi,
            "total_onboarding_cost": total_onboarding_cost,
            "estimated_savings": estimated_savings,
            "success_rate": overall_success_rate,
            "department_roi": dept_roi,
        }

    def _analyze_integration_speed(self, metrics_df: pd.DataFrame, group_by: str) -> dict:
        """Analyze speed of integration (time to productivity).

        Args:
            metrics_df: DataFrame with cohort metrics
            group_by: Grouping attribute

        Returns:
            Dictionary with speed analysis
        """
        if metrics_df.empty:
            return {
                "avg_weeks_to_integration": 0.0,
                "fastest_departments": [],
                "slowest_departments": [],
            }

        # Time to integration (weeks until integration_score >= 0.6)
        speed_metrics = []

        for _, row in metrics_df.iterrows():
            timeline = row.get("integration_timeline", [])
            if not timeline:
                continue

            # Find first week where integration_score >= 0.6
            weeks_to_integration = None
            for week_data in timeline:
                if week_data.get("integration_score", 0.0) >= 0.6:
                    weeks_to_integration = week_data.get("week", None)
                    break

            if weeks_to_integration is None:
                # Never reached threshold, use max weeks
                weeks_to_integration = row["weeks_elapsed"]

            speed_metrics.append(
                {
                    "person_id": row["person_id"],
                    group_by: row[group_by],
                    "weeks_to_integration": weeks_to_integration,
                }
            )

        if not speed_metrics:
            return {
                "avg_weeks_to_integration": 0.0,
                "fastest_departments": [],
                "slowest_departments": [],
            }

        speed_df = pd.DataFrame(speed_metrics)

        avg_weeks = speed_df["weeks_to_integration"].mean()

        if group_by in speed_df.columns:
            dept_speed = (
                speed_df.groupby(group_by)["weeks_to_integration"]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "avg_weeks", "count": "hire_count"})
                .sort_values("avg_weeks")
            )

            fastest_departments = (
                dept_speed.head(10).to_dict("records") if not dept_speed.empty else []
            )
            slowest_departments = (
                dept_speed.tail(10).to_dict("records") if not dept_speed.empty else []
            )
        else:
            fastest_departments = []
            slowest_departments = []

        return {
            "avg_weeks_to_integration": avg_weeks,
            "fastest_departments": fastest_departments,
            "slowest_departments": slowest_departments,
            "speed_distribution": {
                "fast (<=4 weeks)": len(speed_df[speed_df["weeks_to_integration"] <= 4]),
                "medium (5-8 weeks)": len(
                    speed_df[
                        (speed_df["weeks_to_integration"] > 4)
                        & (speed_df["weeks_to_integration"] <= 8)
                    ]
                ),
                "slow (9+ weeks)": len(speed_df[speed_df["weeks_to_integration"] > 8]),
            },
        }

    def _calculate_success_metrics(self, metrics_df: pd.DataFrame) -> dict:
        """Calculate overall success metrics.

        Args:
            metrics_df: DataFrame with cohort metrics

        Returns:
            Dictionary with success metrics
        """
        if metrics_df.empty:
            return {
                "total_hires": 0,
                "success_rate": 0.0,
                "avg_integration_score": 0.0,
            }

        total_hires = len(metrics_df)
        success_count = len(metrics_df[metrics_df["integration_score"] >= 0.6])
        success_rate = success_count / total_hires if total_hires > 0 else 0.0
        avg_integration_score = metrics_df["integration_score"].mean()

        # Status distribution
        status_dist = (
            metrics_df["status"].value_counts().to_dict() if "status" in metrics_df.columns else {}
        )

        return {
            "total_hires": total_hires,
            "success_count": success_count,
            "success_rate": success_rate,
            "avg_integration_score": avg_integration_score,
            "status_distribution": status_dist,
            "well_integrated_count": status_dist.get("well_integrated", 0),
            "at_risk_count": status_dist.get("at_risk", 0),
        }

    def _generate_dashboard_recommendations(
        self, dept_performance: dict, roi_analysis: dict, speed_analysis: dict
    ) -> list[dict]:
        """Generate recommendations from dashboard analysis.

        Args:
            dept_performance: Department performance analysis
            roi_analysis: ROI analysis
            speed_analysis: Speed analysis

        Returns:
            List of recommendation dictionaries
        """
        recommendations = []

        # Top performers: learn from them
        top_depts = dept_performance.get("top_departments", [])
        if top_depts:
            top_dept = top_depts[0]
            recommendations.append(
                {
                    "priority": "high",
                    "category": "best_practices",
                    "title": f"Learn from Top Performer: {top_dept.get('department', 'Department')}",
                    "description": (
                        f"{top_dept.get('department', 'Department')} has the highest "
                        f"onboarding success rate ({top_dept.get('success_rate', 0):.1%}). "
                        "Consider replicating their onboarding approach."
                    ),
                    "metric": f"Success Rate: {top_dept.get('success_rate', 0):.1%}",
                }
            )

        # Bottom performers: need improvement
        bottom_depts = dept_performance.get("bottom_departments", [])
        if bottom_depts:
            bottom_dept = bottom_depts[0]
            if bottom_dept.get("success_rate", 1.0) < 0.6:
                recommendations.append(
                    {
                        "priority": "high",
                        "category": "improvement",
                        "title": f"Improve Onboarding: {bottom_dept.get('department', 'Department')}",
                        "description": (
                            f"{bottom_dept.get('department', 'Department')} has low onboarding "
                            f"success rate ({bottom_dept.get('success_rate', 0):.1%}). "
                            "Immediate intervention recommended."
                        ),
                        "metric": f"Success Rate: {bottom_dept.get('success_rate', 0):.1%}",
                    }
                )

        fastest = speed_analysis.get("fastest_departments", [])
        slowest = speed_analysis.get("slowest_departments", [])
        if fastest and slowest:
            fastest_dept = fastest[0]
            slowest_dept = slowest[0]
            if slowest_dept.get("avg_weeks", 0) > fastest_dept.get("avg_weeks", 0) + 4:
                recommendations.append(
                    {
                        "priority": "medium",
                        "category": "speed",
                        "title": f"Speed Up Integration: {slowest_dept.get('department', 'Department')}",
                        "description": (
                            f"{slowest_dept.get('department', 'Department')} takes "
                            f"{slowest_dept.get('avg_weeks', 0):.1f} weeks to integrate new hires, "
                            f"compared to {fastest_dept.get('department', 'Department')}'s "
                            f"{fastest_dept.get('avg_weeks', 0):.1f} weeks. "
                            "Consider reviewing onboarding timeline."
                        ),
                        "metric": f"Integration Time: {slowest_dept.get('avg_weeks', 0):.1f} weeks",
                    }
                )

        # ROI recommendations
        total_roi = roi_analysis.get("total_roi", 0.0)
        if total_roi < 1.0:
            recommendations.append(
                {
                    "priority": "medium",
                    "category": "roi",
                    "title": "Improve Onboarding ROI",
                    "description": (
                        f"Current onboarding ROI is {total_roi:.2f}x. "
                        "Consider optimizing onboarding programs to improve ROI."
                    ),
                    "metric": f"ROI: {total_roi:.2f}x",
                }
            )

        return recommendations
