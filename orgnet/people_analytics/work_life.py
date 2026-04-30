"""Work-life balance metrics for HR insights.

This module tracks after-hours digital exhaust from calendar, Slack, and email
to provide companies with a "Burnout Risk Score" based on network overload.
"""

import pandas as pd

from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


class WorkLifeBalanceAnalyzer:
    """Work-life balance analysis based on after-hours activity tracking.

    Refines calendar and Slack ingestion to track "after-hours" digital exhaust,
    providing companies with a "Burnout Risk Score" based on network overload.
    """

    def __init__(
        self,
        business_hours: tuple[int, int] = (9, 17),
        weekend_weight: float = 1.5,
        after_hours_weight: float = 1.2,
        **params,
    ):
        """Initialize work-life balance analyzer.

        Args:
            business_hours: Tuple of (start_hour, end_hour) for business hours (default: 9-17)
            weekend_weight: Weight multiplier for weekend activity (default: 1.5)
            after_hours_weight: Weight multiplier for after-hours activity (default: 1.2)
        """
        self.business_hours = business_hours
        self.weekend_weight = weekend_weight
        self.after_hours_weight = after_hours_weight

    def compute(
        self, events: pd.DataFrame, nodes: pd.DataFrame | None = None, **kwargs
    ) -> pd.DataFrame:
        """Compute work-life balance metrics for each person.

        Args:
            events: Event table with columns: timestamp, actor, target, event_type
            nodes: Optional node table (not used here but required by BaseMetric)

        Returns:
            DataFrame with columns: person_id, burnout_risk_score, work_life_score,
                after_hours_ratio, weekend_ratio, avg_hours_per_week, risk_level
        """
        if events.empty or "timestamp" not in events.columns:
            return pd.DataFrame(
                columns=[
                    "person_id",
                    "burnout_risk_score",
                    "work_life_score",
                    "after_hours_ratio",
                    "weekend_ratio",
                    "avg_hours_per_week",
                    "risk_level",
                ]
            )

        events = events.copy()

        # Extract temporal features
        events["hour"] = events["timestamp"].dt.hour
        events["day_of_week"] = events["timestamp"].dt.dayofweek
        events["is_weekend"] = (events["day_of_week"] >= 5).astype(int)
        events["is_business_hours"] = (
            (events["hour"] >= self.business_hours[0])
            & (events["hour"] < self.business_hours[1])
            & (events["is_weekend"] == 0)
        ).astype(int)
        events["is_after_hours"] = (~events["is_business_hours"]).astype(int)

        # Group by person
        person_ids = events["actor"].unique()

        results = []

        for person_id in person_ids:
            person_events = events[events["actor"] == person_id]

            if len(person_events) < 5:  # Minimum events required
                continue

            # Compute metrics
            metrics = self._compute_person_metrics(person_id, person_events)

            results.append(metrics)

        return pd.DataFrame(results)

    def _compute_person_metrics(self, person_id: str, person_events: pd.DataFrame) -> dict:
        """Compute work-life balance metrics for a single person.

        Args:
            person_id: Person identifier
            person_events: Events for this person

        Returns:
            Dictionary with work-life balance metrics
        """
        total_events = len(person_events)

        # After-hours ratio
        after_hours_count = person_events["is_after_hours"].sum()
        after_hours_ratio = after_hours_count / total_events if total_events > 0 else 0.0

        # Weekend ratio
        weekend_count = person_events["is_weekend"].sum()
        weekend_ratio = weekend_count / total_events if total_events > 0 else 0.0

        # Average hours per week (based on event timestamps)
        weeks_covered = self._compute_weeks_covered(person_events)
        avg_hours_per_week = total_events / weeks_covered if weeks_covered > 0 else 0.0

        # Burnout risk score (0-1, higher = more risk)
        burnout_risk = self._compute_burnout_risk(
            after_hours_ratio, weekend_ratio, avg_hours_per_week
        )

        # Work-life balance score (0-1, higher = better balance)
        work_life_score = 1.0 - burnout_risk

        # Risk level classification
        risk_level = self._classify_risk(burnout_risk)

        return {
            "person_id": person_id,
            "burnout_risk_score": burnout_risk,
            "work_life_score": work_life_score,
            "after_hours_ratio": after_hours_ratio,
            "weekend_ratio": weekend_ratio,
            "avg_hours_per_week": avg_hours_per_week,
            "total_events": total_events,
            "weeks_covered": weeks_covered,
            "risk_level": risk_level,
        }

    def _compute_weeks_covered(self, person_events: pd.DataFrame) -> float:
        """Compute number of weeks covered by events.

        Args:
            person_events: Events for a person

        Returns:
            Number of weeks (float)
        """
        if person_events.empty:
            return 1.0

        min_date = person_events["timestamp"].min()
        max_date = person_events["timestamp"].max()

        days_covered = (max_date - min_date).days + 1
        weeks_covered = days_covered / 7.0

        return max(weeks_covered, 1.0)  # At least 1 week

    def _compute_burnout_risk(
        self, after_hours_ratio: float, weekend_ratio: float, avg_hours_per_week: float
    ) -> float:
        """Compute burnout risk score from work-life balance metrics.

        Args:
            after_hours_ratio: Ratio of events outside business hours
            weekend_ratio: Ratio of events on weekends
            avg_hours_per_week: Average events per week

        Returns:
            Burnout risk score between 0 (low risk) and 1 (high risk)
        """
        # Base risk from after-hours activity
        after_hours_risk = min(after_hours_ratio * self.after_hours_weight, 1.0)

        # Weekend activity risk (weighted higher)
        weekend_risk = min(weekend_ratio * self.weekend_weight, 1.0)

        # Volume risk (high activity = potential overload)
        # Normalize: 50 events/week = 0.5 risk, 100+ events/week = 1.0 risk
        volume_risk = min(avg_hours_per_week / 100.0, 1.0)

        # Weighted combination
        burnout_risk = (after_hours_risk * 0.4) + (weekend_risk * 0.3) + (volume_risk * 0.3)

        return min(burnout_risk, 1.0)

    def _classify_risk(self, burnout_risk: float) -> str:
        """Classify burnout risk level.

        Args:
            burnout_risk: Burnout risk score between 0 and 1

        Returns:
            Risk level: 'low', 'medium', 'high', 'critical'
        """
        if burnout_risk >= 0.75:
            return "critical"
        elif burnout_risk >= 0.5:
            return "high"
        elif burnout_risk >= 0.25:
            return "medium"
        else:
            return "low"

    def generate_burnout_risk_report(self, events: pd.DataFrame, top_n: int = 20) -> dict:
        """Generate comprehensive burnout risk report.

        Args:
            events: Event table
            top_n: Number of top-risk individuals to include

        Returns:
            Dictionary with report data
        """
        metrics_df = self.compute(events)

        if metrics_df.empty:
            return {
                "summary": "No data available",
                "at_risk_count": 0,
                "critical_count": 0,
                "top_risks": [],
            }

        # Summary statistics
        at_risk_count = len(metrics_df[metrics_df["risk_level"].isin(["high", "critical"])])
        critical_count = len(metrics_df[metrics_df["risk_level"] == "critical"])

        # Top risks
        top_risks = metrics_df.nlargest(top_n, "burnout_risk_score")[
            ["person_id", "burnout_risk_score", "risk_level", "after_hours_ratio", "weekend_ratio"]
        ].to_dict("records")

        return {
            "summary": f"{at_risk_count} individuals at risk ({critical_count} critical)",
            "at_risk_count": at_risk_count,
            "critical_count": critical_count,
            "total_analyzed": len(metrics_df),
            "top_risks": top_risks,
            "metrics_df": metrics_df,
        }
