"""Automated reporting for executive dashboards."""

import json
from datetime import datetime
from pathlib import Path

from orgnet.utils.logging import get_logger
from orgnet.visualization.executive.dashboard import ExecutiveDashboard

logger = get_logger(__name__)


class AutomatedReporter:
    """Automated reporting for executive dashboards.

    Provides automated report generation that can be integrated with
    HRIS systems and scheduled for regular delivery.
    """

    def __init__(self, dashboard: ExecutiveDashboard):
        """Initialize automated reporter.

        Args:
            dashboard: ExecutiveDashboard instance
        """
        self.dashboard = dashboard

    def generate_json_report(self, output_path: str | None = None) -> str:
        """Generate JSON report.

        Args:
            output_path: Optional path to save report

        Returns:
            JSON string representation of dashboard
        """
        dashboard_data = self.dashboard.export_to_dict()
        json_str = json.dumps(dashboard_data, indent=2, default=str)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(json_str)
            logger.info(f"Saved JSON report to {output_path}")

        return json_str

    def generate_summary_report(self) -> dict:
        """Generate executive summary report (one-page overview).

        Returns:
            Dictionary with key metrics and recommendations
        """
        full_dashboard = self.dashboard.generate_executive_dashboard()

        return {
            "timestamp": datetime.now().isoformat(),
            "key_metrics": full_dashboard.get("key_metrics", {}),
            "top_recommendations": full_dashboard.get("recommendations", [])[:5],
            "organizational_status": full_dashboard.get("executive_summary", {}).get(
                "status", "unknown"
            ),
            "critical_alerts": self._extract_critical_alerts(full_dashboard),
        }

    def _extract_critical_alerts(self, dashboard: dict) -> list[dict]:
        """Extract critical alerts from dashboard.

        Args:
            dashboard: Full dashboard dictionary

        Returns:
            List of critical alerts
        """
        alerts = []

        # Burnout alerts
        burnout_critical = (
            dashboard.get("people_analytics", {}).get("burnout_risk", {}).get("critical_count", 0)
        )
        if burnout_critical > 0:
            alerts.append(
                {
                    "severity": "critical",
                    "category": "burnout",
                    "message": f"{burnout_critical} individuals at critical burnout risk",
                    "action_required": "immediate",
                }
            )

        # Health alerts
        health_status = dashboard.get("executive_summary", {}).get("status", "")
        if health_status == "needs_attention":
            alerts.append(
                {
                    "severity": "high",
                    "category": "organizational_health",
                    "message": "Organizational health requires attention",
                    "action_required": "soon",
                }
            )

        return alerts
