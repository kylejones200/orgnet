"""Executive Dashboard with automated reporting.

This module provides executive-ready dashboards and automated reporting that
goes beyond the current Flask-based API to a robust web-based visualization suite.
"""

from datetime import datetime

import networkx as nx
import pandas as pd

from orgnet.innovation.index import InnovationIndex
from orgnet.innovation.structural_holes import StructuralHoleMonetizer
from orgnet.insights.interventions import InterventionFramework
from orgnet.people_analytics.burnout import BurnoutPredictor
from orgnet.people_analytics.work_life import WorkLifeBalanceAnalyzer
from orgnet.utils.logging import get_logger
from orgnet.visualization.dashboards import DashboardGenerator

logger = get_logger(__name__)


class ExecutiveDashboard:
    """Executive Dashboard with comprehensive organizational insights.

    Provides executive-ready dashboards that integrate all productization features:
    - People Analytics (burnout, work-life balance)
    - Innovation & Knowledge Flow (innovation index, expertise, structural holes)
    - Intervention Tracking
    - Organizational Health
    - ROI Metrics
    """

    def __init__(
        self,
        graph: nx.Graph,
        events: pd.DataFrame | None = None,
        people: list | None = None,
        interactions: list | None = None,
    ):
        """Initialize executive dashboard.

        Args:
            graph: Organizational network graph
            events: Optional event table for temporal analysis
            people: Optional list of Person objects
            interactions: Optional list of Interaction objects
        """
        self.graph = graph
        self.events = events
        self.people = people or []
        self.interactions = interactions or []

        # Core dashboard generator
        self.dashboard_generator = DashboardGenerator(graph)

        # Productization components (lazy-loaded)
        self._burnout_predictor: BurnoutPredictor | None = None
        self._work_life_analyzer: WorkLifeBalanceAnalyzer | None = None
        self._innovation_index: InnovationIndex | None = None
        self._structural_hole_monetizer: StructuralHoleMonetizer | None = None
        self._intervention_framework: InterventionFramework | None = None

    @property
    def burnout_predictor(self) -> BurnoutPredictor:
        """Lazy-load burnout predictor."""
        if self._burnout_predictor is None and self.events is not None:
            self._burnout_predictor = BurnoutPredictor()
            self._burnout_predictor.fit(self.events)
        return self._burnout_predictor

    @property
    def work_life_analyzer(self) -> WorkLifeBalanceAnalyzer:
        """Lazy-load work-life balance analyzer."""
        if self._work_life_analyzer is None and self.events is not None:
            self._work_life_analyzer = WorkLifeBalanceAnalyzer()
        return self._work_life_analyzer

    @property
    def innovation_index(self) -> InnovationIndex:
        """Lazy-load innovation index."""
        if self._innovation_index is None:
            self._innovation_index = InnovationIndex(self.graph)
        return self._innovation_index

    @property
    def structural_hole_monetizer(self) -> StructuralHoleMonetizer:
        """Lazy-load structural hole monetizer."""
        if self._structural_hole_monetizer is None:
            self._structural_hole_monetizer = StructuralHoleMonetizer(self.graph)
        return self._structural_hole_monetizer

    def generate_executive_dashboard(self) -> dict:
        """Generate comprehensive executive dashboard.

        Returns:
            Dictionary with executive dashboard data
        """
        logger.info("Generating executive dashboard...")

        # Core organizational health
        health_dashboard = self.dashboard_generator.generate_health_dashboard()
        executive_summary = self.dashboard_generator.generate_executive_summary()

        # People Analytics
        people_analytics = self._generate_people_analytics_section()

        # Innovation & Knowledge Flow
        innovation_analytics = self._generate_innovation_section()

        # Structural Holes & ROI
        structural_holes = self._generate_structural_holes_section()

        # Intervention Tracking
        intervention_tracking = self._generate_intervention_section()

        # Key Metrics Summary
        key_metrics = self._generate_key_metrics_summary(
            health_dashboard, people_analytics, innovation_analytics, structural_holes
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "key_metrics": key_metrics,
            "organizational_health": health_dashboard,
            "executive_summary": executive_summary,
            "people_analytics": people_analytics,
            "innovation_analytics": innovation_analytics,
            "structural_holes": structural_holes,
            "intervention_tracking": intervention_tracking,
            "recommendations": self._generate_recommendations(
                health_dashboard, people_analytics, innovation_analytics, structural_holes
            ),
        }

    def _generate_people_analytics_section(self) -> dict:
        """Generate people analytics section."""
        if self.events is None or self.events.empty:
            return {
                "burnout_risk": {"summary": "No event data available"},
                "work_life_balance": {"summary": "No event data available"},
            }

        try:
            # Burnout risk
            burnout_df = self.burnout_predictor.predict(self.events, self.graph)
            critical_risk = (
                burnout_df[burnout_df["risk_level"] == "critical"]
                if not burnout_df.empty
                else pd.DataFrame()
            )

            # Work-life balance
            work_life_df = self.work_life_analyzer.compute(self.events)
            work_life_report = self.work_life_analyzer.generate_burnout_risk_report(self.events)

            return {
                "burnout_risk": {
                    "summary": f"{len(critical_risk)} individuals at critical burnout risk",
                    "total_analyzed": len(burnout_df),
                    "critical_count": len(critical_risk),
                    "high_count": (
                        len(burnout_df[burnout_df["risk_level"] == "high"])
                        if not burnout_df.empty
                        else 0
                    ),
                    "top_risks": (
                        critical_risk.head(10).to_dict("records") if not critical_risk.empty else []
                    ),
                },
                "work_life_balance": {
                    "summary": work_life_report.get("summary", "No data"),
                    "at_risk_count": work_life_report.get("at_risk_count", 0),
                    "critical_count": work_life_report.get("critical_count", 0),
                    "top_risks": work_life_report.get("top_risks", [])[:10],
                },
            }
        except Exception as e:
            logger.warning(f"Error generating people analytics: {e}", exc_info=True)
            return {
                "burnout_risk": {"summary": f"Error: {str(e)}"},
                "work_life_balance": {"summary": f"Error: {str(e)}"},
            }

    def _generate_innovation_section(self) -> dict:
        """Generate innovation analytics section."""
        try:
            innovation_df = self.innovation_index.compute()
            top_innovators = self.innovation_index.get_top_innovators(top_n=10)

            return {
                "innovation_index": {
                    "summary": f"Analyzed {len(innovation_df)} individuals for innovation potential",
                    "avg_innovation_index": (
                        innovation_df["innovation_index"].mean() if not innovation_df.empty else 0.0
                    ),
                    "top_innovators": (
                        top_innovators.to_dict("records") if not top_innovators.empty else []
                    ),
                    "innovation_distribution": {
                        "innovator": (
                            len(innovation_df[innovation_df["innovation_level"] == "innovator"])
                            if not innovation_df.empty
                            else 0
                        ),
                        "broker": (
                            len(innovation_df[innovation_df["innovation_level"] == "broker"])
                            if not innovation_df.empty
                            else 0
                        ),
                        "connector": (
                            len(innovation_df[innovation_df["innovation_level"] == "connector"])
                            if not innovation_df.empty
                            else 0
                        ),
                    },
                },
            }
        except Exception as e:
            logger.warning(f"Error generating innovation analytics: {e}", exc_info=True)
            return {"innovation_index": {"summary": f"Error: {str(e)}"}}

    def _generate_structural_holes_section(self) -> dict:
        """Generate structural holes and ROI section."""
        try:
            roi_report = self.structural_hole_monetizer.generate_roi_report(min_roi=2.0)
            high_roi_holes = self.structural_hole_monetizer.get_high_roi_holes(
                min_roi=3.0, top_n=10
            )

            return {
                "structural_holes": {
                    "summary": roi_report.get("summary", "No data"),
                    "total_holes": roi_report.get("total_holes", 0),
                    "high_roi_count": roi_report.get("high_roi_count", 0),
                    "total_potential_benefit": roi_report.get("total_potential_benefit", 0.0),
                    "avg_roi": roi_report.get("avg_roi", 0.0),
                    "top_opportunities": (
                        high_roi_holes.to_dict("records") if not high_roi_holes.empty else []
                    ),
                    "recommendations": roi_report.get("recommendations", []),
                },
            }
        except Exception as e:
            logger.warning(f"Error generating structural holes: {e}", exc_info=True)
            return {"structural_holes": {"summary": f"Error: {str(e)}"}}

    def _generate_intervention_section(self) -> dict:
        """Generate intervention tracking section."""
        if self._intervention_framework is None:
            self._intervention_framework = InterventionFramework()

        try:
            measurement_report = self._intervention_framework.generate_measurement_report()
            outcomes_df = self._intervention_framework.get_intervention_outcomes()

            return {
                "interventions": {
                    "summary": measurement_report.get("summary", "No interventions"),
                    "total_interventions": measurement_report.get("total_interventions", 0),
                    "completed_count": measurement_report.get("completed_count", 0),
                    "success_rate": measurement_report.get("success_rate", 0.0),
                    "avg_roi": measurement_report.get("avg_roi"),
                    "outcomes": measurement_report.get("outcomes", {}),
                    "top_performers": measurement_report.get("top_performers", []),
                },
            }
        except Exception as e:
            logger.warning(f"Error generating intervention tracking: {e}", exc_info=True)
            return {"interventions": {"summary": f"Error: {str(e)}"}}

    def _generate_key_metrics_summary(
        self, health: dict, people: dict, innovation: dict, structural: dict
    ) -> dict:
        """Generate key metrics summary."""
        return {
            "organizational_health_score": health.get("network_density", 0.0),
            "burnout_at_risk": people.get("burnout_risk", {}).get("critical_count", 0),
            "innovation_index": innovation.get("innovation_index", {}).get(
                "avg_innovation_index", 0.0
            ),
            "structural_holes_roi": structural.get("structural_holes", {}).get(
                "total_potential_benefit", 0.0
            ),
            "intervention_success_rate": (
                self._intervention_framework.generate_measurement_report().get("success_rate", 0.0)
                if self._intervention_framework
                else 0.0
            ),
        }

    def _generate_recommendations(
        self, health: dict, people: dict, innovation: dict, structural: dict
    ) -> list[dict]:
        """Generate prioritized recommendations."""
        recommendations = []

        # Burnout recommendations
        burnout_count = people.get("burnout_risk", {}).get("critical_count", 0)
        if burnout_count > 0:
            recommendations.append(
                {
                    "priority": "high",
                    "category": "people_analytics",
                    "title": f"Address {burnout_count} Critical Burnout Risks",
                    "description": f"{burnout_count} individuals at critical burnout risk require immediate attention",
                    "estimated_impact": "high",
                    "effort": "medium",
                }
            )

        # Structural holes recommendations
        high_roi_count = structural.get("structural_holes", {}).get("high_roi_count", 0)
        if high_roi_count > 0:
            potential_benefit = structural.get("structural_holes", {}).get(
                "total_potential_benefit", 0.0
            )
            recommendations.append(
                {
                    "priority": "high",
                    "category": "innovation",
                    "title": f"Plug {high_roi_count} High-ROI Structural Holes",
                    "description": f"Potential benefit: ${potential_benefit:,.0f} from plugging structural holes",
                    "estimated_impact": "high",
                    "effort": "medium",
                    "roi": structural.get("structural_holes", {}).get("avg_roi", 0.0),
                }
            )

        # Innovation recommendations
        avg_innovation = innovation.get("innovation_index", {}).get("avg_innovation_index", 0.0)
        if avg_innovation < 0.4:
            recommendations.append(
                {
                    "priority": "medium",
                    "category": "innovation",
                    "title": "Increase Bridging for Innovation",
                    "description": "Low average innovation index suggests need for more cross-group connections",
                    "estimated_impact": "medium",
                    "effort": "high",
                }
            )

        return recommendations

    def export_to_dict(self) -> dict:
        """Export dashboard to dictionary for JSON serialization.

        Returns:
            Dictionary representation of dashboard
        """
        dashboard = self.generate_executive_dashboard()

        # Convert DataFrames to dict records for JSON serialization
        for section_name, section_data in dashboard.items():
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    if isinstance(value, pd.DataFrame):
                        dashboard[section_name][key] = value.to_dict("records")
                    elif isinstance(value, dict) and isinstance(
                        value.get("top_risks"), pd.DataFrame
                    ):
                        dashboard[section_name][key]["top_risks"] = value["top_risks"].to_dict(
                            "records"
                        )

        return dashboard
