"""Efficiency and Innovation Report Generator.

This module generates an "Efficiency and Innovation Report" that highlights
"high-risk silos" where duplication is likely occurring. This is tailored
specifically for COOs focused on innovation and work deduplication.
"""

from datetime import datetime

import networkx as nx

from orgnet.efficiency.cross_pollination import CrossPollinationEngine
from orgnet.efficiency.deduplication_roi import DeduplicationROICalculator
from orgnet.efficiency.knowledge_graph import KnowledgeGraph
from orgnet.efficiency.redundancy_detection import RedundancyDetector
from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


class EfficiencyInnovationReporter:
    """Efficiency and Innovation Report Generator for COOs.

    Generates comprehensive reports that highlight "high-risk silos" where
    duplication is likely occurring, providing quantitative evidence for
    cross-pollination initiatives.
    """

    def __init__(
        self,
        graph: nx.Graph,
        redundancy_detector: RedundancyDetector,
        cross_pollination_engine: CrossPollinationEngine,
        deduplication_roi: DeduplicationROICalculator,
        knowledge_graph: KnowledgeGraph,
    ):
        """Initialize efficiency and innovation reporter.

        Args:
            graph: Organizational network graph
            redundancy_detector: RedundancyDetector instance
            cross_pollination_engine: CrossPollinationEngine instance
            deduplication_roi: DeduplicationROICalculator instance
            knowledge_graph: KnowledgeGraph instance
        """
        self.graph = graph
        self.redundancy_detector = redundancy_detector
        self.cross_pollination_engine = cross_pollination_engine
        self.deduplication_roi = deduplication_roi
        self.knowledge_graph = knowledge_graph

    def generate_report(
        self,
        include_redundancy: bool = True,
        include_cross_pollination: bool = True,
        include_roi: bool = True,
        include_knowledge_graph: bool = True,
    ) -> dict:
        """Generate comprehensive Efficiency and Innovation Report for COO.

        This report highlights "high-risk silos" where duplication is likely
        occurring and provides actionable recommendations for deduplication.

        Args:
            include_redundancy: Include redundancy detection (default: True)
            include_cross_pollination: Include cross-pollination recommendations (default: True)
            include_roi: Include ROI calculations (default: True)
            include_knowledge_graph: Include knowledge graph insights (default: True)

        Returns:
            Dictionary with comprehensive efficiency report
        """
        logger.info("Generating Efficiency and Innovation Report...")

        report_sections = {}

        # Section 1: Redundancy Detection
        if include_redundancy:
            redundancy_data = self.redundancy_detector.compute()
            redundancy_alert = self.redundancy_detector.generate_redundancy_alert(redundancy_data)

            report_sections["redundancy_detection"] = {
                "summary": redundancy_alert.get("alert", "No redundancy detected"),
                "redundancy_count": redundancy_alert.get("redundancy_count", 0),
                "high_risk_count": redundancy_alert.get("high_risk_count", 0),
                "avg_topic_similarity": redundancy_alert.get("avg_topic_similarity", 0.0),
                "avg_bridging_ratio": redundancy_alert.get("avg_bridging_ratio", 0.0),
                "top_redundancies": redundancy_alert.get("top_redundancies", []),
            }

        # Section 2: Cross-Pollination Recommendations
        if include_cross_pollination:
            cross_pollination_data = self.cross_pollination_engine.compute()
            cross_pollination_report = (
                self.cross_pollination_engine.generate_cross_pollination_report(
                    cross_pollination_data
                )
            )

            report_sections["cross_pollination"] = {
                "summary": cross_pollination_report.get(
                    "summary", "No cross-pollination opportunities"
                ),
                "recommendation_count": cross_pollination_report.get("recommendation_count", 0),
                "high_priority_count": cross_pollination_report.get("high_priority_count", 0),
                "total_estimated_roi": cross_pollination_report.get("total_estimated_roi", 0.0),
                "avg_roi": cross_pollination_report.get("avg_roi", 0.0),
                "top_recommendations": cross_pollination_report.get("top_recommendations", []),
                "action_items": cross_pollination_report.get("action_items", []),
            }

        # Section 3: Deduplication ROI
        if include_roi:
            roi_data = self.deduplication_roi.compute()
            roi_report = self.deduplication_roi.generate_deduplication_roi_report(roi_data)

            report_sections["deduplication_roi"] = {
                "summary": roi_report.get("summary", "No ROI opportunities"),
                "total_waste_cost": roi_report.get("total_waste_cost", 0.0),
                "total_deduplication_cost": roi_report.get("total_deduplication_cost", 0.0),
                "total_net_savings": roi_report.get("total_net_savings", 0.0),
                "avg_roi": roi_report.get("avg_roi", 0.0),
                "high_priority_count": roi_report.get("high_priority_count", 0),
                "waste_breakdown": roi_report.get("waste_breakdown", {}),
                "top_opportunities": roi_report.get("top_opportunities", []),
                "key_insights": roi_report.get("key_insights", []),
            }

        # Section 4: Knowledge Graph
        if include_knowledge_graph:
            kg_data = self.knowledge_graph.compute()
            kg_report = self.knowledge_graph.generate_knowledge_graph_report()

            report_sections["knowledge_graph"] = {
                "summary": kg_report.get("summary", "Knowledge graph not available"),
                "total_problems": kg_report.get("total_problems", 0),
                "total_solutions": kg_report.get("total_solutions", 0),
                "solution_coverage": kg_report.get("solution_coverage", 0.0),
                "avg_solutions_per_problem": kg_report.get("avg_solutions_per_problem", 0.0),
            }

        # Executive Summary
        executive_summary = self._generate_executive_summary(report_sections)

        # High-Risk Silos (communities with high redundancy and low bridging)
        high_risk_silos = self._identify_high_risk_silos(report_sections)

        # Priority Recommendations
        priority_recommendations = self._generate_priority_recommendations(report_sections)

        return {
            "timestamp": datetime.now().isoformat(),
            "executive_summary": executive_summary,
            "high_risk_silos": high_risk_silos,
            "priority_recommendations": priority_recommendations,
            "sections": report_sections,
            "report_type": "Efficiency and Innovation Report",
            "target_audience": "COO",
            "key_takeaways": self._extract_key_takeaways(report_sections),
        }

    def _generate_executive_summary(self, report_sections: dict) -> str:
        """Generate executive summary for the report.

        Args:
            report_sections: Report sections dictionary

        Returns:
            Executive summary string
        """
        summary_parts = []

        # Redundancy summary
        if "redundancy_detection" in report_sections:
            redundancy = report_sections["redundancy_detection"]
            redundancy_count = redundancy.get("redundancy_count", 0)
            high_risk = redundancy.get("high_risk_count", 0)

            if redundancy_count > 0:
                summary_parts.append(
                    f"Detected {redundancy_count} redundancy opportunities, "
                    f"{high_risk} high-risk cases requiring immediate attention."
                )

        # ROI summary
        if "deduplication_roi" in report_sections:
            roi = report_sections["deduplication_roi"]
            total_waste = roi.get("total_waste_cost", 0.0)
            net_savings = roi.get("total_net_savings", 0.0)

            if total_waste > 0:
                summary_parts.append(
                    f"Total estimated waste: ${total_waste:,.0f}. "
                    f"Net savings potential: ${net_savings:,.0f} after deduplication costs."
                )

        # Cross-pollination summary
        if "cross_pollination" in report_sections:
            cp = report_sections["cross_pollination"]
            recommendation_count = cp.get("recommendation_count", 0)
            high_priority = cp.get("high_priority_count", 0)

            if recommendation_count > 0:
                summary_parts.append(
                    f"Identified {recommendation_count} cross-pollination opportunities, "
                    f"{high_priority} high-priority for immediate action."
                )

        if not summary_parts:
            return "No significant efficiency opportunities detected at this time."

        return " | ".join(summary_parts)

    def _identify_high_risk_silos(self, report_sections: dict) -> list[dict]:
        """Identify high-risk silos where duplication is likely occurring.

        Args:
            report_sections: Report sections dictionary

        Returns:
            List of high-risk silo dictionaries
        """
        high_risk_silos = []

        # Get redundancy data
        if "redundancy_detection" not in report_sections:
            return high_risk_silos

        top_redundancies = report_sections["redundancy_detection"].get("top_redundancies", [])

        for redundancy in top_redundancies:
            severity = redundancy.get("severity", "low")
            topic_similarity = redundancy.get("topic_similarity", 0.0)
            bridging_ratio = redundancy.get("bridging_ratio", 0.0)

            # High-risk: high similarity + very low bridging
            if severity in ["high", "critical"] or (
                topic_similarity >= 0.8 and bridging_ratio < 0.05
            ):
                high_risk_silos.append(
                    {
                        "community_a": redundancy.get("community_a"),
                        "community_b": redundancy.get("community_b"),
                        "topic_similarity": topic_similarity,
                        "bridging_ratio": bridging_ratio,
                        "redundancy_score": redundancy.get("redundancy_score", 0.0),
                        "shared_topics": redundancy.get("shared_topic_names", []),
                        "risk_level": (
                            "critical"
                            if topic_similarity >= 0.9 and bridging_ratio < 0.02
                            else "high"
                        ),
                        "recommendation": redundancy.get("recommendation", ""),
                    }
                )

        # Sort by risk level and redundancy score
        high_risk_silos.sort(
            key=lambda x: (
                0 if x["risk_level"] == "critical" else 1,
                -x["redundancy_score"],
            )
        )

        return high_risk_silos[:10]  # Top 10 high-risk silos

    def _generate_priority_recommendations(self, report_sections: dict) -> list[dict]:
        """Generate priority recommendations from all report sections.

        Args:
            report_sections: Report sections dictionary

        Returns:
            List of priority recommendation dictionaries
        """
        recommendations = []

        # Recommendations from cross-pollination
        if "cross_pollination" in report_sections:
            cp = report_sections["cross_pollination"]
            top_recommendations = cp.get("top_recommendations", [])

            for rec in top_recommendations[:5]:  # Top 5
                if rec.get("priority") == "high":
                    recommendations.append(
                        {
                            "source": "cross_pollination",
                            "priority": "high",
                            "title": (
                                f"Connect Communities {rec.get('from_community')} and "
                                f"{rec.get('to_community')} for Knowledge Transfer"
                            ),
                            "description": rec.get("recommendation", ""),
                            "estimated_roi": rec.get("transfer_roi", 0.0),
                            "action": (
                                f"Schedule knowledge transfer session between "
                                f"Communities {rec.get('from_community')} and {rec.get('to_community')}"
                            ),
                        }
                    )

        # Recommendations from ROI
        if "deduplication_roi" in report_sections:
            roi = report_sections["deduplication_roi"]
            top_opportunities = roi.get("top_opportunities", [])

            for opp in top_opportunities[:5]:  # Top 5
                if opp.get("priority") == "high":
                    recommendations.append(
                        {
                            "source": "deduplication_roi",
                            "priority": "high",
                            "title": (
                                f"Address Redundancy: Communities {opp.get('community_a')} "
                                f"and {opp.get('community_b')}"
                            ),
                            "description": opp.get("recommendation", ""),
                            "estimated_roi": opp.get("roi_ratio", 0.0),
                            "estimated_savings": opp.get("net_savings", 0.0),
                            "time_to_value": f"{opp.get('time_to_value_weeks', 0):.1f} weeks",
                            "action": (
                                f"Implement deduplication initiative for Communities "
                                f"{opp.get('community_a')} and {opp.get('community_b')}"
                            ),
                        }
                    )

        # Sort by priority and ROI
        recommendations.sort(
            key=lambda x: (
                0 if x["priority"] == "high" else 1,
                -x.get("estimated_roi", 0.0),
            )
        )

        return recommendations[:10]  # Top 10 recommendations

    def _extract_key_takeaways(self, report_sections: dict) -> list[str]:
        """Extract key takeaways from report sections.

        Args:
            report_sections: Report sections dictionary

        Returns:
            List of key takeaway strings
        """
        takeaways = []

        # Redundancy takeaways
        if "redundancy_detection" in report_sections:
            redundancy = report_sections["redundancy_detection"]
            high_risk = redundancy.get("high_risk_count", 0)

            if high_risk > 0:
                takeaways.append(
                    f"🚨 {high_risk} high-risk redundancy cases detected. "
                    "Immediate action recommended to prevent duplicate work."
                )

        # ROI takeaways
        if "deduplication_roi" in report_sections:
            roi = report_sections["deduplication_roi"]
            total_waste = roi.get("total_waste_cost", 0.0)
            net_savings = roi.get("total_net_savings", 0.0)
            avg_roi = roi.get("avg_roi", 0.0)

            if total_waste > 0:
                takeaways.append(
                    f"💰 Organization wasting ${total_waste:,.0f} annually on redundant work. "
                    f"Net savings potential: ${net_savings:,.0f} (avg ROI: {avg_roi:.2f}x)."
                )

        # Cross-pollination takeaways
        if "cross_pollination" in report_sections:
            cp = report_sections["cross_pollination"]
            high_priority = cp.get("high_priority_count", 0)
            total_roi = cp.get("total_estimated_roi", 0.0)

            if high_priority > 0:
                takeaways.append(
                    f"🔄 {high_priority} high-priority cross-pollination opportunities identified. "
                    f"Total estimated ROI: {total_roi:.2f}x."
                )

        # Knowledge graph takeaways
        if "knowledge_graph" in report_sections:
            kg = report_sections["knowledge_graph"]
            solution_coverage = kg.get("solution_coverage", 0.0)

            if solution_coverage > 0:
                takeaways.append(
                    f"📚 Knowledge graph: {solution_coverage:.1%} solution coverage. "
                    "Searchable problem-solution mappings available."
                )

        if not takeaways:
            takeaways.append("No significant efficiency opportunities detected at this time.")

        return takeaways

    def export_report_html(self, report_data: dict | None = None) -> str:
        """Export report as HTML format.

        Args:
            report_data: Optional pre-computed report data

        Returns:
            HTML string
        """
        if report_data is None:
            report_data = self.generate_report()

        # Generate HTML
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>Efficiency and Innovation Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "h1 { color: #333; }",
            "h2 { color: #666; margin-top: 30px; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            ".high-risk { background-color: #ffcccc; }",
            ".medium-risk { background-color: #fff4cc; }",
            ".low-risk { background-color: #ccffcc; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{report_data.get('report_type', 'Efficiency and Innovation Report')}</h1>",
            f"<p><strong>Generated:</strong> {report_data.get('timestamp', 'N/A')}</p>",
            f"<p><strong>Target Audience:</strong> {report_data.get('target_audience', 'COO')}</p>",
            "<hr>",
            "<h2>Executive Summary</h2>",
            f"<p>{report_data.get('executive_summary', 'N/A')}</p>",
        ]

        # Add key takeaways
        key_takeaways = report_data.get("key_takeaways", [])
        if key_takeaways:
            html_parts.append("<h2>Key Takeaways</h2>")
            html_parts.append("<ul>")
            for takeaway in key_takeaways:
                html_parts.append(f"<li>{takeaway}</li>")
            html_parts.append("</ul>")

        # Add high-risk silos
        high_risk_silos = report_data.get("high_risk_silos", [])
        if high_risk_silos:
            html_parts.append("<h2>High-Risk Silos</h2>")
            html_parts.append("<table>")
            html_parts.append(
                "<tr><th>Community A</th><th>Community B</th><th>Topic Similarity</th>"
                "<th>Bridging Ratio</th><th>Risk Level</th><th>Recommendation</th></tr>"
            )
            for silo in high_risk_silos[:10]:
                risk_class = f"{silo['risk_level']}-risk"
                html_parts.append(
                    f"<tr class='{risk_class}'>"
                    f"<td>{silo['community_a']}</td>"
                    f"<td>{silo['community_b']}</td>"
                    f"<td>{silo['topic_similarity']:.1%}</td>"
                    f"<td>{silo['bridging_ratio']:.1%}</td>"
                    f"<td>{silo['risk_level']}</td>"
                    f"<td>{silo['recommendation']}</td>"
                    "</tr>"
                )
            html_parts.append("</table>")

        # Add priority recommendations
        priority_recs = report_data.get("priority_recommendations", [])
        if priority_recs:
            html_parts.append("<h2>Priority Recommendations</h2>")
            html_parts.append("<ol>")
            for rec in priority_recs[:10]:
                html_parts.append(f"<li><strong>{rec.get('title', 'N/A')}</strong><br>")
                html_parts.append(f"{rec.get('description', 'N/A')}<br>")
                html_parts.append(f"<em>Action: {rec.get('action', 'N/A')}</em></li>")
            html_parts.append("</ol>")

        html_parts.extend(["</body>", "</html>"])

        return "\n".join(html_parts)
