"""Core Organizational Network Analyzer."""

import networkx as nx
from typing import Dict, List, Optional

from orgnet.config import Config
from orgnet.data.ingestion import DataIngester
from orgnet.graph.builder import GraphBuilder
from orgnet.graph.temporal import TemporalGraph
from orgnet.insights.ego_networks import EgoNetworkAnalyzer
from orgnet.insights.interventions import InterventionFramework
from orgnet.insights.team_stability import TeamStabilityAnalyzer
from orgnet.insights.validation import CrossModalValidator
from orgnet.metrics.anomaly import AnomalyDetector
from orgnet.metrics.bonding_bridging import BondingBridgingAnalyzer
from orgnet.metrics.centrality import CentralityAnalyzer
from orgnet.metrics.community import CommunityDetector
from orgnet.metrics.structural import StructuralAnalyzer
from orgnet.utils.logging import get_logger

try:
    from orgnet.visualization.dashboards import DashboardGenerator
except ImportError:
    DashboardGenerator = None  # Optional dependency

logger = get_logger(__name__)


class OrganizationalNetworkAnalyzer:
    """Main class for organizational network analysis."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ONA analyzer.

        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        self.data_ingester = DataIngester(self.config)
        self.graph_builder = GraphBuilder(self.config)

        # Data storage
        self.people: List = []
        self.interactions: List = []
        self.meetings: List = []
        self.documents: List = []
        self.commits: List = []
        self.hris_records: List = []

        # Graph
        self.graph: Optional[nx.Graph] = None
        self.temporal_graph: Optional[TemporalGraph] = None

        # Analysis results
        self.centrality_results: Optional[Dict] = None
        self.structural_results: Optional[Dict] = None
        self.community_results: Optional[Dict] = None

    def load_data(self, data_paths: Optional[Dict[str, str]] = None):
        """
        Load data from various sources.

        Args:
            data_paths: Dictionary mapping source name to file path
                       Keys: 'email', 'slack', 'calendar', 'documents', 'code', 'hris'
        """
        if data_paths is None:
            data_paths = {}

        # Load HRIS first (needed for people)
        if "hris" in data_paths:
            self.hris_records = self.data_ingester.ingest_hris(data_path=data_paths["hris"])
            self.people = self.data_ingester.create_people_from_hris(self.hris_records)

        # Load other data sources using loop for cleaner code
        source_loaders = {
            "email": lambda path: self.interactions.extend(
                self.data_ingester.ingest_email(data_path=path)
            ),
            "slack": lambda path: self.interactions.extend(
                self.data_ingester.ingest_slack(data_path=path)
            ),
            "calendar": lambda path: setattr(
                self, "meetings", self.data_ingester.ingest_calendar(data_path=path)
            ),
            "documents": lambda path: setattr(
                self, "documents", self.data_ingester.ingest_documents(data_path=path)
            ),
            "code": lambda path: setattr(
                self, "commits", self.data_ingester.ingest_code(data_path=path)
            ),
        }

        for source, loader in source_loaders.items():
            if source in data_paths:
                loader(data_paths[source])

    def build_graph(self) -> nx.Graph:
        """
        Build organizational graph from loaded data.

        Returns:
            NetworkX graph
        """
        if not self.people:
            raise ValueError("No people data loaded. Load HRIS data first.")

        self.graph = self.graph_builder.build_graph(
            people=self.people,
            interactions=self.interactions if self.interactions else None,
            meetings=self.meetings if self.meetings else None,
            documents=self.documents if self.documents else None,
            commits=self.commits if self.commits else None,
        )

        return self.graph

    def analyze(self) -> Dict:
        """
        Run comprehensive analysis.

        Returns:
            Dictionary with all analysis results
        """
        if self.graph is None:
            self.build_graph()

        results = {}

        # Centrality analysis
        centrality_analyzer = CentralityAnalyzer(self.graph)
        self.centrality_results = centrality_analyzer.compute_all_centralities()
        results["centrality"] = self.centrality_results

        # Structural analysis
        structural_analyzer = StructuralAnalyzer(self.graph)
        results["constraint"] = structural_analyzer.compute_constraint()
        results["core_periphery"] = structural_analyzer.compute_core_periphery()

        # Community detection
        community_detector = CommunityDetector(self.graph)
        method = self.config.analysis_config.get("community", {}).get("method", "louvain")
        resolution = self.config.analysis_config.get("community", {}).get("resolution", 1.0)
        self.community_results = community_detector.detect_communities(
            method=method, resolution=resolution
        )
        results["communities"] = self.community_results

        # Broker identification
        betweenness_df = self.centrality_results["betweenness"]
        constraint_df = results["constraint"]
        results["brokers"] = structural_analyzer.identify_brokers(betweenness_df, constraint_df)

        return results

    def detect_anomalies(self) -> Dict:
        """
        Detect anomalies in the organizational network.

        Returns:
            Dictionary with anomaly detection results
        """
        if self.graph is None:
            self.build_graph()

        anomaly_detector = AnomalyDetector(self.graph, self.people)
        anomalies = anomaly_detector.detect_all_anomalies()

        return anomalies

    def analyze_ego_network(self, node_id: str) -> Dict:
        """
        Analyze ego network for a specific individual.

        Args:
            node_id: Person ID

        Returns:
            Dictionary with ego network metrics
        """
        if self.graph is None:
            self.build_graph()

        ego_analyzer = EgoNetworkAnalyzer(self.graph)
        return ego_analyzer.analyze_ego_network(node_id)

    def create_interventions(self, findings: Optional[List[Dict]] = None) -> InterventionFramework:
        """
        Create intervention framework and suggest interventions.

        Args:
            findings: Optional list of findings. If None, will use detected anomalies.

        Returns:
            InterventionFramework instance
        """
        if findings is None:
            # Use anomalies as findings
            anomalies = self.detect_anomalies()
            findings = []

            # Convert node anomalies to findings (vectorized)
            if "node" in anomalies and not anomalies["node"].empty:
                findings = anomalies["node"][["type", "description"]].to_dict("records")

        framework = InterventionFramework()
        framework.suggest_interventions_from_findings(findings)

        return framework

    def validate_insights_cross_modal(
        self, finding: str, evidence_by_modality: Dict[str, bool]
    ) -> Dict:
        """
        Validate insights across multiple data modalities.

        Args:
            finding: Description of the finding
            evidence_by_modality: Dictionary mapping modality to evidence (True/False)

        Returns:
            Validation result
        """
        validator = CrossModalValidator()
        return validator.validate_finding(finding, evidence_by_modality)

    def analyze_team_stability(self, team_attribute: str = "team") -> Dict:
        """
        Analyze team stability based on size and tenure (Time-Size Paradox).

        Args:
            team_attribute: Attribute to group by ('team', 'department', etc.)

        Returns:
            Dictionary with stability analysis results
        """
        if self.graph is None:
            self.build_graph()

        analyzer = TeamStabilityAnalyzer(self.graph, self.people)
        stability_df = analyzer.analyze_team_stability(team_attribute)
        at_risk = analyzer.identify_at_risk_teams()
        relationship = analyzer.analyze_size_tenure_relationship()

        return {
            "stability_metrics": stability_df,
            "at_risk_teams": at_risk,
            "size_tenure_relationship": relationship,
        }

    def analyze_bonding_bridging(self, use_formal_structure: bool = False) -> Dict:
        """
        Analyze bonding (within-group) and bridging (between-group) connections.

        Args:
            use_formal_structure: If True, use formal structure instead of detected communities

        Returns:
            Dictionary with bonding/bridging analysis results
        """
        if self.graph is None:
            self.build_graph()

        analyzer = BondingBridgingAnalyzer(self.graph)

        # Get communities if needed
        communities = None
        if not use_formal_structure and self.community_results:
            communities = self.community_results

        node_analysis = analyzer.analyze_bonding_bridging(
            communities=communities, use_formal_structure=use_formal_structure
        )
        group_analysis = analyzer.analyze_group_bonding_bridging(communities=communities)
        network_ratio = analyzer.calculate_network_bonding_bridging_ratio()

        return {
            "node_analysis": node_analysis,
            "group_analysis": group_analysis,
            "network_ratio": network_ratio,
            "top_bridges": analyzer.identify_bridges(),
            "top_bonders": analyzer.identify_bonders(),
        }

    def generate_report(self, output_path: str = "ona_report.html"):
        """
        Generate comprehensive HTML report.

        Args:
            output_path: Path to save report
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build_graph() first.")

        # Run analysis if not done
        if self.centrality_results is None:
            self.analyze()

        # Generate dashboard
        if DashboardGenerator is None:
            raise ImportError(
                "DashboardGenerator requires visualization dependencies. Install with: pip install matplotlib"
            )
        dashboard = DashboardGenerator(self.graph)
        executive_summary = dashboard.generate_executive_summary()
        health_metrics = dashboard.generate_health_dashboard()

        # Create HTML report
        html_content = self._create_html_report(executive_summary, health_metrics)

        # Save report
        with open(output_path, "w") as f:
            f.write(html_content)

        return output_path

    def _create_html_report(self, executive_summary: Dict, health_metrics: Dict) -> str:
        """Create HTML report content."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Organizational Network Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #ecf0f1; border-radius: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
        .metric-label {{ font-size: 12px; color: #7f8c8d; }}
        .finding {{ margin: 10px 0; padding: 10px; border-left: 4px solid #3498db; background: #ebf5fb; }}
        .warning {{ border-left-color: #e74c3c; background: #fadbd8; }}
        .info {{ border-left-color: #f39c12; background: #fef5e7; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
    </style>
</head>
<body>
    <h1>Organizational Network Analysis Report</h1>
    <p>Generated: {executive_summary['timestamp']}</p>

    <h2>Executive Summary</h2>
    <p><strong>Overall Status:</strong> {executive_summary['status'].upper()}</p>

    <h2>Health Metrics</h2>
    <div class="metric">
        <div class="metric-value">{health_metrics['network_density']:.4f}</div>
        <div class="metric-label">Network Density</div>
    </div>
    <div class="metric">
        <div class="metric-value">{health_metrics['modularity']:.2f}</div>
        <div class="metric-label">Modularity</div>
    </div>
    <div class="metric">
        <div class="metric-value">{health_metrics['avg_path_length']:.2f}</div>
        <div class="metric-label">Avg Path Length</div>
    </div>
    <div class="metric">
        <div class="metric-value">{health_metrics['num_communities']}</div>
        <div class="metric-label">Communities</div>
    </div>

    <h2>Key Findings</h2>
"""

        for finding in executive_summary.get("key_findings", []):
            finding_class = finding.get("type", "info")
            html += f"""
    <div class="finding {finding_class}">
        <h3>{finding['title']}</h3>
        <p>{finding['description']}</p>
        <p><strong>Recommendation:</strong> {finding['recommendation']}</p>
    </div>
"""

        html += """
    <h2>Top Brokers</h2>
    <table>
        <tr>
            <th>Person ID</th>
            <th>Broker Score</th>
            <th>Betweenness</th>
            <th>Constraint</th>
        </tr>
"""

        for broker in executive_summary.get("top_brokers", [])[:10]:
            html += f"""
        <tr>
            <td>{broker.get('node_id', 'N/A')}</td>
            <td>{broker.get('broker_score', 0):.3f}</td>
            <td>{broker.get('betweenness_centrality', 0):.3f}</td>
            <td>{broker.get('constraint', 0):.3f}</td>
        </tr>
"""

        html += """
    </table>
</body>
</html>
"""

        return html
