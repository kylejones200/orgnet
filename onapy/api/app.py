"""Flask API application."""

from flask import Flask, jsonify, request
from flask_cors import CORS
from typing import Optional

from onapy.core import OrganizationalNetworkAnalyzer
from onapy.config import Config
from onapy.utils.logging import get_logger

logger = get_logger(__name__)


def create_app(config_path: Optional[str] = None):
    """
    Create Flask application.

    Args:
        config_path: Path to configuration file

    Returns:
        Flask app instance
    """
    app = Flask(__name__)
    CORS(app)

    # Initialize analyzer
    analyzer = OrganizationalNetworkAnalyzer(config_path)

    @app.route("/api/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        logger.info("Health check requested")
        return jsonify({"status": "healthy"})

    @app.route("/api/graph", methods=["GET"])
    def get_graph():
        """Get organizational graph data."""
        if analyzer.graph is None:
            logger.warning("Graph requested but not built")
            return jsonify({"error": "Graph not built"}), 400

        logger.info(
            f"Graph requested: {analyzer.graph.number_of_nodes()} nodes, {analyzer.graph.number_of_edges()} edges"
        )

        # Convert graph to JSON format using list comprehensions
        nodes = [{"id": node_id, **data} for node_id, data in analyzer.graph.nodes(data=True)]

        edges = [
            {"source": u, "target": v, "weight": data.get("weight", 1.0)}
            for u, v, data in analyzer.graph.edges(data=True)
        ]

        return jsonify({"nodes": nodes, "edges": edges})

    @app.route("/api/metrics", methods=["GET"])
    def get_metrics():
        """Get network metrics."""
        if analyzer.graph is None:
            return jsonify({"error": "Graph not built"}), 400

        # Run analysis if needed
        if analyzer.centrality_results is None:
            analyzer.analyze()

        # Convert to JSON-serializable format
        metrics = {
            "centrality": {
                "degree": analyzer.centrality_results["degree"].to_dict("records"),
                "betweenness": analyzer.centrality_results["betweenness"].to_dict("records"),
                "eigenvector": analyzer.centrality_results["eigenvector"].to_dict("records"),
                "closeness": analyzer.centrality_results["closeness"].to_dict("records"),
            }
        }

        return jsonify(metrics)

    @app.route("/api/communities", methods=["GET"])
    def get_communities():
        """Get community detection results."""
        if analyzer.graph is None:
            return jsonify({"error": "Graph not built"}), 400

        if analyzer.community_results is None:
            analyzer.analyze()

        return jsonify(
            {
                "method": analyzer.community_results["method"],
                "num_communities": analyzer.community_results["num_communities"],
                "modularity": analyzer.community_results["modularity"],
                "node_to_community": analyzer.community_results["node_to_community"],
            }
        )

    @app.route("/api/insights", methods=["GET"])
    def get_insights():
        """Get organizational insights."""
        if analyzer.graph is None:
            return jsonify({"error": "Graph not built"}), 400

        from onapy.visualization.dashboards import DashboardGenerator

        dashboard = DashboardGenerator(analyzer.graph)
        summary = dashboard.generate_executive_summary()

        return jsonify(summary)

    @app.route("/api/load_data", methods=["POST"])
    def load_data():
        """Load data from provided paths."""
        data = request.json
        data_paths = data.get("data_paths", {})

        try:
            analyzer.load_data(data_paths)
            return jsonify({"status": "success", "message": "Data loaded successfully"})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/build_graph", methods=["POST"])
    def build_graph():
        """Build organizational graph."""
        try:
            graph = analyzer.build_graph()
            return jsonify(
                {
                    "status": "success",
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                }
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    return app


if __name__ == "__main__":
    app = create_app()
    config = Config()
    api_config = config.api_config

    app.run(
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 5000),
        debug=api_config.get("debug", False),
    )
