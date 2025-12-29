"""
Example usage of the Organizational Network Analysis platform.

This script demonstrates how to use the ONA platform to analyze
organizational networks from various data sources.
"""

from onapy.core import OrganizationalNetworkAnalyzer
from onapy.visualization.network import NetworkVisualizer
from onapy.ml.embeddings import NodeEmbedder
from onapy.ml.link_prediction import LinkPredictor
from onapy.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Main example function."""
    # Initialize analyzer
    logger.info("Initializing Organizational Network Analyzer...")
    analyzer = OrganizationalNetworkAnalyzer(config_path="config.yaml")

    # Load data (example paths - replace with your actual data)
    logger.info("Loading data...")
    data_paths = {
        "hris": "data/hris.csv",  # Required: HRIS data for people
        "email": "data/email.csv",  # Optional: Email interactions
        "slack": "data/slack.csv",  # Optional: Slack messages
        "calendar": "data/calendar.csv",  # Optional: Meeting data
        "documents": "data/documents.csv",  # Optional: Document collaboration
        "code": "data/code.csv",  # Optional: Code commits
    }

    try:
        analyzer.load_data(data_paths)
        logger.info(f"Loaded {len(analyzer.people)} people")
        logger.info(f"Loaded {len(analyzer.interactions)} interactions")
        logger.info(f"Loaded {len(analyzer.meetings)} meetings")
    except FileNotFoundError as e:
        logger.warning(f"Data files not found: {e}")
        logger.info("Please provide data files in the expected format.")
        return

    # Build graph
    logger.info("Building organizational graph...")
    graph = analyzer.build_graph()
    logger.info(f"Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Run analysis
    logger.info("Running analysis...")
    results = analyzer.analyze()

    # Display key metrics
    logger.info("=== Analysis Results ===")
    logger.info(f"Number of communities: {results['communities']['num_communities']}")
    logger.info(f"Modularity: {results['communities']['modularity']:.3f}")

    # Top central nodes
    logger.info("=== Top 5 by Betweenness Centrality ===")
    top_betweenness = results["centrality"]["betweenness"].head(5)
    for _, row in top_betweenness.iterrows():
        logger.info(f"{row['node_id']}: {row['betweenness_centrality']:.3f}")

    # Top brokers
    logger.info("=== Top 5 Brokers ===")
    top_brokers = results["brokers"].head(5)
    for _, row in top_brokers.iterrows():
        logger.info(f"{row['node_id']}: Broker Score = {row['broker_score']:.3f}")

    # Generate report
    logger.info("Generating report...")
    report_path = analyzer.generate_report("ona_report.html")
    logger.info(f"Report saved to: {report_path}")

    # Create network visualization
    logger.info("Creating network visualization...")
    visualizer = NetworkVisualizer(graph)
    viz_path = visualizer.create_interactive_network("network_visualization.html")
    logger.info(f"Visualization saved to: {viz_path}")

    # Node embeddings (optional, requires node2vec)
    try:
        logger.info("Generating node embeddings...")
        embedder = NodeEmbedder(graph)
        embeddings = embedder.fit_node2vec(dimensions=64)
        logger.info(f"Generated embeddings: {embeddings.shape}")

        # Find similar nodes
        if graph.number_of_nodes() > 0:
            sample_node = list(graph.nodes())[0]
            similar = embedder.find_similar_nodes(sample_node, top_k=5)
            logger.info(f"Nodes similar to {sample_node}:")
            for node_id, similarity in similar:
                logger.info(f"  {node_id}: {similarity:.3f}")
    except ImportError:
        logger.warning("Node2Vec not available. Skipping embeddings.")

    # Link prediction (optional)
    try:
        logger.info("Running link prediction...")
        predictor = LinkPredictor(graph)
        predictions = predictor.predict_links(top_k=10)
        logger.info("Top 10 predicted links:")
        for _, row in predictions.iterrows():
            logger.info(f"{row['node1']} <-> {row['node2']}: {row['predicted_score']:.3f}")
    except Exception as e:
        logger.error(f"Link prediction error: {e}")

    logger.info("=== Analysis Complete ===")


if __name__ == "__main__":
    main()
