"""
Happy Path Tutorial: Complete ONA Analysis from CSV to Report

This tutorial demonstrates the complete workflow from raw CSV files
to final HTML report and dashboard, using a tiny fake organization.
"""

from orgnet.core import OrganizationalNetworkAnalyzer
from orgnet.utils.logging import get_logger
import pandas as pd
from datetime import datetime, timedelta
import os

logger = get_logger(__name__)


def create_sample_data():
    """Create sample data for a tiny fake organization."""
    # Create data directory in examples/
    tutorial_data_dir = os.path.join(os.path.dirname(__file__), "tutorial_data")
    os.makedirs(tutorial_data_dir, exist_ok=True)

    # Sample HRIS data
    hris_data = {
        "person_id": ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"],
        "department": [
            "Engineering",
            "Engineering",
            "Product",
            "Product",
            "Sales",
            "Sales",
            "Engineering",
            "Product",
        ],
        "role": [
            "Engineer",
            "Manager",
            "Product Manager",
            "Designer",
            "Sales Rep",
            "Sales Manager",
            "Engineer",
            "Product Manager",
        ],
        "start_date": [
            "2024-01-01",
            "2023-06-01",
            "2024-02-01",
            "2023-12-01",
            "2024-03-01",
            "2023-01-01",
            "2024-04-01",
            "2023-09-01",
        ],
    }
    hris_df = pd.DataFrame(hris_data)
    hris_path = os.path.join(tutorial_data_dir, "hris.csv")
    hris_df.to_csv(hris_path, index=False)
    logger.info(f"Created {hris_path}")

    # Sample email interactions
    base_date = datetime(2024, 1, 1)
    email_data = []
    for i in range(100):
        date = base_date + timedelta(days=i % 30)
        sender = f"p{(i % 8) + 1}"
        receiver = f"p{((i + 1) % 8) + 1}"
        if sender != receiver:
            email_data.append(
                {
                    "sender_id": sender,
                    "receiver_id": receiver,
                    "timestamp": date.isoformat(),
                    "subject": f"Email {i}",
                }
            )

    email_df = pd.DataFrame(email_data)
    email_path = os.path.join(tutorial_data_dir, "email.csv")
    email_df.to_csv(email_path, index=False)
    logger.info(f"Created {email_path}")

    # Sample calendar data
    calendar_data = []
    for i in range(20):
        date = base_date + timedelta(days=i * 2)
        organizer = f"p{(i % 8) + 1}"
        attendees = [f"p{((i + j) % 8) + 1}" for j in range(3)]
        calendar_data.append(
            {
                "organizer_id": organizer,
                "attendee_ids": ",".join(attendees),
                "start_time": date.isoformat(),
                "duration_minutes": 60,
            }
        )

    calendar_df = pd.DataFrame(calendar_data)
    calendar_path = os.path.join(tutorial_data_dir, "calendar.csv")
    calendar_df.to_csv(calendar_path, index=False)
    logger.info(f"Created {calendar_path}")

    logger.info(f"Sample data created in {tutorial_data_dir}/ directory")


def run_tutorial():
    """Run the complete happy path tutorial."""
    logger.info("=" * 60)
    logger.info("ORGANIZATIONAL NETWORK ANALYSIS - HAPPY PATH TUTORIAL")
    logger.info("=" * 60)

    # Step 1: Create sample data
    logger.info("\n[Step 1] Creating sample data...")
    create_sample_data()

    # Step 2: Initialize analyzer
    logger.info("\n[Step 2] Initializing Organizational Network Analyzer...")
    analyzer = OrganizationalNetworkAnalyzer(config_path="config.yaml")

    # Step 3: Load data
    logger.info("\n[Step 3] Loading data from CSV files...")
    tutorial_data_dir = os.path.join(os.path.dirname(__file__), "tutorial_data")
    data_paths = {
        "hris": os.path.join(tutorial_data_dir, "hris.csv"),
        "email": os.path.join(tutorial_data_dir, "email.csv"),
        "calendar": os.path.join(tutorial_data_dir, "calendar.csv"),
    }

    try:
        analyzer.load_data(data_paths)
        logger.info(f"✓ Loaded {len(analyzer.people)} people")
        logger.info(f"✓ Loaded {len(analyzer.interactions)} interactions")
        logger.info(f"✓ Loaded {len(analyzer.meetings)} meetings")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    # Step 4: Build graph
    logger.info("\n[Step 4] Building organizational graph...")
    try:
        graph = analyzer.build_graph()
        logger.info(
            f"✓ Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
    except Exception as e:
        logger.error(f"Error building graph: {e}")
        return

    # Step 5: Run analysis
    logger.info("\n[Step 5] Running comprehensive analysis...")
    try:
        results = analyzer.analyze()
        logger.info("✓ Analysis complete")
        logger.info(f"  - Found {results['communities']['num_communities']} communities")
        logger.info(f"  - Modularity: {results['communities']['modularity']:.3f}")
        logger.info(f"  - Top 5 by betweenness:")
        top_betweenness = results["centrality"]["betweenness"].head(5)
        for _, row in top_betweenness.iterrows():
            logger.info(
                f"    {row['node_id']}: {row.get('value', row.get('betweenness_centrality', 0)):.3f}"
            )
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        return

    # Step 6: Generate report
    logger.info("\n[Step 6] Generating HTML report...")
    try:
        output_dir = os.path.dirname(__file__)
        report_path = analyzer.generate_report(os.path.join(output_dir, "tutorial_report.html"))
        logger.info(f"✓ Report generated: {report_path}")
        logger.info(
            "  Report includes: network map, centrality tables, community list, and insights"
        )
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return

    # Step 7: Create network visualization
    logger.info("\n[Step 7] Creating network visualization...")
    try:
        from orgnet.visualization.network import NetworkVisualizer

        output_dir = os.path.dirname(__file__)
        visualizer = NetworkVisualizer(graph)
        viz_path = visualizer.create_interactive_network(
            os.path.join(output_dir, "tutorial_network.html")
        )
        logger.info(f"✓ Network visualization: {viz_path}")
    except Exception as e:
        logger.warning(f"Visualization not available: {e}")

    # Step 8: Generate dashboard
    logger.info("\n[Step 8] Generating dashboard...")
    try:
        from orgnet.visualization.dashboard_html import generate_dashboard_html

        output_dir = os.path.dirname(__file__)
        dashboard_path = generate_dashboard_html(
            api_base_url="http://localhost:5000",
            output_path=os.path.join(output_dir, "tutorial_dashboard.html"),
        )
        logger.info(f"✓ Dashboard generated: {dashboard_path}")
        logger.info("  Dashboard includes: global overview, team view, and people view")
    except Exception as e:
        logger.warning(f"Dashboard generation: {e}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TUTORIAL COMPLETE!")
    logger.info("=" * 60)
    output_dir = os.path.dirname(__file__)
    logger.info("\nGenerated files:")
    logger.info(
        f"  - {os.path.join(output_dir, 'tutorial_report.html')} (comprehensive analysis report)"
    )
    logger.info(
        f"  - {os.path.join(output_dir, 'tutorial_network.html')} (interactive network map)"
    )
    logger.info(f"  - {os.path.join(output_dir, 'tutorial_dashboard.html')} (lean dashboard)")
    logger.info("\nNext steps:")
    logger.info("  1. Open tutorial_report.html in a web browser")
    logger.info("  2. Review the network map, centrality tables, and insights")
    logger.info("  3. Explore the dashboard views")
    logger.info(
        f"  4. Try with your own data by replacing {os.path.join(output_dir, 'tutorial_data')}/*.csv files"
    )


if __name__ == "__main__":
    run_tutorial()
