"""Generate lean dashboard HTML with three views.

This module creates a simple dashboard that reads from API endpoints
and shows: global overview, team view, and people view.
"""

from __future__ import annotations

from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


def generate_dashboard_html(
    api_base_url: str = "http://localhost:5000",
    output_path: str = "dashboard.html",
) -> str:
    """
    Generate lean dashboard HTML with three views.

    Args:
        api_base_url: Base URL for API endpoints
        output_path: Path to save dashboard HTML

    Returns:
        Path to saved dashboard file
    """
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Organizational Network Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; }}
        .header h1 {{ margin: 0; }}
        .nav {{ background: #34495e; padding: 10px; }}
        .nav button {{ background: #3498db; color: white; border: none; padding: 10px 20px; margin: 5px; cursor: pointer; border-radius: 4px; }}
        .nav button:hover {{ background: #2980b9; }}
        .nav button.active {{ background: #27ae60; }}
        .view {{ display: none; padding: 20px; }}
        .view.active {{ display: block; }}
        .kpi {{ display: inline-block; background: white; padding: 20px; margin: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); min-width: 150px; }}
        .kpi-value {{ font-size: 32px; font-weight: bold; color: #2980b9; }}
        .kpi-label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; margin-top: 5px; }}
        .network-container {{ background: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .table-container {{ background: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); overflow-x: auto; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #3498db; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .loading {{ text-align: center; padding: 40px; color: #7f8c8d; }}
        .error {{ background: #e74c3c; color: white; padding: 15px; border-radius: 4px; margin: 10px 0; }}
        .insight {{ background: #ebf5fb; border-left: 4px solid #3498db; padding: 15px; margin: 10px 0; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Organizational Network Dashboard</h1>
        <p>Real-time organizational network analysis</p>
    </div>

    <div class="nav">
        <button class="active" onclick="showView('overview')">Global Overview</button>
        <button onclick="showView('team')">Team View</button>
        <button onclick="showView('people')">People View</button>
    </div>

    <!-- Global Overview View -->
    <div id="overview" class="view active">
        <h2>Global Overview</h2>
        <div id="overview-kpis" class="loading">Loading KPIs...</div>
        <div class="network-container">
            <h3>Network Map</h3>
            <p><em>Network visualization loads from /api/graph endpoint</em></p>
            <div id="network-map" style="height: 600px; background: #f9f9f9; border: 1px solid #ddd; border-radius: 4px; display: flex; align-items: center; justify-content: center;">
                <p>Network visualization would be embedded here</p>
            </div>
        </div>
        <div class="table-container">
            <h3>Key Insights</h3>
            <div id="overview-insights" class="loading">Loading insights...</div>
        </div>
    </div>

    <!-- Team View -->
    <div id="team" class="view">
        <h2>Team View</h2>
        <div class="table-container">
            <h3>Cross-Team Connections</h3>
            <div id="team-connections" class="loading">Loading team data...</div>
        </div>
        <div class="table-container">
            <h3>Team Bottlenecks</h3>
            <div id="team-bottlenecks" class="loading">Analyzing bottlenecks...</div>
        </div>
    </div>

    <!-- People View -->
    <div id="people" class="view">
        <h2>People View</h2>
        <div style="margin: 20px 0;">
            <input type="text" id="person-search" placeholder="Search person ID..." style="padding: 10px; width: 300px; border: 1px solid #ddd; border-radius: 4px;">
            <button onclick="loadPersonView()" style="padding: 10px 20px; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; margin-left: 10px;">Load</button>
        </div>
        <div id="people-content" class="loading">Enter a person ID to view their network</div>
    </div>

    <script>
        const API_BASE = '{api_base_url}';

        function showView(viewName) {{
            // Hide all views
            document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
            document.querySelectorAll('.nav button').forEach(b => b.classList.remove('active'));

            // Show selected view
            document.getElementById(viewName).classList.add('active');
            event.target.classList.add('active');

            // Load data for view
            if (viewName === 'overview') loadOverview();
            else if (viewName === 'team') loadTeamView();
        }}

        async function loadOverview() {{
            try {{
                // Load summary
                const summaryRes = await fetch(API_BASE + '/api/summary');
                const summary = await summaryRes.json();

                // Display KPIs
                const kpis = summary.health_metrics || {{}};
                document.getElementById('overview-kpis').innerHTML = `
                    <div class="kpi">
                        <div class="kpi-value">${{kpis.num_nodes || 0}}</div>
                        <div class="kpi-label">People</div>
                    </div>
                    <div class="kpi">
                        <div class="kpi-value">${{kpis.num_edges || 0}}</div>
                        <div class="kpi-label">Connections</div>
                    </div>
                    <div class="kpi">
                        <div class="kpi-value">${{(kpis.network_density || 0).toFixed(3)}}</div>
                        <div class="kpi-label">Density</div>
                    </div>
                    <div class="kpi">
                        <div class="kpi-value">${{(kpis.modularity || 0).toFixed(2)}}</div>
                        <div class="kpi-label">Modularity</div>
                    </div>
                    <div class="kpi">
                        <div class="kpi-value">${{kpis.num_communities || 0}}</div>
                        <div class="kpi-label">Communities</div>
                    </div>
                `;

                // Display insights
                const insights = summary.key_findings || [];
                let insightsHtml = '';
                insights.forEach(finding => {{
                    insightsHtml += `
                        <div class="insight">
                            <h4>${{finding.title}}</h4>
                            <p>${{finding.description}}</p>
                            <p><strong>Recommendation:</strong> ${{finding.recommendation}}</p>
                        </div>
                    `;
                }});
                document.getElementById('overview-insights').innerHTML = insightsHtml || '<p>No insights available</p>';

            }} catch (error) {{
                document.getElementById('overview-kpis').innerHTML = `<div class="error">Error loading data: ${{error.message}}</div>`;
            }}
        }}

        async function loadTeamView() {{
            try {{
                // Load communities
                const commRes = await fetch(API_BASE + '/api/communities');
                const communities = await commRes.json();

                let html = '<table><tr><th>Community ID</th><th>Size</th><th>Modularity</th></tr>';
                html += `<tr><td>Overall</td><td>${{communities.num_communities || 0}}</td><td>${{(communities.modularity || 0).toFixed(3)}}</td></tr>`;
                html += '</table>';

                document.getElementById('team-connections').innerHTML = html;
                document.getElementById('team-bottlenecks').innerHTML = '<p>Bottleneck analysis would be displayed here</p>';

            }} catch (error) {{
                document.getElementById('team-connections').innerHTML = `<div class="error">Error: ${{error.message}}</div>`;
            }}
        }}

        async function loadPersonView() {{
            const personId = document.getElementById('person-search').value;
            if (!personId) {{
                alert('Please enter a person ID');
                return;
            }}

            try {{
                // Load metrics
                const metricsRes = await fetch(API_BASE + '/api/metrics');
                const metrics = await metricsRes.json();

                // Find person in metrics
                let personData = null;
                for (const metricType in metrics.centrality) {{
                    const metricData = metrics.centrality[metricType];
                    const person = metricData.find(p => p.node_id === personId);
                    if (person) {{
                        personData = person;
                        break;
                    }}
                }}

                if (personData) {{
                    document.getElementById('people-content').innerHTML = `
                        <div class="table-container">
                            <h3>Person: ${{personId}}</h3>
                            <p><strong>Role in Network:</strong> This person serves as a key connector in the network.</p>
                            <table>
                                <tr><th>Metric</th><th>Value</th></tr>
                                <tr><td>Betweenness</td><td>${{personData.value || personData.betweenness_centrality || 'N/A'}}</td></tr>
                                <tr><td>Rank</td><td>${{personData.rank || 'N/A'}}</td></tr>
                                <tr><td>Top 5%</td><td>${{personData.top_percentile_flag ? 'Yes' : 'No'}}</td></tr>
                            </table>
                        </div>
                    `;
                }} else {{
                    document.getElementById('people-content').innerHTML = `<div class="error">Person ${{personId}} not found</div>`;
                }}

            }} catch (error) {{
                document.getElementById('people-content').innerHTML = `<div class="error">Error: ${{error.message}}</div>`;
            }}
        }}

        // Load overview on page load
        window.onload = function() {{
            loadOverview();
        }};
    </script>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"Dashboard saved to {output_path}")
    return output_path
