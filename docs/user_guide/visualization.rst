Visualization
=============

Network Visualization
---------------------

Create interactive network visualizations:

.. code-block:: python

   from orgnet.visualization.network import NetworkVisualizer
   
   visualizer = NetworkVisualizer(graph)
   
   # Create interactive HTML visualization
   viz_path = visualizer.create_interactive_network("network.html")

Note: Visualization optional dependencies include Matplotlib, Seaborn, PlotSmith (for styled static charts on Python 3.12+), and Pyvis (interactive network HTML). Plotly is not used.

.. code-block:: bash

   pip install orgnet[viz]

Dashboards
----------

Generate executive dashboards:

.. code-block:: python

   from orgnet.visualization.dashboards import DashboardGenerator
   
   dashboard = DashboardGenerator(graph)
   summary = dashboard.generate_executive_summary()
   health = dashboard.generate_health_dashboard()

Reports
-------

Generate HTML reports:

.. code-block:: python

   analyzer.generate_report("report.html")


