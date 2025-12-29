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

Note: Visualization requires optional dependencies. Install with:

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

