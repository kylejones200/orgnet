Graph Construction
==================

Building Organizational Networks
--------------------------------

After loading data, build the organizational graph:

.. code-block:: python

   graph = analyzer.build_graph()
   
   print(f"Nodes: {graph.number_of_nodes()}")
   print(f"Edges: {graph.number_of_edges()}")

Graph Structure
---------------

The graph is a weighted NetworkX graph where:
- **Nodes**: Represent people in the organization
- **Edges**: Represent relationships (communication, collaboration, meetings)
- **Weights**: Edge weights reflect relationship strength

Edge Weighting
--------------

Edge weights are computed from:
- Communication frequency (email, Slack)
- Meeting co-attendance
- Document collaboration
- Code collaboration
- Recency (more recent interactions weighted higher)

