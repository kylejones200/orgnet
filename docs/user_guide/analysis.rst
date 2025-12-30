Analysis
========

Running Analysis
----------------

After building the graph, run comprehensive analysis:

.. code-block:: python

   results = analyzer.analyze()

Analysis Results
----------------

The analysis includes:

- **Centrality Metrics**: Degree, betweenness, eigenvector, closeness, PageRank
- **Community Detection**: Identifies teams and groups
- **Structural Analysis**: Core-periphery, structural holes
- **Anomaly Detection**: Isolation, overload, temporal anomalies

Accessing Results
-----------------

.. code-block:: python

   # Centrality results
   centrality = results["centrality"]
   top_betweenness = centrality["betweenness"].head(10)
   
   # Community results
   communities = results["communities"]
   num_communities = communities["num_communities"]
   modularity = communities["modularity"]
   
   # Structural analysis
   brokers = results["brokers"]


