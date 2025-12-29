Architecture
============

System Overview
---------------

orgnet is designed with a modular architecture that separates concerns and allows for flexible extension.

Key Components
--------------

Data Layer
~~~~~~~~~~

- **Data Ingestion**: Multi-source data loading (email, Slack, calendar, etc.)
- **Data Models**: Structured data models for people, interactions, meetings, documents
- **Data Processors**: Utilities for data transformation and processing

Graph Layer
~~~~~~~~~~~

- **Graph Builder**: Constructs organizational networks from data
- **Edge Weights**: Sophisticated weighting algorithms
- **Temporal Graphs**: Time-aware graph construction

Analysis Layer
~~~~~~~~~~~~~~

- **Metrics**: Centrality, structural, community detection
- **ML**: Graph neural networks, embeddings, link prediction
- **NLP**: Topic modeling, expertise inference
- **Temporal**: Change detection, onboarding analysis

Visualization Layer
~~~~~~~~~~~~~~~~~~~

- **Network Visualization**: Interactive network graphs
- **Dashboards**: Executive summaries and health metrics

API Layer
~~~~~~~~~

- **REST API**: Flask-based API for programmatic access
- **Web Interface**: Interactive dashboards

For detailed architecture documentation, see `ARCHITECTURE.md <https://github.com/kylejones200/orgnet/blob/main/ARCHITECTURE.md>`_.

