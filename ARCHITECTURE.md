# Architecture Overview

This document describes the architecture and design of the Organizational Network Analysis (ONA) platform.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │
│  │  Email   │ │  Slack/  │ │ Calendar │ │  Docs    │         │
│  │          │ │  Teams   │ │          │ │          │         │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘         │
│       │            │            │            │                │
│       └────────────┴────────────┴────────────┘                │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  DATA INGESTION LAYER                         │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  DataIngester                                          │  │
│  │  - Email ingestion                                     │  │
│  │  - Slack/Teams ingestion                              │  │
│  │  - Calendar ingestion                                 │  │
│  │  - Document ingestion                                 │  │
│  │  - Code repository ingestion                          │  │
│  │  - HRIS ingestion                                     │  │
│  └────────────────────────────────────────────────────────┘  │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  DATA PROCESSING LAYER                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  DataProcessor                                         │  │
│  │  - Interaction matrices                               │  │
│  │  - Co-attendance matrices                             │  │
│  │  - Collaboration matrices                             │  │
│  │  - Response time computation                          │  │
│  │  - Reciprocity computation                            │  │
│  └────────────────────────────────────────────────────────┘  │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  GRAPH CONSTRUCTION LAYER                     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  GraphBuilder                                          │  │
│  │  - Multi-layer graph construction                      │  │
│  │  - Edge weight calculation                            │  │
│  │  - Layer fusion                                       │  │
│  │  - Temporal graph management                          │  │
│  └────────────────────────────────────────────────────────┘  │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    ANALYTICS LAYER                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │  Centrality │ │  Community  │ │  Structural │          │
│  │  Metrics    │ │  Detection  │ │  Analysis   │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │    ML       │ │  Temporal   │ │    NLP      │          │
│  │  Models     │ │  Analysis   │ │  Pipeline   │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  PRESENTATION LAYER                          │
│  ┌─────────────────┐ ┌─────────────────┐                  │
│  │   Dashboards    │ │   Visualizations │                  │
│  │   (HTML/API)    │ │   (Interactive)   │                  │
│  └─────────────────┘ └─────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

## Module Structure

### `onapy.data`
**Purpose**: Data ingestion and processing

- **`models.py`**: Data models (Person, Interaction, Meeting, Document, etc.)
- **`ingestion.py`**: Data ingestion from various sources
- **`processors.py`**: Data processing utilities (matrices, statistics)

### `onapy.graph`
**Purpose**: Graph construction and manipulation

- **`builder.py`**: Builds organizational graphs from data
- **`weights.py`**: Calculates edge weights with recency, reciprocity, responsiveness
- **`temporal.py`**: Manages temporal graph snapshots

### `onapy.metrics`
**Purpose**: Network metrics computation

- **`centrality.py`**: Centrality measures (degree, betweenness, eigenvector, closeness, PageRank)
- **`structural.py`**: Structural analysis (constraint, core-periphery, brokers)
- **`community.py`**: Community detection (Louvain, Infomap, Label Propagation, SBM)

### `onapy.ml`
**Purpose**: Machine learning models

- **`embeddings.py`**: Node embeddings (Node2Vec)
- **`link_prediction.py`**: Link prediction models
- **`gnn.py`**: Graph Neural Networks (GCN, GAT)

### `onapy.nlp`
**Purpose**: Natural language processing

- **`topics.py`**: Topic modeling (BERTopic, LDA)
- **`expertise.py`**: Expertise inference from communications

### `onapy.temporal`
**Purpose**: Temporal analysis

- **`change_detection.py`**: Change point detection in network evolution
- **`onboarding.py`**: New hire integration analysis

### `onapy.visualization`
**Purpose**: Visualization and reporting

- **`network.py`**: Network visualizations (matplotlib, Pyvis)
- **`dashboards.py`**: Dashboard generation and executive summaries

### `onapy.api`
**Purpose**: Web API interface

- **`app.py`**: Flask API with REST endpoints

### `onapy.core`
**Purpose**: Main orchestrator

- **`core.py`**: `OrganizationalNetworkAnalyzer` - main class that coordinates all modules

## Data Flow

1. **Ingestion**: Raw data from various sources → Data models
2. **Processing**: Data models → Interaction/collaboration matrices
3. **Graph Construction**: Matrices → Weighted graph with multiple layers
4. **Analysis**: Graph → Metrics, communities, insights
5. **Visualization**: Analysis results → Reports and dashboards

## Key Design Decisions

### 1. Modular Architecture
- Each module is independent and can be used separately
- Clear separation of concerns
- Easy to extend with new data sources or analysis methods

### 2. Configuration-Driven
- All settings in `config.yaml`
- Easy to customize without code changes
- Supports different deployment scenarios

### 3. Privacy-First
- Aggregation before storage
- Configurable retention policies
- Focus on patterns, not individuals

### 4. Extensible
- Plugin-style architecture for new data sources
- Multiple algorithm options for each analysis type
- Easy to add new metrics or visualizations

### 5. Production-Ready
- Error handling throughout
- Graceful degradation for missing dependencies
- API interface for integration
- Comprehensive documentation

## Dependencies

### Core (Required)
- `networkx`: Graph operations
- `pandas`: Data manipulation
- `numpy`: Numerical computations

### Analysis (Optional but Recommended)
- `python-igraph`: Advanced community detection
- `node2vec`: Node embeddings
- `scikit-learn`: Machine learning utilities

### ML (Optional)
- `torch`, `torch-geometric`: Graph Neural Networks
- `transformers`, `sentence-transformers`: NLP models

### NLP (Optional)
- `bertopic`: Modern topic modeling
- `gensim`: LDA topic modeling
- `spacy`: NLP processing

### Visualization (Optional)
- `matplotlib`: Static plots
- `pyvis`: Interactive network visualization
- `plotly`: Interactive charts

### API (Optional)
- `flask`: Web framework
- `flask-cors`: CORS support

## Performance Considerations

- **Graph Size**: Optimized for organizations up to ~10,000 people
- **Memory**: Scales with graph size; consider batching for very large orgs
- **Computation**: Most algorithms are O(n²) or better
- **Storage**: Minimal - processes data in memory

## Security & Privacy

- No data persistence by default
- Configurable retention policies
- Aggregation before analysis
- Anonymization support
- Access control via API authentication (to be implemented)

## Future Enhancements

- Real-time data streaming
- Database backend for large-scale deployments
- Advanced GNN models
- More visualization options
- Integration with common enterprise tools (Slack API, Microsoft Graph, etc.)
- User authentication and authorization
- Multi-tenant support

