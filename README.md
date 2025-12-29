# Organizational Network Analysis (ONA) Platform

A comprehensive Python application for analyzing organizational networks using machine learning and graph analytics. This platform transforms digital exhaust (emails, Slack messages, calendar data, code commits, etc.) into actionable insights about organizational structure, collaboration patterns, and network dynamics.

## Features

- **Multi-source Data Ingestion**: Supports email, Slack/Teams, calendar, documents, code repositories, and HRIS systems
- **Graph Construction**: Builds weighted, temporal organizational graphs with sophisticated edge weighting
- **Network Metrics**: Computes centrality measures (degree, betweenness, eigenvector, closeness), structural holes, and core-periphery analysis
- **Community Detection**: Multiple algorithms (Louvain, Infomap, Label Propagation, SBM)
- **Machine Learning**: Graph Neural Networks (GCN, GAT), Node2Vec embeddings, link prediction
- **NLP Analysis**: Topic modeling (LDA, BERTopic), expertise inference, sentiment analysis
- **Temporal Analysis**: Change point detection, onboarding integration tracking, network evolution
- **Multi-modal Fusion**: Combines insights from multiple data sources
- **Anomaly Detection**: Detects isolation, overload, and temporal anomalies
- **Intervention Framework**: FINDING → HYPOTHESIS → INTERVENTION → MEASUREMENT workflow
- **Ego Network Analysis**: Personal network analysis for individuals
- **Cross-Modal Validation**: Validates insights across different data sources
- **Team Stability Analysis**: Time-Size Paradox analysis for team retention and stability
- **Bonding/Bridging Analysis**: Analyzes within-group (bonding) vs between-group (bridging) connections
- **Interactive Dashboards**: Web-based visualization and reporting
- **Privacy-First**: Built with privacy and ethics considerations

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

## Quick Start

```python
from orgnet.core import OrganizationalNetworkAnalyzer

# Initialize analyzer
analyzer = OrganizationalNetworkAnalyzer(config_path="config.yaml")

# Load data
analyzer.load_data()

# Build graph
graph = analyzer.build_graph()

# Run analysis
results = analyzer.analyze()

# Generate report
analyzer.generate_report(output_path="report.html")
```

## Testing

Run the test suite:

```bash
pytest
```

For coverage report:

```bash
pytest --cov=orgnet --cov-report=html
```

See [tests/README.md](tests/README.md) for more details.

## Code Quality

Format code with black:

```bash
black orgnet/ tests/ example.py --line-length 100
```

Check code style with flake8:

```bash
flake8 orgnet/ tests/ example.py --max-line-length=100
```

Or use the Makefile:

```bash
make format   # Format code
make lint     # Check style
make test     # Run tests
make check    # Format, lint, and test
```

## Project Structure

```
ONA/
├── orgnet/
│   ├── __init__.py
│   ├── core.py                 # Main analyzer class
│   ├── config.py               # Configuration management
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py        # Data ingestion layer
│   │   ├── models.py           # Data models
│   │   └── processors.py       # Data processing utilities
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── builder.py          # Graph construction
│   │   ├── weights.py          # Edge weight calculations
│   │   └── temporal.py         # Temporal graph handling
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── centrality.py       # Centrality measures
│   │   ├── structural.py       # Structural holes, core-periphery
│   │   └── community.py        # Community detection
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── gnn.py              # Graph Neural Networks
│   │   ├── embeddings.py       # Node2Vec, etc.
│   │   └── link_prediction.py  # Link prediction models
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── topics.py           # Topic modeling
│   │   └── expertise.py        # Expertise inference
│   ├── temporal/
│   │   ├── __init__.py
│   │   ├── change_detection.py # Change point detection
│   │   └── onboarding.py      # Onboarding analysis
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── network.py          # Network visualizations
│   │   └── dashboards.py       # Dashboard components
│   └── api/
│       ├── __init__.py
│       ├── app.py              # Flask API
│       └── routes.py           # API routes
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Configuration

Edit `config.yaml` to configure:
- Data source settings
- Graph construction parameters
- Analysis preferences
- Privacy settings
- API configuration

## API Usage

Start the API server:

```bash
python -m orgnet.api.app
```

The API provides endpoints for:
- `/api/graph` - Get organizational graph
- `/api/metrics` - Compute network metrics
- `/api/communities` - Community detection
- `/api/insights` - Generate insights
- `/api/visualization` - Get visualization data

## Privacy & Ethics

This tool is designed with privacy in mind:
- Aggregates data before storage
- Respects retention policies
- Focuses on patterns, not individual monitoring
- Configurable anonymization

**Important**: Always obtain proper consent and follow organizational policies before deploying.

## License

MIT License

## Contributing

Contributions welcome! Please read the contributing guidelines first.


