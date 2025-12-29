# Quick Start Guide

Get started with the Organizational Network Analysis platform in minutes.

## Installation

1. **Clone or download the repository**
   ```bash
   cd ONA
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy language model** (for NLP features)
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Basic Usage

### 1. Prepare Your Data

Create CSV files in the `data/` directory following the format described in `DATA_FORMAT.md`.

**Minimum required**: HRIS data file (`data/hris.csv`)

**Optional but recommended**: Email, Slack, Calendar, Documents, and Code data.

### 2. Run Analysis

#### Option A: Using Python Script

```python
from orgnet.core import OrganizationalNetworkAnalyzer

# Initialize
analyzer = OrganizationalNetworkAnalyzer()

# Load data
analyzer.load_data({
    'hris': 'data/hris.csv',
    'email': 'data/email.csv',
    'slack': 'data/slack.csv'
})

# Build graph
graph = analyzer.build_graph()

# Run analysis
results = analyzer.analyze()

# Generate report
analyzer.generate_report('report.html')
```

#### Option B: Using Example Script

```bash
python example.py
```

### 3. Start API Server

```bash
python -m orgnet.api.app
```

The API will be available at `http://localhost:5000`

**API Endpoints:**
- `GET /api/health` - Health check
- `GET /api/graph` - Get graph data
- `GET /api/metrics` - Get network metrics
- `GET /api/communities` - Get community detection results
- `GET /api/insights` - Get organizational insights
- `POST /api/load_data` - Load data files
- `POST /api/build_graph` - Build graph

## Configuration

Edit `config.yaml` to customize:

- **Data sources**: Enable/disable specific data sources
- **Graph construction**: Adjust edge weight parameters
- **Analysis settings**: Choose community detection method, number of topics, etc.
- **Privacy settings**: Configure data retention and anonymization
- **API settings**: Configure server host and port

## Example Workflow

1. **Data Collection**: Export data from your systems (email, Slack, calendar, etc.)

2. **Data Preparation**: Convert to CSV format following `DATA_FORMAT.md`

3. **Configuration**: Edit `config.yaml` to match your needs

4. **Analysis**: Run the analyzer to generate insights

5. **Review**: Check the generated HTML report and visualizations

6. **Action**: Use insights to make organizational improvements

## Output Files

After running analysis, you'll get:

- `ona_report.html` - Comprehensive HTML report with metrics and insights
- `network_visualization.html` - Interactive network visualization
- Console output with key metrics and findings

## Key Features

### Network Metrics
- **Centrality**: Degree, betweenness, eigenvector, closeness, PageRank
- **Structural Analysis**: Constraint (structural holes), core-periphery
- **Community Detection**: Louvain, Infomap, Label Propagation

### Machine Learning
- **Node Embeddings**: Node2Vec for finding similar organizational roles
- **Link Prediction**: Predict missing or future connections
- **Graph Neural Networks**: GCN and GAT models (optional)

### NLP Analysis
- **Topic Modeling**: BERTopic or LDA to identify discussion topics
- **Expertise Inference**: Map people to knowledge domains

### Temporal Analysis
- **Change Detection**: Identify organizational restructuring events
- **Onboarding Tracking**: Monitor new hire integration

### Visualization
- **Interactive Networks**: Pyvis-based network visualizations
- **Dashboards**: Organizational health metrics
- **Reports**: Executive summaries with actionable insights

## Troubleshooting

### Import Errors
If you get import errors for optional dependencies:
- Some features require additional packages (e.g., `node2vec`, `bertopic`)
- Install them separately: `pip install node2vec bertopic`
- Or use features that don't require them

### Data Format Issues
- Ensure CSV files follow the format in `DATA_FORMAT.md`
- Check that person IDs match across all files
- Verify timestamp formats are parseable by pandas

### Memory Issues
- For large organizations (>1000 people), consider:
  - Processing data in batches
  - Using a subset of data sources
  - Increasing system memory

## Next Steps

1. Read the full `README.md` for detailed documentation
2. Review `DATA_FORMAT.md` for data requirements
3. Explore the `example.py` script for more usage examples
4. Customize `config.yaml` for your organization's needs

## Support

For issues or questions:
- Check the documentation in `README.md`
- Review example code in `example.py`
- Ensure all dependencies are installed correctly

## Privacy & Ethics

⚠️ **Important**: Before deploying:
- Obtain proper consent from employees
- Follow organizational privacy policies
- Comply with data protection regulations (GDPR, etc.)
- Use aggregated insights, not individual monitoring
- Review the privacy settings in `config.yaml`

