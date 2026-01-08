# orgnet Examples

This directory contains example notebooks and sample data generation scripts to help you get started with orgnet.

## Quick Start

### 1. Install Dependencies

First, make sure you have the required packages:

```bash
# Install orgnet and dependencies
pip install -r ../requirements.txt

# Or install just what you need for examples
pip install faker jupyter pandas matplotlib
```

### 2. Generate Sample Data

Generate realistic sample data using Faker:

```bash
cd examples
python generate_sample_data.py
```

This will create sample CSV files in `examples/data/` directory:
- `hris.csv` - 50 people across 7 departments
- `email.csv` - ~250 emails over 90 days
- `slack.csv` - ~400 Slack messages
- `calendar.csv` - ~100 meetings
- `documents.csv` - ~150 documents
- `code.csv` - ~500 code commits

### 3. Run Example Notebooks

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open and run the notebooks in order:

1. **01_basic_analysis.ipynb** - Getting started with orgnet
2. **02_network_metrics.ipynb** - Deep dive into network metrics
3. **03_community_detection.ipynb** - Finding communities and teams
4. **04_visualization.ipynb** - Creating network visualizations

### 4. Run the Happy Path Tutorial

For a complete end-to-end walkthrough, run the happy path tutorial:

```bash
cd examples
python tutorial_happy_path.py
```

This tutorial demonstrates the complete workflow from raw CSV files to final HTML report and dashboard, using a tiny fake organization. It's perfect for first-time users who want to see the full pipeline in action.

## Example Notebooks

### 01_basic_analysis.ipynb
**Getting Started with orgnet**

Learn the basics of:
- Loading data from CSV files
- Building organizational graphs
- Running basic analysis
- Understanding results
- Generating reports

**Perfect for:** First-time users, understanding the workflow

### 02_network_metrics.ipynb
**Network Metrics Deep Dive**

Explore:
- Centrality measures (degree, betweenness, eigenvector, closeness, PageRank)
- Structural analysis (constraint, core-periphery)
- Identifying key brokers and connectors
- Understanding what metrics mean

**Perfect for:** Understanding network structure, finding key people

### 03_community_detection.ipynb
**Community Detection**

Discover:
- Different community detection algorithms (Louvain, Infomap, Label Propagation)
- Identifying teams and groups
- Analyzing community structure
- Cross-community connections
- Comparing algorithms

**Perfect for:** Finding teams, understanding organizational structure

### 04_visualization.ipynb
**Network Visualization**

Create:
- Interactive network visualizations
- Custom layouts and styling
- Export visualizations
- Dashboard generation
- Customizing visualizations

**Perfect for:** Presenting results, exploring networks visually

## Sample Data Details

The generated sample data includes:

- **50 people** across 7 departments (Engineering, Product, Design, Sales, Marketing, Operations, HR)
- **~250 emails** with response times and reciprocity
- **~400 Slack messages** (30% DMs, 70% channels)
- **~100 meetings** with various types (standups, 1-on-1s, team meetings)
- **~150 documents** with collaboration patterns
- **~500 code commits** with reviewers

All data is generated using Faker for realistic but synthetic data. The data spans the last 90 days.

## Customizing Sample Data

You can modify `generate_sample_data.py` to:

- Change the number of people (`NUM_PEOPLE`)
- Adjust the time period (`NUM_DAYS`)
- Modify departments and roles
- Add custom data patterns
- Change interaction frequencies

## Using Your Own Data

To use your own data:

1. Format your data according to `DATA_FORMAT.md`
2. Place CSV files in `examples/data/` directory
3. Update the data paths in the notebooks
4. Run the notebooks as normal

## Troubleshooting

### Import Errors
If you get import errors:
```bash
# Make sure you're in the project root
cd /path/to/orgnet
pip install -e .
```

### Data Not Found
If notebooks can't find data:
```bash
# Generate sample data first
cd examples
python generate_sample_data.py
```

### Jupyter Not Found
```bash
pip install jupyter
jupyter notebook
```

## Next Steps

After running the examples:

1. **Try with your own data** - See `DATA_FORMAT.md` for format requirements
2. **Explore advanced features** - Check the main README.md
3. **Customize analysis** - Modify `config.yaml` for different parameters
4. **Build custom reports** - Use the visualization and dashboard modules
5. **Integrate with your workflow** - Use the API or import modules directly

## Additional Resources

- **Main Documentation**: `../README.md`
- **Data Format Guide**: `../DATA_FORMAT.md`
- **Quick Start Guide**: `../QUICKSTART.md`
- **Architecture**: `../ARCHITECTURE.md`
