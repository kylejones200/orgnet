# Quick Setup Guide

## Step 1: Install Dependencies

```bash
# From project root
pip install faker jupyter pandas matplotlib
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

## Step 2: Generate Sample Data

```bash
cd examples
python generate_sample_data.py
```

This creates:
- `data/hris.csv` - People data (required)
- `data/email.csv` - Email interactions
- `data/slack.csv` - Slack messages
- `data/calendar.csv` - Meeting data
- `data/documents.csv` - Document collaboration
- `data/code.csv` - Code commits

## Step 3: Start Jupyter

```bash
jupyter notebook
```

Then open the notebooks in order:
1. `01_basic_analysis.ipynb`
2. `02_network_metrics.ipynb`
3. `03_community_detection.ipynb`
4. `04_visualization.ipynb`

## Troubleshooting

**Import errors?**
```bash
# Install orgnet in development mode
pip install -e .
```

**Data not found?**
```bash
# Make sure you generated the data
cd examples
python generate_sample_data.py
```

**Jupyter not found?**
```bash
pip install jupyter
```

