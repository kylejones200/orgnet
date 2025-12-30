# Examples

This directory contains example scripts demonstrating how to use the Enterprise Email Analytics Platform.

## Available Examples

### `basic_usage.py`
Basic example showing how to process emails and build an analytics model. Includes thread analysis and classification results.

### `network_analysis.py`
Demonstrates network analysis capabilities including influence metrics, community detection, and bridge node identification.

### `sentiment_analysis.py`
Shows sentiment classification, emotion detection, and tone analysis features.

### `reporting.py`
Example of generating comprehensive reports and dashboards with multi-format export.

### `knowledge_base_example.py`
Demonstrates how to create and use knowledge bases for improved entity extraction with canonical names and aliases.

## Running Examples

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run an example
python examples/basic_usage.py
python examples/network_analysis.py
python examples/sentiment_analysis.py
```

## Prerequisites

- Emails in `maildir/` directory or `emails.csv` file
- All dependencies installed (see main README.md)
- Virtual environment activated

