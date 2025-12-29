# ONA Test Suite

Lightweight test suite for the Organizational Network Analysis package.

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=onapy --cov-report=html

# Run specific test file
pytest tests/test_metrics.py

# Run with verbose output
pytest -v
```

## Test Structure

- `conftest.py` - Shared fixtures (sample data, graphs)
- `test_metrics.py` - Tests for centrality, community detection, structural analysis
- `test_graph.py` - Tests for graph building and weight calculations
- `test_core.py` - Tests for main analyzer workflow

## Test Coverage

The test suite covers:
- Core analyzer initialization and workflow
- Graph building from data
- Centrality metrics computation
- Community detection
- Structural analysis
- Edge weight calculations
- Layer fusion (vectorized operations)

Tests are intentionally lightweight - focusing on verifying core functionality works correctly after refactoring.

