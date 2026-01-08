# Contributing to orgnet

Thank you for your interest in contributing to orgnet! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/kylejones200/orgnet.git
   cd orgnet
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

4. **Install pre-commit hooks** (optional but recommended)
   ```bash
   ./scripts/install-git-hooks.sh
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and modular

## Testing

- Write tests for new features
- Ensure all tests pass: `pytest`
- Aim for good test coverage

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Add tests if applicable
4. Update documentation if needed
5. Run the test suite: `pytest`
6. Run cleanup script: `./scripts/cleanup.sh`
7. Submit a pull request with a clear description

## Project Structure

```
orgnet/
├── orgnet/          # Main package
│   ├── core.py      # Main analyzer class
│   ├── data/        # Data ingestion and processing
│   ├── graph/       # Graph construction
│   ├── metrics/     # Network metrics
│   ├── ml/          # Machine learning models
│   ├── temporal/    # Temporal analysis
│   └── ...
├── tests/           # Test suite
├── examples/        # Example notebooks
└── docs/            # Documentation
```

## Questions?

Feel free to open an issue for questions or discussions!

