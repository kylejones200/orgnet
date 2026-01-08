#!/bin/bash
# Cleanup script for orgnet repository
# Removes build artifacts, cache files, and generated outputs

set -e

echo "🧹 Cleaning up orgnet repository..."

# Remove Python cache files
echo "  Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

# Remove build artifacts
echo "  Removing build artifacts..."
rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .cache/ 2>/dev/null || true

# Remove system files
echo "  Removing system files..."
find . -name ".DS_Store" -delete 2>/dev/null || true
find . -name "Thumbs.db" -delete 2>/dev/null || true

# Remove generated output files
echo "  Removing generated output files..."
rm -f examples/*.html tutorial_*.html test_*.html 2>/dev/null || true
rm -rf tutorial_data/ 2>/dev/null || true

# Remove log files
echo "  Removing log files..."
find . -name "*.log" -not -path "./wip/*" -delete 2>/dev/null || true

echo "✅ Cleanup complete!"
echo ""
echo "Note: The following are preserved:"
echo "  - wip/ directory (archived work)"
echo "  - examples/data/*.csv (sample data for examples)"
echo "  - examples/lib/ (visualization dependencies)"

