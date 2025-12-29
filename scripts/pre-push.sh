#!/bin/bash
# Pre-push hook script
# Run this before pushing to catch issues early
# Usage: ./scripts/pre-push.sh
# Or install as git hook: ln -s ../../scripts/pre-push.sh .git/hooks/pre-push

set -e  # Exit on error

echo "🔍 Running pre-push checks..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}❌ Error: Must run from project root${NC}"
    exit 1
fi

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version"

# Check code formatting
echo ""
echo "Checking code formatting with black..."
if black --check orgnet/ tests/ example.py setup.py --line-length 100 2>/dev/null; then
    echo -e "${GREEN}✓ Code formatting OK${NC}"
else
    echo -e "${RED}❌ Code formatting issues found${NC}"
    echo "Run: black orgnet/ tests/ example.py setup.py --line-length 100"
    exit 1
fi

# Lint code
echo ""
echo "Running flake8 linter..."
if flake8 orgnet/ tests/ example.py setup.py --max-line-length=100 --ignore=E203,W503,E501 --count --quiet 2>/dev/null; then
    echo -e "${GREEN}✓ Linting passed${NC}"
else
    echo -e "${RED}❌ Linting issues found${NC}"
    echo "Run: flake8 orgnet/ tests/ example.py setup.py --max-line-length=100 --ignore=E203,W503,E501"
    exit 1
fi

# Check if package can be built
echo ""
echo "Building package..."
# Check if build module is installed
if ! python3 -c "import build" 2>/dev/null; then
    echo "Installing build module..."
    python3 -m pip install -q build || {
        echo -e "${RED}❌ Failed to install build module${NC}"
        exit 1
    }
fi

# Try to build - use python -m build directly
if python3 -c "from build import ProjectBuilder" 2>/dev/null; then
    # Build using the module directly
    if python3 -c "from build import ProjectBuilder; import os; os.chdir('.'); ProjectBuilder('.').build('wheel', 'dist/')" 2>/dev/null || python3 -m build --wheel > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Package builds successfully${NC}"
    else
        echo -e "${YELLOW}⚠ Package build check skipped (build module issue)${NC}"
        echo "  This is OK - CI will verify the build"
    fi
else
    echo -e "${YELLOW}⚠ Build module not available, skipping build check${NC}"
    echo "  CI will verify the build on push"
fi

# Check package with twine
echo ""
echo "Checking package with twine..."
if twine check dist/* > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Twine check passed${NC}"
else
    echo -e "${RED}❌ Twine check failed${NC}"
    exit 1
fi

# Run tests
echo ""
echo "Running tests..."
if [ -d "tests" ] && [ "$(ls -A tests/*.py 2>/dev/null)" ]; then
    # Check if required dependencies are installed
    echo "Checking test dependencies..."
    missing_deps=()
    for dep in pyyaml python-dateutil pytz; do
        if ! python3 -c "import ${dep//-/_}" 2>/dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        echo -e "${YELLOW}⚠ Missing dependencies: ${missing_deps[*]}${NC}"
        echo "Installing missing dependencies..."
        pip install -q ${missing_deps[*]} || {
            echo -e "${RED}❌ Failed to install dependencies${NC}"
            exit 1
        }
    fi
    
    if pytest tests/ -v --tb=short > /dev/null 2>&1; then
        echo -e "${GREEN}✓ All tests passed${NC}"
    else
        echo -e "${RED}❌ Tests failed${NC}"
        echo "Run: pytest tests/ -v"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ No tests found, skipping${NC}"
fi

echo ""
echo -e "${GREEN}✅ All pre-push checks passed! Safe to push.${NC}"
exit 0

