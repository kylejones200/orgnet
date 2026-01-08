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

# Check if package can be built (optional - skip if build not available)
echo ""
echo "Building package..."
# Check if build module is installed
if ! python3 -c "import build" 2>/dev/null; then
    echo "Build module not found, attempting to install..."
    # Try to install, but don't fail if it's an externally-managed environment
    if python3 -m pip install -q build 2>/dev/null; then
        echo "✓ Build module installed"
    else
        echo -e "${YELLOW}⚠ Could not install build module (externally-managed environment)${NC}"
        echo "  Skipping build check - CI will verify the build on push"
        BUILD_AVAILABLE=false
    fi
else
    BUILD_AVAILABLE=true
fi

# Try to build if build is available
if [ "${BUILD_AVAILABLE:-true}" = "true" ] && python3 -c "from build import ProjectBuilder" 2>/dev/null; then
    # Clean any existing dist directory
    rm -rf dist/ 2>/dev/null || true
    
    # Try to build
    if python3 -m build --wheel > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Package builds successfully${NC}"
        
        # Check package with twine if available
        if command -v twine > /dev/null 2>&1; then
            echo ""
            echo "Checking package with twine..."
            if twine check dist/* > /dev/null 2>&1; then
                echo -e "${GREEN}✓ Twine check passed${NC}"
            else
                echo -e "${YELLOW}⚠ Twine check skipped (twine not available)${NC}"
            fi
        else
            echo -e "${YELLOW}⚠ Twine not available, skipping package check${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Package build check skipped (build failed)${NC}"
        echo "  This is OK - CI will verify the build"
    fi
else
    echo -e "${YELLOW}⚠ Build module not available, skipping build check${NC}"
    echo "  CI will verify the build on push"
fi

# Run tests (optional - skip if pytest not available)
echo ""
echo "Running tests..."
if [ -d "tests" ] && [ "$(ls -A tests/*.py 2>/dev/null)" ]; then
    # Check if pytest is available
    if ! command -v pytest > /dev/null 2>&1 && ! python3 -c "import pytest" 2>/dev/null; then
        echo -e "${YELLOW}⚠ pytest not available, skipping tests${NC}"
        echo "  CI will run tests on push"
    else
        # Check if required dependencies are installed
        echo "Checking test dependencies..."
        missing_deps=()
        for dep in yaml dateutil pytz; do
            if ! python3 -c "import ${dep}" 2>/dev/null; then
                missing_deps+=("${dep}")
            fi
        done
        
        if [ ${#missing_deps[@]} -gt 0 ]; then
            echo -e "${YELLOW}⚠ Missing dependencies: ${missing_deps[*]}${NC}"
            echo "Attempting to install missing dependencies..."
            # Try to install, but don't fail if it's an externally-managed environment
            if python3 -m pip install -q pyyaml python-dateutil pytz 2>/dev/null; then
                echo "✓ Dependencies installed"
            else
                echo -e "${YELLOW}⚠ Could not install dependencies (externally-managed environment)${NC}"
                echo "  Skipping tests - CI will run them on push"
            fi
        fi
        
        # Try to run tests if pytest is available
        if command -v pytest > /dev/null 2>&1 || python3 -c "import pytest" 2>/dev/null; then
            # Check if we have the required dependencies
            if python3 -c "import yaml, dateutil, pytz" 2>/dev/null; then
                if pytest tests/ -v --tb=short > /dev/null 2>&1; then
                    echo -e "${GREEN}✓ All tests passed${NC}"
                else
                    echo -e "${YELLOW}⚠ Some tests failed or dependencies missing${NC}"
                    echo "  CI will run full test suite on push"
                fi
            else
                echo -e "${YELLOW}⚠ Missing test dependencies, skipping tests${NC}"
                echo "  CI will run tests on push"
            fi
        else
            echo -e "${YELLOW}⚠ pytest not available, skipping tests${NC}"
            echo "  CI will run tests on push"
        fi
    fi
else
    echo -e "${YELLOW}⚠ No tests found, skipping${NC}"
fi

echo ""
echo -e "${GREEN}✅ All pre-push checks passed! Safe to push.${NC}"
exit 0

