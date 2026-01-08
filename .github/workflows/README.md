# GitHub Actions Workflows

## CI Tests and Checks

The `ci.yml` workflow runs comprehensive tests on every push and pull request:
- Tests on Python 3.8, 3.9, 3.10, 3.11, 3.12
- Code formatting checks (black)
- Linting (flake8)
- Test suite with coverage
- Package build verification
- Import verification

## Pre-Push Checks

The `pre-push.yml` workflow runs quick validation checks:
- Code formatting
- Linting
- Package build
- Twine check
- Test suite

This runs on every push to catch issues early.

## Publish to PyPI

The `publish.yml` workflow automatically publishes the package to PyPI when:
- A GitHub release is published
- The workflow is manually triggered via workflow_dispatch

### Setup Required

**See [PYPI_SETUP.md](./PYPI_SETUP.md) for detailed setup instructions.**

Quick setup:
1. **Configure Trusted Publisher on PyPI** (Account settings → Trusted publishers):
   - PyPI project: `orgnet`
   - Owner: `kylejones200`
   - Repository: `orgnet`
   - Workflow: `.github/workflows/publish.yml`
   - Environment: `pypi`

2. **Create GitHub Environment** (Settings → Environments):
   - Name: `pypi`
   - (Optional) Add protection rules

3. **The workflow will automatically**:
   - Build the package (wheel and source distribution)
   - Check the package with twine
   - Publish to PyPI using OIDC (no API tokens needed!)

### Manual Trigger

You can also trigger the workflow manually:
1. Go to Actions tab
2. Select "Publish to PyPI"
3. Click "Run workflow"
4. Enter version number
5. Click "Run workflow"

