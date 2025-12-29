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

1. **Configure Trusted Publishing in PyPI**:
   - Go to https://pypi.org/manage/projects/
   - Select your project (orgnet)
   - Go to "Publishing" → "Add a new pending publisher"
   - Select "GitHub" as publisher
   - Repository: `kylejones200/orgnet`
   - Workflow filename: `.github/workflows/publish.yml`
   - Environment: `release` (or leave empty)
   - Save

2. **The workflow will automatically**:
   - Build the package (wheel and source distribution)
   - Check the package with twine
   - Publish to PyPI using trusted publishing (no API tokens needed!)

### Manual Trigger

You can also trigger the workflow manually:
1. Go to Actions tab
2. Select "Publish to PyPI"
3. Click "Run workflow"
4. Enter version number
5. Click "Run workflow"

