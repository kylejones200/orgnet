# Pre-Push Scripts

This directory contains scripts to help catch issues before pushing code.

## Scripts

### `pre-push.sh`

Runs comprehensive checks before pushing:
- Code formatting (black)
- Linting (flake8)
- Package build verification
- Twine package check
- Test suite

**Usage:**
```bash
./scripts/pre-push.sh
```

Or use the Makefile:
```bash
make pre-push
```

### `install-git-hooks.sh`

Installs git hooks to automatically run pre-push checks before each `git push`.

**Usage:**
```bash
./scripts/install-git-hooks.sh
```

Or use the Makefile:
```bash
make install-hooks
```

After installation, the pre-push hook will run automatically. To skip it (not recommended):
```bash
git push --no-verify
```

## GitHub Actions

The `.github/workflows/pre-push.yml` workflow runs the same checks automatically on:
- Every push to main/develop branches
- Every pull request

This ensures code quality even if local hooks are skipped.

## Manual Checks

You can also run individual checks:

```bash
# Format code
make format

# Lint code
make lint

# Run tests
make test

# Build package
make build

# Run all checks
make check
```

