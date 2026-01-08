# PyPI Publishing Setup with OpenID Connect (OIDC)

This repository uses GitHub Actions with OpenID Connect (OIDC) for secure, passwordless publishing to PyPI.

## Prerequisites

1. A PyPI account (create at https://pypi.org/account/register/)
2. Admin access to this GitHub repository
3. GitHub Actions enabled for the repository

## Setup Instructions

### Step 1: Configure Trusted Publisher on PyPI

1. Log in to your PyPI account: https://pypi.org/manage/account/
2. Go to **Account settings** → **Trusted publishers**
3. Click **Add a new trusted publisher**
4. Fill in the form:
   - **PyPI project name**: `orgnet`
   - **Owner**: `kylejones200` (your GitHub username/organization)
   - **Repository name**: `orgnet`
   - **Workflow filename**: `.github/workflows/publish.yml`
   - **Environment name**: `pypi` (optional, but recommended for security)
5. Click **Add**

### Step 2: Create GitHub Environment (Optional but Recommended)

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Environments**
3. Click **New environment**
4. Name it: `pypi`
5. (Optional) Add environment protection rules:
   - Required reviewers (for extra security)
   - Deployment branches (only allow specific branches)
6. Click **Configure environment**

### Step 3: Verify Workflow Configuration

The workflow file (`.github/workflows/publish.yml`) is already configured with:
- ✅ `id-token: write` permission (required for OIDC)
- ✅ `pypa/gh-action-pypi-publish@release/v1` action (supports OIDC)
- ✅ Environment name: `pypi`

### Step 4: Test the Setup

1. **Create a test release:**
   ```bash
   git tag v0.1.2
   git push origin v0.1.2
   ```

2. **Or trigger manually:**
   - Go to **Actions** → **Publish to PyPI**
   - Click **Run workflow** → **Run workflow**

3. **Check the workflow run:**
   - Go to **Actions** tab
   - Click on the workflow run
   - Verify it completes successfully

## How It Works

1. When you push a tag (e.g., `v0.1.2`) or create a GitHub release, the workflow triggers
2. GitHub generates an OIDC token with the required permissions
3. The `pypa/gh-action-pypi-publish` action uses this token to authenticate with PyPI
4. PyPI verifies the token against your trusted publisher configuration
5. The package is published to PyPI

## Security Benefits

- ✅ **No secrets needed**: No API tokens or passwords stored in GitHub
- ✅ **Automatic rotation**: Tokens are generated per-run and expire quickly
- ✅ **Fine-grained permissions**: Only the specific workflow can publish
- ✅ **Audit trail**: All publishes are logged in PyPI

## Troubleshooting

### "Trusted publisher not found"
- Verify the trusted publisher configuration on PyPI matches exactly:
  - Owner: `kylejones200`
  - Repository: `orgnet`
  - Workflow file: `.github/workflows/publish.yml`
  - Environment: `pypi` (if using environments)

### "Environment not found"
- Create the `pypi` environment in GitHub repository settings
- Or remove the `environment:` section from the workflow (less secure)

### "Package already exists"
- Update the version in `setup.py` and `pyproject.toml`
- Create a new tag with the new version

## Version Management

Update version in two places:
1. `setup.py`: `version="0.1.2"`
2. `pyproject.toml`: `version = "0.1.2"`

Then create a tag:
```bash
git tag v0.1.2
git push origin v0.1.2
```

## References

- [PyPI Trusted Publishers Documentation](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions OIDC](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect)
- [pypa/gh-action-pypi-publish](https://github.com/pypa/gh-action-pypi-publish)

