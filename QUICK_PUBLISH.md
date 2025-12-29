# Quick PyPI Publishing Guide

## ✅ What's Already Set Up

1. **pyproject.toml** - Complete build configuration with metadata
2. **MANIFEST.in** - Includes all necessary files in the package
3. **LICENSE** - MIT License file
4. **GitHub Actions Workflow** - Automated publishing via trusted publishing
5. **setup.py** - Updated with complete metadata

## 🚀 To Publish (3 Steps)

### Step 1: Set Up Trusted Publishing in PyPI

1. Go to: https://pypi.org/manage/projects/
2. Find or create your `orgnet` project
3. Go to "Publishing" → "Add a new pending publisher"
4. Configure:
   - **Publisher**: GitHub
   - **Repository**: `kylejones200/orgnet`
   - **Workflow filename**: `.github/workflows/publish.yml`
   - **Environment**: (leave empty or set to `release`)
5. Click "Add"

### Step 2: Update Version (if needed)

Edit both files:
- `pyproject.toml`: `version = "0.1.0"`
- `setup.py`: `version="0.1.0"`

### Step 3: Create GitHub Release

1. Commit and push your changes:
   ```bash
   git add .
   git commit -m "Prepare for release v0.1.0"
   git push
   ```

2. Create a GitHub Release:
   - Go to your GitHub repo
   - Click "Releases" → "Create a new release"
   - Tag: `v0.1.0` (must match version)
   - Title: `Release v0.1.0`
   - Click "Publish release"

3. **That's it!** GitHub Actions will automatically publish to PyPI.

## 🔍 Verify

After publishing, check:
- PyPI: https://pypi.org/project/orgnet/
- Install test: `pip install orgnet`

## 📝 Notes

- The workflow uses **trusted publishing** - no API tokens needed!
- Publishing happens automatically when you create a GitHub release
- You can also trigger manually via "Actions" → "Run workflow"

## 🐛 Troubleshooting

If publishing fails:
1. Check GitHub Actions logs
2. Verify trusted publishing is configured correctly in PyPI
3. Ensure version numbers match in both files
4. Check that the repository name matches in PyPI settings

