#!/bin/bash
# Script to create GitHub release for v0.1.1
# Usage: ./create_release.sh

set -e

TAG="v0.1.1"
TITLE="v0.1.1 - Patch Release"

echo "Creating GitHub release for $TAG..."

# Check if gh is authenticated
if ! gh auth status &>/dev/null; then
    echo "GitHub CLI not authenticated. Please run: gh auth login"
    echo ""
    echo "Or create the release manually at:"
    echo "https://github.com/kylejones200/orgnet/releases/new"
    echo ""
    echo "Select tag: $TAG"
    echo "Title: $TITLE"
    exit 1
fi

# Create the release with inline notes
gh release create "$TAG" \
    --title "$TITLE" \
    --notes "## Initial Release of orgnet

Organizational Network Analysis Platform using ML and Graph Analytics

### Features
- Multi-source data ingestion (email, Slack, calendar, documents, code, HRIS)
- Graph construction with weighted, temporal edges
- Network metrics (centrality, structural holes, core-periphery)
- Community detection (Louvain, Infomap, Label Propagation, SBM)
- Machine learning (GCN, GAT, Node2Vec, link prediction)
- NLP analysis (topic modeling, expertise inference)
- Temporal analysis (change detection, onboarding tracking)
- Interactive dashboards and visualization
- Privacy-first design

### Installation
\`\`\`bash
pip install orgnet
\`\`\`

See README.md for full documentation."

echo "✓ Release created successfully!"
echo "The PyPI publish workflow should now trigger automatically."


