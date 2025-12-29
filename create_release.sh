#!/bin/bash
# Script to create GitHub release for v0.1.0
# Usage: ./create_release.sh

set -e

TAG="v0.1.0"
TITLE="v0.1.0 - Initial Release"
NOTES_FILE="RELEASE_NOTES_v0.1.0.md"

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
    echo "Copy contents from: $NOTES_FILE"
    exit 1
fi

# Create the release
gh release create "$TAG" \
    --title "$TITLE" \
    --notes-file "$NOTES_FILE"

echo "✓ Release created successfully!"
echo "The PyPI publish workflow should now trigger automatically."

