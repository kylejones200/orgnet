#!/bin/bash
# Install git hooks
# This will set up the pre-push hook to run automatically

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOOKS_DIR="$PROJECT_ROOT/.git/hooks"

echo "Installing git hooks..."

# Create hooks directory if it doesn't exist
mkdir -p "$HOOKS_DIR"

# Install pre-push hook
if [ -f "$HOOKS_DIR/pre-push" ]; then
    echo "⚠ Pre-push hook already exists. Backing up to pre-push.backup"
    mv "$HOOKS_DIR/pre-push" "$HOOKS_DIR/pre-push.backup"
fi

cat > "$HOOKS_DIR/pre-push" << 'EOF'
#!/bin/bash
# Pre-push git hook
# Runs pre-push checks before allowing push

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../scripts" && pwd)"
if [ -f "$SCRIPT_DIR/pre-push.sh" ]; then
    exec "$SCRIPT_DIR/pre-push.sh"
else
    echo "⚠ Pre-push script not found at $SCRIPT_DIR/pre-push.sh"
    echo "Skipping pre-push checks..."
    exit 0
fi
EOF

chmod +x "$HOOKS_DIR/pre-push"

echo "✓ Pre-push hook installed successfully!"
echo ""
echo "The hook will now run automatically before each push."
echo "To skip the hook (not recommended), use: git push --no-verify"

