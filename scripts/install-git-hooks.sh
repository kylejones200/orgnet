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

cat > "$HOOKS_DIR/pre-push" << 'HOOKEOF'
#!/bin/bash
# Pre-push git hook
# Runs pre-push checks before allowing push

# Get the project root (two levels up from .git/hooks/)
HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HOOK_DIR/../.." && pwd)"
PRE_PUSH_SCRIPT="$PROJECT_ROOT/scripts/pre-push.sh"

if [ -f "$PRE_PUSH_SCRIPT" ]; then
    exec "$PRE_PUSH_SCRIPT"
else
    echo "⚠ Pre-push script not found at $PRE_PUSH_SCRIPT"
    echo "Skipping pre-push checks..."
    exit 0
fi
HOOKEOF

chmod +x "$HOOKS_DIR/pre-push"

echo "✓ Pre-push hook installed successfully!"
echo ""
echo "The hook will now run automatically before each push."
echo "To skip the hook (not recommended), use: git push --no-verify"

