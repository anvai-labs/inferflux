#!/usr/bin/env bash
# install-hooks.sh: configure git to use project-level hooks in .githooks/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOOKS_DIR="$REPO_ROOT/.githooks"

if [ ! -d "$HOOKS_DIR" ]; then
  echo "ERROR: $HOOKS_DIR not found. Run this script from the repo root."
  exit 1
fi

# Make hooks executable
chmod +x "$HOOKS_DIR/pre-commit"
chmod +x "$HOOKS_DIR/commit-msg"

# Wire git to use project hooks directory
git -C "$REPO_ROOT" config core.hooksPath .githooks

echo "Hooks installed. Project hooks in .githooks/ are now active."
echo ""
echo "  pre-commit : clang-format, cmake build (CPU-only Release), all unit/tagged"
echo "               subtests, StubIntegration, SSECancel"
echo "  commit-msg : blocks AI agent attribution and co-author trailers"
echo ""
echo "Skip variables (export before committing):"
echo "  SKIP_FORMAT=1  — skip clang-format check"
echo "  SKIP_BUILD=1   — skip cmake build"
echo "  SKIP_TESTS=1   — skip all ctest (doc-only commits)"
echo ""
echo "CI-only gates (not run locally — too slow or hardware-specific):"
echo "  coverage      : ENABLE_COVERAGE=ON Debug build + lcov + Codecov upload"
echo "  build-check-mps  : macOS macos-latest MPS compile + unit tests"
echo "  build-check-cuda : CUDA 12.3 toolkit install + inferfluxd compile-check"
echo ""
echo "To uninstall, run:"
echo "  git config --unset core.hooksPath"
