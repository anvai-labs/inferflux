#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/smoke.sh <command> [args...]

Commands:
  gguf-native      Run the canonical native GGUF smoke test
  backend-identity Run backend identity contract checks against a live server

Examples:
  scripts/smoke.sh gguf-native --model-dir models/qwen2.5-3b-instruct
  scripts/smoke.sh backend-identity --help
EOF
}

cmd="${1:-help}"
case "$cmd" in
  gguf-native)
    shift
    exec python3 "$ROOT_DIR/test_gguf_native_smoke.py" "$@"
    ;;
  backend-identity)
    shift
    exec python3 "$ROOT_DIR/check_backend_identity.py" "$@"
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    echo "Unknown smoke command: $cmd" >&2
    echo >&2
    usage >&2
    exit 1
    ;;
esac
