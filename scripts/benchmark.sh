#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/benchmark.sh <command> [args...]

Commands:
  gguf-compare     Compare inferflux_cuda vs llama_cpp_cuda on one GGUF model
  multi-backend    Compare all backends compatible with the supplied model format
  throughput-gate  Run the throughput gate and emit the standard JSON report

Examples:
  scripts/benchmark.sh gguf-compare models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf
  BUILD_DIR=./build-cuda scripts/benchmark.sh multi-backend models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf
  BUILD_DIR=./build-cuda scripts/benchmark.sh multi-backend models/qwen2.5-3b-instruct-safetensors
  scripts/benchmark.sh throughput-gate --require-cuda-lanes
EOF
}

cmd="${1:-help}"
case "$cmd" in
  gguf-compare)
    shift
    exec "$ROOT_DIR/run_gguf_comparison_benchmark.sh" "$@"
    ;;
  multi-backend)
    shift
    exec "$ROOT_DIR/benchmark_multi_backend_comparison.sh" "$@"
    ;;
  throughput-gate)
    shift
    exec python3 "$ROOT_DIR/run_throughput_gate.py" "$@"
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    echo "Unknown benchmark command: $cmd" >&2
    echo >&2
    usage >&2
    exit 1
    ;;
esac
