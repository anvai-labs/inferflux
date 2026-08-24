#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/profile.sh <command> [args...]

Commands:
  backend         Run the standard Nsight Systems backend profiler
  backend-ncu     Run the Nsight Compute backend profiler
  phase-timing    Parse native phase timing lines from a server log
  phase-compare   Run inferflux_cuda vs llama_cpp_cuda side-by-side timing comparison
  analyze-nsys    Summarize exported Nsight Systems CSV reports

Examples:
  scripts/profile.sh backend inferflux_cuda
  scripts/profile.sh backend-ncu inferflux_cuda
  scripts/profile.sh phase-timing gguf_benchmark_results/server_inferflux_cuda.log
  scripts/profile.sh phase-compare --model /path/to/model.gguf
  scripts/profile.sh analyze-nsys nsys_backend_profiles/phaseb_inferflux_cuda
EOF
}

cmd="${1:-help}"
case "$cmd" in
  backend)
    shift
    exec "$ROOT_DIR/profile_backend.sh" "$@"
    ;;
  backend-ncu)
    shift
    exec "$ROOT_DIR/profile_backend_ncu.sh" "$@"
    ;;
  phase-timing)
    shift
    exec python3 "$ROOT_DIR/parse_native_phase_timing.py" "$@"
    ;;
  phase-compare)
    shift
    exec "$ROOT_DIR/phase_compare.sh" "$@"
    ;;
  analyze-nsys)
    shift
    exec python3 "$ROOT_DIR/analyze_nsys_results.py" "$@"
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    echo "Unknown profile command: $cmd" >&2
    echo >&2
    usage >&2
    exit 1
    ;;
esac
