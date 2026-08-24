#!/usr/bin/env bash
# phase-compare: Run both inferflux_cuda and llama_cpp_cuda backends on the
# same model/prompt and produce a side-by-side phase timing comparison.
#
# Prerequisites:
#   - Built inferfluxd with CUDA support
#   - INFERFLUX_MODEL_PATH set to a GGUF model
#
# Usage:
#   scripts/phase_compare.sh [--model PATH] [--prompt TEXT] [--max-tokens N]
set -euo pipefail

MODEL_PATH="${INFERFLUX_MODEL_PATH:-}"
PROMPT="${INFERFLUX_PHASE_COMPARE_PROMPT:-Explain the Fibonacci sequence in three sentences.}"
MAX_TOKENS="${INFERFLUX_PHASE_COMPARE_MAX_TOKENS:-32}"
SERVER_BIN="${INFERFLUX_SERVER_BIN:-./build/inferfluxd}"
CLI_BIN="${INFERFLUX_CLI_BIN:-./build/inferctl}"
PORT_BASE=18990
CONFIG="${INFERFLUX_CONFIG:-config/server.cuda.yaml}"
API_KEY="${INFERCTL_API_KEY:-dev-key-123}"
OUT_DIR="phase_compare_results_$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)  MODEL_PATH="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  echo "Error: Set INFERFLUX_MODEL_PATH or pass --model PATH" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

run_backend() {
  local backend="$1"
  local port="$2"
  local log_file="$OUT_DIR/${backend}.log"
  local timing_file="$OUT_DIR/${backend}_timing.txt"

  echo "=== Running $backend on port $port ==="

  INFERFLUX_MODEL_PATH="$MODEL_PATH" \
  INFERFLUX_CUDA_PHASE_TIMING=1 \
  INFERFLUX_PORT_OVERRIDE="$port" \
  INFERFLUX_BACKEND_PREFER_INFERFLUX=$( [[ "$backend" == "inferflux_cuda" ]] && echo 1 || echo 0 ) \
    "$SERVER_BIN" --config "$CONFIG" > "$log_file" 2>&1 &
  local pid=$!

  # Wait for server to be ready (up to 30 seconds)
  local ready=false
  for i in $(seq 1 30); do
    if curl -sf "http://localhost:$port/healthz" >/dev/null 2>&1; then
      ready=true
      break
    fi
    sleep 1
  done

  if ! $ready; then
    echo "Error: $backend server did not become ready" >&2
    kill "$pid" 2>/dev/null || true
    return 1
  fi

  # Send completion request
  curl -sf "http://localhost:$port/v1/completions" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"$PROMPT\", \"max_tokens\": $MAX_TOKENS}" \
    > "$OUT_DIR/${backend}_response.json" 2>&1 || true

  sleep 1
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true

  # Extract phase timing lines
  grep '\[phase_timing\]' "$log_file" > "$timing_file" 2>/dev/null || true
  echo "  Timing lines: $(wc -l < "$timing_file")"
}

# Run each backend
run_backend "inferflux_cuda" "$PORT_BASE"
run_backend "llama_cpp_cuda" "$((PORT_BASE + 1))"

# Print side-by-side summary
echo ""
echo "=== Phase Timing Comparison ==="
echo ""
echo "--- inferflux_cuda ---"
cat "$OUT_DIR/inferflux_cuda_timing.txt" 2>/dev/null || echo "(no timing data)"
echo ""
echo "--- llama_cpp_cuda ---"
cat "$OUT_DIR/llama_cpp_cuda_timing.txt" 2>/dev/null || echo "(no timing data)"
echo ""
echo "Results saved to: $OUT_DIR/"
