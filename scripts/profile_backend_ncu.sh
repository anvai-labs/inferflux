#!/usr/bin/env bash
# Profile an InferFlux CUDA backend with Nsight Compute on a bounded curl workload.

set -euo pipefail

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
BOLD='\033[1m'

log() { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err() { echo -e "${RED}[ERR]${NC} $1"; }
header() { echo -e "\n${BOLD}$1${NC}"; echo "$(printf '=%.0s' $(seq 1 ${#1}))"; }

BACKEND_INPUT="${1:-inferflux_cuda}"
MODEL_PATH="${2:-models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf}"
OUTPUT_DIR="${3:-./ncu_backend_profile_${BACKEND_INPUT}_$(date +%Y%m%d_%H%M%S)}"
BUILD_DIR="${BUILD_DIR:-./build-cuda}"
SERVER_BIN="$BUILD_DIR/inferfluxd"
API_KEY="${INFERCTL_API_KEY:-dev-key-123}"
PORT=""
HOST="127.0.0.1"
CONCURRENCY="${CONCURRENCY:-4}"
REQUESTS="${REQUESTS:-8}"
MAX_TOKENS="${MAX_TOKENS:-32}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
MIN_BATCH_SIZE="${MIN_BATCH_SIZE:-1}"
BATCH_ACCUMULATION_MS="${BATCH_ACCUMULATION_MS:-2}"
DECODE_BURST_TOKENS="${DECODE_BURST_TOKENS:-0}"
ENABLE_BATCHED_DECODE="${ENABLE_BATCHED_DECODE:-1}"
BENCH_LOG_LEVEL="${BENCH_LOG_LEVEL:-warning}"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-180}"
STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-60}"
WORKLOAD_PROMPT="${WORKLOAD_PROMPT:-Explain why throughput guardrails matter.}"
NCU_SET="${NCU_SET:-default}"
NCU_SECTIONS="${NCU_SECTIONS:-}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-120}"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-0}"
PROFILE_PID=""

normalize_backend() {
  case "$BACKEND_INPUT" in
    native|inferflux|inferflux_cuda|cuda_native|native_cuda)
      BACKEND_ID="inferflux_cuda"
      BACKEND_PORT_DEFAULT=18088
      PREFER_INFERFLUX="true"
      ALLOW_LLAMA_FALLBACK="false"
      STRICT_NATIVE="1"
      DISABLE_PARITY_DELEGATE="1"
      ;;
    llamacpp|llama_cpp|llama_cpp_cuda|cuda_llama_cpp|cuda_llama)
      BACKEND_ID="llama_cpp_cuda"
      BACKEND_PORT_DEFAULT=18089
      PREFER_INFERFLUX="false"
      ALLOW_LLAMA_FALLBACK="true"
      STRICT_NATIVE="0"
      DISABLE_PARITY_DELEGATE="0"
      ;;
    *)
      log_err "Unknown backend '$BACKEND_INPUT' (use inferflux_cuda or llama_cpp_cuda)"
      exit 1
      ;;
  esac
  PORT="${INFERFLUX_PROFILE_PORT:-$BACKEND_PORT_DEFAULT}"
}

require_tools() {
  command -v ncu >/dev/null 2>&1 || { log_err "ncu not found"; exit 1; }
  command -v curl >/dev/null 2>&1 || { log_err "curl not found"; exit 1; }
  [ -x "$SERVER_BIN" ] || { log_err "inferfluxd not found at $SERVER_BIN"; exit 1; }
  [ -f "$MODEL_PATH" ] || { log_err "model not found at $MODEL_PATH"; exit 1; }
}

write_config() {
  cat > "$OUTPUT_DIR/config.yaml" <<CFG
server:
  host: "$HOST"
  http_port: $PORT
  max_concurrent: 128
  enable_metrics: true

models:
  - id: bench-model
    path: "$(realpath "$MODEL_PATH")"
    format: gguf
    backend: $BACKEND_ID
    default: true

runtime:
  backend_priority: [cuda, cpu]
  cuda:
    enabled: true
    attention:
      kernel: auto
    flash_attention:
      enabled: true
    phase_overlap:
      enabled: true
  backend_exposure:
    prefer_inferflux: $PREFER_INFERFLUX
    allow_llama_cpp_fallback: $ALLOW_LLAMA_FALLBACK
  scheduler:
    max_batch_size: 32
    max_batch_tokens: 16384
    min_batch_size: $MIN_BATCH_SIZE
    batch_accumulation_ms: $BATCH_ACCUMULATION_MS
    decode_burst_tokens: $DECODE_BURST_TOKENS
  disaggregated:
    prefill_pool_size: 1
    decode_pool_size: 1
    kv_channel_capacity: 64
    kv_enqueue_max_retries: 3
  paged_kv:
    cpu_pages: 4096
    eviction: lru

auth:
  api_keys:
    - key: $API_KEY
      scopes: [generate, read, admin]
  rate_limit_per_minute: 600

guardrails:
  blocklist: []

logging:
  level: $BENCH_LOG_LEVEL
  format: text
CFG
}

tail_log() {
  local file=$1
  [ -f "$file" ] && tail -20 "$file"
}

wait_ready() {
  local waited=0
  while [ "$waited" -lt "$STARTUP_TIMEOUT_SEC" ]; do
    if [ -n "$PROFILE_PID" ] && ! kill -0 "$PROFILE_PID" 2>/dev/null; then
      return 1
    fi
    if curl -sf -H "Authorization: Bearer $API_KEY" "http://$HOST:$PORT/livez" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  return 1
}

stop_profiled_server() {
  if [ -n "$PROFILE_PID" ] && kill -0 "$PROFILE_PID" 2>/dev/null; then
    kill -TERM "$PROFILE_PID" 2>/dev/null || true
    wait "$PROFILE_PID" 2>/dev/null || true
  fi
  PROFILE_PID=""
}

cleanup() {
  stop_profiled_server
}
trap cleanup EXIT

start_profiled_server() {
  local server_log="$OUTPUT_DIR/server.log"
  local report_prefix="$OUTPUT_DIR/profile"
  local ncu_home="$OUTPUT_DIR/home"
  mkdir -p "$ncu_home"

  local ncu_args=(
    --target-processes all
    --null-stdin
    --graph-profiling node
    --profile-from-start yes
    --launch-count "$NCU_LAUNCH_COUNT"
    --launch-skip "$NCU_LAUNCH_SKIP"
    --kernel-name-base demangled
    --clock-control none
    --cache-control none
    --export "$report_prefix"
    --force-overwrite
  )
  if [ -n "$NCU_SECTIONS" ]; then
    IFS=',' read -ra _ncu_sections <<< "$NCU_SECTIONS"
    for section in "${_ncu_sections[@]}"; do
      ncu_args+=(--section "$section")
    done
  else
    ncu_args+=(--set "$NCU_SET")
  fi

  env \
    HOME="$ncu_home" \
    INFERFLUX_PORT_OVERRIDE="$PORT" \
    INFERCTL_API_KEY="$API_KEY" \
    INFERFLUX_LOG_LEVEL="$BENCH_LOG_LEVEL" \
    INFERFLUX_ENABLE_BATCHED_DECODE="$ENABLE_BATCHED_DECODE" \
    INFERFLUX_CUDA_STRICT="$STRICT_NATIVE" \
    INFERFLUX_CUDA_DISABLE_PARITY_DELEGATE="$DISABLE_PARITY_DELEGATE" \
    ncu \
      "${ncu_args[@]}" \
      "$SERVER_BIN" --config "$OUTPUT_DIR/config.yaml" \
      > "$server_log" 2>&1 &
  PROFILE_PID=$!

  if ! wait_ready; then
    log_err "Server failed to become ready under ncu. Last log lines:"
    tail_log "$server_log"
    return 1
  fi
  log_ok "$BACKEND_ID ready under ncu (PID $PROFILE_PID)"
}

send_profile_request() {
  local request_id=$1
  local output_file=$2
  local prompt_json
  prompt_json=$(printf '%s [req-%s]' "$WORKLOAD_PROMPT" "$request_id" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')
  local response
  response=$(curl -sf -X POST "http://$HOST:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    -d "{\"model\":\"default\",\"prompt\":$prompt_json,\"max_tokens\":$MAX_TOKENS,\"temperature\":0.0}" \
    --max-time "$REQUEST_TIMEOUT_SEC" 2>>"$OUTPUT_DIR/workload.log") || {
      echo "ERROR" > "$output_file"
      return 1
    }
  python3 - <<'PY' "$response" "$output_file"
import json
import sys

payload = json.loads(sys.argv[1])
out_path = sys.argv[2]
choice = payload.get("choices", [{}])[0]
text = choice.get("message", {}).get("content", "") or choice.get("text", "")
tokens = payload.get("usage", {}).get("completion_tokens", 0)
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump({"text": text.strip(), "tokens": tokens}, handle)
PY
}

run_workload() {
  local results_dir="$OUTPUT_DIR/responses"
  mkdir -p "$results_dir"
  : > "$OUTPUT_DIR/workload.log"

  log "Warmup..."
  for i in $(seq 0 $((WARMUP_REQUESTS - 1))); do
    send_profile_request "warmup-$i" "/dev/null" || true
  done

  log "Running profiled workload..."
  local pids=()
  for i in $(seq 0 $((REQUESTS - 1))); do
    send_profile_request "$i" "$results_dir/req_$i.json" &
    pids+=($!)
    if [ ${#pids[@]} -ge "$CONCURRENCY" ]; then
      wait "${pids[0]}" 2>/dev/null || true
      pids=("${pids[@]:1}")
    fi
  done
  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done

  curl -sf -H "Authorization: Bearer $API_KEY" \
    "http://$HOST:$PORT/metrics" > "$OUTPUT_DIR/metrics_prometheus.txt" 2>/dev/null || true
  curl -sf -H "Authorization: Bearer $API_KEY" \
    "http://$HOST:$PORT/v1/admin/cache" > "$OUTPUT_DIR/admin_cache.json" 2>/dev/null || true
}

export_reports() {
  local report_file="$OUTPUT_DIR/profile.ncu-rep"
  if [ ! -f "$report_file" ]; then
    log_warn "ncu report not found at $report_file"
    return 0
  fi
  ncu --import "$report_file" --csv --page raw > "$OUTPUT_DIR/profile_raw.csv"
}

main() {
  normalize_backend
  require_tools
  mkdir -p "$OUTPUT_DIR"

  header "Nsight Compute Backend Profile"
  echo "  Backend:      $BACKEND_ID"
  echo "  Build dir:    $BUILD_DIR"
  echo "  Model:        $MODEL_PATH"
  echo "  Output:       $OUTPUT_DIR"
  echo "  Host/port:    $HOST:$PORT"
  echo "  Workload:     requests=$REQUESTS concurrency=$CONCURRENCY max_tokens=$MAX_TOKENS"
  echo "  Prompt:       $WORKLOAD_PROMPT"
  echo "  Scheduler:    min_batch=$MIN_BATCH_SIZE accum_ms=$BATCH_ACCUMULATION_MS decode_burst=$DECODE_BURST_TOKENS batched_decode=$ENABLE_BATCHED_DECODE"
  echo "  NCU set:      $NCU_SET"
  if [ -n "$NCU_SECTIONS" ]; then
    echo "  NCU sections: $NCU_SECTIONS"
  fi
  echo "  Launch limit: $NCU_LAUNCH_COUNT"

  write_config
  start_profiled_server
  run_workload
  stop_profiled_server
  export_reports

  log_ok "NCU artifacts saved under $OUTPUT_DIR"
  echo "  Inspect raw CSV with: sed -n '1,120p' $OUTPUT_DIR/profile_raw.csv"
}

main "$@"
