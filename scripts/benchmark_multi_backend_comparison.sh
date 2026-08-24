#!/usr/bin/env bash
#
# Multi-Backend Comparison Benchmark
#
# Benchmarks inferflux_cuda, llama_cpp_cuda, Ollama, LM Studio, vLLM, and
# SGLang backends.
# Measures throughput, latency percentiles, and GPU memory consumption across
# multiple concurrency levels to generate scaling curves.
#
# Usage:
#   ./scripts/benchmark_multi_backend_comparison.sh [model.gguf|model.safetensors|model_dir]
#   ./scripts/benchmark_multi_backend_comparison.sh models/qwen2.5-3b-instruct-q4_k_m.gguf
#   CONCURRENCY_LEVELS="1,2,4,8" ./scripts/benchmark_multi_backend_comparison.sh model_dir
#
# The script auto-detects the supplied model format and only benchmarks
# backends that can serve that format.
#
# Environment Variables:
#   OLLAMA_HOST           - Ollama server URL (default: http://192.168.1.20:11434)
#   LMSTUDIO_HOST         - LM Studio server URL (default: http://192.168.1.20:1234)
#   LMSTUDIO_MODEL        - LM Studio model id (default: auto-discover first /v1/models entry)
#   VLLM_HOST             - vLLM server URL (default: http://127.0.0.1:8000)
#   VLLM_MODEL            - vLLM model id (default: auto-discover first /v1/models entry)
#   VLLM_MODEL_PATH       - vLLM local model path for AUTOSTART_VLLM=true
#   VLLM_BIN              - vLLM CLI path (default: ./.venv-vllm/bin/vllm)
#   VLLM_LAUNCH_ARGS      - Extra args appended to `vllm serve`
#   AUTOSTART_VLLM        - Launch/teardown local vLLM for this benchmark (default: false)
#   SGLANG_HOST           - SGLang server URL (default: http://127.0.0.1:30000)
#   SGLANG_MODEL          - SGLang model id (default: auto-discover first /v1/models entry)
#   SGLANG_MODEL_PATH     - SGLang local model path for AUTOSTART_SGLANG=true
#   SGLANG_PYTHON         - SGLang venv python path (default: ./.venv-sglang/bin/python)
#   SGLANG_LAUNCH_ARGS    - Extra args appended to `sglang serve`
#   AUTOSTART_SGLANG      - Launch/teardown local SGLang for this benchmark (default: false)
#   SAFETENSORS_MODEL_PATH - Shared local safetensors model dir for vLLM/SGLang autostart
#   CONCURRENCY_LEVELS    - Comma-separated concurrency levels (default: 1,2,4,8,16)
#   NUM_REQUESTS          - Requests per concurrency level (default: 32)
#   MAX_TOKENS            - Max tokens per request (default: 64)
#   OUTPUT_DIR            - Results directory (default: ./multi_backend_benchmark_results)
#   SKIP_OLLAMA           - Skip Ollama benchmark (default: false)
#   SKIP_LMSTUDIO         - Skip LM Studio benchmark (default: false)
#   SKIP_VLLM             - Skip vLLM benchmark (default: false)
#   SKIP_SGLANG           - Skip SGLang benchmark (default: false)
#   BUILD_DIR             - Build directory (default: auto-detect ./build or ./build-cuda)
#   PORT_NATIVE           - Port for inferflux_cuda (default: 18090)
#   PORT_LLAMA            - Port for llama_cpp_cuda (default: 18091)
#   RESET_CUDA_BETWEEN_BACKENDS - Reset CUDA device after local InferFlux/llama.cpp
#                                 backend teardown (default: true)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_PROMPT_SUITE="$SCRIPT_DIR/../tests/data/benchmarks/prompt_suite_32.json"
DEFAULT_SAFETENSORS_MODEL="$REPO_ROOT/models/qwen2.5-3b-instruct-safetensors"

# ============================================================================
# Configuration
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

DEFAULT_MODEL="models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL_PATH="${1:-${MODEL_PATH:-$DEFAULT_MODEL}}"
BUILD_DIR="${BUILD_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./multi_backend_benchmark_results}"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1,2,4,8,16}"
NUM_REQUESTS="${NUM_REQUESTS:-32}"
MAX_TOKENS="${MAX_TOKENS:-64}"
API_KEY="${API_KEY:-dev-key-123}"
PROMPT_SUITE_PATH="${INFERFLUX_BENCH_PROMPT_SUITE:-$DEFAULT_PROMPT_SUITE}"
SAFETENSORS_MODEL_PATH_INPUT="${SAFETENSORS_MODEL_PATH:-}"
SAFETENSORS_MODEL_PATH="${SAFETENSORS_MODEL_PATH:-$DEFAULT_SAFETENSORS_MODEL}"

# Backend ports
PORT_NATIVE="${PORT_NATIVE:-18090}"
PORT_LLAMA="${PORT_LLAMA:-18091}"
OLLAMA_HOST="${OLLAMA_HOST:-http://192.168.1.20:11434}"
LMSTUDIO_HOST="${LMSTUDIO_HOST:-http://192.168.1.20:1234}"
LMSTUDIO_MODEL="${LMSTUDIO_MODEL:-}"
VLLM_HOST="${VLLM_HOST:-http://127.0.0.1:8000}"
VLLM_MODEL="${VLLM_MODEL:-}"
VLLM_MODEL_PATH_INPUT="${VLLM_MODEL_PATH:-}"
VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-$SAFETENSORS_MODEL_PATH}"
VLLM_BIN="${VLLM_BIN:-$REPO_ROOT/.venv-vllm/bin/vllm}"
VLLM_LAUNCH_ARGS="${VLLM_LAUNCH_ARGS:-}"
AUTOSTART_VLLM="${AUTOSTART_VLLM:-false}"
SGLANG_HOST="${SGLANG_HOST:-http://127.0.0.1:30000}"
SGLANG_MODEL="${SGLANG_MODEL:-}"
SGLANG_MODEL_PATH_INPUT="${SGLANG_MODEL_PATH:-}"
SGLANG_MODEL_PATH="${SGLANG_MODEL_PATH:-$SAFETENSORS_MODEL_PATH}"
SGLANG_PYTHON="${SGLANG_PYTHON:-$REPO_ROOT/.venv-sglang/bin/python}"
SGLANG_LAUNCH_ARGS="${SGLANG_LAUNCH_ARGS:-}"
AUTOSTART_SGLANG="${AUTOSTART_SGLANG:-false}"
RESET_CUDA_BETWEEN_BACKENDS="${RESET_CUDA_BETWEEN_BACKENDS:-true}"
INFERFLUX_BENCH_CHILD_MODE="${INFERFLUX_BENCH_CHILD_MODE:-0}"
INFERFLUX_BENCH_SINGLE_BACKEND="${INFERFLUX_BENCH_SINGLE_BACKEND:-}"

ALL_BACKENDS=(inferflux_cuda llama_cpp_cuda ollama lmstudio vllm sglang)

# Skip Ollama?
SKIP_OLLAMA="${SKIP_OLLAMA:-false}"
SKIP_LMSTUDIO="${SKIP_LMSTUDIO:-false}"
SKIP_VLLM="${SKIP_VLLM:-false}"
SKIP_SGLANG="${SKIP_SGLANG:-false}"

# Logging
log()      { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
log_ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
log_err()  { echo -e "${RED}[ERR]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
header()   { echo -e "\n${BOLD}$1${NC}"; echo "$(printf '=%.0s' $(seq 1 ${#1}))"; }

resolve_build_dir() {
    if [ -n "$BUILD_DIR" ]; then
        echo "$BUILD_DIR"
        return 0
    fi

    if [ -f "./build/inferfluxd" ]; then
        echo "./build"
        return 0
    fi

    if [ -f "./build-cuda/inferfluxd" ]; then
        echo "./build-cuda"
        return 0
    fi

    echo "./build"
}

# ============================================================================
# GPU Memory Measurement
# ============================================================================

gpu_mem_mb() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

gpu_mem_total_mb() {
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

gpu_name() {
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1
}

gpu_utilization() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

port_has_listener() {
    local port=$1
    if command -v ss >/dev/null 2>&1; then
        ss -ltnH "sport = :$port" 2>/dev/null | grep -q LISTEN
        return $?
    fi

    python3 - "$port" <<'PYEOF'
import socket
import sys

port = int(sys.argv[1])
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.2)
    sys.exit(0 if sock.connect_ex(("127.0.0.1", port)) == 0 else 1)
except PermissionError:
    sys.exit(1)
finally:
    try:
        sock.close()
    except Exception:
        pass
PYEOF
}

wait_for_port_free() {
    local port=$1
    local waited=0
    while [ $waited -lt 15 ]; do
        if ! port_has_listener "$port"; then
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    return 1
}

require_port_free() {
    local backend=$1 port=$2
    if port_has_listener "$port"; then
        log_err "$backend cannot start: port $port already has a listener"
        log_err "Free the port or override PORT_NATIVE/PORT_LLAMA before rerunning."
        return 1
    fi
    return 0
}

url_host() {
    python3 - "$1" <<'PYEOF'
from urllib.parse import urlparse
import sys

parsed = urlparse(sys.argv[1])
print(parsed.hostname or "")
PYEOF
}

url_port() {
    python3 - "$1" <<'PYEOF'
from urllib.parse import urlparse
import sys

parsed = urlparse(sys.argv[1])
if parsed.port:
    print(parsed.port)
elif parsed.scheme == "https":
    print(443)
else:
    print(80)
PYEOF
}

is_local_url() {
    local host
    host=$(url_host "$1")
    [ "$host" = "127.0.0.1" ] || [ "$host" = "localhost" ] || [ "$host" = "0.0.0.0" ]
}

detect_model_format() {
    python3 - "$1" <<'PYEOF'
from pathlib import Path
import sys

path = Path(sys.argv[1]).expanduser()
name = path.name.lower()

if path.is_file():
    if name.endswith(".gguf"):
        print("gguf")
        raise SystemExit(0)
    if name.endswith(".safetensors"):
        print("safetensors")
        raise SystemExit(0)

if path.is_dir():
    if any(path.glob("*.gguf")):
        print("gguf")
        raise SystemExit(0)
    if (path / "model.safetensors.index.json").exists():
        print("safetensors")
        raise SystemExit(0)
    if any(path.glob("*.safetensors")):
        print("safetensors")
        raise SystemExit(0)

print("unknown")
PYEOF
}

backend_supports_model_format() {
    local backend=$1
    local format=$2
    case "${backend}:${format}" in
        inferflux_cuda:gguf|inferflux_cuda:safetensors|\
        llama_cpp_cuda:gguf|\
        ollama:gguf|\
        lmstudio:gguf|lmstudio:safetensors|\
        vllm:safetensors|\
        sglang:safetensors)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

managed_backend_pidfile() {
    echo "$OUTPUT_DIR/$1.pid"
}

managed_backend_logfile() {
    echo "$OUTPUT_DIR/server_$1.log"
}

validate_local_openai_backend_launch() {
    local label=$1 host=$2 bin_path=$3 model_path=$4
    if ! is_local_url "$host"; then
        log_err "$label autostart requires a local host URL, got: $host"
        return 1
    fi
    if [ ! -x "$bin_path" ]; then
        log_err "$label CLI not found or not executable: $bin_path"
        return 1
    fi
    if [ ! -e "$model_path" ]; then
        log_err "$label model path not found: $model_path"
        return 1
    fi
    return 0
}

start_vllm_server() {
    local backend=vllm
    local host
    host=$(url_host "$VLLM_HOST")
    local port
    port=$(url_port "$VLLM_HOST")
    local model_path
    model_path=$(realpath "$VLLM_MODEL_PATH")
    local served_model="${VLLM_MODEL:-$(basename "$model_path")}"
    local pidfile
    pidfile=$(managed_backend_pidfile "$backend")
    local log_file
    log_file=$(managed_backend_logfile "$backend")
    local vllm_bin_dir
    vllm_bin_dir=$(dirname "$VLLM_BIN")

    validate_local_openai_backend_launch "vLLM" "$VLLM_HOST" "$VLLM_BIN" "$model_path" || return 1
    stop_managed_openai_backend "$backend" >/dev/null 2>&1 || true
    if ! wait_for_port_free "$port"; then
        log_err "vLLM cannot start: port $port did not become free after cleanup"
        return 1
    fi
    require_port_free "vLLM" "$port" || return 1

    local -a cmd=("$VLLM_BIN" serve "$model_path" --host "$host" --port "$port" --served-model-name "$served_model")
    if [ -n "$VLLM_LAUNCH_ARGS" ]; then
        local -a extra_args=()
        read -r -a extra_args <<< "$VLLM_LAUNCH_ARGS"
        cmd+=("${extra_args[@]}")
    fi

    log "Starting vLLM on $VLLM_HOST using $model_path..."
    env \
        -u VLLM_HOST \
        -u VLLM_MODEL \
        -u VLLM_MODEL_PATH \
        -u VLLM_BIN \
        -u VLLM_LAUNCH_ARGS \
        -u AUTOSTART_VLLM \
        PATH="$vllm_bin_dir:$PATH" \
        "${cmd[@]}" > "$log_file" 2>&1 &
    local pid=$!
    echo "$pid" > "$pidfile"

    local waited=0
    while [ $waited -lt 120 ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            log_err "vLLM exited early. Last log lines:"
            tail -20 "$log_file" || true
            return 1
        fi
        if curl -sf "$VLLM_HOST/v1/models" >/dev/null 2>&1; then
            sleep 1
            if ! kill -0 "$pid" 2>/dev/null; then
                log_err "vLLM exited immediately after readiness. Last log lines:"
                tail -20 "$log_file" || true
                return 1
            fi
            if [ -z "$VLLM_MODEL" ]; then
                VLLM_MODEL="$served_model"
            fi
            log_ok "vLLM ready (PID $pid, model=$VLLM_MODEL)"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done

    log_err "vLLM did not start in 120s. Last log lines:"
    tail -20 "$log_file" || true
    kill "$pid" 2>/dev/null || true
    return 1
}

start_sglang_server() {
    local backend=sglang
    local host
    host=$(url_host "$SGLANG_HOST")
    local port
    port=$(url_port "$SGLANG_HOST")
    local model_path
    model_path=$(realpath "$SGLANG_MODEL_PATH")
    local served_model="${SGLANG_MODEL:-$(basename "$model_path")}"
    local pidfile
    pidfile=$(managed_backend_pidfile "$backend")
    local log_file
    log_file=$(managed_backend_logfile "$backend")
    local sglang_bin_dir
    sglang_bin_dir=$(dirname "$SGLANG_PYTHON")

    validate_local_openai_backend_launch "SGLang" "$SGLANG_HOST" "$SGLANG_PYTHON" "$model_path" || return 1
    stop_managed_openai_backend "$backend" >/dev/null 2>&1 || true
    if ! wait_for_port_free "$port"; then
        log_err "SGLang cannot start: port $port did not become free after cleanup"
        return 1
    fi
    require_port_free "SGLang" "$port" || return 1

    local -a cmd=("$SGLANG_PYTHON" -m sglang.launch_server --model-path "$model_path" --host "$host" --port "$port" --served-model-name "$served_model" --load-format safetensors)
    if [ -n "$SGLANG_LAUNCH_ARGS" ]; then
        local -a extra_args=()
        read -r -a extra_args <<< "$SGLANG_LAUNCH_ARGS"
        cmd+=("${extra_args[@]}")
    fi

    log "Starting SGLang on $SGLANG_HOST using $model_path..."
    env \
        -u SGLANG_HOST \
        -u SGLANG_MODEL \
        -u SGLANG_MODEL_PATH \
        -u SGLANG_PYTHON \
        -u SGLANG_LAUNCH_ARGS \
        -u AUTOSTART_SGLANG \
        PATH="$sglang_bin_dir:$PATH" \
        "${cmd[@]}" > "$log_file" 2>&1 &
    local pid=$!
    echo "$pid" > "$pidfile"

    local waited=0
    while [ $waited -lt 120 ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            log_err "SGLang exited early. Last log lines:"
            tail -20 "$log_file" || true
            return 1
        fi
        if curl -sf "$SGLANG_HOST/v1/models" >/dev/null 2>&1; then
            sleep 1
            if ! kill -0 "$pid" 2>/dev/null; then
                log_err "SGLang exited immediately after readiness. Last log lines:"
                tail -20 "$log_file" || true
                return 1
            fi
            if [ -z "$SGLANG_MODEL" ]; then
                SGLANG_MODEL="$served_model"
            fi
            log_ok "SGLang ready (PID $pid, model=$SGLANG_MODEL)"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done

    log_err "SGLang did not start in 120s. Last log lines:"
    tail -20 "$log_file" || true
    kill "$pid" 2>/dev/null || true
    return 1
}

stop_managed_openai_backend() {
    local backend=$1
    local pidfile
    pidfile=$(managed_backend_pidfile "$backend")
    [ -f "$pidfile" ] || return 0
    local host_url=""
    case "$backend" in
        vllm) host_url="$VLLM_HOST" ;;
        sglang) host_url="$SGLANG_HOST" ;;
        *) return 0 ;;
    esac
    local port
    port=$(url_port "$host_url")
    if [ -f "$pidfile" ]; then
        local pid
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            log "Stopping $backend (PID $pid)..."
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
        rm -f "$pidfile"
    fi
    if ! wait_for_port_free "$port"; then
        log_warn "$backend port $port is still busy after shutdown"
        return 1
    fi
    return 0
}

stop_stale_benchmark_inferflux_servers() {
    # Only target benchmark-generated configs, not general manual server configs.
    pkill -f 'inferfluxd --config .*config_inferflux_cuda.yaml' 2>/dev/null || true
    pkill -f 'inferfluxd --config .*config_llama_cpp_cuda.yaml' 2>/dev/null || true
}

load_prompt_suite() {
    local suite_path=$1
    python3 - "$suite_path" <<'PYEOF'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    suite = json.load(f)
for entry in suite.get("prompts", []):
    prompt = entry.get("prompt", "")
    if prompt:
        print(prompt)
PYEOF
}

write_prompt_suite_artifacts() {
    local suite_copy="$OUTPUT_DIR/prompt_suite.json"
    local summary_file="$OUTPUT_DIR/prompt_suite_summary.json"
    if [ -n "${INFERFLUX_BENCH_SINGLE_PROMPT:-}" ]; then
        python3 - "$summary_file" <<'PYEOF'
import json
import sys

summary = {
    "suite_id": "single_prompt_override",
    "prompt_count": 1,
    "categories": {"override": 1},
    "output_modes": {"text": 1},
    "length_buckets": {"custom": 1},
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
PYEOF
        return 0
    fi

    cp "$PROMPT_SUITE_PATH" "$suite_copy"
    python3 - "$PROMPT_SUITE_PATH" "$summary_file" <<'PYEOF'
import collections
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    suite = json.load(f)
prompts = suite.get("prompts", [])
summary = {
    "suite_id": suite.get("suite_id", "unknown"),
    "prompt_count": len(prompts),
    "categories": dict(collections.Counter(p.get("category", "unknown") for p in prompts)),
    "output_modes": dict(collections.Counter(p.get("output_mode", "unknown") for p in prompts)),
    "length_buckets": dict(collections.Counter(p.get("prompt_length_bucket", "unknown") for p in prompts)),
}
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
PYEOF
}

write_prompt_rotation_plan() {
    local backend=$1 concurrency=$2
    local plan_file="$OUTPUT_DIR/prompt_plan_${backend}_c${concurrency}.json"
    if [ -n "${INFERFLUX_BENCH_SINGLE_PROMPT:-}" ]; then
        python3 - "$plan_file" "$NUM_REQUESTS" <<'PYEOF'
import json
import sys

count = int(sys.argv[2])
plan = [{
    "request_index": i,
    "prompt_id": "single_prompt_override",
    "category": "override",
    "prompt_length_bucket": "custom",
    "output_mode": "text",
} for i in range(count)]
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(plan, f, indent=2)
PYEOF
        return 0
    fi

    python3 - "$PROMPT_SUITE_PATH" "$NUM_REQUESTS" "$plan_file" <<'PYEOF'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    prompts = json.load(f).get("prompts", [])
count = int(sys.argv[2])
plan = []
for i in range(count):
    entry = prompts[i % len(prompts)]
    plan.append({
        "request_index": i,
        "prompt_id": entry.get("id", f"prompt_{i}"),
        "category": entry.get("category", "unknown"),
        "prompt_length_bucket": entry.get("prompt_length_bucket", "unknown"),
        "output_mode": entry.get("output_mode", "unknown"),
    })
with open(sys.argv[3], "w", encoding="utf-8") as f:
    json.dump(plan, f, indent=2)
PYEOF
}

# Deterministic benchmark prompts
if [ -n "${INFERFLUX_BENCH_SINGLE_PROMPT:-}" ]; then
    PROMPTS=("$INFERFLUX_BENCH_SINGLE_PROMPT")
else
    mapfile -t PROMPTS < <(load_prompt_suite "$PROMPT_SUITE_PATH")
fi

# ============================================================================
# InferFlux Server Management
# ============================================================================

write_inferflux_config() {
    local backend=$1 port=$2 config_file=$3
    cat > "$config_file" <<EOF
server:
  host: "127.0.0.1"
  http_port: $port
  max_concurrent: 128
  enable_metrics: true

models:
  - id: bench-model
    path: "$(realpath "$MODEL_PATH")"
    format: $MODEL_FORMAT
    backend: $backend
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
    prefer_inferflux: $([ "$backend" = "inferflux_cuda" ] && echo "true" || echo "false")
    allow_llama_cpp_fallback: $([ "$backend" = "inferflux_cuda" ] && echo "false" || echo "true")
  scheduler:
    max_batch_size: 32
    max_batch_tokens: 16384
    min_batch_size: 1
    batch_accumulation_ms: 2
  disaggregated:
    prefill_pool_size: ${INFERFLUX_SCHED_PREFILL_POOL_SIZE:-1}
    decode_pool_size: ${INFERFLUX_SCHED_DECODE_POOL_SIZE:-1}
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
  level: warning
  format: text
EOF
}

start_inferflux_server() {
    local backend=$1 port=$2
    local config_file="$OUTPUT_DIR/config_${backend}.yaml"
    local log_file="$OUTPUT_DIR/server_${backend}.log"

    stop_inferflux_server "$backend" >/dev/null 2>&1 || true
    if ! wait_for_port_free "$port"; then
        log_err "$backend cannot start: port $port did not become free after cleanup"
        return 1
    fi
    if ! require_port_free "$backend" "$port"; then
        return 1
    fi

    write_inferflux_config "$backend" "$port" "$config_file"

    log "Starting $backend on port $port..."

    # Set environment variables for native backend
    local kv_batch="${INFERFLUX_CUDA_KV_MAX_BATCH:-16}"
    local kv_seq="${INFERFLUX_CUDA_KV_MAX_SEQ:-2048}"
    local strict="$([ "$backend" = "inferflux_cuda" ] && echo "1" || echo "0")"
    local enable_batched_decode="${INFERFLUX_ENABLE_BATCHED_DECODE:-1}"

    INFERFLUX_PORT_OVERRIDE=$port INFERCTL_API_KEY=$API_KEY \
        INFERFLUX_LOG_LEVEL=warning \
        INFERFLUX_ENABLE_BATCHED_DECODE=$enable_batched_decode \
        INFERFLUX_CUDA_KV_MAX_BATCH=$kv_batch \
        INFERFLUX_CUDA_KV_MAX_SEQ=$kv_seq \
        INFERFLUX_CUDA_STRICT=$strict \
        INFERFLUX_CUDA_DISABLE_PARITY_DELEGATE=$strict \
        "$BUILD_DIR/inferfluxd" --config "$config_file" \
        > "$log_file" 2>&1 &
    local pid=$!
    echo "$pid" > "$OUTPUT_DIR/${backend}.pid"

    # Wait for readiness (max 60s)
    local waited=0
    while [ $waited -lt 60 ]; do
        if ! kill -0 $pid 2>/dev/null; then
            log_err "$backend server exited early. Last log lines:"
            tail -20 "$log_file"
            return 1
        fi
        if curl -sf -H "Authorization: Bearer $API_KEY" \
            "http://127.0.0.1:$port/livez" >/dev/null 2>&1; then
            sleep 1
            if ! kill -0 $pid 2>/dev/null; then
                log_err "$backend exited immediately after readiness. Last log lines:"
                tail -20 "$log_file"
                return 1
            fi
            log_ok "$backend ready (PID $pid)"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done

    log_err "$backend did not start in 60s. Last log lines:"
    tail -20 "$log_file"
    kill $pid 2>/dev/null || true
    return 1
}

stop_inferflux_server() {
    local backend=$1
    local pidfile="$OUTPUT_DIR/${backend}.pid"
    local port=""
    case "$backend" in
        inferflux_cuda) port="$PORT_NATIVE" ;;
        llama_cpp_cuda) port="$PORT_LLAMA" ;;
    esac
    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            log "Stopping $backend (PID $pid)..."
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
        rm -f "$pidfile"
    fi
    if [ -n "$port" ]; then
        if ! wait_for_port_free "$port"; then
            log_warn "$backend port $port is still busy after shutdown"
            return 1
        fi
    fi
    local log_file="$OUTPUT_DIR/server_${backend}.log"
    if has_fatal_runtime_signature "$log_file"; then
        log_err "$backend log contains fatal runtime signature; treat this backend result as invalid"
        return 1
    fi
    return 0
}

reset_cuda_device() {
    log "Resetting CUDA device..."
    if python3 - <<'PYEOF' >/dev/null 2>&1
import ctypes
import ctypes.util
import sys

libname = ctypes.util.find_library("cudart")
if not libname:
    sys.exit(1)
try:
    cudart = ctypes.CDLL(libname)
except OSError:
    sys.exit(1)
if cudart.cudaDeviceReset() != 0:
    sys.exit(2)
PYEOF
    then
        log_ok "CUDA device reset"
    else
        log_warn "CUDA device reset unavailable; GPU state may persist until OS cleanup"
    fi
}

reset_benchmark_artifacts() {
    local backend=$1 concurrency=$2
    rm -rf "$OUTPUT_DIR/responses_${backend}/c${concurrency}"
    rm -f "$OUTPUT_DIR/stats_${backend}_c${concurrency}.json" \
          "$OUTPUT_DIR/mem_trace_${backend}_c${concurrency}.txt" \
          "$OUTPUT_DIR/admin_cache_${backend}_c${concurrency}.json" \
          "$OUTPUT_DIR/metrics_${backend}_c${concurrency}.txt" \
          "$OUTPUT_DIR/inferflux_cuda_bucket_winners_${backend}_c${concurrency}.json"
}

capture_inferflux_metrics_snapshot() {
    local backend=$1 port=$2 concurrency=$3
    local metrics_file="$OUTPUT_DIR/metrics_${backend}_c${concurrency}.txt"

    if ! curl -sf -H "Authorization: Bearer $API_KEY" \
        "http://127.0.0.1:$port/metrics" > "$metrics_file" 2>/dev/null; then
        log_warn "  Failed to capture metrics snapshot for $backend @ c=$concurrency"
        return 1
    fi
    return 0
}

capture_inferflux_cuda_bucket_winners() {
    local backend=$1 concurrency=$2
    local metrics_file="$OUTPUT_DIR/metrics_${backend}_c${concurrency}.txt"
    local summary_file="$OUTPUT_DIR/inferflux_cuda_bucket_winners_${backend}_c${concurrency}.json"
    local stats_file="$OUTPUT_DIR/stats_${backend}_c${concurrency}.json"

    if [ "$backend" != "inferflux_cuda" ] || [ ! -f "$metrics_file" ] || [ ! -f "$stats_file" ]; then
        return 0
    fi

    python3 scripts/extract_native_dispatch_winners.py \
        "$metrics_file" "$summary_file" --stats "$stats_file"
}

has_fatal_runtime_signature() {
    local log_file=$1
    [ -f "$log_file" ] || return 1
    grep -Eqi 'double free|corruption \(|segmentation fault|addresssanitizer|terminate called after throwing|fatal glibc error|aborted \(core dumped\)' "$log_file"
}

capture_inferflux_admin_cache_snapshot() {
    local backend=$1 port=$2 concurrency=$3
    local snapshot_file="$OUTPUT_DIR/admin_cache_${backend}_c${concurrency}.json"
    local stats_file="$OUTPUT_DIR/stats_${backend}_c${concurrency}.json"

    if ! curl -sf -H "Authorization: Bearer $API_KEY" \
        "http://127.0.0.1:$port/v1/admin/cache" > "$snapshot_file" 2>/dev/null; then
        log_warn "  Failed to capture admin cache snapshot for $backend @ c=$concurrency"
        return 0
    fi

    python3 - "$snapshot_file" "$stats_file" <<'PYEOF'
import json, sys

snapshot_path, stats_path = sys.argv[1], sys.argv[2]
with open(snapshot_path) as f:
    snapshot = json.load(f)
with open(stats_path) as f:
    stats = json.load(f)

memory = snapshot.get("memory", {})
stats["cache_snapshot"] = {
    "size": snapshot.get("size", 0),
    "capacity": snapshot.get("capacity", 0),
    "hits": snapshot.get("hits", 0),
    "misses": snapshot.get("misses", 0),
    "hit_rate": snapshot.get("hit_rate", 0.0),
    "partial_hits": snapshot.get("partial_hits", 0),
    "matched_tokens": snapshot.get("matched_tokens", 0),
    "kv_reuse_count": snapshot.get("kv_reuse_count", 0),
    "kv_reuse_tokens": snapshot.get("kv_reuse_tokens", 0),
}
stats["memory_snapshot"] = memory

with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)

inferflux_cuda_model = memory.get("inferflux_cuda_model", {})
inferflux_cuda_kv = memory.get("inferflux_cuda_kv", {})
paged_kv = memory.get("paged_kv", {})

print("inferflux_cuda_model_reserved_bytes=" +
      str(inferflux_cuda_model.get("reserved_bytes", 0)))
print("inferflux_cuda_model_in_use_bytes=" +
      str(inferflux_cuda_model.get("in_use_bytes", 0)))
print("inferflux_cuda_kv_active_bytes=" +
      str(inferflux_cuda_kv.get("active_bytes", 0)))
print("inferflux_cuda_kv_prefix_retained_bytes=" +
      str(inferflux_cuda_kv.get("prefix_retained_bytes", 0)))
print("paged_kv_used_bytes=" + str(paged_kv.get("used_bytes", 0)))
print("paged_kv_prefix_retained_bytes=" +
      str(paged_kv.get("prefix_retained_bytes", 0)))
PYEOF
}

# ============================================================================
# Ollama Availability Check
# ============================================================================

check_ollama_available() {
    log "Checking Ollama availability at $OLLAMA_HOST..."

    if ! curl -sf "$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
        log_warn "Ollama not available at $OLLAMA_HOST"
        log_warn "Set OLLAMA_HOST=http://your-host:port or SKIP_OLLAMA=true"
        return 1
    fi

    log_ok "Ollama is available"
    return 0
}

check_lmstudio_available() {
    check_openai_backend_available "LM Studio" "$LMSTUDIO_HOST" LMSTUDIO_HOST LMSTUDIO_MODEL SKIP_LMSTUDIO
}

check_vllm_available() {
    check_openai_backend_available "vLLM" "$VLLM_HOST" VLLM_HOST VLLM_MODEL SKIP_VLLM
}

check_sglang_available() {
    check_openai_backend_available "SGLang" "$SGLANG_HOST" SGLANG_HOST SGLANG_MODEL SKIP_SGLANG
}

should_autostart_backend() {
    case "$1" in
        vllm) [ "$AUTOSTART_VLLM" = "true" ] ;;
        sglang) [ "$AUTOSTART_SGLANG" = "true" ] ;;
        *) return 1 ;;
    esac
}

check_openai_backend_available() {
    local label=$1 host=$2 host_var=$3 model_var=$4 skip_var=$5
    local skip_value=${!skip_var:-false}

    if [ "$skip_value" = "true" ]; then
        log_warn "Skipping $label benchmark ($skip_var=true)"
        return 2
    fi

    log "Checking $label availability at $host..."

    local models_json
    models_json=$(curl -sf "$host/v1/models" 2>/dev/null) || {
        log_warn "$label not available at $host"
        log_warn "Set ${host_var}=http://your-host:port or ${skip_var}=true"
        return 1
    }

    local model_value=${!model_var:-}
    if [ -z "$model_value" ]; then
        model_value=$(printf '%s' "$models_json" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    models = data.get('data', [])
    print(models[0].get('id', '') if models else '')
except Exception:
    print('')
")
        printf -v "$model_var" '%s' "$model_value"
    fi

    if [ -z "${!model_var:-}" ]; then
        log_warn "$label responded but no model id could be discovered from /v1/models"
        log_warn "Set ${model_var} explicitly"
        return 1
    fi

    log_ok "$label is available (model=${!model_var})"
    return 0
}

# ============================================================================
# Request Runner
# ============================================================================

send_inferflux_request() {
    local port=$1 prompt=$2 max_tokens=$3 output_file=$4 request_id=$5

    mkdir -p "$(dirname "$output_file")"

    local start_ns=$(date +%s%N)
    local prompt_json
    prompt_json=$(printf '%s' "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')

    local response
    response=$(curl -sf -X POST "http://127.0.0.1:$port/v1/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        -H "x-inferflux-request-id: $request_id" \
        -d "{\"model\":\"default\",\"prompt\":$prompt_json,\"max_tokens\":$max_tokens,\"temperature\":0.0}" \
        --max-time 120 2>/dev/null) || {
        echo "ERROR" > "$output_file"
        return 1
    }

    local end_ns=$(date +%s%N)
    local latency_ms=$(( (end_ns - start_ns) / 1000000 ))

    local text
    text=$(echo "$response" | python3 -c "
import json, sys
d = json.load(sys.stdin)
text = d.get('choices', [{}])[0].get('text', '')
tokens = d.get('usage', {}).get('completion_tokens', 0)
print(json.dumps({'request_id': '$request_id', 'text': text.strip(), 'tokens': tokens, 'latency_ms': $latency_ms}))
" 2>/dev/null) || {
        echo "PARSE_ERROR" > "$output_file"
        return 1
    }

    echo "$text" > "$output_file"
}

send_ollama_request() {
    local prompt=$1 max_tokens=$2 output_file=$3 request_id=$4

    mkdir -p "$(dirname "$output_file")"

    local start_ns=$(date +%s%N)

    local response
    response=$(curl -sf -X POST "${OLLAMA_HOST}/api/generate" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${OLLAMA_MODEL:-qwen2.5:3b}\",\"prompt\":$(printf '%s' "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),\"options\":{\"num_predict\":$max_tokens,\"temperature\":0.0},\"stream\":false}" \
        --max-time 120 2>/dev/null) || {
        echo "ERROR" > "$output_file"
        return 1
    }

    local end_ns=$(date +%s%N)
    local latency_ms=$(( (end_ns - start_ns) / 1000000 ))

    local text
    text=$(echo "$response" | python3 -c "
import json, sys
d = json.load(sys.stdin)
text = d.get('response', '')
tokens = len(text.split())  # Ollama doesn't return token count
print(json.dumps({'request_id': '$request_id', 'text': text.strip(), 'tokens': tokens, 'latency_ms': $latency_ms}))
" 2>/dev/null) || {
        echo "PARSE_ERROR" > "$output_file"
        return 1
    }

    echo "$text" > "$output_file"
}

send_lmstudio_request() {
    send_openai_request "$LMSTUDIO_HOST" "$LMSTUDIO_MODEL" "$1" "$2" "$3" "$4"
}

send_vllm_request() {
    send_openai_request "$VLLM_HOST" "$VLLM_MODEL" "$1" "$2" "$3" "$4"
}

send_sglang_request() {
    send_openai_request "$SGLANG_HOST" "$SGLANG_MODEL" "$1" "$2" "$3" "$4"
}

send_openai_request() {
    local host=$1 model=$2 prompt=$3 max_tokens=$4 output_file=$5 request_id=$6

    mkdir -p "$(dirname "$output_file")"

    local start_ns=$(date +%s%N)
    local prompt_json
    prompt_json=$(printf '%s' "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')

    local response
    response=$(curl -sf -X POST "${host}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${model}\",\"prompt\":$prompt_json,\"max_tokens\":$max_tokens,\"temperature\":0.0}" \
        --max-time 120 2>/dev/null) || {
        echo "ERROR" > "$output_file"
        return 1
    }

    local end_ns=$(date +%s%N)
    local latency_ms=$(( (end_ns - start_ns) / 1000000 ))

    local text
    text=$(echo "$response" | python3 -c "
import json, sys
d = json.load(sys.stdin)
text = d.get('choices', [{}])[0].get('text', '')
tokens = d.get('usage', {}).get('completion_tokens', 0)
print(json.dumps({'request_id': '$request_id', 'text': text.strip(), 'tokens': tokens, 'latency_ms': $latency_ms}))
" 2>/dev/null) || {
        echo "PARSE_ERROR" > "$output_file"
        return 1
    }

    echo "$text" > "$output_file"
}

# ============================================================================
# Benchmark Runner
# ============================================================================

run_benchmark() {
    local backend=$1 concurrency=$2 port_or_url=$3
    local backend_kind=$4

    local results_dir="$OUTPUT_DIR/responses_${backend}/c${concurrency}"
    mkdir -p "$results_dir"
    write_prompt_rotation_plan "$backend" "$concurrency"

    log "  Warmup (2 requests)..."

    # Measure GPU memory after warmup
    sleep 1
    local mem_loaded=$(gpu_mem_mb)

    # Run benchmark
    log "  Running $NUM_REQUESTS requests (concurrency=$concurrency)..."
    local start_time=$(date +%s%N)

    # Track peak memory and GPU utilization.
    # Seed the trace with an initial sample so transient nvidia-smi failures do
    # not collapse the whole row to zero.
    local mem_peak=$mem_loaded
    local gpu_util_peak=0
    local initial_util
    initial_util=$(gpu_utilization || true)
    initial_util=${initial_util:-0}
    local mem_trace_file="$OUTPUT_DIR/mem_trace_${backend}_c${concurrency}.txt"
    printf '%s %s\n' "${mem_loaded:-0}" "$initial_util" > "$mem_trace_file"
    (
        set +e
        while true; do
            local m u
            m=$(gpu_mem_mb 2>/dev/null || true)
            u=$(gpu_utilization 2>/dev/null || true)
            [ -n "$m" ] || m=${mem_loaded:-0}
            [ -n "$u" ] || u=0
            printf '%s %s\n' "$m" "$u"
            sleep 0.2 || break
        done
    ) >> "$mem_trace_file" &
    local monitor_pid=$!

    local driver_backend_kind=""
    local driver_endpoint=""
    local driver_model=""
    case "$backend_kind" in
        ollama)
            driver_backend_kind="ollama"
            driver_endpoint="${OLLAMA_HOST}/api/generate"
            driver_model="${OLLAMA_MODEL:-qwen2.5:3b}"
            ;;
        lmstudio)
            driver_backend_kind="openai"
            driver_endpoint="${LMSTUDIO_HOST}/v1/completions"
            driver_model="${LMSTUDIO_MODEL}"
            ;;
        vllm)
            driver_backend_kind="openai"
            driver_endpoint="${VLLM_HOST}/v1/completions"
            driver_model="${VLLM_MODEL}"
            ;;
        sglang)
            driver_backend_kind="openai"
            driver_endpoint="${SGLANG_HOST}/v1/completions"
            driver_model="${SGLANG_MODEL}"
            ;;
        *)
            driver_backend_kind="inferflux"
            driver_endpoint="http://127.0.0.1:${port_or_url}/v1/completions"
            driver_model="default"
            ;;
    esac

    local -a driver_cmd=(
        python3 "$SCRIPT_DIR/benchmark_request_driver.py"
        --backend-kind "$driver_backend_kind"
        --endpoint "$driver_endpoint"
        --model "$driver_model"
        --api-key "$API_KEY"
        --num-requests "$NUM_REQUESTS"
        --max-tokens "$MAX_TOKENS"
        --concurrency "$concurrency"
        --output-dir "$results_dir"
    )
    if [ -n "${INFERFLUX_BENCH_SINGLE_PROMPT:-}" ]; then
        driver_cmd+=(--single-prompt "$INFERFLUX_BENCH_SINGLE_PROMPT")
    else
        driver_cmd+=(--prompt-suite "$PROMPT_SUITE_PATH")
    fi

    local driver_summary
    driver_summary=$("${driver_cmd[@]}")

    local end_time=$(date +%s%N)
    local total_ms=$(( (end_time - start_time) / 1000000 ))

    # Stop memory monitor
    kill $monitor_pid 2>/dev/null || true
    wait $monitor_pid 2>/dev/null || true

    # Compute peak memory and GPU utilization
    if [ -f "$mem_trace_file" ]; then
        mem_peak=$(awk '{print $1}' "$mem_trace_file" | sed '/^$/d' | sort -rn | head -1)
        gpu_util_peak=$(awk '{print $2}' "$mem_trace_file" | sed '/^$/d' | sort -rn | head -1)
    fi

    # Aggregate results
    local total_tokens
    total_tokens=$(printf '%s' "$driver_summary" | python3 -c "import json,sys; print(json.load(sys.stdin)['total_tokens'])")
    local total_latency
    total_latency=$(printf '%s' "$driver_summary" | python3 -c "import json,sys; print(json.load(sys.stdin)['total_latency_ms'])")
    local success_count
    success_count=$(printf '%s' "$driver_summary" | python3 -c "import json,sys; print(json.load(sys.stdin)['success_count'])")
    local latencies
    latencies=$(printf '%s' "$driver_summary" | python3 -c "import json,sys; print(' '.join(str(v) for v in json.load(sys.stdin)['latencies']))")
    local wall_time_ms
    wall_time_ms=$(printf '%s' "$driver_summary" | python3 -c "import json,sys; print(json.load(sys.stdin).get('wall_time_ms', 0))")
    if [ "${wall_time_ms:-0}" -gt 0 ]; then
        total_ms=$wall_time_ms
    fi

    # Compute stats
    local tok_per_sec=0
    if [ $total_ms -gt 0 ]; then
        tok_per_sec=$(python3 -c "print(f'{$total_tokens / ($total_ms / 1000.0):.1f}')")
    fi

    local avg_latency=0
    local p50=0 p95=0 p99=0
    if [ $success_count -gt 0 ]; then
        avg_latency=$((total_latency / success_count))
        local sorted_lats=$(echo $latencies | tr ' ' '\n' | sort -n)
        p50=$(echo "$sorted_lats" | awk "NR==int($(echo $success_count)*0.50+0.5){print}")
        p95=$(echo "$sorted_lats" | awk "NR==int($(echo $success_count)*0.95+0.5){print}")
        p99=$(echo "$sorted_lats" | awk "NR==int($(echo $success_count)*0.99+0.5){print}")
    fi

    # Store results
    cat > "$OUTPUT_DIR/stats_${backend}_c${concurrency}.json" <<EOF
{
    "backend": "$backend",
    "concurrency": $concurrency,
    "tok_per_sec": $tok_per_sec,
    "avg_latency_ms": $avg_latency,
    "p50_latency_ms": ${p50:-0},
    "p95_latency_ms": ${p95:-0},
    "p99_latency_ms": ${p99:-0},
    "total_tokens": $total_tokens,
    "total_time_ms": $total_ms,
    "success_count": $success_count,
    "total_requests": $NUM_REQUESTS,
    "gpu_mem_loaded_mb": $mem_loaded,
    "gpu_mem_peak_mb": ${mem_peak:-0},
    "gpu_util_peak_percent": ${gpu_util_peak:-0}
}
EOF

    local memory_summary=""
    if [ "$backend_kind" = "inferflux" ]; then
        capture_inferflux_metrics_snapshot "$backend" "$port_or_url" "$concurrency" >/dev/null 2>&1 || true
        memory_summary=$(capture_inferflux_admin_cache_snapshot \
            "$backend" "$port_or_url" "$concurrency" 2>/dev/null || true)
    fi

    log_ok "  $backend @ c=$concurrency: $success_count/$NUM_REQUESTS OK, ${tok_per_sec} tok/s, avg ${avg_latency}ms, GPU ${gpu_util_peak}%"
    if [ -n "${memory_summary:-}" ]; then
        log "  Memory snapshot:"
        printf '%s\n' "$memory_summary" | sed 's/^/    /'
    fi
    if [ "$backend" = "inferflux_cuda" ]; then
        local bucket_winner_summary=""
        bucket_winner_summary=$(capture_inferflux_cuda_bucket_winners "$backend" "$concurrency" 2>/dev/null || true)
        if [ -n "${bucket_winner_summary:-}" ]; then
            log "  Native dispatch bucket winners:"
            printf '%s\n' "$bucket_winner_summary" | sed 's/^/    /'
        fi
    fi

    if [ $success_count -lt $NUM_REQUESTS ]; then
        return 1
    fi
}

# ============================================================================
# Response Similarity
# ============================================================================

compare_responses() {
    python3 - "$OUTPUT_DIR" "$CONCURRENCY_LEVELS" <<'PYEOF'
import json
import os
import re
import sys

results_dir = sys.argv[1]
concurrency_levels = [int(x) for x in sys.argv[2].split(',')]
backends = ["llama_cpp_cuda", "inferflux_cuda"]

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def jaccard(a, b):
    sa, sb = set(a), set(b)
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0

def overlap(a, b):
    sa, sb = set(a), set(b)
    total = len(sa) + len(sb)
    return 2.0 * len(sa & sb) / total if total else 1.0

for concurrency in concurrency_levels:
    responses = {}
    for backend in backends:
        resp_dir = os.path.join(results_dir, f"responses_{backend}", f"c{concurrency}")
        if not os.path.isdir(resp_dir):
            continue
        responses[backend] = {}
        for name in sorted(os.listdir(resp_dir)):
            if not name.endswith(".json"):
                continue
            try:
                idx = int(name.split("_")[1].split(".")[0])
                with open(os.path.join(resp_dir, name), "r", encoding="utf-8") as f:
                    data = json.load(f)
                responses[backend][idx] = data.get("text", "")
            except Exception:
                continue

    if len(responses) < 2:
        continue

    lhs = responses[backends[0]]
    rhs = responses[backends[1]]
    common = sorted(set(lhs.keys()) & set(rhs.keys()))
    if not common:
        continue

    exact = 0
    jaccards = []
    overlaps = []
    for idx in common:
        ta, tb = lhs[idx], rhs[idx]
        if ta == tb:
            exact += 1
        toks_a = tokenize(ta)
        toks_b = tokenize(tb)
        jaccards.append(jaccard(toks_a, toks_b))
        overlaps.append(overlap(toks_a, toks_b))

    comp = {
        "concurrency": concurrency,
        "backends": backends,
        "num_compared": len(common),
        "exact_match_rate": exact / len(common),
        "mean_jaccard": sum(jaccards) / len(jaccards),
        "mean_overlap": sum(overlaps) / len(overlaps),
    }

    sim_path = os.path.join(results_dir, f"similarity_c{concurrency}.json")
    with open(sim_path, "w", encoding="utf-8") as f:
        json.dump(comp, f, indent=2)

    for backend, peer in ((backends[0], backends[1]), (backends[1], backends[0])):
        stats_path = os.path.join(results_dir, f"stats_{backend}_c{concurrency}.json")
        if not os.path.exists(stats_path):
            continue
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        stats["comparison_backend"] = peer
        stats["compared_responses"] = comp["num_compared"]
        stats["exact_match_rate"] = comp["exact_match_rate"]
        stats["mean_jaccard"] = comp["mean_jaccard"]
        stats["mean_overlap"] = comp["mean_overlap"]
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
PYEOF
}

print_benchmark_banner() {
    header "Multi-Backend Comparison Benchmark"
    echo "  Model:       $MODEL_PATH"
    echo "  GPU:         $(gpu_name)"
    echo "  GPU Memory:  $(gpu_mem_total_mb) MB total"
    echo "  Requests:    $NUM_REQUESTS per concurrency level"
    echo "  Max tokens:  $MAX_TOKENS"
    echo "  Concurrency: $CONCURRENCY_LEVELS"
    echo "  Build dir:   $BUILD_DIR"
    echo ""
}

invoke_isolated_backend_child() {
    local backend=$1
    log "Launching isolated backend benchmark for $backend"
    env \
        BUILD_DIR="$BUILD_DIR" \
        OUTPUT_DIR="$OUTPUT_DIR" \
        CONCURRENCY_LEVELS="$CONCURRENCY_LEVELS" \
        NUM_REQUESTS="$NUM_REQUESTS" \
        MAX_TOKENS="$MAX_TOKENS" \
        API_KEY="$API_KEY" \
        PROMPT_SUITE_PATH="$PROMPT_SUITE_PATH" \
        MODEL_FORMAT="$MODEL_FORMAT" \
        OLLAMA_HOST="$OLLAMA_HOST" \
        LMSTUDIO_HOST="$LMSTUDIO_HOST" \
        LMSTUDIO_MODEL="$LMSTUDIO_MODEL" \
        VLLM_HOST="$VLLM_HOST" \
        VLLM_MODEL="$VLLM_MODEL" \
        VLLM_MODEL_PATH="$VLLM_MODEL_PATH" \
        VLLM_BIN="$VLLM_BIN" \
        VLLM_LAUNCH_ARGS="$VLLM_LAUNCH_ARGS" \
        AUTOSTART_VLLM="$AUTOSTART_VLLM" \
        SGLANG_HOST="$SGLANG_HOST" \
        SGLANG_MODEL="$SGLANG_MODEL" \
        SGLANG_MODEL_PATH="$SGLANG_MODEL_PATH" \
        SGLANG_PYTHON="$SGLANG_PYTHON" \
        SGLANG_LAUNCH_ARGS="$SGLANG_LAUNCH_ARGS" \
        AUTOSTART_SGLANG="$AUTOSTART_SGLANG" \
        SAFETENSORS_MODEL_PATH="$SAFETENSORS_MODEL_PATH" \
        SKIP_OLLAMA="$SKIP_OLLAMA" \
        SKIP_LMSTUDIO="$SKIP_LMSTUDIO" \
        SKIP_VLLM="$SKIP_VLLM" \
        SKIP_SGLANG="$SKIP_SGLANG" \
        PORT_NATIVE="$PORT_NATIVE" \
        PORT_LLAMA="$PORT_LLAMA" \
        RESET_CUDA_BETWEEN_BACKENDS="$RESET_CUDA_BETWEEN_BACKENDS" \
        INFERFLUX_BENCH_CHILD_MODE=1 \
        INFERFLUX_BENCH_SINGLE_BACKEND="$backend" \
        "${BASH:-/bin/bash}" "$0" "$MODEL_PATH"
}

# ============================================================================
# Main Benchmark Loop
# ============================================================================

main() {
    BUILD_DIR=$(resolve_build_dir)
    print_benchmark_banner

    mkdir -p "$OUTPUT_DIR"
    write_prompt_suite_artifacts
    stop_stale_benchmark_inferflux_servers
    sleep 1

    # Validate model
    if [ ! -e "$MODEL_PATH" ]; then
        log_err "Model not found: $MODEL_PATH"
        exit 1
    fi

    MODEL_FORMAT=$(detect_model_format "$MODEL_PATH")
    if [ "$MODEL_FORMAT" = "unknown" ]; then
        log_err "Unable to detect model format from: $MODEL_PATH"
        log_err "Expected a .gguf file/directory or a safetensors file/directory."
        exit 1
    fi
    echo "  Model fmt:   $MODEL_FORMAT"

    if [ "$MODEL_FORMAT" = "safetensors" ]; then
        if [ -z "$SAFETENSORS_MODEL_PATH_INPUT" ]; then
            SAFETENSORS_MODEL_PATH="$MODEL_PATH"
        fi
        if [ -z "$VLLM_MODEL_PATH_INPUT" ]; then
            VLLM_MODEL_PATH="$SAFETENSORS_MODEL_PATH"
        fi
        if [ -z "$SGLANG_MODEL_PATH_INPUT" ]; then
            SGLANG_MODEL_PATH="$SAFETENSORS_MODEL_PATH"
        fi
    fi

    if [ -n "$INFERFLUX_BENCH_SINGLE_BACKEND" ]; then
        local backend_known=false
        local backend
        for backend in "${ALL_BACKENDS[@]}"; do
            if [ "$backend" = "$INFERFLUX_BENCH_SINGLE_BACKEND" ]; then
                backend_known=true
                break
            fi
        done
        if [ "$backend_known" != "true" ]; then
            log_err "Unknown isolated backend: $INFERFLUX_BENCH_SINGLE_BACKEND"
            exit 1
        fi
    fi

    # Validate inferfluxd
    if [ ! -f "$BUILD_DIR/inferfluxd" ]; then
        log_err "inferfluxd not found at $BUILD_DIR/inferfluxd"
        log "Build with: cmake -S . -B $BUILD_DIR -DENABLE_CUDA=ON && cmake --build $BUILD_DIR -j"
        exit 1
    fi

    # Check nvidia-smi
    if ! nvidia-smi >/dev/null 2>&1; then
        log_err "nvidia-smi not found — CUDA GPU required"
        exit 1
    fi

    if [ "$INFERFLUX_BENCH_CHILD_MODE" != "1" ]; then
        local requested_backends=("${ALL_BACKENDS[@]}")
        if [ -n "$INFERFLUX_BENCH_SINGLE_BACKEND" ]; then
            requested_backends=("$INFERFLUX_BENCH_SINGLE_BACKEND")
        fi

        local backend
        for backend in "${requested_backends[@]}"; do
            if ! invoke_isolated_backend_child "$backend"; then
                log_warn "Isolated backend benchmark failed for $backend"
            fi
        done

        compare_responses

        echo ""
        header "Scaling Analysis Report"
        generate_report
        save_combined_results
        log_ok "Benchmark complete! Results saved to $OUTPUT_DIR"
        return 0
    fi

    local mem_baseline=$(gpu_mem_mb)
    log "GPU baseline memory: ${mem_baseline} MB"
    log "InferFlux backends will run one at a time."
    log "Each backend is torn down before the next one starts."

    # Convert comma-separated concurrency levels to array
    IFS=',' read -ra CONCURRENCY_ARRAY <<< "$CONCURRENCY_LEVELS"

    # Backend configurations
    declare -A BACKEND_PORTS
    declare -A BACKEND_KIND
    declare -A BACKEND_AVAILABLE
    declare -A BACKEND_FORMAT_COMPATIBLE
    local requested_backends=("${ALL_BACKENDS[@]}")
    if [ -n "$INFERFLUX_BENCH_SINGLE_BACKEND" ]; then
        requested_backends=("$INFERFLUX_BENCH_SINGLE_BACKEND")
    fi

    BACKEND_PORTS[inferflux_cuda]=$PORT_NATIVE
    BACKEND_KIND[inferflux_cuda]=inferflux
    BACKEND_AVAILABLE[inferflux_cuda]=true

    BACKEND_PORTS[llama_cpp_cuda]=$PORT_LLAMA
    BACKEND_KIND[llama_cpp_cuda]=inferflux
    BACKEND_AVAILABLE[llama_cpp_cuda]=true

    BACKEND_PORTS[ollama]="$OLLAMA_HOST"
    BACKEND_KIND[ollama]=ollama

    BACKEND_PORTS[lmstudio]="$LMSTUDIO_HOST"
    BACKEND_KIND[lmstudio]=lmstudio

    BACKEND_PORTS[vllm]="$VLLM_HOST"
    BACKEND_KIND[vllm]=vllm

    BACKEND_PORTS[sglang]="$SGLANG_HOST"
    BACKEND_KIND[sglang]=sglang

    local backend
    for backend in "${requested_backends[@]}"; do
        if backend_supports_model_format "$backend" "$MODEL_FORMAT"; then
            BACKEND_FORMAT_COMPATIBLE[$backend]=true
        else
            BACKEND_FORMAT_COMPATIBLE[$backend]=false
            BACKEND_AVAILABLE[$backend]=false
            log_warn "Skipping $backend benchmark (model format $MODEL_FORMAT unsupported)"
        fi
    done

    for backend in "${requested_backends[@]}"; do
        for concurrency in "${CONCURRENCY_ARRAY[@]}"; do
            reset_benchmark_artifacts "$backend" "$concurrency"
        done
    done

    # Check Ollama availability
    if [[ " ${requested_backends[*]} " == *" ollama "* ]] && [ "${BACKEND_FORMAT_COMPATIBLE[ollama]}" != "true" ]; then
        BACKEND_AVAILABLE[ollama]=false
    elif [[ " ${requested_backends[*]} " == *" ollama "* ]] && [ "$SKIP_OLLAMA" = "true" ]; then
        log_warn "Skipping Ollama benchmark (SKIP_OLLAMA=true)"
        BACKEND_AVAILABLE[ollama]=false
    elif [[ " ${requested_backends[*]} " == *" ollama "* ]] && check_ollama_available; then
        BACKEND_AVAILABLE[ollama]=true
    elif [[ " ${requested_backends[*]} " == *" ollama "* ]]; then
        BACKEND_AVAILABLE[ollama]=false
    fi

    if [[ " ${requested_backends[*]} " == *" lmstudio "* ]] && [ "${BACKEND_FORMAT_COMPATIBLE[lmstudio]}" != "true" ]; then
        BACKEND_AVAILABLE[lmstudio]=false
    elif [[ " ${requested_backends[*]} " == *" lmstudio "* ]] && check_lmstudio_available; then
        BACKEND_AVAILABLE[lmstudio]=true
    elif [[ " ${requested_backends[*]} " == *" lmstudio "* ]] && [ "$SKIP_LMSTUDIO" = "true" ]; then
        BACKEND_AVAILABLE[lmstudio]=false
    elif [[ " ${requested_backends[*]} " == *" lmstudio "* ]]; then
        BACKEND_AVAILABLE[lmstudio]=false
    fi

    if [[ " ${requested_backends[*]} " == *" vllm "* ]] && [ "${BACKEND_FORMAT_COMPATIBLE[vllm]}" != "true" ]; then
        BACKEND_AVAILABLE[vllm]=false
    elif [[ " ${requested_backends[*]} " == *" vllm "* ]] && [ "$SKIP_VLLM" = "true" ]; then
        log_warn "Skipping vLLM benchmark (SKIP_VLLM=true)"
        BACKEND_AVAILABLE[vllm]=false
    elif [[ " ${requested_backends[*]} " == *" vllm "* ]] && should_autostart_backend vllm; then
        if validate_local_openai_backend_launch "vLLM" "$VLLM_HOST" "$VLLM_BIN" "$VLLM_MODEL_PATH"; then
            BACKEND_AVAILABLE[vllm]=true
        else
            BACKEND_AVAILABLE[vllm]=false
        fi
    elif [[ " ${requested_backends[*]} " == *" vllm "* ]] && check_vllm_available; then
        BACKEND_AVAILABLE[vllm]=true
    elif [[ " ${requested_backends[*]} " == *" vllm "* ]]; then
        BACKEND_AVAILABLE[vllm]=false
    fi

    if [[ " ${requested_backends[*]} " == *" sglang "* ]] && [ "${BACKEND_FORMAT_COMPATIBLE[sglang]}" != "true" ]; then
        BACKEND_AVAILABLE[sglang]=false
    elif [[ " ${requested_backends[*]} " == *" sglang "* ]] && [ "$SKIP_SGLANG" = "true" ]; then
        log_warn "Skipping SGLang benchmark (SKIP_SGLANG=true)"
        BACKEND_AVAILABLE[sglang]=false
    elif [[ " ${requested_backends[*]} " == *" sglang "* ]] && should_autostart_backend sglang; then
        if validate_local_openai_backend_launch "SGLang" "$SGLANG_HOST" "$SGLANG_PYTHON" "$SGLANG_MODEL_PATH"; then
            BACKEND_AVAILABLE[sglang]=true
        else
            BACKEND_AVAILABLE[sglang]=false
        fi
    elif [[ " ${requested_backends[*]} " == *" sglang "* ]] && check_sglang_available; then
        BACKEND_AVAILABLE[sglang]=true
    elif [[ " ${requested_backends[*]} " == *" sglang "* ]]; then
        BACKEND_AVAILABLE[sglang]=false
    fi

    # Run benchmarks for all backends and concurrency levels
    for backend in "${requested_backends[@]}"; do
        if [ "${BACKEND_AVAILABLE[$backend]}" != "true" ]; then
            continue
        fi

        local port_or_url="${BACKEND_PORTS[$backend]}"
        local backend_kind="${BACKEND_KIND[$backend]}"

        if [ "$backend_kind" = "inferflux" ]; then
            echo ""
            header "Starting: $backend"
            local mem_before_start=$(gpu_mem_mb)
            if ! start_inferflux_server "$backend" "$port_or_url"; then
                log_err "Failed to start $backend"
                continue
            fi
            local mem_after_start=$(gpu_mem_mb)
            local mem_for_model=$((mem_after_start - mem_before_start))
            log "GPU memory after $backend startup: ${mem_after_start} MB (delta: +${mem_for_model} MB)"
        elif [ "$backend" = "vllm" ] && should_autostart_backend vllm; then
            echo ""
            header "Starting: $backend"
            local mem_before_start=$(gpu_mem_mb)
            if ! start_vllm_server; then
                log_err "Failed to start $backend"
                continue
            fi
            local mem_after_start=$(gpu_mem_mb)
            local mem_for_model=$((mem_after_start - mem_before_start))
            log "GPU memory after $backend startup: ${mem_after_start} MB (delta: +${mem_for_model} MB)"
        elif [ "$backend" = "sglang" ] && should_autostart_backend sglang; then
            echo ""
            header "Starting: $backend"
            local mem_before_start=$(gpu_mem_mb)
            if ! start_sglang_server; then
                log_err "Failed to start $backend"
                continue
            fi
            local mem_after_start=$(gpu_mem_mb)
            local mem_for_model=$((mem_after_start - mem_before_start))
            log "GPU memory after $backend startup: ${mem_after_start} MB (delta: +${mem_for_model} MB)"
        fi

        for concurrency in "${CONCURRENCY_ARRAY[@]}"; do
            echo ""
            header "Benchmarking: $backend @ concurrency=$concurrency"

            if ! run_benchmark "$backend" "$concurrency" "$port_or_url" "$backend_kind"; then
                log_warn "  Some requests failed for $backend @ c=$concurrency"
            fi

            # Let GPU memory settle between runs
            sleep 2
        done

        if [ "$backend_kind" = "inferflux" ]; then
            local mem_before_stop=$(gpu_mem_mb)
            if ! stop_inferflux_server "$backend"; then
                log_warn "  $backend shutdown reported a fatal runtime signature"
            fi
            if [ "$RESET_CUDA_BETWEEN_BACKENDS" = "true" ]; then
                reset_cuda_device
            fi
            sleep 3
            local mem_after_stop=$(gpu_mem_mb)
            local mem_freed=$((mem_before_stop - mem_after_stop))
            log "GPU memory after stopping $backend: ${mem_after_stop} MB (freed: ${mem_freed} MB)"
        elif [ "$backend" = "vllm" ] && should_autostart_backend vllm; then
            local mem_before_stop=$(gpu_mem_mb)
            stop_managed_openai_backend "$backend" || true
            sleep 3
            local mem_after_stop=$(gpu_mem_mb)
            local mem_freed=$((mem_before_stop - mem_after_stop))
            log "GPU memory after stopping $backend: ${mem_after_stop} MB (freed: ${mem_freed} MB)"
        elif [ "$backend" = "sglang" ] && should_autostart_backend sglang; then
            local mem_before_stop=$(gpu_mem_mb)
            stop_managed_openai_backend "$backend" || true
            sleep 3
            local mem_after_stop=$(gpu_mem_mb)
            local mem_freed=$((mem_before_stop - mem_after_stop))
            log "GPU memory after stopping $backend: ${mem_after_stop} MB (freed: ${mem_freed} MB)"
        fi
    done

    log_ok "Isolated backend benchmark complete for ${requested_backends[*]}"
}

# ============================================================================
# Report Generation
# ============================================================================

generate_report() {
    python3 - "$OUTPUT_DIR" "$CONCURRENCY_LEVELS" <<'PYEOF'
import json, os, sys

output_dir = sys.argv[1]
concurrency_levels = sys.argv[2].split(',')

backends = ['inferflux_cuda', 'llama_cpp_cuda', 'ollama', 'lmstudio', 'vllm', 'sglang']

# Load all results
results = {}
for backend in backends:
    results[backend] = {}
    for c in concurrency_levels:
        stats_file = os.path.join(output_dir, f"stats_{backend}_c{c}.json")
        if os.path.exists(stats_file):
            with open(stats_file) as f:
                results[backend][int(c)] = json.load(f)

# Print header
print()
print(f"{'Backend':<20} {'C':<4} {'Tok/s':>10} {'Avg(ms)':>10} {'P50(ms)':>10} {'P95(ms)':>10} {'GPU Mem':>10} {'GPU %':>8}")
print("-" * 92)

# Print results sorted by concurrency
for c in sorted([int(x) for x in concurrency_levels]):
    for backend in backends:
        if c in results[backend]:
            r = results[backend][c]
            print(f"{backend:<20} {c:<4} {r['tok_per_sec']:>10.1f} "
                  f"{r['avg_latency_ms']:>10.0f} {r['p50_latency_ms']:>10.0f} "
                  f"{r['p95_latency_ms']:>10.0f} {r['gpu_mem_peak_mb']:>10.0f} "
                  f"{r['gpu_util_peak_percent']:>8.0f}")
    print()

# Print scaling efficiency (speedup from c=1)
print()
print("Scaling Efficiency (Speedup from c=1):")
print("-" * 60)
for backend in backends:
    if 1 in results[backend]:
        baseline = results[backend][1]['tok_per_sec']
        if baseline > 0:
            print(f"{backend:<20}: ", end='')
            for c in sorted([int(x) for x in concurrency_levels if int(x) > 1]):
                if c in results[backend]:
                    speedup = results[backend][c]['tok_per_sec'] / baseline
                    efficiency = (speedup / c) * 100
                    print(f"c={c}={speedup:.2f}x ({efficiency:.0f}%)  ", end='')
            print()

# Print memory efficiency
print()
print("Memory Efficiency:")
print("-" * 60)
for backend in backends:
    if 1 in results[backend]:
        baseline_mem = results[backend][1]['gpu_mem_peak_mb']
        print(f"{backend:<20}: ", end='')
        for c in sorted([int(x) for x in concurrency_levels if int(x) > 1]):
            if c in results[backend]:
                mem = results[backend][c]['gpu_mem_peak_mb']
                mem_ratio = mem / baseline_mem if baseline_mem > 0 else 0
                print(f"c={c}={mem:.0f}MB ({mem_ratio:.2f}x)  ", end='')
        print()

print()
print("Similarity (inferflux_cuda vs llama_cpp_cuda):")
print("-" * 60)
for c in sorted([int(x) for x in concurrency_levels]):
    sim_file = os.path.join(output_dir, f"similarity_c{c}.json")
    if not os.path.exists(sim_file):
        continue
    with open(sim_file) as f:
        sim = json.load(f)
    print(f"c={c:<2} exact={sim.get('exact_match_rate', 0.0):.3f}  "
          f"jaccard={sim.get('mean_jaccard', 0.0):.3f}  "
          f"overlap={sim.get('mean_overlap', 0.0):.3f}  "
          f"compared={sim.get('num_compared', 0)}")

PYEOF
}

# ============================================================================
# Save Combined Results
# ============================================================================

save_combined_results() {
    python3 - "$OUTPUT_DIR" "$CONCURRENCY_LEVELS" "$MODEL_PATH" <<'PYEOF'
import json, os, sys, time

output_dir = sys.argv[1]
concurrency_levels = sys.argv[2].split(',')
model_path = sys.argv[3]

backends = ['inferflux_cuda', 'llama_cpp_cuda', 'ollama', 'lmstudio', 'vllm', 'sglang']

combined = {
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'model': model_path,
    'num_requests': int(os.environ.get('NUM_REQUESTS', 32)),
    'max_tokens': int(os.environ.get('MAX_TOKENS', 64)),
    'concurrency_levels': [int(c) for c in concurrency_levels],
    'backends': {},
    'similarity': {}
}

for backend in backends:
    combined['backends'][backend] = {}
    for c in concurrency_levels:
        stats_file = os.path.join(output_dir, f"stats_{backend}_c{c}.json")
        if os.path.exists(stats_file):
            with open(stats_file) as f:
                combined['backends'][backend][int(c)] = json.load(f)

for c in concurrency_levels:
    sim_file = os.path.join(output_dir, f"similarity_c{c}.json")
    if os.path.exists(sim_file):
        with open(sim_file) as f:
            combined['similarity'][int(c)] = json.load(f)

output_file = os.path.join(output_dir, f"combined_results_{time.strftime('%Y%m%d_%H%M%S')}.json")
with open(output_file, 'w') as f:
    json.dump(combined, f, indent=2)

print(f"Combined results saved to: {output_file}")

# Also generate CSV for easy plotting
csv_file = os.path.join(output_dir, f"scaling_curves_{time.strftime('%Y%m%d_%H%M%S')}.csv")
with open(csv_file, 'w') as f:
    f.write("backend,concurrency,tok_per_sec,avg_latency_ms,p50_latency_ms,p95_latency_ms,p99_latency_ms,gpu_mem_peak_mb,gpu_util_peak_percent,")
    f.write("inferflux_cuda_model_reserved_bytes,inferflux_cuda_model_in_use_bytes,inferflux_cuda_kv_active_bytes,inferflux_cuda_kv_prefix_retained_bytes,")
    f.write("paged_kv_used_bytes,paged_kv_prefix_retained_bytes\n")
    for backend in backends:
        for c in concurrency_levels:
            ci = int(c)
            if ci in combined['backends'].get(backend, {}):
                r = combined['backends'][backend][ci]
                memory = r.get('memory_snapshot', {})
                inferflux_cuda_model = memory.get('inferflux_cuda_model', {})
                inferflux_cuda_kv = memory.get('inferflux_cuda_kv', {})
                paged_kv = memory.get('paged_kv', {})
                f.write(f"{backend},{c},{r['tok_per_sec']},{r['avg_latency_ms']},{r['p50_latency_ms']},")
                f.write(f"{r['p95_latency_ms']},{r['p99_latency_ms']},{r['gpu_mem_peak_mb']},{r['gpu_util_peak_percent']},")
                f.write(f"{inferflux_cuda_model.get('reserved_bytes', 0)},{inferflux_cuda_model.get('in_use_bytes', 0)},")
                f.write(f"{inferflux_cuda_kv.get('active_bytes', 0)},{inferflux_cuda_kv.get('prefix_retained_bytes', 0)},")
                f.write(f"{paged_kv.get('used_bytes', 0)},{paged_kv.get('prefix_retained_bytes', 0)}\n")

print(f"CSV for plotting saved to: {csv_file}")

PYEOF
}

# ============================================================================
# Cleanup
# ============================================================================

cleanup() {
    log "Running cleanup..."
    local mem_before=$(gpu_mem_mb 2>/dev/null || echo "0")
    stop_inferflux_server inferflux_cuda 2>/dev/null || true
    stop_inferflux_server llama_cpp_cuda 2>/dev/null || true
    stop_managed_openai_backend vllm 2>/dev/null || true
    stop_managed_openai_backend sglang 2>/dev/null || true
    stop_stale_benchmark_inferflux_servers
    sleep 2
    local mem_after=$(gpu_mem_mb 2>/dev/null || echo "0")
    local mem_freed=$((mem_before - mem_after))
    if [ $mem_before -gt 0 ] && [ $mem_after -ge 0 ]; then
        log "Cleanup: GPU memory ${mem_before} MB → ${mem_after} MB (freed: ${mem_freed} MB)"
    fi
}

trap cleanup EXIT

# ============================================================================
# Entry Point
# ============================================================================

main "$@"
