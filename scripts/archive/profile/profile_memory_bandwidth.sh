#!/bin/bash
#
# Profile GEMV kernel memory bandwidth with Nsight Compute
#
# Profiles the native CUDA backend to identify memory bottlenecks
# and guide optimization efforts (vectorized loads, cache alignment).
#
# Usage: ./scripts/profile_memory_bandwidth.sh [--model <path>]

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

log()      { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
log_ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
log_err()  { echo -e "${RED}[ERR]${NC} $1"; }
header()  { echo -e "\n${BOLD}$1${NC}"; echo "$(printf '=%.0s' ${#1})"; }

# Parse arguments
MODEL_PATH="${INFERFLUX_MODEL_PATH:-}"
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    *)
      log_err "Unknown option: $1"
      echo "Usage: $0 [--model <path>]"
      exit 1
      ;;
  esac
done

# Default model (TinyLlama for quick profiling)
if [ -z "$MODEL_PATH" ]; then
  # Try to find TinyLlama in common locations
  for path in \
    "/mnt/c/Users/vjsin/.ollama/models/blobs/sha256-summary" \
    "$HOME/.inferflux/models/tinyllama*.gguf" \
    "./models/tinyllama*.gguf"
  do
    if [ -f "$path" ] || ls $path 2>/dev/null; then
      MODEL_PATH="$path"
      break
    fi
  done
fi

header "Memory Bandwidth Profiling Setup"

# Check prerequisites
if ! command -v ncu &> /dev/null; then
    log_err "ncu (Nsight Compute) not found. Install Nsight Compute toolkit."
    exit 1
fi

if ! command -v nvcc &> /dev/null; then
    log_err "nvcc not found. Install CUDA toolkit."
    exit 1
fi

BUILD_DIR="build"
SERVER_BIN="$BUILD_DIR/inferfluxd"

if [ ! -f "$SERVER_BIN" ]; then
    log_err "Server binary not found: $SERVER_BIN"
    log "Build first with: ./scripts/build.sh"
    exit 1
fi

log_ok "Prerequisites validated"

# Create output directory
OUTPUT_DIR="./memory_profile_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Create temporary config
CONFIG_FILE="$OUTPUT_DIR/profile_config.yaml"
cat > "$CONFIG_FILE" <<EOF
# Memory profiling configuration
server:
  host: "127.0.0.1"
  port: 8765

model:
  path: "$MODEL_PATH"
  backend: "cuda"
  format: "gguf"

runtime:
  backend_priority: ["native_cuda", "cuda_llama_cpp"]

# Enable CUDA-specific logging
logging:
  level: "info"
  format: "text"

# Metrics endpoint for monitoring
metrics:
  enabled: true
  port: 9090
EOF

header "Profiling GEMV Kernels with Nsight Compute"

log "Output directory: $OUTPUT_DIR"
log "Config file: $CONFIG_FILE"

# Step 1: Profile single-sequence decode (baseline)
header "Step 1: Single-Sequence Decode Baseline"

log "Starting server for single-sequence profiling..."
"$SERVER_BIN" --config "$CONFIG_FILE" &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Generate a simple completion request
log "Sending single-sequence request..."
curl -s http://127.0.0.1:8765/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev-key-123" \
  -d '{
    "model": "default",
    "prompt": "Hello",
    "max_tokens": 50
  }' > /dev/null

# Kill server
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

log_ok "Single-sequence profiling complete"

# Step 2: Profile concurrent requests
header "Step 2: Concurrent Request Profiling"

log "Starting server for concurrent profiling..."
"$SERVER_BIN" --config "$CONFIG_FILE" &
SERVER_PID=$!
sleep 5

# Send 4 concurrent requests
for i in {1..4}; do
  curl -s http://127.0.0.1:8765/v1/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer dev-key-123" \
    -d '{
      "model": "default",
      "prompt": "Hello",
      "max_tokens": 50
    }' > /dev/null &
done

wait

# Kill server
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

log_ok "Concurrent profiling complete"

header "Analysis Complete"

# Generate analysis document
cat > "$OUTPUT_DIR/ANALYSIS.md" <<EOF
# Memory Bandwidth Profiling Analysis

**Date**: $(date)
**Profile Directory**: $OUTPUT_DIR

## How to Analyze

### 1. Open in Nsight Compute GUI
\`\`\`bash
ncu-ui $OUTPUT_DIR/(single_seq|concurrent).ncu-rep
\`\`\`

### 2. Key Metrics to Check

**Memory Workload**:
- DRAM Frequency (MHz)
- Memory Bandwidth Utilization (%)
- L2 Cache Hit Rate (%)
- Memory Throughput (GB/s)

**Kernel Analysis**:
- `fused_dequant_gemv_q4k` kernels (main compute)
- Memory access patterns (coalesced vs. scattered)
- Shared memory usage

**Occupancy**:
- Warp Occupancy (%)
- Register usage per thread
- Shared memory per block

### 3. Compare Single vs Concurrent

Look for:
- Does memory bandwidth saturate with concurrent requests?
- Is there contention in L2 cache?
- Do kernels show different efficiency patterns?

## Expected Findings

If memory bandwidth is the bottleneck:
- High DRAM utilization (>80%)
- Low L2 cache hit rate (<50%)
- Memory throughput near theoretical max

If compute is the bottleneck:
- Low DRAM utilization
- High warp occupancy
- Memory throughput below theoretical max

## Next Steps

Based on profiling results:

1. **High memory bandwidth utilization**: Focus on reducing memory traffic
   - Implement vectorized loads (float4/uint4)
   - Better cache utilization
   - Fuse operations to reduce global memory writes

2. **Low memory bandwidth utilization**: Focus on compute optimization
   - Better instruction-level parallelism
   - Reduce warp divergence
   - Optimize shared memory usage

3. **L2 cache miss patterns**: Improve data locality
   - Tile data to fit in L2
   - Reduce cache line splits
   - Better memory access patterns

## Commands to Re-run

\`\`\`bash
# Single-sequence profiling
ncu --set full --target-processes all \\
    --export "$OUTPUT_DIR/single_seq.ncu-rep" \\
    --force-overwrite true \\
    $SERVER_BIN --config "$CONFIG_FILE"

# Concurrent profiling (4 requests)
ncu --set full --target-processes all \\
    --export "$OUTPUT_DIR/concurrent.ncu-rep" \\
    --force-overwrite true \\
    $SERVER_BIN --config "$CONFIG_FILE"
\`\`\`

EOF

log_ok "Analysis saved to: $OUTPUT_DIR/ANALYSIS.md"

# Summary
header "Profiling Complete"

log "Output directory: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Analyze results: ncu-ui $OUTPUT_DIR/*.ncu-rep"
echo "  2. Review analysis: cat $OUTPUT_DIR/ANALYSIS.md"
echo "  3. Implement optimizations based on findings"
echo ""
echo "To re-run profiling:"
echo "  $SCRIPT_DIR/profile_memory_bandwidth.sh"
