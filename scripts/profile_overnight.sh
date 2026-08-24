#!/bin/bash
# Overnight profiling: nsys timeline + ncu kernel analysis
# Run: bash scripts/profile_overnight.sh
set -e

OUTDIR="/home/vsingh/code/inferflux/profiling_results"
MODEL="/home/vsingh/code/inferflux/models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf"
SERVER="./build/inferfluxd"
CONFIG="./config/server.cuda.yaml"
PORT=8080

export INFERFLUX_MODEL_PATH="$MODEL"
export INFERFLUX_CUDA_DISABLE_PARITY_DELEGATE=1
export INFERFLUX_BACKEND_PREFER_INFERFLUX=1
export INFERFLUX_ENABLE_FUSED_GATE_UP_SILU=1
export INFERFLUX_LOG_LEVEL=warn
export INFERFLUX_DISABLE_CUDA_GRAPH=1  # Disable graphs so profiler sees real kernels

send_request() {
  curl -s http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer dev-key-123" \
    -d '{"model":"qwen2.5-3b-instruct-q4_k_m","messages":[{"role":"user","content":"Write a detailed essay about the history of computing from the 1940s to present day. Cover mainframes, minicomputers, personal computers, the internet, mobile computing, and artificial intelligence."}],"max_tokens":512,"temperature":0}' > /dev/null 2>&1
}

wait_for_server() {
  for i in $(seq 1 30); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "Server failed to start"
  return 1
}

kill_server() {
  kill $(pgrep -f inferfluxd) 2>/dev/null || true
  sleep 2
}

echo "========================================="
echo "InferFlux Overnight Profiling"
echo "========================================="
echo "Output directory: $OUTDIR"
echo ""

# =========================================
# Phase 1: nsys timeline profile
# =========================================
echo "[Phase 1/3] nsys timeline profiling..."
kill_server

nsys profile \
  --output="$OUTDIR/inferflux_decode" \
  --force-overwrite=true \
  --trace=cuda,nvtx \
  --capture-range=cudaProfilerApi \
  --cudabacktrace=all \
  --stats=true \
  $SERVER --config $CONFIG &
NSYS_PID=$!
sleep 15
wait_for_server

# Warm up (2 requests)
echo "  Warm-up..."
send_request
send_request

# Profile requests (5 requests for stable data)
echo "  Profiling 5 requests..."
for i in $(seq 1 5); do
  send_request
  echo "    Request $i done"
done

kill_server
wait $NSYS_PID 2>/dev/null || true
echo "  nsys report saved to $OUTDIR/inferflux_decode.nsys-rep"

# Generate stats report
nsys stats "$OUTDIR/inferflux_decode.nsys-rep" --report cuda_gpu_kern_sum \
  --format csv --output "$OUTDIR/nsys_kernel_summary" 2>/dev/null || true
echo "  Kernel summary: $OUTDIR/nsys_kernel_summary.csv"
echo ""

# =========================================
# Phase 2: nsys with NVTX ranges (per-layer)
# =========================================
echo "[Phase 2/3] nsys NVTX-annotated profile..."
kill_server

nsys profile \
  --output="$OUTDIR/inferflux_nvtx" \
  --force-overwrite=true \
  --trace=cuda,nvtx \
  --stats=true \
  $SERVER --config $CONFIG &
NSYS_PID=$!
sleep 15
wait_for_server

send_request  # warm up
echo "  Profiling 3 requests with NVTX..."
for i in $(seq 1 3); do
  send_request
  echo "    Request $i done"
done

kill_server
wait $NSYS_PID 2>/dev/null || true
echo "  NVTX report saved to $OUTDIR/inferflux_nvtx.nsys-rep"

# Generate NVTX report
nsys stats "$OUTDIR/inferflux_nvtx.nsys-rep" --report nvtx_pushpop_sum \
  --format csv --output "$OUTDIR/nsys_nvtx_summary" 2>/dev/null || true
echo ""

# =========================================
# Phase 3: ncu kernel profiler (slow, detailed)
# =========================================
echo "[Phase 3/3] ncu kernel profiling (this takes 10-30 minutes)..."
kill_server

# Start server normally first, warm up, then profile a single request
$SERVER --config $CONFIG &
SERVER_PID=$!
sleep 15
wait_for_server
send_request  # warm up
send_request  # warm up (ensure CUDA graph captured)
kill_server

# Now run with ncu — profile only the MMVQ and attention kernels
ncu \
  --target-processes all \
  --set full \
  --kernel-name "inferflux_mmvq|FlashDecode|fused_rmsnorm|SiluMul|ResidualAdd|EmbeddingLookup|BatchedRoPE|BatchedKvAppend" \
  --launch-skip 200 \
  --launch-count 100 \
  --output "$OUTDIR/inferflux_kernels" \
  --force-overwrite \
  $SERVER --config $CONFIG &
NCU_PID=$!
sleep 30
wait_for_server || { echo "Server under ncu failed to start (this is slow, waiting more...)"; sleep 60; wait_for_server; }

echo "  Sending profiled request..."
send_request
echo "  Request done, waiting for ncu to finish..."

kill_server
wait $NCU_PID 2>/dev/null || true
echo "  ncu report saved to $OUTDIR/inferflux_kernels.ncu-rep"
echo ""

# =========================================
# Summary
# =========================================
echo "========================================="
echo "Profiling complete!"
echo "========================================="
echo ""
echo "Files generated:"
ls -lh "$OUTDIR"/*.{nsys-rep,ncu-rep,csv} 2>/dev/null
echo ""
echo "To analyze:"
echo "  nsys stats $OUTDIR/inferflux_decode.nsys-rep --report cuda_gpu_kern_sum"
echo "  ncu --import $OUTDIR/inferflux_kernels.ncu-rep"
echo "  # Or open .nsys-rep / .ncu-rep in Nsight Systems / Nsight Compute GUI"
