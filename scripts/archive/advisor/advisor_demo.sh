#!/bin/bash
# Quick Startup Advisor Demo

set -e

INFERFLUXD="./build/inferfluxd"
LOG_DIR="logs/advisor_tests"
mkdir -p "$LOG_DIR"

cleanup() {
    pkill -f inferfluxd || true
    sleep 1
}
trap cleanup EXIT

echo "=== Startup Advisor Demo ==="
echo

# Test 1: Suboptimal config (FA disabled, small batch)
echo "1. Testing: Suboptimal config (FA disabled, small batch, few KV pages)"
echo "---"

cat > "$LOG_DIR/demo_suboptimal.yaml" << 'EOF'
server:
  host: 0.0.0.0
  http_port: 8080
models:
  - id: test-model
    path: models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf
    format: gguf
    backend: cuda
    default: true
runtime:
  backend_priority: [cuda, cpu]
  cuda:
    enabled: true
    attention:
      kernel: auto
    flash_attention:
      enabled: false
      tile_size: 128
    phase_overlap:
      enabled: false
  scheduler:
    max_batch_size: 4
    max_batch_tokens: 2048
  paged_kv:
    cpu_pages: 16
    eviction: lru
auth:
  api_keys:
    - key: dev-key-123
      scopes: [generate, read, admin]
logging:
  level: info
  format: text
EOF

"$INFERFLUXD" --config "$LOG_DIR/demo_suboptimal.yaml" 2>&1 | tee "$LOG_DIR/demo_suboptimal.log" &
sleep 5
pkill -f inferfluxd || true
sleep 1

echo
echo "Recommendations:"
grep -A 20 "Startup Recommendations" "$LOG_DIR/demo_suboptimal.log" | sed 's/^/  /' || echo "  (None)"
echo

# Test 2: Well-tuned config
echo "2. Testing: Well-tuned config (FA enabled, large batch, many KV pages)"
echo "---"

cat > "$LOG_DIR/demo_well_tuned.yaml" << 'EOF'
server:
  host: 0.0.0.0
  http_port: 8080
models:
  - id: test-model
    path: models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf
    format: gguf
    backend: cuda
    default: true
runtime:
  backend_priority: [cuda, cpu]
  cuda:
    enabled: true
    attention:
      kernel: auto
    flash_attention:
      enabled: true
      tile_size: 128
    phase_overlap:
      enabled: true
  scheduler:
    max_batch_size: 32
    max_batch_tokens: 8192
  paged_kv:
    cpu_pages: 256
    eviction: lru
auth:
  api_keys:
    - key: dev-key-123
      scopes: [generate, read, admin]
logging:
  level: info
  format: text
EOF

"$INFERFLUXD" --config "$LOG_DIR/demo_well_tuned.yaml" 2>&1 | tee "$LOG_DIR/demo_well_tuned.log" &
sleep 5
pkill -f inferfluxd || true
sleep 1

echo
echo "Recommendations:"
grep -A 20 "Startup Recommendations" "$LOG_DIR/demo_well_tuned.log" | sed 's/^/  /' || echo "  (None)"
echo

echo "=== Demo Complete ==="
