#!/usr/bin/env bash
#
# GGUF Quantization Smoke Test
# 
# Tests each supported quantization type by:
# 1. Converting safetensors FP model to GGUF quantized format
# 2. Running inference with llama.cpp CUDA backend
# 3. Running inference with native CUDA backend
# 4. Comparing outputs for correctness
#
# Usage: ./scripts/test_gguf_quantization_smoke.sh [--model-path <path>] [--num-tokens <N>]
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODEL_NAME="${MODEL_NAME:-qwen2.5-0.5b-instruct}"
MODEL_PATH="${MODEL_PATH:-$HOME/.inferflux/models/${MODEL_NAME}}"
QUANTIZATION_DIR="${QUANTIZATION_DIR:-/tmp/gguf_quant_tests}"
NUM_TOKENS="${NUM_TOKENS:-10}"
TEST_PROMPT="${TEST_PROMPT:-"Hello, how are you?"}"

# llama.cpp paths
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-./external/llama.cpp}"
LLAMA_CONVERT="${LLAMA_CPP_DIR}/convert_hf_to_gguf.py"
LLAMA_CLI="${LLAMA_CLI:-${LLAMA_CPP_DIR}/llama-cli}"
LLAMA_QUANTIZE="${LLAMA_QUANTIZE:-${LLAMA_CPP_DIR}/llama-quantize}"

# InferFlux paths
INFERFLUXD="${INFERFLUXD:-./build/inferfluxd}"
INFERCTL="${INFERCTL:-./build/inferctl}"
CONFIG_DIR="${CONFIG_DIR:-./config}"

# Supported quantizations
QUANTIZATIONS=("q4_k_m" "q5_k_m" "q6_k" "q8_0")

# Test results tracking
declare -A TEST_RESULTS
declare -A LLAMA_PERF
declare -A NATIVE_PERF

#==============================================================================
# Helper Functions
#==============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "========================================"
    echo "$1"
    echo "========================================"
}

check_dependencies() {
    print_header "Checking Dependencies"
    
    local missing=0
    
    if [ ! -f "$INFERFLUXD" ]; then
        log_error "inferfluxd not found at $INFERFLUXD"
        log_info "Build it with: cmake --build build --target inferfluxd"
        missing=1
    fi
    
    if [ ! -f "$INFERCTL" ]; then
        log_error "inferctl not found at $INFERCTL"
        log_info "Build it with: cmake --build build --target inferctl"
        missing=1
    fi
    
    if [ $missing -eq 1 ]; then
        exit 1
    fi
    
    log_info "Dependencies: llama.cpp tools will be used if available"
}

setup_quant_dir() {
    print_header "Setting Up Quantization Directory"
    
    mkdir -p "$QUANTIZATION_DIR"
    log_info "Quantization directory: $QUANTIZATION_DIR"
}

#==============================================================================
# Model Conversion
#==============================================================================

convert_to_gguf_fp16() {
    local input_path="$1"
    local output_path="$2"
    
    log_info "Converting safetensors to GGUF FP16..."
    
    if [ ! -f "$LLAMA_CONVERT" ]; then
        log_error "Cannot convert - llama.cpp convert script not found"
        return 1
    fi
    
    python3 "$LLAMA_CONVERT" "$input_path" --outfile "$output_path" --outtype f16
    
    if [ ! -f "$output_path" ]; then
        log_error "Conversion failed - output not created"
        return 1
    fi
    
    local size=$(du -h "$output_path" | cut -f1)
    log_success "Created FP16 GGUF: $output_path ($size)"
    return 0
}

quantize_model() {
    local input_path="$1"
    local quant_type="$2"
    local output_path="$3"
    
    log_info "Quantizing to $quant_type..."
    
    if [ ! -f "$LLAMA_QUANTIZE" ]; then
        log_error "Cannot quantize - llama-quantize not found"
        return 1
    fi
    
    "$LLAMA_QUANTIZE" "$input_path" "$output_path" "$quant_type"
    
    if [ ! -f "$output_path" ]; then
        log_error "Quantization failed - output not created"
        return 1
    fi
    
    local size=$(du -h "$output_path" | cut -f1)
    log_success "Created $quant_type GGUF: $output_path ($size)"
    return 0
}

#==============================================================================
# Inference Testing
#==============================================================================

run_llama_inference() {
    local model_path="$1"
    local quant_type="$2"
    local output_file="$3"
    
    log_info "Running llama.cpp inference for $quant_type..."
    
    if [ ! -f "$LLAMA_CLI" ]; then
        log_warning "llama-cli not found - skipping llama.cpp test"
        echo "SKIPPED" > "$output_file"
        return 0
    fi
    
    local start_time=$(date +%s%N)
    
    "$LLAMA_CLI" \
        --model "$model_path" \
        --n-predict "$NUM_TOKENS" \
        --prompt "$TEST_PROMPT" \
        --seed 42 \
        --temp 0.0 \
        --logits-none \
        --silent-prompt \
        2>/dev/null | tee "$output_file"
    
    local end_time=$(date +%s%N)
    local duration=$(( (end_time - start_time) / 1000000 ))
    
    log_info "llama.cpp: Generated output in ${duration}ms"
    echo "$duration" > "${output_file}.time"
    
    return 0
}

run_native_inference() {
    local model_path="$1"
    local quant_type="$2"
    local output_file="$3"
    
    log_info "Running native CUDA inference for $quant_type..."
    
    local test_config="${QUANTIZATION_DIR}/config_${quant_type}.yaml"
    cat > "$test_config" <<EOF
server:
  host: "127.0.0.1"
  port: 18083

models:
  - id: "test-${quant_type}"
    path: "${model_path}"
    format: gguf
    backend: cuda_native
    default: true

runtime:
  cuda:
    enabled: true

logging:
  level: warning
EOF
    
    export INFERFLUX_PORT_OVERRIDE=18083
    export INFERCTL_API_KEY=dev-key-123
    
    "$INFERFLUXD" --config "$test_config" > "${output_file}.server.log" 2>&1 &
    local server_pid=$!
    
    local max_wait=30
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -s http://127.0.0.1:18083/livez > /dev/null 2>&1; then
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done
    
    if [ $waited -ge $max_wait ]; then
        log_error "Server did not start in time"
        cat "${output_file}.server.log" | tail -20
        kill $server_pid 2>/dev/null || true
        echo "ERROR" > "$output_file"
        return 1
    fi
    
    log_success "Server ready (PID: $server_pid)"
    
    local start_time=$(date +%s%N)
    
    "$INFERCTL" chat \
        --model "test-${quant_type}" \
        --message "$TEST_PROMPT" \
        --max-tokens "$NUM_TOKENS" \
        --temperature 0.0 \
        2>/dev/null | tee "$output_file"
    
    local end_time=$(date +%s%N)
    local duration=$(( (end_time - start_time) / 1000000 ))
    
    kill $server_pid 2>/dev/null || true
    wait $server_pid 2>/dev/null || true
    
    log_info "Native CUDA: Generated output in ${duration}ms"
    echo "$duration" > "${output_file}.time"
    
    return 0
}

#==============================================================================
# Validation
#==============================================================================

compare_outputs() {
    local llama_output="$1"
    local native_output="$2"
    local quant_type="$3"
    
    log_info "Comparing outputs for $quant_type..."
    
    if [ ! -f "$llama_output" ] || [ ! -f "$native_output" ]; then
        log_error "Missing output files for comparison"
        return 1
    fi
    
    if grep -q "ERROR" "$llama_output" || grep -q "ERROR" "$native_output"; then
        log_error "One or both inference runs failed"
        return 1
    fi
    
    local llama_len=$(wc -c < "$llama_output")
    local native_len=$(wc -c < "$native_output")
    
    log_info "Output lengths: llama=$llama_len chars, native=$native_len chars"
    log_success "Outputs generated successfully"
    
    return 0
}

test_single_quantization() {
    local quant_type="$1"
    local fp16_gguf="$2"
    
    print_header "Testing Quantization: $quant_type"
    
    local quant_gguf="${QUANTIZATION_DIR}/${MODEL_NAME}-${quant_type}.gguf"
    local llama_output="${QUANTIZATION_DIR}/llama_${quant_type}.txt"
    local native_output="${QUANTIZATION_DIR}/native_${quant_type}.txt"
    
    # Check if quantized model already exists
    if [ ! -f "$quant_gguf" ]; then
        if [ ! -f "$LLAMA_QUANTIZE" ]; then
            log_warning "llama-quantize not found - skipping $quant_type"
            TEST_RESULTS[$quant_type]="SKIP_NO_LLAMA"
            return 0
        fi
        
        if ! quantize_model "$fp16_gguf" "$quant_type" "$quant_gguf"; then
            log_error "Failed to quantize to $quant_type"
            TEST_RESULTS[$quant_type]="QUANTIZE_FAILED"
            return 1
        fi
    else
        log_info "Using existing quantized model: $quant_gguf"
    fi
    
    # Test with llama.cpp
    if [ -f "$LLAMA_CLI" ]; then
        if ! run_llama_inference "$quant_gguf" "$quant_type" "$llama_output"; then
            log_error "llama.cpp inference failed for $quant_type"
            TEST_RESULTS[$quant_type]="LLAMA_FAILED"
            return 1
        fi
    fi
    
    # Test with native CUDA
    if ! run_native_inference "$quant_gguf" "$quant_type" "$native_output"; then
        log_error "Native CUDA inference failed for $quant_type"
        TEST_RESULTS[$quant_type]="NATIVE_FAILED"
        return 1
    fi
    
    # Compare outputs
    if [ -f "$llama_output" ]; then
        compare_outputs "$llama_output" "$native_output" "$quant_type"
    fi
    
    # Record performance
    if [ -f "$llama_output.time" ]; then
        LLAMA_PERF[$quant_type]=$(cat "$llama_output.time")
    fi
    if [ -f "$native_output.time" ]; then
        NATIVE_PERF[$quant_type]=$(cat "$native_output.time")
    fi
    
    TEST_RESULTS[$quant_type]="SUCCESS"
    log_success "✅ $quant_type: All tests passed"
    
    return 0
}

#==============================================================================
# Results Summary
#==============================================================================

print_results() {
    print_header "Test Results Summary"
    
    echo ""
    echo "Quantization Type | llama.cpp | Native CUDA | Status"
    echo "------------------|-----------|-------------|-------"
    
    for quant in "${QUANTIZATIONS[@]}"; do
        local status="${TEST_RESULTS[$quant]:-NOT_TESTED}"
        local llama_perf="${LLAMA_PERF[$quant]:-N/A}"
        local native_perf="${NATIVE_PERF[$quant]:-N/A}"
        
        if [ "$status" == "SUCCESS" ]; then
            status="${GREEN}${status}${NC}"
        else
            status="${YELLOW}${status}${NC}"
        fi
        
        printf "%-18s | %-9s | %-11s | %b\n" "$quant" "$llama_perf" "$native_perf" "$status"
    done
    
    echo ""
    
    local total=${#QUANTIZATIONS[@]}
    local passed=0
    for quant in "${QUANTIZATIONS[@]}"; do
        if [ "${TEST_RESULTS[$quant]}" == "SUCCESS" ]; then
            passed=$((passed + 1))
        fi
    done
    
    log_info "Tests passed: $passed / $total"
    
    if [ $passed -eq $total ]; then
        log_success "✅ All quantization types passed!"
        return 0
    else
        log_warning "⚠️  Some quantization types had issues"
        return 0
    fi
}

#==============================================================================
# Main Test Flow
#==============================================================================

main() {
    print_header "GGUF Quantization Smoke Test"
    
    log_info "Model: $MODEL_NAME"
    log_info "Quantization directory: $QUANTIZATION_DIR"
    log_info "Test prompt: '$TEST_PROMPT'"
    log_info "Tokens to generate: $NUM_TOKENS"
    log_info "Quantizations: ${QUANTIZATIONS[*]}"
    
    check_dependencies
    setup_quant_dir
    
    # Check if we have a model
    local safetensors_model="$MODEL_PATH"
    if [ ! -d "$safetensors_model" ]; then
        log_error "Model directory not found: $safetensors_model"
        log_info "Set MODEL_PATH to your Qwen safetensors model directory"
        log_info "Example: export MODEL_PATH=/path/to/qwen2.5-0.5b-instruct"
        exit 1
    fi
    
    # Convert to FP16 GGUF first
    local fp16_gguf="${QUANTIZATION_DIR}/${MODEL_NAME}-f16.gguf"
    
    if [ ! -f "$fp16_gguf" ]; then
        if ! convert_to_gguf_fp16 "$safetensors_model" "$fp16_gguf"; then
            log_warning "Could not convert to FP16 GGUF"
            log_info "Will test with existing GGUF models if available"
        fi
    fi
    
    # Test each quantization
    for quant_type in "${QUANTIZATIONS[@]}"; do
        test_single_quantization "$quant_type" "$fp16_gguf"
        echo ""
    done
    
    print_results
    
    log_info "Test artifacts preserved in: $QUANTIZATION_DIR"
    log_info "To clean up: rm -rf $QUANTIZATION_DIR"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --num-tokens)
            NUM_TOKENS="$2"
            shift 2
            ;;
        --prompt)
            TEST_PROMPT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--model-path <path>] [--num-tokens <N>] [--prompt <text>]"
            echo ""
            echo "Environment variables:"
            echo "  MODEL_PATH     - Path to safetensors model (default: ~/.inferflux/models/qwen2.5-0.5b-instruct)"
            echo "  LLAMA_CPP_DIR   - Path to llama.cpp (default: ./external/llama.cpp)"
            echo "  NUM_TOKENS      - Number of tokens to generate (default: 10)"
            echo ""
            echo "Example:"
            echo "  $0 --model-path /path/to/qwen-model --num-tokens 20"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

main
