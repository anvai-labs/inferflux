#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR=${BUILD_DIR:-build}

# Parse command-line flags
ALL_BACKENDS=false
CPU_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --all-backends)  ALL_BACKENDS=true ;;
    --cpu-only)      CPU_ONLY=true ;;
    *)               echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

GENERATOR_ARGS=()
if command -v ninja >/dev/null 2>&1; then
  GENERATOR_ARGS=(-G Ninja)
fi

CMAKE_ARGS=(
  -S .
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE=Release
  -DINFERFLUX_CUDA_DEVICE_LTO=${INFERFLUX_CUDA_DEVICE_LTO:-OFF}
  -DINFERFLUX_CUDA_USE_FAST_MATH=${INFERFLUX_CUDA_USE_FAST_MATH:-OFF}
)

if $CPU_ONLY; then
  CMAKE_ARGS+=(
    -DENABLE_CUDA=OFF
    -DENABLE_ROCM=OFF
    -DENABLE_MPS=OFF
    -DENABLE_VULKAN=OFF
  )
elif $ALL_BACKENDS; then
  CMAKE_ARGS+=(
    -DENABLE_CUDA=ON
    -DENABLE_ROCM=ON
    -DENABLE_MPS=ON
    -DENABLE_VULKAN=ON
  )
fi

# Prefer GCC/G++ as the CUDA host compiler on Linux for maximum compatibility.
# GCC 14+ on Ubuntu 24.04 moved cc1plus to /usr/libexec/ which nvcc cannot find.
# Auto-detect a compatible host compiler.
if [[ -z "${CMAKE_CUDA_HOST_COMPILER:-}" ]]; then
  _cuda_host_cxx=""
  if [[ -x /usr/bin/g++ ]]; then
    _cc1plus=$(/usr/bin/g++ -print-file-name=cc1plus 2>/dev/null || true)
    if [[ -f "$_cc1plus" && "$_cc1plus" == /usr/lib/* ]]; then
      _cuda_host_cxx=/usr/bin/g++
    else
      # System g++ cc1plus not in nvcc search path — find compatible version
      for _gxx in g++-13 g++-12 g++-11; do
        if command -v "$_gxx" >/dev/null 2>&1; then
          _cc1plus=$("$_gxx" -print-file-name=cc1plus 2>/dev/null || true)
          if [[ -f "$_cc1plus" && "$_cc1plus" == /usr/lib/* ]]; then
            _cuda_host_cxx=$(command -v "$_gxx")
            echo "Note: Using $_gxx as CUDA host compiler (system g++ cc1plus not in nvcc search path)"
            break
          fi
        fi
      done
      if [[ -z "$_cuda_host_cxx" ]]; then
        echo "Warning: No compatible g++ found for nvcc. Install g++-12: sudo apt install g++-12"
        _cuda_host_cxx=/usr/bin/g++  # Try anyway, CMake will also attempt detection
      fi
    fi
  fi
  if [[ -n "$_cuda_host_cxx" ]]; then
    CMAKE_ARGS+=(-DCMAKE_CUDA_HOST_COMPILER="$_cuda_host_cxx")
  fi
else
  CMAKE_ARGS+=(-DCMAKE_CUDA_HOST_COMPILER="${CMAKE_CUDA_HOST_COMPILER}")
fi

# Optional per-node CUDA architecture override (example: 89 for Ada RTX 4000).
if [[ -n "${INFERFLUX_CUDA_ARCHS:-}" ]]; then
  CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="${INFERFLUX_CUDA_ARCHS}")
fi

cmake "${GENERATOR_ARGS[@]}" "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
