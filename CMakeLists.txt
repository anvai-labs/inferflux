cmake_minimum_required(VERSION 3.22)
project(InferFlux VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Prevent Windows min/max macros from breaking std::min/std::max.
if(WIN32)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Load dependencies from cmake/Dependencies.cmake
include(cmake/Dependencies.cmake)

option(ENABLE_CUDA "Enable CUDA runtime" ON)
option(ENABLE_ROCM "Enable ROCm runtime" ON)
option(ENABLE_MPS "Enable Metal/MPS runtime" ON)
option(ENABLE_BLAS "Enable BLAS acceleration" ON)
option(ENABLE_VULKAN "Enable Vulkan GPU backend via llama.cpp" ON)
option(ENABLE_WEBUI "Enable embedded WebUI (litehtml-based) support" OFF)
option(ENABLE_MLX "Enable experimental MLX backend" OFF)
# MLX_C_TAG: mlx-c release to fetch when MLX_C_ROOT is not provided.
# Bump this together with the `pip install mlx==X.Y.Z` version in the README.
set(MLX_C_TAG "v0.5.0" CACHE STRING "mlx-c git tag to use when building from source")

option(ENABLE_CPU "Enable CPU runtime" ON)
option(ENABLE_MTMD "Enable multimodal (vision) support via libmtmd" OFF)
option(INFERFLUX_CUDA_DEVICE_LTO "Enable CUDA device LTO profile (requires custom lto gencode)" OFF)
option(INFERFLUX_CUDA_USE_FAST_MATH "Enable --use_fast_math for CUDA kernels (default ON, matches llama.cpp)" ON)
option(INFERFLUX_CUDA_NATIVE_ARCH "Compile CUDA kernels for local GPU only (faster builds, better codegen)" OFF)

# BackendInterface is the canonical base for all backends.
add_compile_definitions(INFERFLUX_USE_COMMON_BACKEND_TYPES)

add_library(inferflux_core STATIC
  runtime/device_context.h
  runtime/core/gguf/igguf_parser.h
  runtime/core/gguf/cpu_gguf_parser.cpp
  runtime/core/gguf/cpu_gguf_parser.h
  runtime/core/gguf/igguf_executor.h
  runtime/core/gguf/cpu_executor.cpp
  runtime/core/gguf/cpu_executor.h
  runtime/kv_cache/paged_kv_cache.cpp
  runtime/kv_cache/paged_kv_cache.h
  runtime/tensors/tensor.h
  runtime/backends/cpu/cpu_backend.cpp
  runtime/backends/cpu/cpu_backend.h
  runtime/backends/cpu/llama_cpp_backend.h
  runtime/backends/llama/llama_cpp_backend.cpp
  runtime/backends/llama/llama_cpp_backend.h
  runtime/backends/native/native_inference_runtime.h
  model/tokenizer_factory.cpp
  model/tokenizer_factory.h
  runtime/backends/backend_capabilities.cpp
  runtime/backends/backend_capabilities.h
  runtime/backends/common/backend_types.h
  runtime/backends/common/backend_interface.h
  runtime/backends/common/batching_utils.h
  runtime/backends/gpu/gpu_device_strategy.h
  runtime/backends/gpu/gpu_accelerated_backend.cpp
  runtime/backends/gpu/gpu_accelerated_backend.h
  runtime/backends/gpu/backend_config_extensions.h
  runtime/backends/gpu/cuda_device_strategy.cpp
  runtime/backends/gpu/cuda_device_strategy.h
  runtime/backends/gpu/native_cuda_device_strategy.cpp
  runtime/backends/gpu/native_cuda_device_strategy.h
  runtime/backends/gpu/rocm_device_strategy.cpp
  runtime/backends/gpu/rocm_device_strategy.h
  runtime/backends/gpu/mps_device_strategy.cpp
  runtime/backends/gpu/mps_device_strategy.h
  runtime/backends/gpu/vulkan_device_strategy.cpp
  runtime/backends/gpu/vulkan_device_strategy.h
  runtime/backends/gpu/cpu_device_strategy.cpp
  runtime/backends/gpu/cpu_device_strategy.h
  runtime/backends/gpu/opencl_device_strategy.cpp
  runtime/backends/gpu/opencl_device_strategy.h
  runtime/backends/vulkan/vulkan_backend.cpp
  runtime/backends/vulkan/vulkan_backend.h
  runtime/backends/mps/mps_backend.cpp
  runtime/backends/mps/mps_backend.h
  runtime/backends/opencl/opencl_backend.cpp
  runtime/backends/opencl/opencl_backend.h
  runtime/backends/llama/llama_backend_traits.cpp
  runtime/backends/llama/llama_backend_traits.h
  runtime/backends/backend_registry.cpp
  runtime/backends/backend_registry.h
  runtime/backends/backend_factory.cpp
  runtime/backends/backend_factory.h
  runtime/backends/backend_manager.cpp
  runtime/backends/backend_manager.h
  runtime/speculative/speculative_decoder.cpp
  runtime/speculative/speculative_decoder.h
  runtime/execution/batch_executor.cpp
  runtime/execution/batch_executor.h
  runtime/execution/unified_batch_lane_dispatcher.cpp
  runtime/execution/unified_batch_lane_dispatcher.h
  runtime/execution/parallel_context.cpp
  runtime/execution/parallel_context.h
  runtime/backends/ep_dispatch.cpp
  runtime/backends/ep_dispatch.h
  runtime/prefix_cache/prefix_cache.cpp
  runtime/prefix_cache/prefix_cache.h
  runtime/prefix_cache/radix_prefix_cache.cpp
  runtime/prefix_cache/radix_prefix_cache.h
  model/tokenizer/simple_tokenizer.cpp
  model/tokenizer/simple_tokenizer.h
  model/tokenizer.h
  model/llama_tokenizer.cpp
  model/llama_tokenizer.h
  model/hf_tokenizer.cpp
  model/hf_tokenizer.h
  runtime/backends/mlx/mlx_tokenizer.cpp
  runtime/backends/mlx/mlx_tokenizer.h
  model/model_format.cpp
  model/model_format.h
  runtime/scheduler/sequence_slot_manager.cpp
  runtime/scheduler/sequence_slot_manager.h
  scheduler/batch_selection_policy.cpp
  scheduler/scheduler.cpp
  scheduler/session_handle_manager.cpp
  scheduler/session_handle_manager.h
  scheduler/fairness_controller.cpp
  scheduler/scheduler.h
  scheduler/model_selection.cpp
  scheduler/model_selection.h
  scheduler/single_model_router.cpp
  scheduler/single_model_router.h
  scheduler/model_registry.cpp
  scheduler/model_registry.h
  server/http/http_server.cpp
  server/http/http_server.h
  server/http/model_json.cpp
  server/http/model_json.h
  server/http/completion_payload.h
  server/http/http_utils.h
  server/auth/api_key_auth.cpp
  server/auth/api_key_auth.h
  server/metrics/metrics.cpp
  server/metrics/metrics.h
  server/auth/rate_limiter.cpp
  server/auth/rate_limiter.h
  server/auth/oidc_validator.cpp
  server/auth/oidc_validator.h
  server/diagnostics/crash_handler.cpp
  server/diagnostics/crash_handler.h
  server/logging/audit_logger.cpp
  server/logging/audit_logger.h
  server/logging/logger.cpp
  server/logging/logger.h
  server/quantization_detection.h
  server/cpu_quantization_detector.cpp
  server/cpu_quantization_detector.h
  server/policy/guardrail.cpp
  server/policy/guardrail.h
  server/startup_advisor.cpp
  server/startup_advisor.h
  policy/policy_store.cpp
  policy/policy_store.h
  policy/opa_client.cpp
  policy/opa_client.h
  runtime/structured_output/structured_constraint.h
  runtime/structured_output/structured_output_adapter.cpp
  runtime/structured_output/structured_output_adapter.h
  runtime/structured_output/json_schema_shim.cpp
  runtime/disaggregated/kv_channel.cpp
  runtime/disaggregated/kv_channel.h
  runtime/disaggregated/shm_kv_transport.cpp
  runtime/disaggregated/shm_kv_transport.h
  runtime/multimodal/image_preprocessor.cpp
  runtime/multimodal/image_preprocessor.h
  external/llama.cpp/common/json-schema-to-grammar.cpp
  io/async_file_writer.cpp
  io/async_file_writer.h
  net/http_client.cpp
  net/http_client.h
)

target_include_directories(inferflux_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/external
)

find_package(OpenSSL REQUIRED)
find_package(CUDAToolkit QUIET)

if(ENABLE_CUDA AND CUDAToolkit_FOUND)
  message(STATUS "CUDA toolkit found: enabling CUDA build path")

  # GCC 14+ on Ubuntu 24.04 moved cc1plus to /usr/libexec/ which nvcc cannot
  # find.  Auto-detect a compatible host compiler if the default g++ fails.
  if(NOT CMAKE_CUDA_HOST_COMPILER)
    execute_process(
      COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=cc1plus
      OUTPUT_VARIABLE _cc1plus_path OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_cc1plus_path STREQUAL "cc1plus" OR NOT EXISTS "${_cc1plus_path}"
       OR NOT _cc1plus_path MATCHES "/usr/lib/")
      # cc1plus is not in the standard path nvcc expects — find an older g++
      foreach(_gxx g++-13 g++-12 g++-11)
        find_program(_compat_gxx ${_gxx})
        if(_compat_gxx)
          execute_process(
            COMMAND ${_compat_gxx} -print-file-name=cc1plus
            OUTPUT_VARIABLE _compat_cc1plus OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
          if(EXISTS "${_compat_cc1plus}" AND _compat_cc1plus MATCHES "/usr/lib/")
            set(CMAKE_CUDA_HOST_COMPILER "${_compat_gxx}" CACHE FILEPATH
                "CUDA host compiler (auto-detected for nvcc compatibility)" FORCE)
            message(STATUS "CUDA host compiler auto-set to ${_compat_gxx} (system g++ cc1plus not in nvcc search path)")
            break()
          endif()
        endif()
        unset(_compat_gxx CACHE)
      endforeach()
      if(NOT CMAKE_CUDA_HOST_COMPILER)
        message(WARNING "System g++ cc1plus not found in nvcc search path and no compatible g++-{13,12,11} found. "
                        "CUDA build may fail. Install g++-12: sudo apt install g++-12")
      endif()
    endif()
  endif()

  if(INFERFLUX_CUDA_NATIVE_ARCH)
    # Keep the cache aligned with the requested mode so reconfigured dev builds
    # do not retain an old numeric architecture list and still pass -arch=native
    # through to nvcc, which falls back to a generic target on some hosts.
    set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures" FORCE)
    message(STATUS "CUDA native arch: probing local GPU architecture")
  elseif(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90" CACHE STRING "CUDA architectures" FORCE)
  endif()
  enable_language(CUDA)
  if(CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
    if(DEFINED CMAKE_CUDA_ARCHITECTURES_NATIVE AND
       CMAKE_CUDA_ARCHITECTURES_NATIVE MATCHES
         "^[0-9]+(a|f)?(-real|-virtual)?(;[0-9]+(a|f)?(-real|-virtual)?)*$")
      set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES_NATIVE}" CACHE STRING
          "CUDA architectures" FORCE)
      message(STATUS "CUDA native arch resolved to ${CMAKE_CUDA_ARCHITECTURES}")
    else()
      set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90" CACHE STRING
          "CUDA architectures" FORCE)
      message(WARNING
        "Failed to resolve native CUDA architecture; falling back to standard multi-arch build")
    endif()
  endif()
  set(LLAMA_CUDA ON CACHE BOOL "" FORCE)
  target_compile_definitions(inferflux_core PUBLIC INFERFLUX_HAS_CUDA=1)
  target_link_libraries(inferflux_core PUBLIC CUDA::cudart)

  # GGUF/quantization components that require CUDA headers
  target_sources(inferflux_core PRIVATE
    runtime/backends/cuda/native/model_loader.cpp
    runtime/backends/cuda/native/model_memory_ledger.cpp
    runtime/backends/cuda/native/quantization_handler.cpp
    runtime/backends/cuda/native/gguf_util.cpp
    runtime/backends/cuda/native/strategy_registry.cpp
    runtime/backends/cuda/native/quantized_weight_map.cpp
  )

  # Native CUDA inference kernels (no llama.cpp dependency)
  add_library(inferflux_cuda_kernels STATIC
    runtime/backends/cuda/native/cuda_kernels.cu
    runtime/backends/cuda/native/kv_cache_gpu.cu
    runtime/backends/cuda/native/hybrid_kv_cache_gpu.cu
    runtime/backends/cuda/native/gpu_sampler.cu
    runtime/backends/cuda/kernels/flash_attention.cu
    # GGUF dequantization kernels
    runtime/backends/cuda/native/kernels/dequantization.cu
  )
  set_target_properties(inferflux_cuda_kernels PROPERTIES
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  )
  target_include_directories(inferflux_cuda_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/external
  )
  # NVTX3 is header-only in CUDA 12+; no library linkage needed.
  target_link_libraries(inferflux_cuda_kernels PUBLIC CUDA::cudart CUDA::cublas CUDA::curand)
  target_include_directories(inferflux_cuda_kernels PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

  # Add native CUDA C++ sources to inferflux_core
  target_sources(inferflux_core PRIVATE
    runtime/backends/cuda/cuda_device_context.cpp
    runtime/backends/cuda/cuda_device_context.h
    runtime/backends/cuda/cuda_backend.cpp
    runtime/backends/cuda/cuda_backend.h
    runtime/backends/cuda/inferflux_cuda_runtime.cpp
    runtime/backends/cuda/inferflux_cuda_runtime.h
    runtime/backends/cuda/inferflux_cuda_backend.cpp
    runtime/backends/cuda/inferflux_cuda_backend.h
    runtime/backends/native/native_gpu_backend.cpp
    runtime/backends/native/native_gpu_backend.h
    runtime/backends/cuda/native/safetensors_parser.cpp
    runtime/backends/cuda/native/safetensors_parser.h
    runtime/backends/cuda/inferflux_cuda_executor.cpp
    runtime/backends/cuda/inferflux_cuda_executor.h
    runtime/backends/cuda/native/native_bootstrap_config.cpp
    runtime/backends/cuda/native/cublas_gemm.cpp
    runtime/backends/cuda/native/native_dispatch_registry.cpp
    runtime/backends/cuda/native/native_dispatch_policy.cpp
    runtime/backends/cuda/native/weight_map.cpp
    runtime/backends/cuda/native/model_forward_factory.cpp
    runtime/backends/cuda/native/transformer_forward.cu
    runtime/backends/cuda/native/fused_quant_gemm.cu
    # GGUF quantization support (Phase 1-5: Foundation, GGUF Parsing, Handlers, CUDA Kernels, Forward Pass)
    runtime/backends/cuda/native/safetensors_adapter.cpp
    runtime/backends/cuda/native/gguf_model_loader.cpp
    runtime/backends/cuda/native/q4_k_m_handler.cpp
    runtime/backends/cuda/native/q5_k_m_handler.cpp
    runtime/backends/cuda/native/q6_k_handler.cpp
    runtime/backends/cuda/native/q8_0_handler.cpp
    runtime/backends/cuda/native/q8_k_handler.cpp
    model/gguf_tokenizer.cpp
  )
  target_link_libraries(inferflux_core PUBLIC inferflux_cuda_kernels CUDA::cublas)
  set_target_properties(inferflux_core PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  if(INFERFLUX_CUDA_DEVICE_LTO)
    message(WARNING
      "INFERFLUX_CUDA_DEVICE_LTO is requested, but automatic -dlto wiring is disabled "
      "when CMAKE_CUDA_ARCHITECTURES uses standard gencode targets. "
      "Use explicit code=lto_<arch> gencode flags for device-LTO builds.")
  endif()
  if(INFERFLUX_CUDA_USE_FAST_MATH)
    target_compile_options(inferflux_cuda_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
    target_compile_options(inferflux_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
    message(STATUS "CUDA fast math enabled (--use_fast_math)")
  endif()
  target_compile_definitions(inferflux_core PUBLIC INFERFLUX_NATIVE_KERNELS_READY=1)
  message(STATUS "Native CUDA kernels enabled (cuBLAS + custom kernels)")

else()
  if(ENABLE_CUDA)
    message(WARNING "ENABLE_CUDA=ON but CUDA toolkit not found; building without GPU support")
  endif()
  # NOTE: GGUF/safetensors stubs require CUDA headers even in stub form.
  # Disabling CPU-only GGUF support until proper CPU GGUF loader is implemented.
  # GGUF tests require CUDA for now.
  message(STATUS "CUDA disabled - GGUF native loader unavailable (GGUF tests require CUDA)")
  set(LLAMA_CUDA OFF CACHE BOOL "" FORCE)
endif()

if(ENABLE_MPS AND CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message(STATUS "Metal/MPS enabled")
  set(LLAMA_METAL ON CACHE BOOL "" FORCE)
  target_compile_definitions(inferflux_core PUBLIC INFERFLUX_HAS_METAL=1)
else()
  set(LLAMA_METAL OFF CACHE BOOL "" FORCE)
endif()

if(ENABLE_BLAS)
  find_package(BLAS)
  if(BLAS_FOUND)
    message(STATUS "BLAS backend found: ${BLAS_LIBRARIES}")
    set(LLAMA_BLAS ON CACHE BOOL "" FORCE)
    target_link_libraries(inferflux_core PUBLIC BLAS::BLAS)
    target_compile_definitions(inferflux_core PUBLIC INFERFLUX_HAS_BLAS=1)
  else()
    message(WARNING "ENABLE_BLAS=ON but no BLAS implementation found; building without BLAS acceleration")
    set(LLAMA_BLAS OFF CACHE BOOL "" FORCE)
  endif()
else()
  set(LLAMA_BLAS OFF CACHE BOOL "" FORCE)
endif()

if(ENABLE_VULKAN)
  find_package(Vulkan QUIET)
  if(Vulkan_FOUND)
    message(STATUS "Vulkan SDK found: enabling Vulkan GPU backend")
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
    target_compile_definitions(inferflux_core PUBLIC INFERFLUX_HAS_VULKAN=1)
    target_link_libraries(inferflux_core PUBLIC Vulkan::Vulkan)
  else()
    message(STATUS "ENABLE_VULKAN=ON but Vulkan SDK not found; building without Vulkan")
    set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
  endif()
else()
  set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
endif()

# Multi-backend: CUDA and ROCm can be enabled simultaneously.  llama.cpp builds
# ggml-cuda and ggml-hip as independent backends; InferFlux native kernels use
# CUDA only.  Use --all-backends in scripts/build.sh or set both options here.
if(ENABLE_ROCM)
  find_package(hip QUIET CONFIG PATHS /opt/rocm)
  find_package(rocblas QUIET CONFIG PATHS /opt/rocm)
  find_package(hipblas QUIET CONFIG PATHS /opt/rocm)
  if(hip_FOUND AND rocblas_FOUND AND hipblas_FOUND)
    message(STATUS "ROCm/HIP found: enabling ROCm build path")
    set(GGML_HIP ON CACHE BOOL "" FORCE)
    target_compile_definitions(inferflux_core PUBLIC INFERFLUX_HAS_ROCM=1)
    target_link_libraries(inferflux_core PUBLIC hip::host roc::rocblas roc::hipblas)

    # Add ROCm backend sources
    target_sources(inferflux_core PRIVATE
      runtime/backends/rocm/rocm_backend.cpp
      runtime/backends/rocm/rocm_backend.h
    )
  else()
    if(ENABLE_ROCM)
      message(WARNING "ENABLE_ROCM=ON but ROCm toolkit not found; building without ROCm support")
    endif()
    set(GGML_HIP OFF CACHE BOOL "" FORCE)
  endif()
else()
  set(GGML_HIP OFF CACHE BOOL "" FORCE)
endif()

if(ENABLE_CUDA AND CUDAToolkit_FOUND AND ENABLE_ROCM AND GGML_HIP)
  message(STATUS "Multi-backend build: CUDA + ROCm enabled simultaneously")
endif()

if(ENABLE_WEBUI)
  message(STATUS "ENABLE_WEBUI=ON — embedding litehtml-based WebUI")
  target_compile_definitions(inferflux_core PUBLIC INFERFLUX_ENABLE_WEBUI=1)
  target_sources(inferflux_core PRIVATE
    webui/ui_renderer.cpp
    webui/ui_bundle.cpp
  )
  target_include_directories(inferflux_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/litehtml_stub
    ${CMAKE_CURRENT_SOURCE_DIR}/webui
  )
else()
  target_compile_definitions(inferflux_core PUBLIC INFERFLUX_ENABLE_WEBUI=0)
endif()

if(ENABLE_MLX)
  target_sources(inferflux_core PRIVATE
    runtime/backends/mlx/mlx_backend.cpp
    runtime/backends/mlx/mlx_loader.cpp
    runtime/backends/mlx/mlx_execution.cpp
  )
  message(STATUS "ENABLE_MLX=ON — configuring mlx-c dependency")
  set(MLX_C_HINTS $ENV{MLX_C_ROOT} ${MLX_C_ROOT})
  if(MLX_C_HINTS)
    # Fast path: user supplied a pre-built mlx-c installation.
    find_path(MLX_C_INCLUDE_DIR mlx/c/mlx.h HINTS ${MLX_C_HINTS} PATH_SUFFIXES include)
    find_library(MLX_C_LIBRARY NAMES mlxc mlx HINTS ${MLX_C_HINTS} PATH_SUFFIXES lib)
  else()
    # Slow path: fetch mlx-c from GitHub at the pinned tag and build it.
    # Requires `pip install mlx==${MLX_C_TAG compatible version}` first.
    message(STATUS "MLX_C_ROOT not set — fetching mlx-c ${MLX_C_TAG} via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
      mlxc
      GIT_REPOSITORY https://github.com/ml-explore/mlx-c.git
      GIT_TAG        ${MLX_C_TAG}
      GIT_SHALLOW    TRUE
    )
    set(MLX_C_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(MLX_C_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(mlxc)
    # FetchContent builds the 'mlxc' target; expose its include + lib paths.
    set(MLX_C_INCLUDE_DIR ${mlxc_SOURCE_DIR})
    set(MLX_C_LIBRARY mlxc)
  endif()
  if(MLX_C_INCLUDE_DIR AND MLX_C_LIBRARY)
    message(STATUS "mlx-c include dir: ${MLX_C_INCLUDE_DIR}")
    target_compile_definitions(inferflux_core PUBLIC INFERFLUX_HAS_MLX=1)
    target_include_directories(inferflux_core PUBLIC ${MLX_C_INCLUDE_DIR})
    target_link_libraries(inferflux_core PUBLIC ${MLX_C_LIBRARY})
    # Record the directory of mlxc so ctest can set DYLD_LIBRARY_PATH / LD_LIBRARY_PATH.
    get_filename_component(MLX_C_LIB_DIR "${MLX_C_LIBRARY}" DIRECTORY)
    set(MLX_C_LIB_DIR "${MLX_C_LIB_DIR}" CACHE INTERNAL "")
    # Also find the MLX C++ library for rpath embedding.
    find_library(MLX_LIBRARY NAMES mlx
                 HINTS ${MLX_C_HINTS} $ENV{MLX_ROOT}
                 PATHS /opt/anaconda3/lib/python3.13/site-packages/mlx
                       /opt/homebrew/lib
                 PATH_SUFFIXES lib ../lib)
    if(MLX_LIBRARY)
      get_filename_component(MLX_LIB_DIR "${MLX_LIBRARY}" DIRECTORY)
      set_target_properties(inferflux_core PROPERTIES
        BUILD_RPATH   "${MLX_C_LIB_DIR};${MLX_LIB_DIR}"
        INSTALL_RPATH "${MLX_C_LIB_DIR};${MLX_LIB_DIR}")
    endif()
  else()
    message(WARNING "ENABLE_MLX=ON but mlx-c headers/libraries not found. Disabling MLX backend.")
    target_compile_definitions(inferflux_core PUBLIC INFERFLUX_HAS_MLX=0)
  endif()
else()
  target_compile_definitions(inferflux_core PUBLIC INFERFLUX_HAS_MLX=0)
endif()

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
# Workaround: llama.cpp ggml-cuda CUDA host compiler version detection can fail
# in WSL2, causing ggml_get_flags() to receive empty CCVER argument.
# Disable ggml compiler warnings flags to skip the problematic detection path.
set(GGML_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
# NOTE: LLAMA_CUDA, GGML_HIP, LLAMA_METAL, and LLAMA_BLAS are set by the
# hardware-detection blocks above. Do not override them here; stale resets can
# silently disable an otherwise detected accelerator.
add_subdirectory(external/llama.cpp EXCLUDE_FROM_ALL)
target_link_libraries(inferflux_core PUBLIC llama)
target_include_directories(inferflux_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp/include
  ${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp/common
  ${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp/vendor
)
target_link_libraries(inferflux_core PUBLIC OpenSSL::Crypto OpenSSL::SSL yaml-cpp)
# POSIX SHM (shm_open/mmap) requires -lrt on Linux; macOS links it automatically.
if(UNIX AND NOT APPLE)
  target_link_libraries(inferflux_core PUBLIC rt)
endif()

# §2.2 Multimodal support via libmtmd (ENABLE_MTMD=OFF by default)
if(ENABLE_MTMD)
  set(MTMD_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp/tools/mtmd)
  target_sources(inferflux_core PRIVATE
    ${MTMD_SRC_DIR}/mtmd.cpp
    ${MTMD_SRC_DIR}/mtmd-audio.cpp
    ${MTMD_SRC_DIR}/mtmd-helper.cpp
    ${MTMD_SRC_DIR}/clip.cpp
    ${MTMD_SRC_DIR}/models/cogvlm.cpp
    ${MTMD_SRC_DIR}/models/conformer.cpp
    ${MTMD_SRC_DIR}/models/glm4v.cpp
    ${MTMD_SRC_DIR}/models/internvl.cpp
    ${MTMD_SRC_DIR}/models/kimivl.cpp
    ${MTMD_SRC_DIR}/models/kimik25.cpp
    ${MTMD_SRC_DIR}/models/nemotron-v2-vl.cpp
    ${MTMD_SRC_DIR}/models/llama4.cpp
    ${MTMD_SRC_DIR}/models/llava.cpp
    ${MTMD_SRC_DIR}/models/minicpmv.cpp
    ${MTMD_SRC_DIR}/models/paddleocr.cpp
    ${MTMD_SRC_DIR}/models/pixtral.cpp
    ${MTMD_SRC_DIR}/models/qwen2vl.cpp
    ${MTMD_SRC_DIR}/models/qwen3vl.cpp
    ${MTMD_SRC_DIR}/models/siglip.cpp
    ${MTMD_SRC_DIR}/models/whisper-enc.cpp
    ${MTMD_SRC_DIR}/models/mobilenetv5.cpp
    ${MTMD_SRC_DIR}/models/youtuvl.cpp
  )
  target_include_directories(inferflux_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp/tools/mtmd
    ${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp/tools
  )
  target_compile_definitions(inferflux_core PUBLIC INFERFLUX_HAS_MTMD=1)
  if(NOT MSVC)
    target_compile_options(inferflux_core PRIVATE -Wno-cast-qual)
  endif()
  message(STATUS "ENABLE_MTMD=ON — libmtmd vision support enabled")
else()
  message(STATUS "ENABLE_MTMD=OFF — multimodal support disabled (pass -DENABLE_MTMD=ON to enable)")
endif()

add_executable(inferfluxd server/main.cpp)
# Use --whole-archive to ensure static self-registration objects (backend
# registry entries in inferflux_cuda_backend.cpp etc.) are linked even when
# nothing else in main.cpp references them directly.
if(MSVC)
  target_link_libraries(inferfluxd PRIVATE inferflux_core)
  target_link_options(inferfluxd PRIVATE /WHOLEARCHIVE:inferflux_core)
elseif(APPLE)
  target_link_libraries(inferfluxd PRIVATE inferflux_core)
  target_link_options(inferfluxd PRIVATE
    "LINKER:-force_load,$<TARGET_FILE:inferflux_core>")
else()
  target_link_libraries(inferfluxd PRIVATE
    -Wl,--whole-archive inferflux_core -Wl,--no-whole-archive)
endif()

add_executable(inferctl cli/main.cpp)
target_link_libraries(inferctl PRIVATE inferflux_core)

if(ENABLE_CUDA AND CUDAToolkit_FOUND)
  add_executable(benchmark_downproj_q81 tests/unit/benchmark_downproj_q81.cu)
  target_include_directories(benchmark_downproj_q81 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/external)
  target_link_libraries(benchmark_downproj_q81 PRIVATE inferflux_core)
  add_executable(benchmark_fused_gate_up_q81
                 tests/unit/benchmark_fused_gate_up_q81.cu)
  target_include_directories(benchmark_fused_gate_up_q81 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/external)
  target_link_libraries(benchmark_fused_gate_up_q81 PRIVATE inferflux_core)
  add_executable(benchmark_grouped_q81_pair
                 tests/unit/benchmark_grouped_q81_pair.cu)
  target_include_directories(benchmark_grouped_q81_pair PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/external)
  target_link_libraries(benchmark_grouped_q81_pair PRIVATE inferflux_core)

  # Fused kernel parity tests: .cpp (Catch2, MSVC) + .cu (test-only kernels, nvcc)
  # Split avoids nvcc/Catch2 operator== conflict with cuda_bf16.hpp on CUDA 13.2.
  add_executable(inferflux_fused_kernel_tests
    tests/unit/test_fused_kernels.cpp
    tests/unit/test_fused_kernels_cuda.cu
    external/catch2/catch_amalgamated.cpp
  )
  target_include_directories(inferflux_fused_kernel_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/external
  )
  target_link_libraries(inferflux_fused_kernel_tests PRIVATE inferflux_core)
  set_target_properties(inferflux_fused_kernel_tests PROPERTIES
    CUDA_STANDARD 17
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  )
endif()

install(TARGETS inferfluxd inferctl
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(FILES README.md docs/Quickstart.md docs/Troubleshooting.md
        DESTINATION ${CMAKE_INSTALL_DOCDIR})

install(FILES config/server.yaml
        DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/inferflux
        RENAME inferflux.yaml)

enable_testing()
add_executable(inferflux_tests
  tests/unit/test_tokenizer.cpp
  tests/unit/test_api_key_auth.cpp
  tests/unit/test_rate_limiter.cpp
  tests/unit/test_crash_handler.cpp
  tests/unit/test_guardrail.cpp
  tests/unit/test_audit_logger.cpp
  tests/unit/test_metrics.cpp
  tests/unit/test_http_server.cpp
  tests/unit/test_scheduler.cpp
  tests/unit/test_batch_selection_policy.cpp
  tests/unit/test_session_handle_manager.cpp
  tests/unit/test_sequence_slot_manager.cpp
  tests/unit/test_policy_store.cpp
  tests/unit/test_oidc_validator.cpp
  tests/unit/test_prefix_cache.cpp
  tests/unit/test_radix_prefix_cache.cpp
  tests/unit/test_span.cpp
  tests/unit/test_kv_channel.cpp
  tests/unit/test_phased_execution.cpp
  tests/unit/test_multimodal.cpp
  tests/unit/test_moe_routing.cpp
  tests/unit/test_flash_attn.cpp
  tests/unit/test_shm_transport.cpp
  tests/unit/test_chat_template.cpp
  tests/unit/test_sampling.cpp
  tests/unit/test_logger.cpp
  tests/unit/test_structured_output.cpp
  tests/unit/test_paged_kv_cache.cpp
  tests/unit/test_http_server_admin_cache.cpp
  tests/unit/test_unified_batching.cpp
  tests/unit/test_lane_dispatcher.cpp
  tests/unit/test_model_registry.cpp
  tests/unit/test_model_selection.cpp
  tests/unit/test_single_model_router.cpp
  tests/unit/test_stop_sequences.cpp
  tests/unit/test_parallel_context.cpp
  tests/unit/test_ep_dispatch.cpp
  tests/unit/test_backend_factory.cpp
  tests/unit/test_backend_registry.cpp
  tests/unit/test_backend_capabilities.cpp
  tests/unit/test_model_format.cpp
  tests/unit/test_model_info_paths.cpp
  tests/unit/test_backend_types.cpp
  tests/unit/test_batch_analyzer.cpp
  tests/unit/test_backend_interface.cpp
  tests/unit/test_native_bootstrap_config.cpp
  tests/unit/test_kv_cache_planner.cpp
  tests/unit/test_tokenizer_factory.cpp
  tests/unit/test_startup_advisor.cpp
  tests/unit/test_result.cpp
  tests/integration/test_http_generation_admission.cpp
  external/catch2/catch_amalgamated.cpp
)

# CUDA-specific tests (require CUDA headers and native kernel support)
if(ENABLE_CUDA AND CUDAToolkit_FOUND)
  target_sources(inferflux_tests PRIVATE
    tests/unit/test_native_forward.cpp
    tests/unit/test_native_rowpair_kernels.cpp
    tests/unit/test_native_dequant_policy.cpp
    tests/unit/test_gpu_kv_cache.cpp
    tests/unit/test_gpu_sampler.cpp
    tests/unit/test_native_batching.cpp
    tests/unit/test_native_gpu_backend.cpp
    tests/unit/test_batched_decode.cpp
    tests/unit/test_model_memory_ledger.cpp
    tests/unit/test_dtype_traits.cpp
    tests/unit/test_gguf_tokenizer.cpp
    tests/unit/test_gguf_parsing.cpp
    tests/unit/test_gguf_model_loader.cpp
    tests/unit/test_quantized_weight_map.cpp
    tests/unit/test_quantized_weight_map_scratch.cpp
    tests/unit/test_strategy_registry.cpp
    tests/integration/test_gguf_quantization.cpp
  )
endif()
target_include_directories(inferflux_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/external
  ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_compile_definitions(inferflux_tests PRIVATE INFERFLUX_TESTING)
target_link_libraries(inferflux_tests PRIVATE inferflux_core)
if(ENABLE_CUDA)
  add_executable(inferflux_first_token_probe
    tests/tools/first_token_probe.cpp
  )
  target_include_directories(inferflux_first_token_probe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/external
  )
  target_link_libraries(inferflux_first_token_probe PRIVATE inferflux_core)

  add_executable(inferflux_attention_profile_probe
    tests/tools/attention_profile_probe.cpp
  )
  target_include_directories(inferflux_attention_profile_probe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/external
  )
  target_link_libraries(inferflux_attention_profile_probe PRIVATE inferflux_core)

  add_executable(inferflux_kernel_count_probe
    tests/tools/kernel_count_probe.cpp
  )
  target_include_directories(inferflux_kernel_count_probe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/external
  )
  target_link_libraries(inferflux_kernel_count_probe PRIVATE inferflux_core)
  find_library(CUPTI_LIBRARY cupti
    HINTS ${CUDAToolkit_LIBRARY_DIR}
          ${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/lib64
          ${CUDA_TOOLKIT_ROOT_DIR}/lib64
          $ENV{CUDA_PATH}/extras/CUPTI/lib64
  )
  find_path(CUPTI_INCLUDE_DIR cupti.h
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/include
          $ENV{CUDA_PATH}/extras/CUPTI/include
  )
  if(CUPTI_LIBRARY AND CUPTI_INCLUDE_DIR)
    target_link_libraries(inferflux_kernel_count_probe PRIVATE ${CUPTI_LIBRARY})
    target_include_directories(inferflux_kernel_count_probe PRIVATE
      ${CUPTI_INCLUDE_DIR}
    )
    target_compile_definitions(inferflux_kernel_count_probe PRIVATE INFERFLUX_HAS_CUPTI=1)
  endif()
endif()
if(ENABLE_MLX)
  target_sources(inferflux_tests PRIVATE
    tests/unit/test_mlx_loader.cpp
    tests/unit/test_mlx_tokenizer.cpp
    tests/unit/test_mlx_execution.cpp
    tests/unit/test_mlx_backend.cpp
  )
endif()
add_test(NAME UnitTests COMMAND inferflux_tests)
add_test(NAME FairnessTests COMMAND inferflux_tests "[fairness]")
set_tests_properties(FairnessTests PROPERTIES LABELS fairness)
add_test(NAME PagedKVTests COMMAND inferflux_tests "[paged_kv]")
set_tests_properties(PagedKVTests PROPERTIES LABELS paged_kv)
add_test(NAME UnifiedBatchTests COMMAND inferflux_tests "[unified_batch]")
set_tests_properties(UnifiedBatchTests PROPERTIES LABELS unified_batch)
add_test(NAME ParallelTests COMMAND inferflux_tests "[parallel]")
set_tests_properties(ParallelTests PROPERTIES LABELS parallel)
add_test(NAME EPTests COMMAND inferflux_tests "[ep]")
set_tests_properties(EPTests PROPERTIES LABELS ep)
add_test(NAME BackendFactoryTests COMMAND inferflux_tests "[backend_factory]")
set_tests_properties(BackendFactoryTests PROPERTIES LABELS backend_factory)
add_test(NAME BackendCapabilitiesTests COMMAND inferflux_tests "[backend_capabilities]")
set_tests_properties(BackendCapabilitiesTests PROPERTIES LABELS backend_capabilities)
if(ENABLE_CUDA AND CUDAToolkit_FOUND)
  add_test(NAME GGUFMemoryContractTests COMMAND inferflux_tests "[memory_contract]")
  set_tests_properties(GGUFMemoryContractTests PROPERTIES LABELS "memory_contract;integration")
endif()
add_test(NAME MoETests COMMAND inferflux_tests "[moe]")
set_tests_properties(MoETests PROPERTIES LABELS moe)
add_test(NAME FlashAttnTests COMMAND inferflux_tests "[flash_attn]")
set_tests_properties(FlashAttnTests PROPERTIES LABELS flash_attn)
add_test(NAME ShmTransportTests COMMAND inferflux_tests "[shm_transport]")
set_tests_properties(ShmTransportTests PROPERTIES LABELS shm_transport)
add_test(NAME HttpServerTests COMMAND inferflux_tests "[http_server]")
set_tests_properties(HttpServerTests PROPERTIES LABELS http_server)
add_test(NAME HttpGenerationAdmissionIntegrationTests
         COMMAND inferflux_tests "[http_server][integration][admission]")
set_tests_properties(HttpGenerationAdmissionIntegrationTests PROPERTIES
  LABELS "http_server;integration;admission")
add_test(NAME ChatTemplateTests COMMAND inferflux_tests "[chat_template]")
set_tests_properties(ChatTemplateTests PROPERTIES LABELS chat_template)
add_test(NAME SamplingTests COMMAND inferflux_tests "[sampling]")
set_tests_properties(SamplingTests PROPERTIES LABELS sampling)
add_test(NAME LoggerTests COMMAND inferflux_tests "[logger]")
set_tests_properties(LoggerTests PROPERTIES LABELS logger)
add_test(NAME StructuredTests COMMAND inferflux_tests "[structured]")
set_tests_properties(StructuredTests PROPERTIES LABELS structured)
add_test(NAME ModelRegistryTests COMMAND inferflux_tests "[model_registry]")
set_tests_properties(ModelRegistryTests PROPERTIES LABELS model_registry)
add_test(NAME ModelFormatTests COMMAND inferflux_tests "[model_format]")
set_tests_properties(ModelFormatTests PROPERTIES LABELS model_format)
add_test(NAME ModelPathTests COMMAND inferflux_tests "[model_paths]")
set_tests_properties(ModelPathTests PROPERTIES LABELS model_paths)
add_test(NAME ModelIdentityTests COMMAND inferflux_tests "[model_identity]")
set_tests_properties(ModelIdentityTests PROPERTIES LABELS model_identity)
add_test(NAME EmbeddingsRoutingTests COMMAND inferflux_tests "[embeddings_routing]")
set_tests_properties(EmbeddingsRoutingTests PROPERTIES LABELS embeddings_routing)
add_test(NAME StopSequenceTests COMMAND inferflux_tests "[stop_sequences]")
set_tests_properties(StopSequenceTests PROPERTIES LABELS stop_sequences)
if(ENABLE_CUDA AND CUDAToolkit_FOUND)
  add_test(NAME NativeForwardTests COMMAND inferflux_tests "[native_forward]")
  set_tests_properties(NativeForwardTests PROPERTIES LABELS native_forward)
  add_test(NAME DtypeTraitsTests COMMAND inferflux_tests "[dtype_traits]")
  set_tests_properties(DtypeTraitsTests PROPERTIES LABELS dtype_traits)
  add_test(NAME GpuKvCacheTests COMMAND inferflux_tests "[gpu_kv_cache]")
  set_tests_properties(GpuKvCacheTests PROPERTIES LABELS gpu_kv_cache)
  add_test(NAME GpuSamplerTests COMMAND inferflux_tests "[gpu_sampler]")
  set_tests_properties(GpuSamplerTests PROPERTIES LABELS gpu_sampler)
  add_test(NAME NativeBatchTests COMMAND inferflux_tests "[native_batch]")
  set_tests_properties(NativeBatchTests PROPERTIES LABELS native_batch)
endif()
add_test(NAME TokenizerFactoryTests COMMAND inferflux_tests "[tokenizer_factory]")
set_tests_properties(TokenizerFactoryTests PROPERTIES LABELS tokenizer_factory)
add_test(NAME StartupAdvisorTests COMMAND inferflux_tests "[startup_advisor]")
set_tests_properties(StartupAdvisorTests PROPERTIES LABELS startup_advisor)
if(ENABLE_CUDA AND CUDAToolkit_FOUND)
  add_test(NAME GGUFTests COMMAND inferflux_tests "[gguf]")
  set_tests_properties(GGUFTests PROPERTIES LABELS gguf)
  add_test(NAME QuantizationTests COMMAND inferflux_tests "[quantization]")
  set_tests_properties(QuantizationTests PROPERTIES LABELS quantization)
endif()
add_test(NAME SequenceSlotManagerTests COMMAND inferflux_tests "[slot_manager]")
set_tests_properties(SequenceSlotManagerTests PROPERTIES LABELS slot_manager)
add_test(NAME BatchSelectionPolicyTests COMMAND inferflux_tests "[batch_selection_policy]")
set_tests_properties(BatchSelectionPolicyTests PROPERTIES LABELS batch_selection_policy)
if(ENABLE_MLX)
  add_test(NAME MlxLoaderTests COMMAND inferflux_tests "[mlx_loader]")
  set_tests_properties(MlxLoaderTests PROPERTIES LABELS mlx_loader)
  if(MLX_C_LIB_DIR)
    # Provide the mlx-c and MLX dylib locations to the test runner on macOS.
    set_tests_properties(MlxLoaderTests PROPERTIES
      ENVIRONMENT "DYLD_LIBRARY_PATH=${MLX_C_LIB_DIR}:$ENV{DYLD_LIBRARY_PATH}")
  endif()
  add_test(NAME MlxTokenizerTests COMMAND inferflux_tests "[mlx_tokenizer]")
  set_tests_properties(MlxTokenizerTests PROPERTIES LABELS mlx_tokenizer)
  add_test(NAME MlxExecutionTests COMMAND inferflux_tests "[mlx_execution]")
  set_tests_properties(MlxExecutionTests PROPERTIES LABELS mlx_execution)
  if(MLX_C_LIB_DIR)
    set_tests_properties(MlxExecutionTests PROPERTIES
      ENVIRONMENT "DYLD_LIBRARY_PATH=${MLX_C_LIB_DIR}:$ENV{DYLD_LIBRARY_PATH}")
  endif()
endif()
if(ENABLE_CUDA AND CUDAToolkit_FOUND)
  add_test(NAME FusedKernelTests COMMAND inferflux_fused_kernel_tests)
  set_tests_properties(FusedKernelTests PROPERTIES LABELS "cuda;fused_kernels")
endif()

# On Windows, llama.cpp/ggml build DLLs into bin/Release/. Tests that link
# inferflux_core need these DLLs on PATH. CUDA runtime DLLs are at
# CUDA_PATH/bin/x64/. Set PATH for all test targets via ctest ENVIRONMENT.
if(WIN32)
  set(_DLL_DIRS "${CMAKE_BINARY_DIR}/bin/$<CONFIG>")
  if(DEFINED ENV{CUDA_PATH})
    set(_DLL_DIRS "${_DLL_DIRS}\;$ENV{CUDA_PATH}/bin/x64")
  endif()
  # Collect all test names that use inferflux_tests or inferflux_fused_kernel_tests
  get_property(_ALL_TESTS DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY TESTS)
  foreach(_TEST ${_ALL_TESTS})
    get_test_property(${_TEST} ENVIRONMENT _EXISTING_ENV)
    if(_EXISTING_ENV)
      set_tests_properties(${_TEST} PROPERTIES
        ENVIRONMENT "${_EXISTING_ENV};PATH=${_DLL_DIRS}\;$ENV{PATH}")
    else()
      set_tests_properties(${_TEST} PROPERTIES
        ENVIRONMENT "PATH=${_DLL_DIRS}\;$ENV{PATH}")
    endif()
  endforeach()
endif()

set(CPACK_PACKAGE_NAME "inferflux")
set(CPACK_PACKAGE_VENDOR "Inferencial Labs")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "InferFlux inference server and CLI")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_CONTACT "support@inferencial.ai")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
set(CPACK_PACKAGE_FILE_NAME
    "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_STRIP_FILES TRUE)
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libssl3, libyaml-cpp0.7")
set(CPACK_DEBIAN_PACKAGE_SECTION "utils")
set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
set(CPACK_RPM_PACKAGE_LICENSE "Apache-2.0")
set(CPACK_RPM_PACKAGE_REQUIRES "openssl, yaml-cpp")
set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION
    "${CMAKE_INSTALL_SYSCONFDIR}/inferflux")
set(CPACK_GENERATOR "TGZ")
include(CPack)

# ---------------------------------------------------------------------------
# SBOM — generate CycloneDX / SPDX bill-of-materials after every build.
#
# The SBOM script (scripts/generate_sbom.py) reads the validated dependency
# evidence in config/dependencies.json and writes two output files to
# ${CMAKE_BINARY_DIR}/sbom/:
#   inferflux-sbom.cdx.json   — CycloneDX 1.5 JSON (for supply-chain tooling)
#   inferflux-sbom.spdx        — SPDX 2.3 tag-value (for license compliance)
#
# The target is always defined but only runs when Python3 is available.
# Run manually: cmake --build build --target sbom
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Coverage — gcov/lcov instrumentation.
# Enable with -DENABLE_COVERAGE=ON (automatically uses Debug build).
# Requires GCC or Clang + lcov installed.  Run:
#   cmake --build build-cov --target coverage
# to produce HTML report at build-cov/coverage/html/index.html and an
# lcov.info trace file for upload to Codecov.
# ---------------------------------------------------------------------------
option(ENABLE_COVERAGE "Instrument build for gcov/lcov coverage" OFF)

if(ENABLE_COVERAGE)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "ENABLE_COVERAGE requires GCC or Clang")
  endif()
  # Force Debug so optimiser doesn't collapse away instrumented lines.
  if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "ENABLE_COVERAGE: overriding build type to Debug")
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
  endif()
  set(COVERAGE_FLAGS --coverage -O0 -fno-inline)
  target_compile_options(inferflux_core PUBLIC ${COVERAGE_FLAGS})
  target_compile_options(inferflux_tests PRIVATE ${COVERAGE_FLAGS})
  target_link_libraries(inferflux_core PUBLIC --coverage)
  target_link_libraries(inferflux_tests PRIVATE --coverage)

  find_program(LCOV_BIN lcov REQUIRED)
  find_program(GENHTML_BIN genhtml REQUIRED)

  set(COVERAGE_OUT_DIR "${CMAKE_BINARY_DIR}/coverage")
  set(LCOV_INFO "${COVERAGE_OUT_DIR}/lcov.info")

  add_custom_target(coverage
    DEPENDS inferflux_tests inferfluxd
    COMMENT "Collecting gcov data and generating coverage report"
    # 0. Ensure output directory exists
    COMMAND ${CMAKE_COMMAND} -E make_directory "${COVERAGE_OUT_DIR}"
    # 1. Zero any stale counters
    COMMAND ${LCOV_BIN} --zerocounters --directory "${CMAKE_BINARY_DIR}" --quiet
    # 2. Capture initial baseline (zero coverage)
    COMMAND ${LCOV_BIN} --capture --initial --directory "${CMAKE_BINARY_DIR}"
            --output-file "${COVERAGE_OUT_DIR}/baseline.info" --quiet
    # 3. Run the test suite
    COMMAND ${CMAKE_CTEST_COMMAND} --test-dir "${CMAKE_BINARY_DIR}"
            --output-on-failure --timeout 90
    # 4. Capture post-test counters
    COMMAND ${LCOV_BIN} --capture --directory "${CMAKE_BINARY_DIR}"
            --output-file "${COVERAGE_OUT_DIR}/tests.info" --quiet
    # 5. Merge baseline + test data
    COMMAND ${LCOV_BIN} --add-tracefile "${COVERAGE_OUT_DIR}/baseline.info"
            --add-tracefile "${COVERAGE_OUT_DIR}/tests.info"
            --output-file "${LCOV_INFO}" --quiet
    # 6. Strip external/vendored and system headers
    COMMAND ${LCOV_BIN} --remove "${LCOV_INFO}"
            "${CMAKE_CURRENT_SOURCE_DIR}/external/*"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/*"
            "/usr/*"
            --output-file "${LCOV_INFO}" --quiet
    # 7. HTML report
    COMMAND ${CMAKE_COMMAND} -E make_directory "${COVERAGE_OUT_DIR}/html"
    COMMAND ${GENHTML_BIN} "${LCOV_INFO}"
            --output-directory "${COVERAGE_OUT_DIR}/html"
            --title "InferFlux coverage"
            --legend --demangle-cpp --quiet
    COMMAND ${CMAKE_COMMAND} -E echo
            "Coverage report: ${COVERAGE_OUT_DIR}/html/index.html"
    COMMAND ${CMAKE_COMMAND} -E echo
            "lcov.info for Codecov: ${LCOV_INFO}"
    VERBATIM
    USES_TERMINAL
  )
  message(STATUS "ENABLE_COVERAGE=ON — run: cmake --build <build-dir> --target coverage")
endif()

option(ENABLE_SBOM "Generate SBOM after build" ON)

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  # INFERFLUX_SERVER_BIN tells integration tests which binary to launch.
  # CMake always knows the correct path (CMAKE_BINARY_DIR/inferfluxd), so we
  # inject it as a ctest environment variable — this makes the tests work
  # regardless of whether the build directory is "build", "build-cov", etc.
  set(_INFERFLUX_SERVER_BIN "${CMAKE_BINARY_DIR}/inferfluxd")
  set(_INFERCTL_BIN "${CMAKE_BINARY_DIR}/inferctl")

  # Focused throughput-gate contract tests (argument parsing + metrics parsing).
  add_test(
    NAME ThroughputGateContractTests
    COMMAND ${Python3_EXECUTABLE} tests/integration/throughput_gate_contract_test.py)
  set_tests_properties(ThroughputGateContractTests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 30)

  # Throughput-gate failure message contract tests.
  add_test(
    NAME ThroughputGateFailureContractTests
    COMMAND ${Python3_EXECUTABLE} tests/integration/throughput_gate_failure_contract_test.py)
  set_tests_properties(ThroughputGateFailureContractTests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 30)

  add_test(
    NAME IntegrationBackendIdentityContract
    COMMAND ${Python3_EXECUTABLE} tests/integration/backend_identity_contract_test.py)
  set_tests_properties(IntegrationBackendIdentityContract PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 30)

  # Stub-mode integration tests — run without a model; always enabled.
  add_test(NAME StubIntegration
           COMMAND ${Python3_EXECUTABLE} tests/integration/stub_integration_test.py)
  set_tests_properties(StubIntegration PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 60
    ENVIRONMENT
      "INFERFLUX_SERVER_BIN=${_INFERFLUX_SERVER_BIN};INFERCTL_BIN=${_INFERCTL_BIN}")

  # Focused CLI contract gate: script-facing model-list output modes +
  # auth failure semantics.
  add_test(
    NAME IntegrationCLIModelListContract
    COMMAND ${Python3_EXECUTABLE} tests/integration/stub_integration_test.py
            StubIntegrationTests.test_inferctl_models_json
            StubIntegrationTests.test_inferctl_models_default_output
            StubIntegrationTests.test_inferctl_admin_models_list_json
            StubIntegrationTests.test_inferctl_admin_models_list_default_output
            StubIntegrationTests.test_inferctl_models_json_auth_failure
            StubIntegrationTests.test_inferctl_admin_models_list_json_auth_failure)
  set_tests_properties(IntegrationCLIModelListContract PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 60
    ENVIRONMENT
      "INFERFLUX_SERVER_BIN=${_INFERFLUX_SERVER_BIN};INFERCTL_BIN=${_INFERCTL_BIN}")

  add_test(
    NAME IntegrationEmbeddingsRoutingContract
    COMMAND ${Python3_EXECUTABLE} tests/integration/stub_integration_test.py
            StubIntegrationTests.test_embeddings_requires_input
            StubIntegrationTests.test_embeddings_backend_unavailable_default_model
            StubIntegrationTests.test_embeddings_explicit_model_not_found
            StubIntegrationTests.test_completion_explicit_model_not_found)
  set_tests_properties(IntegrationEmbeddingsRoutingContract PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 60
    ENVIRONMENT "INFERFLUX_SERVER_BIN=${_INFERFLUX_SERVER_BIN}")

  add_test(
    NAME IntegrationModelIdentityContract
    COMMAND ${Python3_EXECUTABLE} tests/integration/stub_integration_test.py
            StubIntegrationTests.test_models_get_by_id_not_found
            StubIntegrationTests.test_admin_models_load_requires_path
            StubIntegrationTests.test_admin_models_unload_requires_id
            StubIntegrationTests.test_admin_models_unload_not_found
            StubIntegrationTests.test_admin_models_set_default_requires_id
            StubIntegrationTests.test_admin_models_set_default_not_found
            StubIntegrationTests.test_inferctl_models_id_not_found_json
            StubIntegrationTests.test_inferctl_models_id_not_found_default_output
            StubIntegrationTests.test_inferctl_admin_models_unload_not_found
            StubIntegrationTests.test_inferctl_admin_models_set_default_not_found
            StubIntegrationStrictNativePolicyTests.test_admin_models_load_explicit_native_rejected_in_strict_mode)
  set_tests_properties(IntegrationModelIdentityContract PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 60
    ENVIRONMENT
      "INFERFLUX_SERVER_BIN=${_INFERFLUX_SERVER_BIN};INFERCTL_BIN=${_INFERCTL_BIN}")

  # Focused CLI contract gate: strict fail-fast argument semantics for
  # automation-facing admin commands.
  add_test(
    NAME IntegrationCLIAdminArgContract
    COMMAND ${Python3_EXECUTABLE} tests/integration/stub_integration_test.py
            StubIntegrationTests.test_inferctl_models_id_requires_value
            StubIntegrationTests.test_inferctl_admin_models_load_requires_value
            StubIntegrationTests.test_inferctl_admin_models_unload_requires_value
            StubIntegrationTests.test_inferctl_admin_models_set_default_requires_value
            StubIntegrationTests.test_inferctl_admin_models_rejects_multiple_operations
            StubIntegrationTests.test_inferctl_admin_models_default_requires_load
            StubIntegrationTests.test_inferctl_admin_cache_rejects_multiple_operations
            StubIntegrationTests.test_inferctl_admin_cache_tokens_requires_warm
            StubIntegrationTests.test_inferctl_admin_cache_warm_requires_tokens
            StubIntegrationTests.test_inferctl_admin_cache_warm_requires_completion
            StubIntegrationTests.test_inferctl_admin_cache_completion_tokens_requires_integer
            StubIntegrationTests.test_inferctl_admin_cache_status_auth_failure
            StubIntegrationTests.test_inferctl_admin_api_keys_rejects_multiple_operations
            StubIntegrationTests.test_inferctl_admin_api_keys_scopes_requires_add
            StubIntegrationTests.test_inferctl_admin_api_keys_add_requires_scopes
            StubIntegrationTests.test_inferctl_admin_api_keys_add_requires_value
            StubIntegrationTests.test_inferctl_admin_api_keys_remove_requires_value
            StubIntegrationTests.test_inferctl_admin_api_keys_list_auth_failure
            StubIntegrationTests.test_inferctl_admin_guardrails_rejects_multiple_operations
            StubIntegrationTests.test_inferctl_admin_guardrails_set_requires_value
            StubIntegrationTests.test_inferctl_admin_guardrails_list_auth_failure
            StubIntegrationTests.test_inferctl_admin_rate_limit_rejects_multiple_operations
            StubIntegrationTests.test_inferctl_admin_rate_limit_set_requires_value
            StubIntegrationTests.test_inferctl_admin_rate_limit_set_requires_integer
            StubIntegrationTests.test_inferctl_admin_rate_limit_get_auth_failure
            StubIntegrationTests.test_inferctl_admin_pools_requires_get
            StubIntegrationTests.test_inferctl_admin_pools_get_auth_failure
            StubIntegrationTests.test_inferctl_admin_pools_get
            StubIntegrationTests.test_inferctl_admin_routing_rejects_multiple_operations
            StubIntegrationTests.test_inferctl_admin_routing_set_requires_fields
            StubIntegrationTests.test_inferctl_admin_routing_allow_default_requires_set
            StubIntegrationTests.test_inferctl_admin_routing_allow_default_requires_value
            StubIntegrationTests.test_inferctl_admin_routing_allow_default_requires_boolean
            StubIntegrationTests.test_inferctl_admin_routing_require_ready_requires_boolean
            StubIntegrationTests.test_inferctl_admin_routing_fallback_scope_requires_value
            StubIntegrationTests.test_inferctl_admin_routing_fallback_scope_validates_value
            StubIntegrationTests.test_inferctl_admin_routing_get_auth_failure
            StubIntegrationTests.test_inferctl_admin_routing_get
            StubIntegrationTests.test_inferctl_admin_routing_set_scope_success
            StubIntegrationTests.test_inferctl_admin_routing_set_booleans_success)
  set_tests_properties(IntegrationCLIAdminArgContract PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 60
    ENVIRONMENT
      "INFERFLUX_SERVER_BIN=${_INFERFLUX_SERVER_BIN};INFERCTL_BIN=${_INFERCTL_BIN}")

  add_test(
    NAME IntegrationCLIAdminPoolsContract
    COMMAND ${Python3_EXECUTABLE}
            tests/integration/inferctl_admin_pools_contract_test.py)
  set_tests_properties(IntegrationCLIAdminPoolsContract PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 30
    ENVIRONMENT "INFERCTL_BIN=${_INFERCTL_BIN}")

  # Native CUDA backend metrics contract (stub-mode: verifies metric names + format).
  add_test(NAME IntegrationNativeMetrics
           COMMAND ${Python3_EXECUTABLE} tests/integration/native_metrics_test.py)
  set_tests_properties(IntegrationNativeMetrics PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 60
    ENVIRONMENT "INFERFLUX_SERVER_BIN=${_INFERFLUX_SERVER_BIN}")

  add_test(NAME IntegrationNativePhaseTimingParser
           COMMAND ${Python3_EXECUTABLE}
                   tests/integration/native_phase_timing_parser_test.py)
  set_tests_properties(IntegrationNativePhaseTimingParser PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 30)

  add_test(NAME IntegrationBenchmarkResponseClassifier
           COMMAND ${Python3_EXECUTABLE}
                   tests/integration/benchmark_response_classifier_test.py)
  set_tests_properties(IntegrationBenchmarkResponseClassifier PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 30)

  add_test(NAME IntegrationBenchmarkHarnessDefaults
           COMMAND ${Python3_EXECUTABLE}
                   tests/integration/benchmark_harness_defaults_test.py)
  set_tests_properties(IntegrationBenchmarkHarnessDefaults PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 30)

  add_test(NAME IntegrationDecodeTraceParity
           COMMAND ${Python3_EXECUTABLE}
                   tests/integration/decode_trace_parity_test.py)
  set_tests_properties(IntegrationDecodeTraceParity PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 30)

  if(ENABLE_CUDA)
    add_test(NAME IntegrationFirstTokenParityProbe
             COMMAND ${Python3_EXECUTABLE}
                     tests/integration/first_token_parity_probe_test.py)
    set_tests_properties(IntegrationFirstTokenParityProbe PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      TIMEOUT 180
      ENVIRONMENT "INFERFLUX_FIRST_TOKEN_PROBE_BIN=${CMAKE_BINARY_DIR}/inferflux_first_token_probe")
  endif()

  add_test(NAME SSECancel
           COMMAND ${Python3_EXECUTABLE} tests/integration/sse_cancel_test.py)
  set_tests_properties(SSECancel PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 60
    ENVIRONMENT "INFERFLUX_SERVER_BIN=${_INFERFLUX_SERVER_BIN}")

  add_test(NAME ShmSmoke
           COMMAND ${Python3_EXECUTABLE} tests/integration/shm_smoke_test.py)
  set_tests_properties(ShmSmoke PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 60
    ENVIRONMENT "INFERFLUX_SERVER_BIN=${_INFERFLUX_SERVER_BIN}")

  if(NOT "$ENV{INFERFLUX_MODEL_PATH}" STREQUAL "")
    add_test(NAME IntegrationSSE COMMAND ${Python3_EXECUTABLE} tests/integration/sse_metrics_test.py)
    set_tests_properties(IntegrationSSE PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      ENVIRONMENT "INFERFLUX_SERVER_BIN=${_INFERFLUX_SERVER_BIN}")
  else()
    message(STATUS "Skipping IntegrationSSE test (INFERFLUX_MODEL_PATH not set)")
  endif()

  if(ENABLE_SBOM)
    set(SBOM_OUT_DIR "${CMAKE_BINARY_DIR}/sbom")
    add_custom_target(sbom
      COMMAND ${CMAKE_COMMAND} -E make_directory ${SBOM_OUT_DIR}
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_sbom.py
              --source-dir  ${CMAKE_CURRENT_SOURCE_DIR}
              --version     ${PROJECT_VERSION}
              --out-dir     ${SBOM_OUT_DIR}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Generating SBOM (CycloneDX + SPDX) → ${SBOM_OUT_DIR}"
      VERBATIM
    )
    message(STATUS "SBOM target registered — run: cmake --build build --target sbom")
  endif()
else()
  message(STATUS "Python3 interpreter not found; skipping integration tests and SBOM")
endif()
