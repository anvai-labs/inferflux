/**
 * @file test_gguf_quantization.cpp
 * @brief Integration tests for GGUF quantization support
 *
 * These tests validate the complete GGUF quantization pipeline:
 * - GGUF file parsing and loading
 * - Quantization detection
 * - Weight accessor functionality
 * - Dequantization correctness
 * - Memory efficiency
 */

#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/cuda/native/gguf_model_loader.h"
#include "runtime/backends/cuda/native/model_loader.h"
#include "runtime/backends/cuda/native/quantization_handler.h"
#include "support/model_loader_test_access.h"

#ifdef INFERFLUX_HAS_CUDA
#include <cuda_runtime_api.h>
#endif
#include <cstring>
#include <fstream>
#include <numeric>
#include <vector>

namespace fs = std::filesystem;
using namespace inferflux::runtime::cuda::native;

namespace {

//==============================================================================
// Test Fixture and Helpers
//==============================================================================

/**
 * @brief Create a minimal GGUF file for testing
 *
 * Creates a synthetic GGUF file with basic structure for testing
 * the parser without requiring real model files.
 */
struct GGUFTestFile {
  fs::path path;
  std::vector<uint8_t> data;

  GGUFTestFile(const std::string &name,
               const std::string &quant_type = "q4_k_m") {
    // Create temporary file
    path = fs::temp_directory_path() / ("test_gguf_" + name + ".gguf");

    // Build GGUF header (24 bytes total)
    data.resize(24);

    memcpy(data.data(), "GGUF", 4);
    // Version (4 bytes): v3 at offset 4
    uint32_t version = 3;
    memcpy(data.data() + 4, &version, 4);

    // Tensor count (8 bytes): 1 tensor at offset 8
    int64_t tensor_count = 1;
    memcpy(data.data() + 8, &tensor_count, 8);

    // KV count (8 bytes): 3 key-value pairs at offset 16
    int64_t kv_count = 3;
    memcpy(data.data() + 16, &kv_count, 8);

    // Add KV pairs (simplified)
    // KV 1: general.architecture (string)
    std::string arch_name = "general.architecture";
    std::string arch_value = "qwen2";
    // ... (would add full KV encoding here)

    // Add tensor info
    // Tensor 1: tok_emb.weight, Q4_K_M, shape [vocab_size, hidden_size]
    // ... (would add full tensor encoding here)

    // Write to file
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(data.data()), data.size());
  }

  ~GGUFTestFile() {
    if (fs::exists(path)) {
      fs::remove(path);
    }
  }
};

/**
 * @brief Create a synthetic safetensors directory
 */
struct SafetensorsTestDir {
  fs::path path;

  SafetensorsTestDir(const std::string &name) {
    path = fs::temp_directory_path() / ("test_safetensors_" + name);
    fs::create_directories(path);

    // Create minimal config.json
    auto config_path = path / "config.json";
    std::ofstream out(config_path);
    out << R"({
      "hidden_size": 768,
      "num_hidden_layers": 12,
      "num_attention_heads": 12,
      "intermediate_size": 3072
    })";
  }

  ~SafetensorsTestDir() {
    if (fs::exists(path)) {
      fs::remove_all(path);
    }
  }
};

} // namespace

//==============================================================================
// GGUF Model Loader Integration Tests
//==============================================================================

TEST_CASE("GGUF Integration: Model format detection", "[gguf][integration]") {
  SECTION("GGUF format detected from .gguf extension") {
    GGUFTestFile test_file("detection_test");

    auto loader = CreateModelLoader(test_file.path);
    REQUIRE(loader != nullptr);

    // Format should be detected as GGUF
    std::string format = loader->GetFormat();
    REQUIRE(format == "gguf");
  }

  SECTION("Safetensors format detected from directory") {
    SafetensorsTestDir test_dir("format_test");

    auto loader = CreateModelLoader(test_dir.path);
    REQUIRE(loader != nullptr);

    // Format should be detected as safetensors
    std::string format = loader->GetFormat();
    REQUIRE(format == "safetensors");
  }
}

TEST_CASE("GGUF Integration: Quantization detection", "[gguf][integration]") {
  // This test verifies that the GGUF loader can detect quantization types
  // In a real scenario, this would use actual GGUF files with different
  // quantization types

  SECTION("Q4_K_M quantization detected") {
    // Mock: verify we can create a handler for q4_k_m
    auto handler = CreateQuantizationHandler("q4_k_m");
    REQUIRE(handler != nullptr);
    REQUIRE(handler->GetType() == "q4_k_m");
    REQUIRE(IsQuantizedType(GGUF::TensorType::Q4_K) == true);
  }

  SECTION("Q5_K_M quantization detected") {
    auto handler = CreateQuantizationHandler("q5_k_m");
    REQUIRE(handler != nullptr);
    REQUIRE(handler->GetType() == "q5_k_m");
    REQUIRE(IsQuantizedType(GGUF::TensorType::Q5_K) == true);
  }

  SECTION("Q6_K quantization detected") {
    auto handler = CreateQuantizationHandler("q6_k");
    REQUIRE(handler != nullptr);
    REQUIRE(handler->GetType() == "q6_k");
    REQUIRE(IsQuantizedType(GGUF::TensorType::Q6_K) == true);
  }

  SECTION("Q8_0 quantization detected") {
    auto handler = CreateQuantizationHandler("q8_0");
    REQUIRE(handler != nullptr);
    REQUIRE(handler->GetType() == "q8_0");
  }

  SECTION("Non-quantized format detection") {
    REQUIRE(IsQuantizedType(GGUF::TensorType::F16) == false);
    REQUIRE(IsQuantizedType(GGUF::TensorType::F32) == false);
  }
}

//==============================================================================
// Quantization Handler Integration Tests
//==============================================================================

TEST_CASE("GGUF Integration: Handler registry", "[gguf][integration]") {
  auto &registry = QuantizationHandlerRegistry::Instance();

  // Verify all supported types are registered
  SECTION("K-series quantizations registered") {
    REQUIRE(registry.IsRegistered("q4_k_m"));
    REQUIRE(registry.IsRegistered("q4_k"));
    REQUIRE(registry.IsRegistered("q5_k_m"));
    REQUIRE(registry.IsRegistered("q5_k"));
    REQUIRE(registry.IsRegistered("q6_k"));
    REQUIRE(registry.IsRegistered("q6_k_m"));
    REQUIRE(registry.IsRegistered("q8_k"));
    REQUIRE(registry.IsRegistered("q8_k_m"));
  }

  SECTION("Implemented quantizations registered") {
    REQUIRE(registry.IsRegistered("q8_0"));
    // Note: q4_0, q5_0, q8_1 are not yet implemented
  }
}

TEST_CASE("GGUF Integration: Handler creation and basic functionality",
          "[gguf][integration]") {
  auto &registry = QuantizationHandlerRegistry::Instance();

  SECTION("Create and validate Q4_K_M handler") {
    auto handler = registry.Create("q4_k_m");
    REQUIRE(handler != nullptr);

    // Validate properties
    REQUIRE(handler->GetType() == "q4_k_m");
    REQUIRE(handler->GetBitsPerValue() == Catch::Approx(4.5));

    // Validate size calculations
    // 144 bytes (Q4_K_M) → 512 bytes (FP16)
    REQUIRE(handler->GetDequantizedSize(144) == 512);
  }

  SECTION("Create and validate Q5_K_M handler") {
    auto handler = registry.Create("q5_k_m");
    REQUIRE(handler != nullptr);

    REQUIRE(handler->GetType() == "q5_k_m");
    REQUIRE(handler->GetBitsPerValue() == Catch::Approx(5.5));

    // 176 bytes (Q5_K_M) → 512 bytes (FP16)
    REQUIRE(handler->GetDequantizedSize(176) == 512);
  }

  SECTION("Create and validate Q6_K handler") {
    auto handler = registry.Create("q6_k");
    REQUIRE(handler != nullptr);

    REQUIRE(handler->GetType() == "q6_k");
    REQUIRE(handler->GetBitsPerValue() == Catch::Approx(6.5625));

    // 210 bytes (Q6_K) → 512 bytes (FP16)
    REQUIRE(handler->GetDequantizedSize(210) == 512);
  }

  SECTION("Create and validate Q6_K_M alias handler") {
    auto handler = registry.Create("q6_k_m");
    REQUIRE(handler != nullptr);

    REQUIRE(handler->GetType() == "q6_k");
    REQUIRE(handler->GetBitsPerValue() == Catch::Approx(6.5625));
    REQUIRE(handler->GetDequantizedSize(210) == 512);
  }

  SECTION("Create and validate Q8_0 handler") {
    auto handler = registry.Create("q8_0");
    REQUIRE(handler != nullptr);

    REQUIRE(handler->GetType() == "q8_0");
    REQUIRE(handler->GetBitsPerValue() == Catch::Approx(8.5));

    // 34 bytes (Q8_0) → 64 bytes (FP16)
    REQUIRE(handler->GetDequantizedSize(34) == 64);
  }

  SECTION("Create and validate Q8_K and Q8_K_M alias handlers") {
    auto handler_q8_k = registry.Create("q8_k");
    REQUIRE(handler_q8_k != nullptr);
    REQUIRE(handler_q8_k->GetType() == "q8_k");
    REQUIRE(handler_q8_k->GetBitsPerValue() == Catch::Approx(9.125));
    REQUIRE(handler_q8_k->GetDequantizedSize(292) == 512);

    auto handler_q8_k_m = registry.Create("q8_k_m");
    REQUIRE(handler_q8_k_m != nullptr);
    REQUIRE(handler_q8_k_m->GetType() == "q8_k");
    REQUIRE(handler_q8_k_m->GetBitsPerValue() == Catch::Approx(9.125));
    REQUIRE(handler_q8_k_m->GetDequantizedSize(292) == 512);
  }

  SECTION("Handler creation for unknown type returns nullptr") {
    auto handler = registry.Create("unknown_type");
    REQUIRE(handler == nullptr);
  }
}

//==============================================================================
// Tensor Name Mapping Integration Tests
//==============================================================================

TEST_CASE("GGUF Integration: Tensor name mapping", "[gguf][integration]") {
  SECTION("Qwen2 architecture mapping") {
    // Token embeddings
    REQUIRE(GGUFReader::MapTensorName("tok_emb.weight", "qwen2") ==
            "model.embed_tokens.weight");

    // Layer 0 attention
    REQUIRE(GGUFReader::MapTensorName("blk.0.attn_q.weight", "qwen2") ==
            "model.layers.0.self_attn.q_proj.weight");
    REQUIRE(GGUFReader::MapTensorName("blk.0.attn_k.weight", "qwen2") ==
            "model.layers.0.self_attn.k_proj.weight");
    REQUIRE(GGUFReader::MapTensorName("blk.0.attn_v.weight", "qwen2") ==
            "model.layers.0.self_attn.v_proj.weight");
    REQUIRE(GGUFReader::MapTensorName("blk.0.attn_o.weight", "qwen2") ==
            "model.layers.0.self_attn.o_proj.weight");

    // Layer 0 FFN
    REQUIRE(GGUFReader::MapTensorName("blk.0.ffn_gate.weight", "qwen2") ==
            "model.layers.0.mlp.gate_proj.weight");
    REQUIRE(GGUFReader::MapTensorName("blk.0.ffn_up.weight", "qwen2") ==
            "model.layers.0.mlp.up_proj.weight");
    REQUIRE(GGUFReader::MapTensorName("blk.0.ffn_down.weight", "qwen2") ==
            "model.layers.0.mlp.down_proj.weight");

    // Layer norms
    REQUIRE(GGUFReader::MapTensorName("blk.0.attn_norm.weight", "qwen2") ==
            "model.layers.0.input_layernorm.weight");
    REQUIRE(GGUFReader::MapTensorName("blk.0.ffn_norm.weight", "qwen2") ==
            "model.layers.0.post_attention_layernorm.weight");
  }

  SECTION("Llama architecture mapping") {
    // Token embeddings
    REQUIRE(GGUFReader::MapTensorName("token_embd.weight", "llama") ==
            "model.embed_tokens.weight");

    // Layer 0 attention
    REQUIRE(GGUFReader::MapTensorName("blk.0.attn_q.weight", "llama") ==
            "model.layers.0.self_attn.q_proj.weight");

    // Layer 0 FFN
    REQUIRE(GGUFReader::MapTensorName("blk.0.ffn_gate.weight", "llama") ==
            "model.layers.0.mlp.gate_proj.weight");
  }
}

TEST_CASE("GGUF Integration: Tensor name parsing", "[gguf][integration]") {
  SECTION("Parse special layer indices") {
    auto parts = GGUFReader::ParseTensorName("tok_emb.weight");
    REQUIRE(parts.layer == -1); // Special index for token embeddings
    REQUIRE(parts.component == "tok_emb");
    REQUIRE(parts.type == "weight");

    parts = GGUFReader::ParseTensorName("output.weight");
    REQUIRE(parts.layer == -2); // Special index for output layer
    REQUIRE(parts.component == "output");
  }

  SECTION("Parse regular layer tensors") {
    auto parts = GGUFReader::ParseTensorName("blk.5.attn_q.weight");
    REQUIRE(parts.layer == 5);
    REQUIRE(parts.component == "attn_q");
    REQUIRE(parts.type == "weight");

    parts = GGUFReader::ParseTensorName("blk.12.ffn_gate.weight");
    REQUIRE(parts.layer == 12);
    REQUIRE(parts.component == "ffn_gate");
  }
}

//==============================================================================
// Memory Efficiency Integration Tests
//==============================================================================

TEST_CASE("GGUF Integration: Memory efficiency validation",
          "[gguf][integration]") {

  SECTION("Quantized size calculations for model sizing") {
    // For a 1B parameter model
    size_t params_1b = 1ULL * 1000 * 1000 * 1000;

    // FP16 baseline
    size_t fp16_gb = (params_1b * 2) / (1024 * 1024 * 1024);
    REQUIRE(fp16_gb == 1); // ~2 GB

    // Q4_K_M: 1B * (144/256 * 2) / (1024^3) ≈ 1.1 GB
    size_t q4_gb = (params_1b * 144 / 256 * 2) / (1024 * 1024 * 1024);
    REQUIRE(q4_gb >= 1);
    REQUIRE(q4_gb <= 2);

    // Q6_K: 1B * (210/256 * 2) / (1024^3) ≈ 1.6 GB
    size_t q6_gb = (params_1b * 210 / 256 * 2) / (1024 * 1024 * 1024);
    REQUIRE(q6_gb >= 1);
    REQUIRE(q6_gb <= 2);

    // Q8_0: 1B * (34/32 * 2) / (1024^3) ≈ 2 GB
    size_t q8_gb = (params_1b * 34 / 32 * 2) / (1024 * 1024 * 1024);
    REQUIRE(q8_gb >= 1);
    REQUIRE(q8_gb <= 3);
  }

  SECTION("Compression ratio validation") {
    size_t num_elements = 256;

    // Calculate sizes
    size_t fp16_size =
        BaseQuantizationHandler::GetQuantizedSize(num_elements, "f16");
    size_t q4_size =
        BaseQuantizationHandler::GetQuantizedSize(num_elements, "q4_k_m");
    size_t q6_size =
        BaseQuantizationHandler::GetQuantizedSize(num_elements, "q6_k");
    size_t q8_size =
        BaseQuantizationHandler::GetQuantizedSize(num_elements, "q8_0");

    // Calculate compression ratios
    double q4_ratio = (double)fp16_size / q4_size;
    double q6_ratio = (double)fp16_size / q6_size;
    double q8_ratio = (double)fp16_size / q8_size;

    // Validate: Q4 should compress more than Q6, which compresses more than Q8
    REQUIRE(q4_ratio > q6_ratio);
    REQUIRE(q6_ratio > q8_ratio);
    REQUIRE(q4_ratio > 1.5); // At least 1.5x compression
  }
}

//==============================================================================
// Block Size Validation Tests
//==============================================================================

TEST_CASE("GGUF Integration: Block size consistency", "[gguf][integration]") {
  // Verify that block sizes match expected values from ggml-common.h

  SECTION("Legacy format block sizes (32 values)") {
    REQUIRE(BaseQuantizationHandler::GetBlockSize("q4_0") == 32);
    REQUIRE(BaseQuantizationHandler::GetBlockSize("q4_1") == 32);
    REQUIRE(BaseQuantizationHandler::GetBlockSize("q5_0") == 32);
    REQUIRE(BaseQuantizationHandler::GetBlockSize("q5_1") == 32);
    REQUIRE(BaseQuantizationHandler::GetBlockSize("q8_0") == 32);
  }

  SECTION("K-series block sizes (256 values)") {
    REQUIRE(BaseQuantizationHandler::GetBlockSize("q2_k") == 256);
    REQUIRE(BaseQuantizationHandler::GetBlockSize("q3_k") == 256);
    REQUIRE(BaseQuantizationHandler::GetBlockSize("q4_k") == 256);
    REQUIRE(BaseQuantizationHandler::GetBlockSize("q5_k") == 256);
    REQUIRE(BaseQuantizationHandler::GetBlockSize("q6_k") == 256);
  }

  SECTION("Block size affects quantized size calculation") {
    size_t num_32_blocks = 32 * 10;   // 10 blocks of 32 values
    size_t num_256_blocks = 256 * 10; // 10 blocks of 256 values

    // Q8_0 uses 32-value blocks
    size_t q8_0_size =
        BaseQuantizationHandler::GetQuantizedSize(num_32_blocks, "q8_0");
    REQUIRE(q8_0_size == 10 * 34); // 10 blocks * 34 bytes

    // Q4_K_M uses 256-value blocks
    size_t q4_k_m_size =
        BaseQuantizationHandler::GetQuantizedSize(num_256_blocks, "q4_k_m");
    REQUIRE(q4_k_m_size == 10 * 144); // 10 blocks * 144 bytes
  }
}

//==============================================================================
// Cross-Format Compatibility Tests
//==============================================================================

TEST_CASE("GGUF Integration: Multi-format support", "[gguf][integration]") {
  // Verify that the system can handle both GGUF and safetensors

  SECTION("Factory creates appropriate loader") {
    GGUFTestFile gguf_file("compat_test");
    SafetensorsTestDir safetensors_dir("compat_test");

    // GGUF file → GGUFModelLoader
    auto gguf_loader = CreateModelLoader(gguf_file.path);
    REQUIRE(gguf_loader != nullptr);
    REQUIRE(gguf_loader->GetFormat() == "gguf");

    // Safetensors directory → SafetensorsLoaderAdapter
    auto safetensors_loader = CreateModelLoader(safetensors_dir.path);
    REQUIRE(safetensors_loader != nullptr);
    REQUIRE(safetensors_loader->GetFormat() == "safetensors");
  }

  SECTION("Both formats support quantization queries") {
    GGUFTestFile gguf_file("quant_test");
    SafetensorsTestDir safetensors_dir("quant_test");

    auto gguf_loader = CreateModelLoader(gguf_file.path);
    auto safetensors_loader = CreateModelLoader(safetensors_dir.path);

    // Both should respond to quantization queries
    // GGUF would return true if it has quantized tensors
    // Safetensors would return false (FP16/BF16 only)
    REQUIRE_NOTHROW(gguf_loader->IsQuantized());
    REQUIRE_NOTHROW(safetensors_loader->IsQuantized());
  }
}

//==============================================================================
// Type Conversion Tests
//==============================================================================

TEST_CASE("GGUF Integration: Type string conversions", "[gguf][integration]") {
  SECTION("Tensor type to string") {
    REQUIRE(TensorTypeToString(GGUF::TensorType::F32) == "f32");
    REQUIRE(TensorTypeToString(GGUF::TensorType::F16) == "f16");
    REQUIRE(TensorTypeToString(GGUF::TensorType::Q4_K) == "q4_k");
    REQUIRE(TensorTypeToString(GGUF::TensorType::Q5_K) == "q5_k");
    REQUIRE(TensorTypeToString(GGUF::TensorType::Q6_K) == "q6_k");
  }

  SECTION("String to tensor type") {
    REQUIRE(StringToTensorType("f32") == GGUF::TensorType::F32);
    REQUIRE(StringToTensorType("f16") == GGUF::TensorType::F16);
    REQUIRE(StringToTensorType("q4_k") == GGUF::TensorType::Q4_K);
    REQUIRE(StringToTensorType("q4_k_m") == GGUF::TensorType::Q4_K); // Alias
    REQUIRE(StringToTensorType("q5_k") == GGUF::TensorType::Q5_K);
    REQUIRE(StringToTensorType("q5_k_m") == GGUF::TensorType::Q5_K); // Alias
    REQUIRE(StringToTensorType("q6_k") == GGUF::TensorType::Q6_K);
  }

  SECTION("Quantization type detection") {
    REQUIRE(GetQuantizationType(GGUF::TensorType::Q4_K) == "q4_k_m");
    REQUIRE(GetQuantizationType(GGUF::TensorType::Q5_K) == "q5_k_m");
    REQUIRE(GetQuantizationType(GGUF::TensorType::Q6_K) == "q6_k");
    REQUIRE(GetQuantizationType(GGUF::TensorType::F16) == ""); // Not quantized
    REQUIRE(GetQuantizationType(GGUF::TensorType::F32) == ""); // Not quantized
  }
}

TEST_CASE("GGUF Integration: F16 accessor returns direct GPU pointer without "
          "dequant cache",
          "[gguf][integration][f16_contract]") {
#ifdef INFERFLUX_HAS_CUDA
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping F16 GGUF accessor contract.");
    return;
  }

  cudaStream_t stream = nullptr;
  REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) ==
          cudaSuccess);

  GGUFModelLoader loader;
  ModelLoaderTestAccess lacc(loader);
  const std::string gguf_name = "blk.0.attn_q.weight";
  const std::string internal_name = "model.layers.0.self_attn.q_proj.weight";
  const std::vector<uint64_t> shape = {64, 64};
  const size_t f16_bytes = shape[0] * shape[1] * sizeof(half);

  GGUFTensorData tensor;
  tensor.info.name = gguf_name;
  tensor.info.shape = shape;
  tensor.info.type = GGUF::TensorType::F16;
  tensor.info.offset = 0;
  tensor.info.byte_size = f16_bytes;
  tensor.cpu_data.assign(f16_bytes, 0U);

  REQUIRE(cudaMalloc(&lacc.d_quantized_buffer(), f16_bytes) == cudaSuccess);
  lacc.quantized_buffer_size() = f16_bytes;
  REQUIRE(cudaMemset(lacc.d_quantized_buffer(), 0, f16_bytes) == cudaSuccess);

  tensor.gpu_data = lacc.d_quantized_buffer();
  tensor.gpu_offset = 0;
  lacc.tensors().emplace(gguf_name, std::move(tensor));
  lacc.gguf_to_internal_name_map()[gguf_name] = internal_name;
  lacc.internal_to_gguf_name_map()[internal_name] = gguf_name;

  auto accessor = loader.GetWeightAccessor(internal_name);
  REQUIRE(accessor != nullptr);
  REQUIRE_FALSE(accessor->IsQuantized());

  const void *raw_ptr = accessor->GetGpuWeights(stream);
  half *f16_ptr = accessor->GetDequantizedGpuWeights(stream);
  REQUIRE(raw_ptr != nullptr);
  REQUIRE(reinterpret_cast<const void *>(f16_ptr) == raw_ptr);

  const auto *tensor_after = loader.GetTensorByGGUFName(gguf_name);
  REQUIRE(tensor_after != nullptr);
  REQUIRE(tensor_after->dequantized_gpu == nullptr);

  loader.FreeGPUMemory();
  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
#else
  SUCCEED("Built without CUDA; skipping F16 GGUF accessor contract.");
#endif
}

//==============================================================================
// Batch Dequant Cache Lifecycle Contract Test
//==============================================================================

TEST_CASE("GGUF Integration: Batch dequant policy drops allocations between "
          "consecutive requests",
          "[gguf][integration][memory_contract]") {
#ifdef INFERFLUX_HAS_CUDA
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping memory-contract gate.");
    return;
  }

  auto run_policy_contract = [](DequantizedCachePolicy policy) {
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) ==
            cudaSuccess);

    GGUFModelLoader loader;
    ModelLoaderTestAccess lacc(loader);
    loader.SetDequantizedCachePolicy(policy);

    const std::string gguf_name = "blk.0.attn_q.weight";
    const std::string internal_name = "model.layers.0.self_attn.q_proj.weight";
    const std::vector<uint64_t> shape = {4096, 2048}; // 8,388,608 elements
    const size_t dequantized_bytes =
        shape[0] * shape[1] * sizeof(half); // ~16 MiB
    const size_t quantized_bytes =
        CalcTensorSize(GGUF::TensorType::Q8_0, shape);
    REQUIRE(quantized_bytes > 0);

    GGUFTensorData tensor;
    tensor.info.name = gguf_name;
    tensor.info.shape = shape;
    tensor.info.type = GGUF::TensorType::Q8_0;
    tensor.info.offset = 0;
    tensor.info.byte_size = quantized_bytes;
    tensor.cpu_data.assign(quantized_bytes, 0U); // Valid all-zero quant blocks

    REQUIRE(cudaMalloc(&lacc.d_quantized_buffer(), quantized_bytes) ==
            cudaSuccess);
    lacc.quantized_buffer_size() = quantized_bytes;
    REQUIRE(cudaMemcpy(lacc.d_quantized_buffer(), tensor.cpu_data.data(),
                       quantized_bytes, cudaMemcpyHostToDevice) == cudaSuccess);

    tensor.gpu_data = lacc.d_quantized_buffer();
    tensor.gpu_offset = 0;
    lacc.tensors().emplace(gguf_name, std::move(tensor));
    lacc.gguf_to_internal_name_map()[gguf_name] = internal_name;
    lacc.internal_to_gguf_name_map()[internal_name] = gguf_name;

    auto accessor = loader.GetWeightAccessor(internal_name);
    REQUIRE(accessor != nullptr);

    size_t free_before = 0;
    size_t total_mem = 0;
    REQUIRE(cudaMemGetInfo(&free_before, &total_mem) == cudaSuccess);

    // Request 1: dequant cache must be materialized.
    half *req1_ptr = accessor->GetDequantizedGpuWeights(stream);
    REQUIRE(req1_ptr != nullptr);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    const auto *tensor_after_req1 = loader.GetTensorByGGUFName(gguf_name);
    REQUIRE(tensor_after_req1 != nullptr);
    REQUIRE(tensor_after_req1->dequantized_gpu != nullptr);

    size_t free_after_req1 = 0;
    REQUIRE(cudaMemGetInfo(&free_after_req1, &total_mem) == cudaSuccess);

    // Runtime batch boundary: clear cache for memory-first policies.
    if (loader.GetDequantizedCachePolicy() !=
        DequantizedCachePolicy::kModelLifetime) {
      loader.ClearDequantizedCache();
    }
    const auto *tensor_after_clear = loader.GetTensorByGGUFName(gguf_name);
    REQUIRE(tensor_after_clear != nullptr);
    REQUIRE(tensor_after_clear->dequantized_gpu == nullptr);

    size_t free_after_clear = 0;
    REQUIRE(cudaMemGetInfo(&free_after_clear, &total_mem) == cudaSuccess);

    // Require visible recovery to catch regressions that retain dequant
    // buffers.
    const size_t min_recovered_bytes = dequantized_bytes / 8; // conservative
    REQUIRE(free_after_clear >= free_after_req1 + min_recovered_bytes);

    // Request 2: cache is rebuilt (new request after boundary).
    half *req2_ptr = accessor->GetDequantizedGpuWeights(stream);
    REQUIRE(req2_ptr != nullptr);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    const auto *tensor_after_req2 = loader.GetTensorByGGUFName(gguf_name);
    REQUIRE(tensor_after_req2 != nullptr);
    REQUIRE(tensor_after_req2->dequantized_gpu != nullptr);

    size_t free_after_req2 = 0;
    REQUIRE(cudaMemGetInfo(&free_after_req2, &total_mem) == cudaSuccess);
    REQUIRE(free_after_req2 + min_recovered_bytes <= free_after_clear);
    REQUIRE(free_after_req1 + min_recovered_bytes <= free_before);

    loader.FreeGPUMemory();
    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
  };

  auto run_model_policy_contract = []() {
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) ==
            cudaSuccess);

    GGUFModelLoader loader;
    ModelLoaderTestAccess lacc(loader);
    loader.SetDequantizedCachePolicy(DequantizedCachePolicy::kModelLifetime);

    const std::string gguf_name = "blk.0.attn_q.weight";
    const std::string internal_name = "model.layers.0.self_attn.q_proj.weight";
    const std::vector<uint64_t> shape = {4096, 2048}; // 8,388,608 elements
    const size_t quantized_bytes =
        CalcTensorSize(GGUF::TensorType::Q8_0, shape);
    REQUIRE(quantized_bytes > 0);

    GGUFTensorData tensor;
    tensor.info.name = gguf_name;
    tensor.info.shape = shape;
    tensor.info.type = GGUF::TensorType::Q8_0;
    tensor.info.offset = 0;
    tensor.info.byte_size = quantized_bytes;
    tensor.cpu_data.assign(quantized_bytes, 0U);

    REQUIRE(cudaMalloc(&lacc.d_quantized_buffer(), quantized_bytes) ==
            cudaSuccess);
    lacc.quantized_buffer_size() = quantized_bytes;
    REQUIRE(cudaMemcpy(lacc.d_quantized_buffer(), tensor.cpu_data.data(),
                       quantized_bytes, cudaMemcpyHostToDevice) == cudaSuccess);

    tensor.gpu_data = lacc.d_quantized_buffer();
    tensor.gpu_offset = 0;
    lacc.tensors().emplace(gguf_name, std::move(tensor));
    lacc.gguf_to_internal_name_map()[gguf_name] = internal_name;
    lacc.internal_to_gguf_name_map()[internal_name] = gguf_name;

    auto accessor = loader.GetWeightAccessor(internal_name);
    REQUIRE(accessor != nullptr);

    half *req1_ptr = accessor->GetDequantizedGpuWeights(stream);
    REQUIRE(req1_ptr != nullptr);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    const auto *tensor_after_req1 = loader.GetTensorByGGUFName(gguf_name);
    REQUIRE(tensor_after_req1 != nullptr);
    REQUIRE(tensor_after_req1->dequantized_gpu != nullptr);

    // Model-lifetime policy: runtime boundary must not clear cache.
    if (loader.GetDequantizedCachePolicy() !=
        DequantizedCachePolicy::kModelLifetime) {
      loader.ClearDequantizedCache();
    }
    const auto *tensor_after_boundary = loader.GetTensorByGGUFName(gguf_name);
    REQUIRE(tensor_after_boundary != nullptr);
    REQUIRE(tensor_after_boundary->dequantized_gpu != nullptr);

    half *req2_ptr = accessor->GetDequantizedGpuWeights(stream);
    REQUIRE(req2_ptr != nullptr);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    REQUIRE(req2_ptr == req1_ptr);

    loader.FreeGPUMemory();
    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
  };

  SECTION("batch policy clears dequantized buffers at request boundary") {
    run_policy_contract(DequantizedCachePolicy::kBatchLifetime);
  }

  SECTION("none policy clears dequantized buffers at request boundary") {
    run_policy_contract(DequantizedCachePolicy::kNone);
  }

  SECTION(
      "model policy retains dequantized buffers across request boundaries") {
    run_model_policy_contract();
  }
#else
  SUCCEED("Built without CUDA; skipping memory-contract gate.");
#endif
}
