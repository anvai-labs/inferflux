#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/cuda/native/gguf_util.h"
#include "runtime/backends/cuda/native/model_loader.h"
#include "runtime/backends/cuda/native/quantization_handler.h"

#include <cstring>
#include <fstream>
#include <numeric>
#include <thread>
#include <vector>

// Forward declarations for CUDA block structures (when CUDA is available)
#ifdef INFERFLUX_HAS_CUDA
// Block structure constants from ggml-common.h
#define QK_K 256
#define K_SCALE_SIZE 12
#define QK8_0 32

// Block structure declarations (from dequantization.cuh)
extern "C" {
typedef struct {
  unsigned short d;
  unsigned short dmin;
  unsigned char scales[K_SCALE_SIZE];
  unsigned char qs[QK_K / 2];
} block_q4_k;

typedef struct {
  unsigned short d;
  unsigned short dmin;
  unsigned char scales[K_SCALE_SIZE];
  unsigned char qs[QK_K / 8];
  unsigned char qh[QK_K / 2];
} block_q5_k;

typedef struct {
  unsigned short d;
  unsigned char scales[QK_K / 4];
  unsigned char qs[QK_K / 2];
  unsigned char qh[QK_K / 16];
} block_q6_k;

typedef struct {
  unsigned short d;
  signed char qs[QK8_0];
} block_q8_0;
}
#endif

namespace fs = std::filesystem;
using namespace inferflux::runtime::cuda::native;

namespace {

// Helper to create a temporary directory
fs::path CreateTempDir(const std::string &suffix) {
  const auto base =
      fs::temp_directory_path() / ("inferflux_test_" + suffix + "_" +
                                   std::to_string(std::hash<std::thread::id>{}(
                                       std::this_thread::get_id())));
  fs::create_directories(base);
  return base;
}

// Helper to write a binary file
bool WriteBinaryFile(const fs::path &path, const void *data, size_t size) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    return false;
  }
  out.write(static_cast<const char *>(data), size);
  return out.good();
}

// Helper to create a minimal GGUF header
std::vector<uint8_t> CreateGGUFHeader(uint32_t tensor_count = 1,
                                      uint32_t kv_count = 0) {
  std::vector<uint8_t> header(24); // GGUF v3 header is 24 bytes

  // Magic (4 bytes): "GGUF"
  memcpy(header.data(), "GGUF", 4);

  // Version (4 bytes) at offset 4
  uint32_t version = 3;
  memcpy(header.data() + 4, &version, 4);

  // Tensor count (8 bytes) at offset 8
  int64_t tensors_i64 = tensor_count;
  memcpy(header.data() + 8, &tensors_i64, 8);

  // KV count (8 bytes) at offset 16
  int64_t kv_i64 = kv_count;
  memcpy(header.data() + 16, &kv_i64, 8);

  return header;
}

} // namespace

//==============================================================================
// GGUF Utility Tests
//==============================================================================

TEST_CASE("GGUF: Value type to string conversion", "[gguf][gguf_util]") {
  REQUIRE(ValueTypeToString(GGUF::ValueType::UINT8) == "uint8");
  REQUIRE(ValueTypeToString(GGUF::ValueType::FLOAT32) == "float32");
  REQUIRE(ValueTypeToString(GGUF::ValueType::STRING) == "string");
  REQUIRE(ValueTypeToString(GGUF::ValueType::ARRAY) == "array");
}

TEST_CASE("GGUF: Tensor type to string conversion", "[gguf][gguf_util]") {
  REQUIRE(TensorTypeToString(GGUF::TensorType::F32) == "f32");
  REQUIRE(TensorTypeToString(GGUF::TensorType::F16) == "f16");
  REQUIRE(TensorTypeToString(GGUF::TensorType::Q4_K) == "q4_k");
  REQUIRE(TensorTypeToString(GGUF::TensorType::Q5_K) == "q5_k");
  REQUIRE(TensorTypeToString(GGUF::TensorType::Q6_K) == "q6_k");
}

TEST_CASE("GGUF: String to tensor type conversion", "[gguf][gguf_util]") {
  REQUIRE(StringToTensorType("f32") == GGUF::TensorType::F32);
  REQUIRE(StringToTensorType("q4_k") == GGUF::TensorType::Q4_K);
  REQUIRE(StringToTensorType("q4_k_m") == GGUF::TensorType::Q4_K);
  REQUIRE(StringToTensorType("q5_k_m") == GGUF::TensorType::Q5_K);
  REQUIRE(StringToTensorType("q6_k") == GGUF::TensorType::Q6_K);
  REQUIRE(StringToTensorType("q6_k_m") == GGUF::TensorType::Q6_K);
  REQUIRE(StringToTensorType("q8_k") == GGUF::TensorType::Q8_K);
  REQUIRE(StringToTensorType("q8_k_m") == GGUF::TensorType::Q8_K);
}

TEST_CASE("GGUF: Is quantized type detection", "[gguf][gguf_util]") {
  REQUIRE(IsQuantizedType(GGUF::TensorType::F32) == false);
  REQUIRE(IsQuantizedType(GGUF::TensorType::F16) == false);
  REQUIRE(IsQuantizedType(GGUF::TensorType::Q4_K) == true);
  REQUIRE(IsQuantizedType(GGUF::TensorType::Q5_K) == true);
  REQUIRE(IsQuantizedType(GGUF::TensorType::Q6_K) == true);
}

TEST_CASE("GGUF: Get quantization type string", "[gguf][gguf_util]") {
  REQUIRE(GetQuantizationType(GGUF::TensorType::Q4_0) == "q4_0");
  REQUIRE(GetQuantizationType(GGUF::TensorType::Q4_1) == "q4_1");
  REQUIRE(GetQuantizationType(GGUF::TensorType::Q4_K) == "q4_k_m");
  REQUIRE(GetQuantizationType(GGUF::TensorType::Q5_0) == "q5_0");
  REQUIRE(GetQuantizationType(GGUF::TensorType::Q5_1) == "q5_1");
  REQUIRE(GetQuantizationType(GGUF::TensorType::Q5_K) == "q5_k_m");
  REQUIRE(GetQuantizationType(GGUF::TensorType::Q6_K) == "q6_k");
  REQUIRE(GetQuantizationType(GGUF::TensorType::Q8_0) == "q8_0");
  REQUIRE(GetQuantizationType(GGUF::TensorType::Q8_1) == "q8_1");
  REQUIRE(GetQuantizationType(GGUF::TensorType::F16) == "");
  REQUIRE(GetQuantizationType(GGUF::TensorType::F32) == "");
}

TEST_CASE("GGUF: Calculate tensor size", "[gguf][gguf_util]") {
  // Q4_K: 256 values per block, 144 bytes per block
  std::vector<uint64_t> shape_256 = {256, 1}; // 256 values
  size_t size_256 = CalcTensorSize(GGUF::TensorType::Q4_K, shape_256);
  REQUIRE(size_256 == 144); // 1 block of Q4_K

  std::vector<uint64_t> shape_512 = {512, 1}; // 512 values
  size_t size_512 = CalcTensorSize(GGUF::TensorType::Q4_K, shape_512);
  REQUIRE(size_512 == 288); // 2 blocks of Q4_K

  // F16: 2 bytes per value
  std::vector<uint64_t> shape_f16 = {1024, 1};
  size_t size_f16 = CalcTensorSize(GGUF::TensorType::F16, shape_f16);
  REQUIRE(size_f16 == 2048); // 1024 * 2 bytes

  // F32: 4 bytes per value
  std::vector<uint64_t> shape_f32 = {512, 1};
  size_t size_f32 = CalcTensorSize(GGUF::TensorType::F32, shape_f32);
  REQUIRE(size_f32 == 2048); // 512 * 4 bytes
}

TEST_CASE("GGUF: Tensor name parsing", "[gguf][gguf_util]") {
  // Token embeddings
  auto parts1 = GGUFReader::ParseTensorName("tok_emb.weight");
  REQUIRE(parts1.layer == -1);
  REQUIRE(parts1.component == "tok_emb");
  REQUIRE(parts1.type == "weight");

  // Output layer
  auto parts2 = GGUFReader::ParseTensorName("output.weight");
  REQUIRE(parts2.layer == -2);
  REQUIRE(parts2.component == "output");
  REQUIRE(parts2.type == "weight");

  // Layer 0 attention
  auto parts3 = GGUFReader::ParseTensorName("blk.0.attn_q.weight");
  REQUIRE(parts3.layer == 0);
  REQUIRE(parts3.component == "attn_q");
  REQUIRE(parts3.type == "weight");

  // Layer 5 FFN
  auto parts4 = GGUFReader::ParseTensorName("blk.5.ffn_gate.weight");
  REQUIRE(parts4.layer == 5);
  REQUIRE(parts4.component == "ffn_gate");
  REQUIRE(parts4.type == "weight");
}

TEST_CASE("GGUF: Tensor name mapping for Qwen2", "[gguf][gguf_util]") {
  // Token embeddings
  REQUIRE(GGUFReader::MapTensorName("tok_emb.weight", "qwen2") ==
          "model.embed_tokens.weight");

  // Layer 0 attention
  REQUIRE(GGUFReader::MapTensorName("blk.0.attn_q.weight", "qwen2") ==
          "model.layers.0.self_attn.q_proj.weight");

  // Layer 0 FFN
  REQUIRE(GGUFReader::MapTensorName("blk.0.ffn_gate.weight", "qwen2") ==
          "model.layers.0.mlp.gate_proj.weight");

  // Layer norms
  REQUIRE(GGUFReader::MapTensorName("blk.0.attn_norm.weight", "qwen2") ==
          "model.layers.0.input_layernorm.weight");
  REQUIRE(GGUFReader::MapTensorName("blk.0.ffn_norm.weight", "qwen2") ==
          "model.layers.0.post_attention_layernorm.weight");

  // Non-zero layer mapping is generated dynamically (not only layer 0).
  REQUIRE(GGUFReader::MapTensorName("blk.5.attn_q.weight", "qwen2") ==
          "model.layers.5.self_attn.q_proj.weight");
  REQUIRE(GGUFReader::MapTensorName("blk.12.ffn_down.weight", "qwen2") ==
          "model.layers.12.mlp.down_proj.weight");
  REQUIRE(GGUFReader::MapTensorName("output_norm.weight", "qwen2") ==
          "model.norm.weight");
}

TEST_CASE("GGUF: Tensor name mapping for Llama", "[gguf][gguf_util]") {
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

//==============================================================================
// Quantization Handler Tests
//==============================================================================

TEST_CASE("Quantization: Handler registration", "[quantization][handler]") {
  // Check that handlers are registered
  auto &registry = QuantizationHandlerRegistry::Instance();

  REQUIRE(registry.IsRegistered("q4_k_m"));
  REQUIRE(registry.IsRegistered("q4_k"));
  REQUIRE(registry.IsRegistered("q5_k_m"));
  REQUIRE(registry.IsRegistered("q5_k"));
  REQUIRE(registry.IsRegistered("q6_k"));
  REQUIRE(registry.IsRegistered("q6_k_m"));
  REQUIRE(registry.IsRegistered("q8_0"));
  REQUIRE(registry.IsRegistered("q8_k"));
  REQUIRE(registry.IsRegistered("q8_k_m"));
}

TEST_CASE("Quantization: Handler creation", "[quantization][handler]") {
  auto handler_q4 = CreateQuantizationHandler("q4_k_m");
  REQUIRE(handler_q4 != nullptr);
  REQUIRE(handler_q4->GetType() == "q4_k_m");
  REQUIRE(handler_q4->GetBitsPerValue() == Catch::Approx(4.5));

  auto handler_q5 = CreateQuantizationHandler("q5_k_m");
  REQUIRE(handler_q5 != nullptr);
  REQUIRE(handler_q5->GetType() == "q5_k_m");
  REQUIRE(handler_q5->GetBitsPerValue() == Catch::Approx(5.5));

  auto handler_q6 = CreateQuantizationHandler("q6_k");
  REQUIRE(handler_q6 != nullptr);
  REQUIRE(handler_q6->GetType() == "q6_k");
  REQUIRE(handler_q6->GetBitsPerValue() == Catch::Approx(6.5625));

  auto handler_q6_m = CreateQuantizationHandler("q6_k_m");
  REQUIRE(handler_q6_m != nullptr);
  REQUIRE(handler_q6_m->GetType() == "q6_k");
  REQUIRE(handler_q6_m->GetBitsPerValue() == Catch::Approx(6.5625));

  auto handler_q8 = CreateQuantizationHandler("q8_0");
  REQUIRE(handler_q8 != nullptr);
  REQUIRE(handler_q8->GetType() == "q8_0");
  REQUIRE(handler_q8->GetBitsPerValue() == Catch::Approx(8.5));

  auto handler_q8_k = CreateQuantizationHandler("q8_k");
  REQUIRE(handler_q8_k != nullptr);
  REQUIRE(handler_q8_k->GetType() == "q8_k");
  REQUIRE(handler_q8_k->GetBitsPerValue() == Catch::Approx(9.125));

  auto handler_q8_k_m = CreateQuantizationHandler("q8_k_m");
  REQUIRE(handler_q8_k_m != nullptr);
  REQUIRE(handler_q8_k_m->GetType() == "q8_k");
  REQUIRE(handler_q8_k_m->GetBitsPerValue() == Catch::Approx(9.125));

  // Non-quantized handler
  auto handler_none = CreateQuantizationHandler("none");
  REQUIRE(handler_none != nullptr);
  REQUIRE(handler_none->GetType() == "none");
}

TEST_CASE("Quantization: Get dequantized size", "[quantization][handler]") {
  auto handler_q4 = CreateQuantizationHandler("q4_k_m");
  REQUIRE(handler_q4 != nullptr);

  // Q4_K_M: 144 bytes per 256 values → 512 bytes (256 * 2)
  REQUIRE(handler_q4->GetDequantizedSize(144) == 512);

  auto handler_q5 = CreateQuantizationHandler("q5_k_m");
  REQUIRE(handler_q5 != nullptr);

  // Q5_K_M: 176 bytes per 256 values → 512 bytes
  REQUIRE(handler_q5->GetDequantizedSize(176) == 512);

  auto handler_q6 = CreateQuantizationHandler("q6_k");
  REQUIRE(handler_q6 != nullptr);

  // Q6_K: 210 bytes per 256 values → 512 bytes
  REQUIRE(handler_q6->GetDequantizedSize(210) == 512);

  auto handler_q6_m = CreateQuantizationHandler("q6_k_m");
  REQUIRE(handler_q6_m != nullptr);
  REQUIRE(handler_q6_m->GetDequantizedSize(210) == 512);

  auto handler_q8_k = CreateQuantizationHandler("q8_k");
  REQUIRE(handler_q8_k != nullptr);

  // Q8_K: 292 bytes per 256 values → 512 bytes
  REQUIRE(handler_q8_k->GetDequantizedSize(292) == 512);

  auto handler_q8_k_m = CreateQuantizationHandler("q8_k_m");
  REQUIRE(handler_q8_k_m != nullptr);
  REQUIRE(handler_q8_k_m->GetDequantizedSize(292) == 512);
}

TEST_CASE("Quantization: Block size calculation", "[quantization][handler]") {
  // Block sizes from ggml-common.h
  REQUIRE(BaseQuantizationHandler::GetBlockSize("q4_0") == 32);
  REQUIRE(BaseQuantizationHandler::GetBlockSize("q4_1") == 32);
  REQUIRE(BaseQuantizationHandler::GetBlockSize("q5_0") == 32);
  REQUIRE(BaseQuantizationHandler::GetBlockSize("q5_1") == 32);
  REQUIRE(BaseQuantizationHandler::GetBlockSize("q8_0") == 32);

  REQUIRE(BaseQuantizationHandler::GetBlockSize("q2_k") == 256);
  REQUIRE(BaseQuantizationHandler::GetBlockSize("q3_k") == 256);
  REQUIRE(BaseQuantizationHandler::GetBlockSize("q4_k") == 256);
  REQUIRE(BaseQuantizationHandler::GetBlockSize("q5_k") == 256);
  REQUIRE(BaseQuantizationHandler::GetBlockSize("q6_k") == 256);
  REQUIRE(BaseQuantizationHandler::GetBlockSize("q6_k_m") == 256);
  REQUIRE(BaseQuantizationHandler::GetBlockSize("q8_k") == 256);
  REQUIRE(BaseQuantizationHandler::GetBlockSize("q8_k_m") == 256);
}

TEST_CASE("Quantization: Quantized size calculation",
          "[quantization][handler]") {
  // Calculate quantized size for different types
  size_t num_elements = 256; // One block

  // Q4_K_M: 144 bytes per 256 values
  size_t q4_k_m_size =
      BaseQuantizationHandler::GetQuantizedSize(num_elements, "q4_k_m");
  REQUIRE(q4_k_m_size == 144);

  // Q5_K_M: 176 bytes per 256 values
  size_t q5_k_m_size =
      BaseQuantizationHandler::GetQuantizedSize(num_elements, "q5_k_m");
  REQUIRE(q5_k_m_size == 176);

  // Q6_K: 210 bytes per 256 values
  size_t q6_k_size =
      BaseQuantizationHandler::GetQuantizedSize(num_elements, "q6_k");
  REQUIRE(q6_k_size == 210);

  size_t q6_k_m_size =
      BaseQuantizationHandler::GetQuantizedSize(num_elements, "q6_k_m");
  REQUIRE(q6_k_m_size == 210);

  // Q8_0: 34 bytes per 32 values
  // For 256 values: 8 blocks * 34 bytes = 272 bytes
  size_t q8_0_size =
      BaseQuantizationHandler::GetQuantizedSize(num_elements, "q8_0");
  REQUIRE(q8_0_size == 272);

  // Q8_K: 320 bytes per 256 values
  size_t q8_k_size =
      BaseQuantizationHandler::GetQuantizedSize(num_elements, "q8_k");
  REQUIRE(q8_k_size == 292);

  size_t q8_k_m_size =
      BaseQuantizationHandler::GetQuantizedSize(num_elements, "q8_k_m");
  REQUIRE(q8_k_m_size == 292);

  // Non-quantized should use FP16 size
  size_t fp16_size =
      BaseQuantizationHandler::GetQuantizedSize(num_elements, "f16");
  REQUIRE(fp16_size == 512); // 256 * 2 bytes
}

//==============================================================================
// Model Loader Factory Tests
//==============================================================================

TEST_CASE("ModelLoader: Factory detects format from path",
          "[model_loader][factory]") {
  const auto temp_dir = CreateTempDir("model_loader");

  // Create safetensors directory marker
  auto safetensors_dir = temp_dir / "safetensors_test";
  fs::create_directories(safetensors_dir);

  // Create config.json (required for safetensors detection)
  auto config_path = safetensors_dir / "config.json";
  {
    std::ofstream out(config_path);
    out << "{}";
  }

  // Test safetensors detection
  auto loader1 = CreateModelLoader(safetensors_dir);
  REQUIRE(loader1 != nullptr);
  REQUIRE(loader1->GetFormat() == "safetensors");

  // Test GGUF file detection
  auto gguf_file = temp_dir / "model.gguf";
  {
    auto header = CreateGGUFHeader();
    WriteBinaryFile(gguf_file, header.data(), header.size());
  }

  auto loader2 = CreateModelLoader(gguf_file);
  REQUIRE(loader2 != nullptr);
  REQUIRE(loader2->GetFormat() == "gguf");

  // Cleanup
  fs::remove_all(temp_dir);
}

TEST_CASE("ModelLoader: Factory returns null for unknown format",
          "[model_loader][factory]") {
  const auto temp_dir = CreateTempDir("unknown_format");

  // Create directory with no recognizable files
  auto loader = CreateModelLoader(temp_dir);
  REQUIRE(loader == nullptr);

  fs::remove_all(temp_dir);
}

//==============================================================================
// Q4_K_M Block Structure Tests (CUDA only)
//==============================================================================

#ifdef INFERFLUX_HAS_CUDA

TEST_CASE("Q4_K_M: Block structure size", "[gguf][block_q4_k]") {
  // Block structure from ggml-common.h
  REQUIRE(sizeof(block_q4_k) == 2 * sizeof(half) + K_SCALE_SIZE + QK_K / 2);

  // Verify: 4 + 12 + 128 = 144 bytes
  REQUIRE(2 * sizeof(half) == 4); // d + dmin
  REQUIRE(K_SCALE_SIZE == 12);    // scales + mins
  REQUIRE(QK_K / 2 == 128);       // quants

  size_t expected = 4 + 12 + 128;
  REQUIRE(sizeof(block_q4_k) == expected);
}

TEST_CASE("Q5_K_M: Block structure size", "[gguf][block_q5_k]") {
  // Verify: 4 + 12 + 32 + 128 = 176 bytes
  REQUIRE(sizeof(block_q5_k) ==
          2 * sizeof(half) + K_SCALE_SIZE + QK_K / 8 + QK_K / 2);

  size_t expected = 4 + 12 + 32 + 128;
  REQUIRE(sizeof(block_q5_k) == expected);
}

TEST_CASE("Q6_K: Block structure size", "[gguf][block_q6_k]") {
  // Verify: 2 + 128 + 64 + 16 = 210 bytes
  REQUIRE(sizeof(block_q6_k) == sizeof(half) + QK_K / 2 + QK_K / 4 + QK_K / 16);

  size_t expected = 2 + 128 + 64 + 16;
  REQUIRE(sizeof(block_q6_k) == expected);
}

TEST_CASE("Q8_0: Block structure size", "[gguf][block_q8_0]") {
  // Verify: 2 + 32 = 34 bytes
  REQUIRE(sizeof(block_q8_0) == sizeof(half) + QK8_0);

  size_t expected = 2 + 32;
  REQUIRE(sizeof(block_q8_0) == expected);
}

#endif // INFERFLUX_HAS_CUDA

//==============================================================================
// Dequantization Tests (CPU-side validation)
//==============================================================================

TEST_CASE("Dequantization: Q4_K_M bits per value", "[gguf][dequantization]") {
  auto handler = CreateQuantizationHandler("q4_k_m");
  REQUIRE(handler != nullptr);

  // Q4_K_M: 256 values stored in 144 bytes
  // Total bits = 144 * 8 = 1152 bits
  // Bits per value = 1152 / 256 = 4.5
  double bpp = handler->GetBitsPerValue();
  REQUIRE(bpp == Catch::Approx(4.5));
}

TEST_CASE("Dequantization: Q5_K_M bits per value", "[gguf][dequantization]") {
  auto handler = CreateQuantizationHandler("q5_k_m");
  REQUIRE(handler != nullptr);

  // Q5_K_M: 256 values stored in 176 bytes
  // Total bits = 176 * 8 = 1408 bits
  // Bits per value = 1408 / 256 = 5.5
  double bpp = handler->GetBitsPerValue();
  REQUIRE(bpp == Catch::Approx(5.5));
}

TEST_CASE("Dequantization: Q6_K bits per value", "[gguf][dequantization]") {
  auto handler = CreateQuantizationHandler("q6_k");
  REQUIRE(handler != nullptr);

  // Q6_K: 256 values stored in 210 bytes
  // Total bits = 210 * 8 = 1680 bits
  // Bits per value = 1680 / 256 = 6.5625
  double bpp = handler->GetBitsPerValue();
  REQUIRE(bpp == Catch::Approx(6.5625));
}

TEST_CASE("Dequantization: Q8_0 bits per value", "[gguf][dequantization]") {
  auto handler = CreateQuantizationHandler("q8_0");
  REQUIRE(handler != nullptr);

  // Q8_0: 32 values stored in 34 bytes
  // Total bits = 34 * 8 = 272 bits
  // Bits per value = 272 / 32 = 8.5
  double bpp = handler->GetBitsPerValue();
  REQUIRE(bpp == Catch::Approx(8.5));
}

//==============================================================================
// Memory Efficiency Tests
//==============================================================================

TEST_CASE("Memory: Quantization compression ratio", "[gguf][memory]") {
  // For a 3B parameter model:
  // BF16: 3B * 2 bytes = 6 GB
  // Q6_K: 3B * (210/256 * 2) ≈ 4.9 GB
  // Q5_K_M: 3B * (176/256 * 2) ≈ 4.1 GB
  // Q4_K_M: 3B * (144/256 * 2) ≈ 3.4 GB

  size_t params_3b = 3ULL * 1000 * 1000 * 1000;

  size_t bf16_size = params_3b * 2; // 2 bytes per param
  size_t q4_k_m_size = params_3b * 144 / 256 * 2;
  size_t q5_k_m_size = params_3b * 176 / 256 * 2;
  size_t q6_k_size = params_3b * 210 / 256 * 2;

  // Verify compression ratios
  double q4_ratio = (double)bf16_size / q4_k_m_size;
  double q5_ratio = (double)bf16_size / q5_k_m_size;
  double q6_ratio = (double)bf16_size / q6_k_size;

  // Q4_K_M: 2 / (144/256*2) = 2 / 1.125 = 1.777... (16/9)
  REQUIRE(q4_ratio == Catch::Approx(1.7777).epsilon(0.01));
  // Q5_K_M: 2 / (176/256*2) = 2 / 1.375 = 1.4545... (32/22)
  REQUIRE(q5_ratio == Catch::Approx(1.4545).epsilon(0.01));
  // Q6_K: 2 / (210/256*2) = 2 / 1.640625 = 1.2195... (512/420)
  REQUIRE(q6_ratio == Catch::Approx(1.2195).epsilon(0.01));
}

TEST_CASE("Memory: Model size comparison for 3B model", "[gguf][memory]") {
  size_t params_3b = 3ULL * 1000 * 1000 * 1000;

  size_t bf16_gb = (params_3b * 2) / (1024 * 1024 * 1024);
  size_t q4_gb = (params_3b * 144 / 256 * 2) / (1024 * 1024 * 1024);
  size_t q5_gb = (params_3b * 176 / 256 * 2) / (1024 * 1024 * 1024);
  size_t q6_gb = (params_3b * 210 / 256 * 2) / (1024 * 1024 * 1024);

  // BF16: ~6 GB
  REQUIRE(bf16_gb >= 5);
  REQUIRE(bf16_gb <= 7);

  // Q4_K_M: ~3.4 GB
  REQUIRE(q4_gb >= 3);
  REQUIRE(q4_gb <= 4);

  // Q5_K_M: ~3.8 GB (not 4.1 GB as originally estimated)
  REQUIRE(q5_gb >= 3);
  REQUIRE(q5_gb <= 4);

  // Q6_K: ~4.5 GB (not 4.9 GB as originally estimated)
  REQUIRE(q6_gb >= 4);
  REQUIRE(q6_gb <= 5);
}
