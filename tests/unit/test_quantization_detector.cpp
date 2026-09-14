#include <catch2/catch_amalgamated.hpp>

#include "runtime/core/gguf/igguf_parser.h"
#include "server/cpu_quantization_detector.h"
#include "server/quantization_detection.h"
#include "server/startup_advisor.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

using namespace inferflux;

namespace {

void WriteU32(std::ofstream &out, uint32_t v) {
  out.write(reinterpret_cast<const char *>(&v), 4);
}
void WriteU64(std::ofstream &out, uint64_t v) {
  out.write(reinterpret_cast<const char *>(&v), 8);
}
void WriteString(std::ofstream &out, const std::string &s) {
  WriteU64(out, s.size());
  out.write(s.data(), static_cast<std::streamsize>(s.size()));
}
void WriteKvKey(std::ofstream &out, const std::string &key) {
  WriteString(out, key);
}
void WriteTensorInfo(std::ofstream &out, const std::string &name,
                     uint32_t ggml_type) {
  WriteString(out, name);
  WriteU32(out, 2); // n_dims
  WriteU64(out, 256);
  WriteU64(out, 256);
  WriteU32(out, ggml_type);
  WriteU64(out, 0); // offset
}

// Minimal GGUF v3: header + KV metadata (string, array-of-string, u32) + two
// Q4_K tensor infos. Mirrors real model files closely enough that the old
// detector (which never skipped the KV section) failed on it with
// "Tensor name too long".
void WriteSyntheticGguf(const std::filesystem::path &path) {
  std::ofstream out(path, std::ios::binary);
  WriteU32(out, 0x46554747); // "GGUF" little-endian
  WriteU32(out, 3);          // version
  WriteU64(out, 2);          // tensor_count
  WriteU64(out, 3);          // kv_count

  // KV 1: architecture string.
  WriteKvKey(out, "general.architecture");
  WriteU32(out, 8); // STRING
  WriteString(out, "llama");

  // KV 2: array of strings (exercises the ARRAY skip path).
  WriteKvKey(out, "general.tags");
  WriteU32(out, 9); // ARRAY
  WriteU32(out, 8); // element type = STRING
  WriteU64(out, 2); // count
  WriteString(out, "text");
  WriteString(out, "chat");

  // KV 3: u32 scalar.
  WriteKvKey(out, "context.length");
  WriteU32(out, 4); // UINT32
  WriteU32(out, 4096);

  WriteTensorInfo(out, "blk.0.attn_q.weight", 12); // Q4_K
  WriteTensorInfo(out, "blk.0.attn_v.weight", 12); // Q4_K
}

} // namespace

TEST_CASE("QuantizationDetector detects Q4_K_M from synthetic GGUF with KV "
          "metadata (issue #161 follow-up)",
          "[quantization][gguf]") {
  const auto path = std::filesystem::temp_directory_path() /
                    "inferflux_test_quant_detector.gguf";
  WriteSyntheticGguf(path);

  server::CpuQuantizationDetector detector;
  const auto detected = detector.DetectFromGgufMetadata(path);
  REQUIRE(detected == QuantizationType::kQ4_K_M);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST_CASE("QuantizationDetector returns Fp32 for an all-F16 GGUF",
          "[quantization][gguf]") {
  const auto path = std::filesystem::temp_directory_path() /
                    "inferflux_test_quant_detector_f16.gguf";
  std::ofstream out(path, std::ios::binary);
  WriteU32(out, 0x46554747);
  WriteU32(out, 3);
  WriteU64(out, 1);                                 // tensor_count
  WriteU64(out, 0);                                 // kv_count
  WriteTensorInfo(out, "tok_embeddings.weight", 1); // F16
  out.close();

  server::CpuQuantizationDetector detector;
  const auto detected = detector.DetectFromGgufMetadata(path);
  REQUIRE(detected == QuantizationType::kFp32);

  std::error_code ec;
  std::filesystem::remove(path, ec);
}

TEST_CASE("CpuGgufParser SkipValue advances per GGUF spec",
          "[quantization][gguf]") {
  auto parser = runtime::core::gguf::CreateCpuGgufParser();
  REQUIRE(parser != nullptr);

  std::FILE *file = tmpfile();
  REQUIRE(file != nullptr);
  // STRING "hello": 8-byte length + 5 bytes = 13.
  const uint64_t len = 5;
  fwrite(&len, 8, 1, file);
  fwrite("hello", 1, 5, file);
  // ARRAY of 2 x UINT32: u32 elem type (4), u64 count (2), 2 x 4 bytes = 20.
  const uint32_t elem_type = 4;
  const uint64_t count = 2;
  const uint32_t values[2] = {7, 9};
  fwrite(&elem_type, 4, 1, file);
  fwrite(&count, 8, 1, file);
  fwrite(values, 4, 2, file);
  fflush(file);
  rewind(file);

  long before = ftell(file);
  REQUIRE(parser->SkipValue(file, runtime::core::gguf::GgufValueType::STRING));
  REQUIRE(ftell(file) - before == 13);

  before = ftell(file);
  REQUIRE(parser->SkipValue(file, runtime::core::gguf::GgufValueType::ARRAY));
  REQUIRE(ftell(file) - before == 20);

  fclose(file);
}
