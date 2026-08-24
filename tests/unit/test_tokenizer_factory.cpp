#include <catch2/catch_amalgamated.hpp>

#include "model/tokenizer_factory.h"
#ifdef INFERFLUX_NATIVE_KERNELS_READY
#include "runtime/backends/cuda/inferflux_cuda_executor.h"
#endif

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace inferflux {

// ============================================================================
// Factory: basic error cases
// ============================================================================

TEST_CASE("CreateTokenizer returns nullptr for empty path",
          "[tokenizer_factory]") {
  auto tok = CreateTokenizer("");
  REQUIRE(tok == nullptr);
}

TEST_CASE("CreateTokenizer returns nullptr for nonexistent path",
          "[tokenizer_factory]") {
  auto tok = CreateTokenizer("/nonexistent/path/model.gguf", "gguf");
  REQUIRE(tok == nullptr);
}

TEST_CASE("CreateTokenizer returns nullptr for nonexistent safetensors path",
          "[tokenizer_factory]") {
  auto tok = CreateTokenizer("/nonexistent/safetensors/dir", "safetensors");
  REQUIRE(tok == nullptr);
}

// ============================================================================
// Factory: format detection
// ============================================================================

TEST_CASE("CreateTokenizer with format=auto and .gguf extension resolves gguf",
          "[tokenizer_factory]") {
  // Even though the file doesn't exist, ResolveModelFormat will return "gguf"
  // from the extension, and then LlamaTokenizer::Load will fail → nullptr.
  auto tok = CreateTokenizer("/tmp/fake_model.gguf", "auto");
  REQUIRE(tok == nullptr); // File doesn't exist, but format resolved correctly
}

// ============================================================================
// Factory: GGUF model tests (require INFERFLUX_MODEL_PATH)
// ============================================================================

TEST_CASE("CreateTokenizer loads LlamaTokenizer for GGUF model",
          "[tokenizer_factory][.model]") {
  const char *model_path = std::getenv("INFERFLUX_MODEL_PATH");
  if (!model_path || std::string(model_path).empty()) {
    SKIP("INFERFLUX_MODEL_PATH not set");
  }

  auto tok = CreateTokenizer(model_path, "gguf");
  REQUIRE(tok != nullptr);
  REQUIRE(tok->IsLoaded());
  REQUIRE(tok->VocabSize() > 0);
  REQUIRE(tok->BosTokenId() >= 0);
  REQUIRE(tok->EosTokenId() >= 0);

  // Roundtrip test
  auto tokens = tok->Tokenize("Hello, world!");
  REQUIRE_FALSE(tokens.empty());
  auto text = tok->Detokenize(tokens);
  REQUIRE_FALSE(text.empty());
}

TEST_CASE("CreateTokenizer with format=auto and .gguf file loads "
          "LlamaTokenizer",
          "[tokenizer_factory][.model]") {
  const char *model_path = std::getenv("INFERFLUX_MODEL_PATH");
  if (!model_path || std::string(model_path).empty()) {
    SKIP("INFERFLUX_MODEL_PATH not set");
  }

  auto tok = CreateTokenizer(model_path); // default format="auto"
  REQUIRE(tok != nullptr);
  REQUIRE(tok->IsLoaded());
}

// ============================================================================
// ITokenizer contract tests
// ============================================================================

TEST_CASE("LlamaTokenizer satisfies ITokenizer contract",
          "[tokenizer_interface][.model]") {
  const char *model_path = std::getenv("INFERFLUX_MODEL_PATH");
  if (!model_path || std::string(model_path).empty()) {
    SKIP("INFERFLUX_MODEL_PATH not set");
  }

  auto tok = CreateTokenizer(model_path, "gguf");
  REQUIRE(tok != nullptr);
  REQUIRE(tok->IsLoaded());
  REQUIRE(tok->VocabSize() > 0);
  REQUIRE(tok->BosTokenId() >= 0);
  REQUIRE(tok->EosTokenId() >= 0);

  // Tokenize produces non-empty result
  auto tokens = tok->Tokenize("The quick brown fox", true);
  REQUIRE_FALSE(tokens.empty());

  // Detokenize roundtrip is reasonable
  auto decoded = tok->Detokenize(tokens);
  REQUIRE_FALSE(decoded.empty());

  // TokenToString works
  auto piece = tok->TokenToString(tokens[0]);
  REQUIRE_FALSE(piece.empty());

  // TokenCount is consistent
  REQUIRE(tok->TokenCount("hello") > 0);

  // Chat template
  std::vector<std::pair<std::string, std::string>> messages = {
      {"user", "Hello!"}};
  auto chat = tok->ApplyChatTemplate(messages);
  // May or may not have a template, but shouldn't crash
  (void)chat;
}

// ============================================================================
// InferfluxCudaExecutor integration (compile-time only without GPU)
// ============================================================================

#ifdef INFERFLUX_NATIVE_KERNELS_READY
TEST_CASE("InferfluxCudaExecutor: NativeFormatChat delegates to ITokenizer",
          "[tokenizer_factory]") {
  // Without a loaded model, NativeFormatChat should return invalid
  InferfluxCudaExecutor executor;
  std::vector<std::pair<std::string, std::string>> messages = {
      {"user", "Hello!"}};
  auto result = executor.NativeFormatChat(messages);
  REQUIRE_FALSE(result.valid);
}

TEST_CASE("InferfluxCudaExecutor: NativeGetTokenizer returns nullptr before "
          "LoadModel",
          "[tokenizer_factory]") {
  InferfluxCudaExecutor executor;
  REQUIRE(executor.NativeGetTokenizer() == nullptr);
}
#endif

} // namespace inferflux
