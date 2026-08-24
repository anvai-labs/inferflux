#include <catch2/catch_amalgamated.hpp>

#include "model/gguf_tokenizer.h"
#include "model/llama_tokenizer.h"
#include "runtime/backends/llama/llama_cpp_backend.h"
#include "runtime/backends/cuda/native/gguf_model_loader.h"

#include <filesystem>
#include <unordered_map>
#include <string>
#include <vector>

using namespace inferflux;

namespace {

std::vector<std::string> ByteLevelVocab() {
  std::vector<std::string> vocab(20);
  vocab[0] = "<unk>";
  vocab[1] = "<s>";
  vocab[2] = "</s>";
  vocab[3] = "h";
  vocab[4] = "e";
  vocab[5] = "l";
  vocab[6] = "o";
  vocab[7] = "w";
  vocab[8] = "r";
  vocab[9] = "d";
  vocab[10] = "\xc4\xa0";       // Ġ
  vocab[11] = "he";
  vocab[12] = "hel";
  vocab[13] = "hell";
  vocab[14] = "hello";
  vocab[15] = "\xc4\xa0w";
  vocab[16] = "\xc4\xa0wo";
  vocab[17] = "\xc4\xa0wor";
  vocab[18] = "\xc4\xa0worl";
  vocab[19] = "\xc4\xa0world";
  return vocab;
}

std::vector<std::string> ByteLevelMerges() {
  return {
      "h e",            "he l",           "hel l",    "hell o",
      "\xc4\xa0 w",     "\xc4\xa0w o",    "\xc4\xa0wo r",
      "\xc4\xa0wor l",  "\xc4\xa0worl d",
  };
}

} // namespace

TEST_CASE("GGUFTokenizer initializes from byte-level metadata",
          "[gguf][tokenizer_factory]") {
  GGUFTokenizer tok;
  REQUIRE(tok.InitializeFromMetadata(ByteLevelVocab(), ByteLevelMerges(),
                                     "qwen2", 1, 2));

  auto encoded = tok.Tokenize("hello world");
  REQUIRE(encoded == std::vector<int>({1, 14, 19}));
  REQUIRE(tok.Detokenize({14, 19}) == "hello world");
  REQUIRE(tok.TokenToString(19) == " world");
  REQUIRE(tok.BosTokenId() == 1);
  REQUIRE(tok.EosTokenId() == 2);
}

TEST_CASE("GGUFTokenizer rejects empty metadata vocabulary",
          "[gguf][tokenizer_factory]") {
  GGUFTokenizer tok;
  REQUIRE_FALSE(tok.InitializeFromMetadata({}, {}, "qwen2", 1, 2));
}

TEST_CASE("GGUFTokenizer respects metadata add_bos_token=false",
          "[gguf][tokenizer_factory]") {
  GGUFTokenizer tok;
  REQUIRE(tok.InitializeFromMetadata(ByteLevelVocab(), ByteLevelMerges(),
                                     "qwen2", 1, 2,
                                     /*add_bos_token=*/false));

  auto encoded = tok.Tokenize("hello world", /*add_bos=*/true);
  REQUIRE(encoded == std::vector<int>({14, 19}));
}

TEST_CASE("GGUFTokenizer treats control tokens as terminal generation tokens",
          "[gguf][tokenizer_factory]") {
  GGUFTokenizer tok;
  REQUIRE(tok.InitializeFromMetadata(ByteLevelVocab(), ByteLevelMerges(),
                                     "qwen2", 1, 2));

  REQUIRE(tok.IsSpecialToken(1));
  REQUIRE(tok.IsSpecialToken(2));
  REQUIRE(tok.IsTerminalGeneratedToken(1));
  REQUIRE(tok.IsTerminalGeneratedToken(2));
  REQUIRE(tok.IsTerminalGeneratedToken(-1));
  REQUIRE_FALSE(tok.IsTerminalGeneratedToken(14));
}

TEST_CASE("GGUFTokenizer matches llama tokenizer on local Qwen GGUF prompts",
          "[gguf][tokenizer_factory][local_model]") {
  const std::string model_path =
      "models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf";
  if (!std::filesystem::exists(model_path)) {
    SUCCEED("Local Qwen GGUF model not present; skipping tokenizer parity check");
    return;
  }

  GGUFTokenizer gguf_tok;
  if (!gguf_tok.Load(model_path)) {
    SUCCEED("GGUF model lacks embedded tokenizer metadata; skipping parity check");
    return;
  }

  LlamaTokenizer llama_tok;
  REQUIRE(llama_tok.Load(model_path));

  const std::vector<std::string> prompts = {
      "A hash table is a data structure that",
      "The function should be named `fibonacci` and return the nth number.",
      "Paris",
      "Hola mundo",
      "Three prime numbers greater than 10 are",
      "List the highest-impact ways to reduce cloud infrastructure cost for a GPU inference service without causing major reliability regressions.",
  };

  for (const auto &prompt : prompts) {
    CAPTURE(prompt);
    const auto gguf_tokens = gguf_tok.Tokenize(prompt, /*add_bos=*/true);
    const auto llama_tokens = llama_tok.Tokenize(prompt, /*add_bos=*/true);
    REQUIRE(gguf_tokens == llama_tokens);
  }
}

TEST_CASE("LlamaCppBackend treats GGUF end-of-generation tokens as terminal",
          "[gguf][tokenizer_factory][local_model]") {
  const std::string model_path =
      "models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf";
  if (!std::filesystem::exists(model_path)) {
    SUCCEED("Local Qwen GGUF model not present; skipping terminal-token check");
    return;
  }

  runtime::cuda::native::GGUFModelLoader loader;
  if (!loader.Load(model_path)) {
    SUCCEED("GGUF model lacks valid metadata; skipping terminal-token check");
    return;
  }

  GGUFTokenizer gguf_tok;
  if (!gguf_tok.Load(model_path)) {
    SUCCEED("GGUF model lacks tokenizer metadata; skipping terminal-token check");
    return;
  }

  LlamaCppBackend backend;
  LlamaBackendConfig config;
  config.gpu_layers = 0;
  REQUIRE(backend.LoadModel(model_path, config));

  std::vector<int> terminal_ids = {gguf_tok.EosTokenId()};
  const std::unordered_map<std::string, int> known_eog_pieces = {
      {"<|im_end|>", -1},
      {"<|eot_id|>", -1},
      {"<|endoftext|>", -1},
  };
  for (const auto &[piece, _] : known_eog_pieces) {
    for (std::size_t i = 0; i < loader.TokenizerPieces().size(); ++i) {
      if (loader.TokenizerPieces()[i] == piece) {
        terminal_ids.push_back(static_cast<int>(i));
        break;
      }
    }
  }

  for (int token_id : terminal_ids) {
    CAPTURE(token_id);
    REQUIRE(backend.IsTerminalGeneratedToken(token_id));
  }

  const auto content_tokens = gguf_tok.Tokenize("hello", /*add_bos=*/false);
  REQUIRE_FALSE(content_tokens.empty());
  REQUIRE_FALSE(backend.IsTerminalGeneratedToken(content_tokens.front()));
}
