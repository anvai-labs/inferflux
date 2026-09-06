#include <catch2/catch_amalgamated.hpp>

#include "model/hf_tokenizer.h"

#include <filesystem>
#include <fstream>
#include <string>

#include "nlohmann/json.hpp"

using namespace inferflux;
namespace fs = std::filesystem;

// Minimal ByteLevel tokenizer.json + config.json, no tokenizer_config.json —
// mirrors a real safetensors conversion that ships only tokenizer.json and
// config.json (the shape that originally broke both chat-template rendering
// and EOS resolution; see model/hf_tokenizer.cpp and
// runtime/backends/mlx/mlx_tokenizer.cpp).
static fs::path WriteMinimalSafetensorsModelDir(const fs::path &dir) {
  fs::create_directories(dir);

  nlohmann::json vocab;
  vocab["<unk>"] = 0;
  vocab["<s>"] = 1;
  vocab["</s>"] = 2;
  vocab["h"] = 3;
  vocab["i"] = 4;
  vocab["hi"] = 5;

  nlohmann::json tok;
  tok["model"]["type"] = "BPE";
  tok["model"]["vocab"] = vocab;
  tok["model"]["merges"] = nlohmann::json::array({"h i"});
  tok["pre_tokenizer"]["type"] = "ByteLevel";
  tok["pre_tokenizer"]["add_prefix_space"] = false;
  std::ofstream tok_f(dir / "tokenizer.json");
  tok_f << tok.dump(2);

  // Deliberately distinct from MlxTokenizer::Reset()'s defaults (1, 2) so
  // a test asserting these values proves the config.json fallback actually
  // ran, rather than coincidentally matching the unresolved default.
  nlohmann::json cfg;
  cfg["bos_token_id"] = 1;
  cfg["eos_token_id"] = 99;
  std::ofstream cfg_f(dir / "config.json");
  cfg_f << cfg.dump(2);

  return dir;
}

TEST_CASE("HFTokenizer ApplyChatTemplate renders a real prompt, not empty",
          "[hf_tokenizer]") {
  // Regression test: HFTokenizer::ApplyChatTemplate used to unconditionally
  // return {valid=false}, forcing every safetensors chat request through a
  // generic non-instruct-aware fallback path (observed in production as the
  // model generating fake "user:"/"chatbot:" conversation turns instead of
  // stopping after answering).
  const auto dir = fs::temp_directory_path() / "ifx_hf_tok_chat";
  WriteMinimalSafetensorsModelDir(dir);

  HFTokenizer tok;
  REQUIRE(tok.Load(dir.string()));

  auto result =
      tok.ApplyChatTemplate({{"user", "hi"}}, /*add_assistant_prefix=*/true);
  REQUIRE(result.valid);
  // No chat_template.jinja/tokenizer_config.json present, so this exercises
  // the ChatML default fallback (matching GGUFTokenizer's behavior).
  REQUIRE(result.prompt ==
          "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");

  fs::remove_all(dir);
}

TEST_CASE("HFTokenizer ApplyChatTemplate returns invalid for empty messages",
          "[hf_tokenizer]") {
  const auto dir = fs::temp_directory_path() / "ifx_hf_tok_empty";
  WriteMinimalSafetensorsModelDir(dir);

  HFTokenizer tok;
  REQUIRE(tok.Load(dir.string()));

  auto result = tok.ApplyChatTemplate({}, true);
  REQUIRE_FALSE(result.valid);

  fs::remove_all(dir);
}

TEST_CASE("HFTokenizer resolves EOS from config.json when "
          "tokenizer_config.json is absent",
          "[hf_tokenizer]") {
  // Regression test: without this fallback, EosTokenId() silently returns
  // MlxTokenizer::Reset()'s default (2) instead of the model's declared
  // eos_token_id, breaking generation stopping for any safetensors model
  // directory that ships only tokenizer.json + config.json.
  const auto dir = fs::temp_directory_path() / "ifx_hf_tok_eos";
  WriteMinimalSafetensorsModelDir(dir);

  HFTokenizer tok;
  REQUIRE(tok.Load(dir.string()));
  REQUIRE(tok.EosTokenId() == 99);
  REQUIRE(tok.BosTokenId() == 1);

  fs::remove_all(dir);
}
