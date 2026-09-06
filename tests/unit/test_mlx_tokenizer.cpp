#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/mlx/mlx_tokenizer.h"

#include <filesystem>
#include <fstream>
#include <string>

#include "nlohmann/json.hpp"

using namespace inferflux;
namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Helpers — write synthetic tokenizer files to a temp directory
// ---------------------------------------------------------------------------

// Write a minimal tokenizer.json with a Metaspace pre-tokenizer and BPE vocab.
// Vocab: <unk>=0, <s>=1, </s>=2, individual ASCII chars 3-39, then merged
// tokens.
static fs::path WriteMetaspaceTokenizer(const fs::path &dir) {
  fs::create_directories(dir);

  // Build a small vocabulary: special tokens + chars + a few merged tokens.
  nlohmann::json vocab;
  vocab["<unk>"] = 0;
  vocab["<s>"] = 1;
  vocab["</s>"] = 2;
  // Individual character tokens (▁ must be in vocab for Metaspace to work).
  vocab["\xe2\x96\x81"] = 3; // ▁
  vocab["h"] = 4;
  vocab["e"] = 5;
  vocab["l"] = 6;
  vocab["o"] = 7;
  vocab["w"] = 8;
  vocab["r"] = 9;
  vocab["d"] = 10;
  vocab["!"] = 11;
  // Merged tokens produced by merges below.
  vocab["\xe2\x96\x81h"] = 12;     // ▁h
  vocab["\xe2\x96\x81he"] = 13;    // ▁he
  vocab["\xe2\x96\x81hel"] = 14;   // ▁hel
  vocab["\xe2\x96\x81hell"] = 15;  // ▁hell
  vocab["\xe2\x96\x81hello"] = 16; // ▁hello
  vocab["\xe2\x96\x81w"] = 17;     // ▁w
  vocab["\xe2\x96\x81wo"] = 18;    // ▁wo
  vocab["\xe2\x96\x81wor"] = 19;   // ▁wor
  vocab["\xe2\x96\x81worl"] = 20;  // ▁worl
  vocab["\xe2\x96\x81world"] = 21; // ▁world

  // Merges (rank = index in array).
  nlohmann::json merges = nlohmann::json::array({
      "\xe2\x96\x81 h",     // ▁ + h  → ▁h
      "\xe2\x96\x81h e",    // ▁h + e → ▁he
      "\xe2\x96\x81he l",   // ▁he + l → ▁hel
      "\xe2\x96\x81hel l",  // ▁hel + l → ▁hell
      "\xe2\x96\x81hell o", // ▁hell + o → ▁hello
      "\xe2\x96\x81 w",     // ▁ + w  → ▁w
      "\xe2\x96\x81w o",    // ▁w + o → ▁wo
      "\xe2\x96\x81wo r",   // ▁wo + r → ▁wor
      "\xe2\x96\x81wor l",  // ▁wor + l → ▁worl
      "\xe2\x96\x81worl d", // ▁worl + d → ▁world
  });

  nlohmann::json tok;
  tok["model"]["type"] = "BPE";
  tok["model"]["vocab"] = vocab;
  tok["model"]["merges"] = merges;
  tok["pre_tokenizer"]["type"] = "Metaspace";
  tok["pre_tokenizer"]["add_prefix_space"] = true;
  tok["pre_tokenizer"]["replacement"] = "\xe2\x96\x81";
  tok["added_tokens"] = nlohmann::json::array({
      {{"id", 0}, {"content", "<unk>"}, {"special", true}},
      {{"id", 1}, {"content", "<s>"}, {"special", true}},
      {{"id", 2}, {"content", "</s>"}, {"special", true}},
  });

  const auto path = dir / "tokenizer.json";
  std::ofstream f(path);
  f << tok.dump(2);
  return dir;
}

// Write tokenizer_config.json alongside.
static void WriteTokenizerConfig(const fs::path &dir,
                                 const std::string &bos = "<s>",
                                 const std::string &eos = "</s>") {
  nlohmann::json cfg;
  cfg["bos_token"] = bos;
  cfg["eos_token"] = eos;
  std::ofstream f(dir / "tokenizer_config.json");
  f << cfg.dump();
}

// Write a minimal ByteLevel tokenizer.json.
// Small vocab: individual byte-encoded chars + a few merged tokens for
// "Ġhello".
static fs::path WriteByteLevelTokenizer(const fs::path &dir) {
  fs::create_directories(dir);

  // Build vocab: byte tokens for chars h,e,l,o,w,r,d,space(Ġ), then merged.
  // In ByteLevel encoding: byte 0x20 (space) → Ġ (U+0120) = "\xc4\xa0"
  nlohmann::json vocab;
  vocab["<unk>"] = 0;
  vocab["<s>"] = 1;
  vocab["</s>"] = 2;
  vocab["h"] = 3;
  vocab["e"] = 4;
  vocab["l"] = 5;
  vocab["o"] = 6;
  vocab["w"] = 7;
  vocab["r"] = 8;
  vocab["d"] = 9;
  // Ġ = encoded space (byte 0x20 → U+0120 → UTF-8: C4 A0).
  vocab["\xc4\xa0"] = 10; // Ġ
  // Merged tokens.
  vocab["he"] = 11;
  vocab["hel"] = 12;
  vocab["hell"] = 13;
  vocab["hello"] = 14;
  vocab["\xc4\xa0w"] = 15;     // Ġw
  vocab["\xc4\xa0wo"] = 16;    // Ġwo
  vocab["\xc4\xa0wor"] = 17;   // Ġwor
  vocab["\xc4\xa0worl"] = 18;  // Ġworl
  vocab["\xc4\xa0world"] = 19; // Ġworld

  nlohmann::json merges = nlohmann::json::array({
      "h e",            // h + e → he
      "he l",           // he + l → hel
      "hel l",          // hel + l → hell
      "hell o",         // hell + o → hello
      "\xc4\xa0 w",     // Ġ + w → Ġw
      "\xc4\xa0w o",    // Ġw + o → Ġwo
      "\xc4\xa0wo r",   // Ġwo + r → Ġwor
      "\xc4\xa0wor l",  // Ġwor + l → Ġworl
      "\xc4\xa0worl d", // Ġworl + d → Ġworld
  });

  nlohmann::json tok;
  tok["model"]["type"] = "BPE";
  tok["model"]["vocab"] = vocab;
  tok["model"]["merges"] = merges;
  tok["pre_tokenizer"]["type"] = "ByteLevel";
  tok["pre_tokenizer"]["add_prefix_space"] = false;
  tok["added_tokens"] = nlohmann::json::array({
      {{"id", 0}, {"content", "<unk>"}, {"special", true}},
      {{"id", 1}, {"content", "<s>"}, {"special", true}},
      {{"id", 2}, {"content", "</s>"}, {"special", true}},
  });

  const auto path = dir / "tokenizer.json";
  std::ofstream f(path);
  f << tok.dump(2);
  return dir;
}

// Same vocab/merges as WriteByteLevelTokenizer, but with pre_tokenizer
// wrapped as Sequence[Split, ByteLevel] — the form HuggingFace actually
// emits for Qwen2.5, GPT-2, Llama-3, Phi-3, and most other ByteLevel-BPE
// tokenizers. Regression fixture for the bug where the wrapped form was
// never unwrapped, leaving pre_tok_ at Unknown and Decode() returning raw,
// undecoded token text (literal Ġ/Ċ instead of spaces/newlines).
static fs::path WriteByteLevelSequenceTokenizer(const fs::path &dir) {
  fs::create_directories(dir);

  nlohmann::json vocab;
  vocab["<unk>"] = 0;
  vocab["<s>"] = 1;
  vocab["</s>"] = 2;
  vocab["h"] = 3;
  vocab["e"] = 4;
  vocab["l"] = 5;
  vocab["o"] = 6;
  vocab["w"] = 7;
  vocab["r"] = 8;
  vocab["d"] = 9;
  vocab["\xc4\xa0"] = 10; // Ġ
  vocab["he"] = 11;
  vocab["hel"] = 12;
  vocab["hell"] = 13;
  vocab["hello"] = 14;
  vocab["\xc4\xa0w"] = 15;     // Ġw
  vocab["\xc4\xa0wo"] = 16;    // Ġwo
  vocab["\xc4\xa0wor"] = 17;   // Ġwor
  vocab["\xc4\xa0worl"] = 18;  // Ġworl
  vocab["\xc4\xa0world"] = 19; // Ġworld

  nlohmann::json merges = nlohmann::json::array({
      "h e",
      "he l",
      "hel l",
      "hell o",
      "\xc4\xa0 w",
      "\xc4\xa0w o",
      "\xc4\xa0wo r",
      "\xc4\xa0wor l",
      "\xc4\xa0worl d",
  });

  nlohmann::json tok;
  tok["model"]["type"] = "BPE";
  tok["model"]["vocab"] = vocab;
  tok["model"]["merges"] = merges;
  tok["pre_tokenizer"]["type"] = "Sequence";
  tok["pre_tokenizer"]["pretokenizers"] = nlohmann::json::array({
      {{"type", "Split"}, {"behavior", "Isolated"}, {"invert", false}},
      {{"type", "ByteLevel"}, {"add_prefix_space", false}},
  });
  tok["added_tokens"] = nlohmann::json::array({
      {{"id", 0}, {"content", "<unk>"}, {"special", true}},
      {{"id", 1}, {"content", "<s>"}, {"special", true}},
      {{"id", 2}, {"content", "</s>"}, {"special", true}},
  });

  const auto path = dir / "tokenizer.json";
  std::ofstream f(path);
  f << tok.dump(2);
  return dir;
}

// ---------------------------------------------------------------------------
// MlxTokenizerResult defaults
// ---------------------------------------------------------------------------

TEST_CASE("MlxTokenizerResult defaults", "[mlx_tokenizer]") {
  MlxTokenizerResult r;
  REQUIRE_FALSE(r.ok);
  REQUIRE(r.ids.empty());
}

// ---------------------------------------------------------------------------
// Load error cases
// ---------------------------------------------------------------------------

TEST_CASE("MlxTokenizer not loaded by default", "[mlx_tokenizer]") {
  MlxTokenizer tok;
  REQUIRE_FALSE(tok.Loaded());
  REQUIRE(tok.VocabSize() == 0);
}

TEST_CASE("MlxTokenizer load from nonexistent dir returns false",
          "[mlx_tokenizer]") {
  MlxTokenizer tok;
  REQUIRE_FALSE(tok.Load("/tmp/ifx_no_such_tok_dir_xyz"));
}

TEST_CASE("MlxTokenizer load from dir without tokenizer.json returns false",
          "[mlx_tokenizer]") {
  const auto dir = fs::temp_directory_path() / "ifx_tok_no_json";
  fs::create_directories(dir);
  MlxTokenizer tok;
  REQUIRE_FALSE(tok.Load(dir));
  fs::remove_all(dir);
}

TEST_CASE("MlxTokenizer Encode returns ok=false when not loaded",
          "[mlx_tokenizer]") {
  MlxTokenizer tok;
  auto r = tok.Encode("hello");
  REQUIRE_FALSE(r.ok);
}

// ---------------------------------------------------------------------------
// Metaspace tokenizer
// ---------------------------------------------------------------------------

TEST_CASE("MlxTokenizer load Metaspace tokenizer", "[mlx_tokenizer]") {
  const auto dir = fs::temp_directory_path() / "ifx_tok_meta";
  WriteMetaspaceTokenizer(dir);
  WriteTokenizerConfig(dir);

  MlxTokenizer tok;
  REQUIRE(tok.Load(dir));
  REQUIRE(tok.Loaded());
  REQUIRE(tok.VocabSize() >= 22);
  REQUIRE(tok.BosId() == 1);
  REQUIRE(tok.EosId() == 2);

  fs::remove_all(dir);
}

TEST_CASE("MlxTokenizer Metaspace encode 'hello world' with BOS",
          "[mlx_tokenizer]") {
  const auto dir = fs::temp_directory_path() / "ifx_tok_meta_enc";
  WriteMetaspaceTokenizer(dir);
  WriteTokenizerConfig(dir);

  MlxTokenizer tok;
  REQUIRE(tok.Load(dir));

  auto r = tok.Encode("hello world");
  REQUIRE(r.ok);
  // Expected: [BOS=1, ▁hello=16, ▁world=21]
  REQUIRE(r.ids.size() == 3);
  REQUIRE(r.ids[0] == 1);  // BOS
  REQUIRE(r.ids[1] == 16); // ▁hello
  REQUIRE(r.ids[2] == 21); // ▁world

  fs::remove_all(dir);
}

TEST_CASE("MlxTokenizer Metaspace encode without BOS", "[mlx_tokenizer]") {
  const auto dir = fs::temp_directory_path() / "ifx_tok_meta_nobos";
  WriteMetaspaceTokenizer(dir);

  MlxTokenizer tok;
  REQUIRE(tok.Load(dir));

  auto r = tok.Encode("hello", /*add_bos=*/false);
  REQUIRE(r.ok);
  REQUIRE(r.ids[0] == 16); // ▁hello directly, no BOS

  fs::remove_all(dir);
}

TEST_CASE("MlxTokenizer Metaspace decode round-trip", "[mlx_tokenizer]") {
  const auto dir = fs::temp_directory_path() / "ifx_tok_meta_dec";
  WriteMetaspaceTokenizer(dir);
  WriteTokenizerConfig(dir);

  MlxTokenizer tok;
  REQUIRE(tok.Load(dir));

  const std::string original = "hello world";
  auto r = tok.Encode(original, /*add_bos=*/false);
  REQUIRE(r.ok);
  const std::string decoded = tok.Decode(r.ids);
  REQUIRE(decoded == original);

  fs::remove_all(dir);
}

TEST_CASE("MlxTokenizer Metaspace decode skips special tokens",
          "[mlx_tokenizer]") {
  const auto dir = fs::temp_directory_path() / "ifx_tok_meta_spec";
  WriteMetaspaceTokenizer(dir);
  WriteTokenizerConfig(dir);

  MlxTokenizer tok;
  REQUIRE(tok.Load(dir));

  // Encode adds BOS; decode with skip_special should remove it.
  auto r = tok.Encode("hello", /*add_bos=*/true);
  REQUIRE(r.ok);
  REQUIRE(r.ids.front() == 1); // BOS

  const std::string decoded = tok.Decode(r.ids, /*skip_special=*/true);
  REQUIRE(decoded == "hello");
  REQUIRE(decoded.find("<s>") == std::string::npos);

  fs::remove_all(dir);
}

TEST_CASE("MlxTokenizer IsSpecial identifies added special tokens",
          "[mlx_tokenizer]") {
  const auto dir = fs::temp_directory_path() / "ifx_tok_meta_isspec";
  WriteMetaspaceTokenizer(dir);

  MlxTokenizer tok;
  REQUIRE(tok.Load(dir));

  REQUIRE(tok.IsSpecial(0));        // <unk>
  REQUIRE(tok.IsSpecial(1));        // <s>
  REQUIRE(tok.IsSpecial(2));        // </s>
  REQUIRE_FALSE(tok.IsSpecial(16)); // ▁hello — not special

  fs::remove_all(dir);
}

// ---------------------------------------------------------------------------
// ByteLevel tokenizer
// ---------------------------------------------------------------------------

TEST_CASE("MlxTokenizer load ByteLevel tokenizer", "[mlx_tokenizer]") {
  const auto dir = fs::temp_directory_path() / "ifx_tok_bytelevel";
  WriteByteLevelTokenizer(dir);
  WriteTokenizerConfig(dir);

  MlxTokenizer tok;
  REQUIRE(tok.Load(dir));
  REQUIRE(tok.Loaded());

  fs::remove_all(dir);
}

TEST_CASE("MlxTokenizer ByteLevel encode 'hello world' with BOS",
          "[mlx_tokenizer]") {
  const auto dir = fs::temp_directory_path() / "ifx_tok_bl_enc";
  WriteByteLevelTokenizer(dir);
  WriteTokenizerConfig(dir);

  MlxTokenizer tok;
  REQUIRE(tok.Load(dir));

  auto r = tok.Encode("hello world");
  REQUIRE(r.ok);
  // Expected: [BOS=1, hello=14, Ġworld=19]
  REQUIRE(r.ids.size() == 3);
  REQUIRE(r.ids[0] == 1);  // BOS
  REQUIRE(r.ids[1] == 14); // hello
  REQUIRE(r.ids[2] == 19); // Ġworld

  fs::remove_all(dir);
}

TEST_CASE("MlxTokenizer ByteLevel decode round-trip", "[mlx_tokenizer]") {
  const auto dir = fs::temp_directory_path() / "ifx_tok_bl_dec";
  WriteByteLevelTokenizer(dir);

  MlxTokenizer tok;
  REQUIRE(tok.Load(dir));

  auto r = tok.Encode("hello world", /*add_bos=*/false);
  REQUIRE(r.ok);
  const std::string decoded = tok.Decode(r.ids, /*skip_special=*/false);
  REQUIRE(decoded == "hello world");

  fs::remove_all(dir);
}

TEST_CASE("MlxTokenizer unwraps Sequence-wrapped ByteLevel pre_tokenizer",
          "[mlx_tokenizer]") {
  // Regression test: HuggingFace's actual tokenizer.json for Qwen2.5,
  // GPT-2, Llama-3, Phi-3, etc. wraps pre_tokenizer as
  // Sequence[Split, ByteLevel], not a flat {"type": "ByteLevel"}. Before
  // the fix, Load() only matched the flat form, so pre_tok_ stayed Unknown
  // and Decode() fell through to raw, undecoded token text — every decoded
  // response contained literal Ġ/Ċ instead of spaces/newlines.
  const auto dir = fs::temp_directory_path() / "ifx_tok_bl_seq";
  WriteByteLevelSequenceTokenizer(dir);

  MlxTokenizer tok;
  REQUIRE(tok.Load(dir));
  REQUIRE(tok.Loaded());

  auto r = tok.Encode("hello world", /*add_bos=*/false);
  REQUIRE(r.ok);
  const std::string decoded = tok.Decode(r.ids, /*skip_special=*/false);
  // Must be real decoded text with a space, not "helloĠworld" or
  // "hello\xc4\xa0world" (the undecoded ByteLevel marker for space).
  REQUIRE(decoded == "hello world");

  fs::remove_all(dir);
}

TEST_CASE("MlxTokenizer does not crash on a malformed non-object entry in "
          "a Sequence pretokenizers array",
          "[mlx_tokenizer]") {
  // Regression test: the Sequence-unwrapping loop must not assume every
  // array entry is a JSON object. A hand-edited or corrupted tokenizer.json
  // with a non-object entry (a bare string, here) previously threw an
  // uncaught nlohmann::json::type_error out of Load() -- a contract
  // violation (Load() must return false on bad input, never throw) that
  // would abort the whole process via std::terminate.
  const auto dir = fs::temp_directory_path() / "ifx_tok_seq_malformed";
  fs::create_directories(dir);
  {
    nlohmann::json vocab;
    vocab["<unk>"] = 0;
    nlohmann::json tok;
    tok["model"]["type"] = "BPE";
    tok["model"]["vocab"] = vocab;
    tok["model"]["merges"] = nlohmann::json::array();
    tok["pre_tokenizer"]["type"] = "Sequence";
    tok["pre_tokenizer"]["pretokenizers"] = nlohmann::json::array(
        {"not_an_object",
         {{"type", "ByteLevel"}, {"add_prefix_space", false}}});
    std::ofstream f(dir / "tokenizer.json");
    f << tok.dump(2);
  }

  MlxTokenizer tok;
  // Must not throw/crash; the well-formed ByteLevel entry after the
  // malformed one should still be picked up.
  REQUIRE_NOTHROW(tok.Load(dir));
  REQUIRE(tok.Loaded());

  fs::remove_all(dir);
}

TEST_CASE("MlxTokenizer skips a vocab entry with a non-numeric id instead "
          "of crashing",
          "[mlx_tokenizer]") {
  // Regression test: the vocab-parsing loop called id_val.get<int32_t>()
  // on every model["vocab"] entry unconditionally. A hand-edited or
  // corrupted tokenizer.json with a non-numeric id (a string, here)
  // previously threw an uncaught nlohmann::json::type_error out of
  // Load() -- Load() must return false on bad input, never throw.
  const auto dir = fs::temp_directory_path() / "ifx_tok_vocab_bad_id";
  fs::create_directories(dir);
  {
    nlohmann::json vocab;
    vocab["<unk>"] = 0;
    vocab["hello"] = 1;
    vocab["bad_token"] = "not_a_number"; // wrong type
    nlohmann::json tok;
    tok["model"]["type"] = "BPE";
    tok["model"]["vocab"] = vocab;
    tok["model"]["merges"] = nlohmann::json::array();
    std::ofstream f(dir / "tokenizer.json");
    f << tok.dump(2);
  }

  MlxTokenizer tok;
  REQUIRE_NOTHROW(tok.Load(dir));
  REQUIRE(tok.Loaded());
  // The well-formed entries must still load; the bad one must not
  // contribute to vocab_size_.
  REQUIRE(tok.VocabSize() == 2);
  // Stronger than a VocabSize() count alone: confirm the bad entry did not
  // get silently inserted under a fallback id (e.g. 0), which would still
  // satisfy VocabSize()==2 without actually being rejected.
  REQUIRE(tok.Decode({0}, /*skip_special=*/false) == "<unk>");

  fs::remove_all(dir);
}

TEST_CASE("MlxTokenizer rejects a negative vocab id instead of corrupting "
          "memory",
          "[mlx_tokenizer]") {
  // Regression test for a SEGV found by adversarial review: a negative id
  // passes the is_number() type check added for the previous fix, but
  // id_to_token_[id] = tok then indexes a std::vector<std::string> with a
  // negative int32_t implicitly converted to a huge size_t -- an
  // out-of-bounds write, reproduced as a crash under ASan.
  const auto dir = fs::temp_directory_path() / "ifx_tok_vocab_negative_id";
  fs::create_directories(dir);
  {
    nlohmann::json vocab;
    vocab["<unk>"] = 0;
    vocab["hello"] = 1;
    vocab["evil"] = -12345;
    nlohmann::json tok;
    tok["model"]["type"] = "BPE";
    tok["model"]["vocab"] = vocab;
    tok["model"]["merges"] = nlohmann::json::array();
    std::ofstream f(dir / "tokenizer.json");
    f << tok.dump(2);
  }

  MlxTokenizer tok;
  REQUIRE_NOTHROW(tok.Load(dir));
  REQUIRE(tok.Loaded());
  REQUIRE(tok.VocabSize() == 2); // negative-id entry rejected, not sized in

  fs::remove_all(dir);
}

TEST_CASE("MlxTokenizer rejects an unreasonably large vocab id instead of "
          "attempting a huge allocation",
          "[mlx_tokenizer]") {
  // Regression test for a memory-exhaustion DoS found by adversarial
  // review: an id like 2,000,000,000 passes the is_number() check, and
  // id_to_token_.assign(max_id + 1, "") then attempts to allocate and
  // default-construct ~2 billion std::string objects from a few hundred
  // bytes of input JSON.
  const auto dir = fs::temp_directory_path() / "ifx_tok_vocab_huge_id";
  fs::create_directories(dir);
  {
    nlohmann::json vocab;
    vocab["<unk>"] = 0;
    vocab["hello"] = 1;
    vocab["huge"] = 2000000000;
    nlohmann::json tok;
    tok["model"]["type"] = "BPE";
    tok["model"]["vocab"] = vocab;
    tok["model"]["merges"] = nlohmann::json::array();
    std::ofstream f(dir / "tokenizer.json");
    f << tok.dump(2);
  }

  MlxTokenizer tok;
  // Must return promptly, not attempt a multi-gigabyte allocation.
  REQUIRE_NOTHROW(tok.Load(dir));
  REQUIRE(tok.Loaded());
  REQUIRE(tok.VocabSize() == 2); // huge-id entry rejected, not sized in

  fs::remove_all(dir);
}

TEST_CASE("MlxTokenizer does not let an id truncate into a valid range and "
          "silently overwrite an existing token",
          "[mlx_tokenizer]") {
  // Regression test for a truncation exploit found by adversarial review:
  // reading the id via get<int32_t>() performs an unchecked static_cast
  // from JSON's internal 64-bit storage, so an id like 2^32 + 1 silently
  // truncates to 1 -- passing an int32-width range check while colliding
  // with (and overwriting) whatever legitimate token already has id 1.
  // The fix reads the id as int64_t and range-checks the untruncated
  // value before narrowing.
  const auto dir = fs::temp_directory_path() / "ifx_tok_vocab_id_truncation";
  fs::create_directories(dir);
  {
    nlohmann::json vocab;
    vocab["<unk>"] = 0;
    vocab["legit_one"] = 1;
    // 2^32 + 1 == 4294967297; truncates to 1 under an unchecked
    // int64_t -> int32_t cast.
    vocab["attacker_wrap"] = 4294967297LL;
    nlohmann::json tok;
    tok["model"]["type"] = "BPE";
    tok["model"]["vocab"] = vocab;
    tok["model"]["merges"] = nlohmann::json::array();
    std::ofstream f(dir / "tokenizer.json");
    f << tok.dump(2);
  }

  MlxTokenizer tok;
  REQUIRE_NOTHROW(tok.Load(dir));
  REQUIRE(tok.Loaded());
  // id 1 must still resolve to the legitimate token, not the attacker's.
  REQUIRE(tok.Decode({1}, /*skip_special=*/false) == "legit_one");
  REQUIRE(tok.VocabSize() == 2); // the wrapping entry was rejected

  fs::remove_all(dir);
}

TEST_CASE("MlxTokenizer rejects an out-of-range added_tokens id instead of "
          "corrupting memory",
          "[mlx_tokenizer]") {
  // Same class of bug as the vocab-loop negative/huge id issue above, for
  // the added_tokens loop's id_to_token_.resize(id + 1) / id_to_token_[id].
  const auto dir = fs::temp_directory_path() / "ifx_tok_added_bad_range";
  fs::create_directories(dir);
  {
    nlohmann::json vocab;
    vocab["<unk>"] = 0;
    nlohmann::json tok;
    tok["model"]["type"] = "BPE";
    tok["model"]["vocab"] = vocab;
    tok["model"]["merges"] = nlohmann::json::array();
    tok["added_tokens"] = nlohmann::json::array({
        {{"id", -1}, {"content", "<neg>"}},
        {{"id", 2000000000}, {"content", "<huge>"}},
        {{"id", 5}, {"content", "<good>"}, {"special", true}},
    });
    std::ofstream f(dir / "tokenizer.json");
    f << tok.dump(2);
  }

  MlxTokenizer tok;
  REQUIRE_NOTHROW(tok.Load(dir));
  REQUIRE(tok.Loaded());
  REQUIRE(tok.VocabSize() == 6); // only the well-formed id=5 entry counted

  fs::remove_all(dir);
}

TEST_CASE("MlxTokenizer skips a merges entry with an unexpected shape "
          "instead of crashing",
          "[mlx_tokenizer]") {
  // Regression test: the merges-parsing loop called m.get<std::string>()
  // on every model["merges"] entry unconditionally. A hand-edited or
  // corrupted tokenizer.json with a non-string, non-array entry (a bare
  // number, here) previously threw an uncaught nlohmann::json::type_error
  // out of Load().
  const auto dir = fs::temp_directory_path() / "ifx_tok_merges_bad_shape";
  fs::create_directories(dir);
  {
    nlohmann::json vocab;
    vocab["<unk>"] = 0;
    vocab["h"] = 1;
    vocab["e"] = 2;
    vocab["he"] = 3;
    nlohmann::json tok;
    tok["model"]["type"] = "BPE";
    tok["model"]["vocab"] = vocab;
    tok["model"]["merges"] = nlohmann::json::array({42, "h e"});
    std::ofstream f(dir / "tokenizer.json");
    f << tok.dump(2);
  }

  MlxTokenizer tok;
  REQUIRE_NOTHROW(tok.Load(dir));
  REQUIRE(tok.Loaded());

  // The well-formed "h e" merge after the malformed entry must still take
  // effect.
  auto r = tok.Encode("he", /*add_bos=*/false);
  REQUIRE(r.ok);
  REQUIRE(r.ids.size() == 1);
  REQUIRE(r.ids[0] == 3); // merged "he" token id

  fs::remove_all(dir);
}

TEST_CASE("MlxTokenizer supports merges in array-of-arrays format",
          "[mlx_tokenizer]") {
  // Compatibility test (not just crash-avoidance): some newer HuggingFace
  // tokenizer.json exports encode merges as an array of 2-element
  // [piece_a, piece_b] arrays rather than the older space-joined
  // "piece_a piece_b" string. Both are legitimate tokenizer.json shapes in
  // the wild; before this fix only the string form worked.
  const auto dir = fs::temp_directory_path() / "ifx_tok_merges_arr";
  fs::create_directories(dir);
  {
    nlohmann::json vocab;
    vocab["<unk>"] = 0;
    vocab["h"] = 1;
    vocab["e"] = 2;
    vocab["l"] = 3;
    vocab["he"] = 4;
    vocab["hel"] = 5;
    nlohmann::json tok;
    tok["model"]["type"] = "BPE";
    tok["model"]["vocab"] = vocab;
    tok["model"]["merges"] = nlohmann::json::array({
        nlohmann::json::array({"h", "e"}),
        nlohmann::json::array({"he", "l"}),
    });
    std::ofstream f(dir / "tokenizer.json");
    f << tok.dump(2);
  }

  MlxTokenizer tok;
  REQUIRE(tok.Load(dir));
  REQUIRE(tok.Loaded());

  auto r = tok.Encode("hel", /*add_bos=*/false);
  REQUIRE(r.ok);
  REQUIRE(r.ids.size() == 1);
  REQUIRE(r.ids[0] == 5); // "hel" merged via array-of-arrays merges

  fs::remove_all(dir);
}

TEST_CASE("MlxTokenizer skips an added_tokens entry with wrong-typed id or "
          "content instead of crashing",
          "[mlx_tokenizer]") {
  // Regression test: the added_tokens loop checked at.contains("id") /
  // at.contains("content") but not that those values were actually the
  // expected JSON type. A hand-edited tokenizer.json with e.g.
  // {"id": "not_a_number", ...} or {"content": 123, ...} previously threw
  // an uncaught nlohmann::json::type_error out of Load().
  const auto dir = fs::temp_directory_path() / "ifx_tok_added_bad_type";
  fs::create_directories(dir);
  {
    nlohmann::json vocab;
    vocab["<unk>"] = 0;
    nlohmann::json tok;
    tok["model"]["type"] = "BPE";
    tok["model"]["vocab"] = vocab;
    tok["model"]["merges"] = nlohmann::json::array();
    tok["added_tokens"] = nlohmann::json::array({
        {{"id", "not_a_number"}, {"content", "<bad_id>"}},
        {{"id", 5}, {"content", 123}}, // content wrong type
        {{"id", 7}, {"content", "<good>"}, {"special", true}},
    });
    std::ofstream f(dir / "tokenizer.json");
    f << tok.dump(2);
  }

  MlxTokenizer tok;
  REQUIRE_NOTHROW(tok.Load(dir));
  REQUIRE(tok.Loaded());
  // The well-formed entry after the two malformed ones must still load.
  REQUIRE(tok.VocabSize() == 8); // ids up to 7

  fs::remove_all(dir);
}

TEST_CASE("MlxTokenizer ignores a non-string pre_tokenizer type field "
          "instead of crashing",
          "[mlx_tokenizer]") {
  // Regression test found during the same audit: apply_pre_tokenizer_type()
  // read pt.value("type", "") directly, which throws
  // nlohmann::json::type_error if "type" is present but not a string (e.g.
  // hand-edited to a number).
  const auto dir = fs::temp_directory_path() / "ifx_tok_pt_bad_type";
  fs::create_directories(dir);
  {
    nlohmann::json vocab;
    vocab["<unk>"] = 0;
    nlohmann::json tok;
    tok["model"]["type"] = "BPE";
    tok["model"]["vocab"] = vocab;
    tok["model"]["merges"] = nlohmann::json::array();
    tok["pre_tokenizer"]["type"] = 42; // wrong type
    std::ofstream f(dir / "tokenizer.json");
    f << tok.dump(2);
  }

  MlxTokenizer tok;
  REQUIRE_NOTHROW(tok.Load(dir));
  REQUIRE(tok.Loaded());

  fs::remove_all(dir);
}

TEST_CASE("MlxTokenizer ignores a non-boolean add_prefix_space field "
          "instead of crashing",
          "[mlx_tokenizer]") {
  // Same class of bug as the pre_tokenizer type field above, for the
  // boolean field read alongside it.
  const auto dir = fs::temp_directory_path() / "ifx_tok_pt_bad_prefix_space";
  fs::create_directories(dir);
  {
    nlohmann::json vocab;
    vocab["<unk>"] = 0;
    nlohmann::json tok;
    tok["model"]["type"] = "BPE";
    tok["model"]["vocab"] = vocab;
    tok["model"]["merges"] = nlohmann::json::array();
    tok["pre_tokenizer"]["type"] = "ByteLevel";
    tok["pre_tokenizer"]["add_prefix_space"] = "yes"; // wrong type
    std::ofstream f(dir / "tokenizer.json");
    f << tok.dump(2);
  }

  MlxTokenizer tok;
  REQUIRE_NOTHROW(tok.Load(dir));
  REQUIRE(tok.Loaded());

  fs::remove_all(dir);
}

TEST_CASE("MlxTokenizer ignores a non-boolean added_tokens special field "
          "instead of crashing",
          "[mlx_tokenizer]") {
  const auto dir = fs::temp_directory_path() / "ifx_tok_added_bad_special";
  fs::create_directories(dir);
  {
    nlohmann::json vocab;
    vocab["<unk>"] = 0;
    nlohmann::json tok;
    tok["model"]["type"] = "BPE";
    tok["model"]["vocab"] = vocab;
    tok["model"]["merges"] = nlohmann::json::array();
    tok["added_tokens"] = nlohmann::json::array({
        {{"id", 5}, {"content", "<x>"}, {"special", "yes"}}, // wrong type
    });
    std::ofstream f(dir / "tokenizer.json");
    f << tok.dump(2);
  }

  MlxTokenizer tok;
  REQUIRE_NOTHROW(tok.Load(dir));
  REQUIRE(tok.Loaded());
  REQUIRE(tok.VocabSize() == 6);

  fs::remove_all(dir);
}

// ---------------------------------------------------------------------------
// tokenizer_config.json — bos/eos resolved from object notation
// ---------------------------------------------------------------------------

TEST_CASE("MlxTokenizer resolves bos/eos from object in tokenizer_config",
          "[mlx_tokenizer]") {
  const auto dir = fs::temp_directory_path() / "ifx_tok_cfg_obj";
  WriteMetaspaceTokenizer(dir);
  // Write config with bos/eos as objects (LLaMA 2 style).
  nlohmann::json cfg;
  cfg["bos_token"] = {{"content", "<s>"}, {"single_word", false}};
  cfg["eos_token"] = {{"content", "</s>"}, {"single_word", false}};
  std::ofstream f(dir / "tokenizer_config.json");
  f << cfg.dump();

  MlxTokenizer tok;
  REQUIRE(tok.Load(dir));
  REQUIRE(tok.BosId() == 1);
  REQUIRE(tok.EosId() == 2);

  fs::remove_all(dir);
}

// ---------------------------------------------------------------------------
// Chat template tests
// ---------------------------------------------------------------------------

TEST_CASE("MlxTokenizer parses chat_template from tokenizer_config",
          "[mlx_tokenizer]") {
  const auto dir = fs::temp_directory_path() / "ifx_tok_chat_tmpl";
  WriteMetaspaceTokenizer(dir);
  {
    nlohmann::json cfg;
    cfg["bos_token"] = "<s>";
    cfg["eos_token"] = "</s>";
    cfg["chat_template"] = "{% for msg in messages %}{{ msg.role }}: {{ "
                           "msg.content }}\n{% endfor %}";
    std::ofstream f(dir / "tokenizer_config.json");
    f << cfg.dump();
    // f closed here (end of scope) so bytes are flushed before Load().
  }

  MlxTokenizer tok;
  REQUIRE(tok.Load(dir));
  REQUIRE(tok.HasChatTemplate());
  REQUIRE(!tok.ChatTemplate().empty());
  REQUIRE(tok.ChatTemplate().find("msg.role") != std::string::npos);

  fs::remove_all(dir);
}

TEST_CASE("MlxTokenizer HasChatTemplate false when key absent",
          "[mlx_tokenizer]") {
  const auto dir = fs::temp_directory_path() / "ifx_tok_no_tmpl";
  WriteMetaspaceTokenizer(dir);
  {
    // tokenizer_config.json without chat_template key.
    nlohmann::json cfg;
    cfg["bos_token"] = "<s>";
    cfg["eos_token"] = "</s>";
    std::ofstream f(dir / "tokenizer_config.json");
    f << cfg.dump();
  }

  MlxTokenizer tok;
  REQUIRE(tok.Load(dir));
  REQUIRE_FALSE(tok.HasChatTemplate());
  REQUIRE(tok.ChatTemplate().empty());

  fs::remove_all(dir);
}

TEST_CASE("MlxTokenizer falls back to standalone chat_template.jinja file "
          "when tokenizer_config.json is absent",
          "[mlx_tokenizer]") {
  // Regression test: transformers v4.44+ / vLLM / SGLang ship the chat
  // template as a standalone chat_template.jinja file when there is no
  // tokenizer_config.json (or it has no embedded chat_template). Before
  // this fallback existed, such a model directory silently loaded with no
  // chat template at all, forcing every caller through their non-instruct
  // fallback formatting.
  const auto dir = fs::temp_directory_path() / "ifx_tok_jinja_file";
  WriteByteLevelTokenizer(dir); // no tokenizer_config.json written
  {
    std::ofstream f(dir / "chat_template.jinja");
    f << "{{ '<|im_start|>' + role }}";
  }

  MlxTokenizer tok;
  REQUIRE(tok.Load(dir));
  REQUIRE(tok.HasChatTemplate());
  REQUIRE(tok.ChatTemplate().find("im_start") != std::string::npos);

  fs::remove_all(dir);
}

TEST_CASE("MlxTokenizer falls back to config.json bos/eos_token_id when "
          "tokenizer_config.json is absent",
          "[mlx_tokenizer]") {
  // Regression test: some safetensors conversions ship only tokenizer.json
  // + config.json (no tokenizer_config.json). Without this fallback,
  // bos_id_/eos_id_ silently kept Reset()'s defaults (1, 2) instead of the
  // model's real special-token IDs, breaking generation stopping.
  const auto dir = fs::temp_directory_path() / "ifx_tok_cfg_json_eos";
  WriteByteLevelTokenizer(dir); // no tokenizer_config.json written
  {
    nlohmann::json cfg;
    cfg["bos_token_id"] = 1;
    cfg["eos_token_id"] = 42; // distinct from Reset()'s default (2)
    std::ofstream f(dir / "config.json");
    f << cfg.dump();
  }

  MlxTokenizer tok;
  REQUIRE(tok.Load(dir));
  REQUIRE(tok.BosId() == 1);
  REQUIRE(tok.EosId() == 42);

  fs::remove_all(dir);
}

TEST_CASE("MlxTokenizer prefers tokenizer_config.json bos/eos over "
          "config.json when both are present",
          "[mlx_tokenizer]") {
  const auto dir = fs::temp_directory_path() / "ifx_tok_cfg_precedence";
  WriteByteLevelTokenizer(dir);
  {
    nlohmann::json tok_cfg;
    tok_cfg["bos_token"] = "<s>";
    tok_cfg["eos_token"] = "</s>";
    std::ofstream f(dir / "tokenizer_config.json");
    f << tok_cfg.dump();
  }
  {
    nlohmann::json model_cfg;
    model_cfg["bos_token_id"] = 99; // should be ignored
    model_cfg["eos_token_id"] = 99; // should be ignored
    std::ofstream f(dir / "config.json");
    f << model_cfg.dump();
  }

  MlxTokenizer tok;
  REQUIRE(tok.Load(dir));
  REQUIRE(tok.BosId() == 1); // from tokenizer_config.json's "<s>"
  REQUIRE(tok.EosId() == 2); // from tokenizer_config.json's "</s>"

  fs::remove_all(dir);
}
