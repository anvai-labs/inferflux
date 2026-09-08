#include <catch2/catch_amalgamated.hpp>

#include "model/model_format.h"
#include "support/scoped_env.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

namespace fs = std::filesystem;

using namespace inferflux;
using inferflux::test::ScopedEnvVar;

namespace {

fs::path MakeTempDir(const std::string &suffix) {
  const auto base =
      fs::temp_directory_path() / ("ifx_model_format_" + suffix + "_" +
                                   std::to_string(std::hash<std::thread::id>{}(
                                       std::this_thread::get_id())));
  fs::create_directories(base);
  return base;
}

void TouchFile(const fs::path &path) {
  std::ofstream out(path);
  out << "stub";
}

} // namespace

TEST_CASE("Model format normalization accepts aliases", "[model_format]") {
  REQUIRE(NormalizeModelFormat("GGUF") == "gguf");
  REQUIRE(NormalizeModelFormat("safe_tensors") == "safetensors");
  REQUIRE(NormalizeModelFormat("huggingface") == "hf");
  REQUIRE(NormalizeModelFormat("unknown").empty());
}

TEST_CASE("ResolveGgufArtifactPath resolves files, directories, and misses",
          "[model_format]") {
  const auto dir = MakeTempDir("gguf_artifact");
  REQUIRE(ResolveGgufArtifactPath(dir.string()).empty());

  // A *.gguf file resolves to itself; non-gguf files do not resolve.
  TouchFile(dir / "weights.gguf");
  REQUIRE(ResolveGgufArtifactPath((dir / "weights.gguf").string()) ==
          (dir / "weights.gguf").string());
  TouchFile(dir / "model.safetensors");
  REQUIRE(ResolveGgufArtifactPath((dir / "model.safetensors").string())
              .empty());
  REQUIRE(ResolveGgufArtifactPath((dir / "does-not-exist.gguf").string())
              .empty());

  // A directory with gguf files resolves to the best-scoring artifact
  // (quant-aware ranking, f16 as the catch-all).
  TouchFile(dir / "model-f16.gguf");
  REQUIRE(ResolveGgufArtifactPath(dir.string()) ==
          (dir / "model-f16.gguf").string());
  TouchFile(dir / "model-q4_k_m.gguf");
  REQUIRE(ResolveGgufArtifactPath(dir.string()) ==
          (dir / "model-q4_k_m.gguf").string());

  // Extension match is case-insensitive.
  TouchFile(dir / "UPPER.GGUF");
  REQUIRE(ResolveGgufArtifactPath((dir / "UPPER.GGUF").string()) ==
          (dir / "UPPER.GGUF").string());

  std::error_code ec;
  fs::remove_all(dir, ec);
}

TEST_CASE("ResolveLlamaLoadPath picks highest-ranked GGUF sidecar",
          "[model_format]") {
  const auto dir = MakeTempDir("ranked");
  TouchFile(dir / "model.Q8_0.gguf");
  TouchFile(dir / "model.Q4_K_M.gguf");

  const auto resolved = ResolveLlamaLoadPath(dir.string(), "safetensors");
  REQUIRE(resolved == (dir / "model.Q4_K_M.gguf").string());

  fs::remove_all(dir);
}

TEST_CASE("ResolveLlamaLoadPath resolves hf URI via local cache",
          "[model_format]") {
  const auto home = MakeTempDir("hfcache");
  const auto repo_dir = home / "models" / "org" / "repo";
  fs::create_directories(repo_dir);
  TouchFile(repo_dir / "model.Q4_K_M.gguf");

  ScopedEnvVar env("INFERFLUX_HOME", home.string());
  const auto resolved = ResolveLlamaLoadPath("hf://org/repo", "hf");
  REQUIRE(resolved == (repo_dir / "model.Q4_K_M.gguf").string());

  fs::remove_all(home);
}

TEST_CASE("ResolveMlxLoadPath maps hf URI and safetensors file",
          "[model_format]") {
  const auto home = MakeTempDir("mlxpath");
  const auto repo_dir = home / "models" / "org" / "repo";
  fs::create_directories(repo_dir);

  ScopedEnvVar env("INFERFLUX_HOME", home.string());
  REQUIRE(ResolveMlxLoadPath("hf://org/repo", "hf") == repo_dir.string());

  const auto sf_dir = MakeTempDir("mlxsf");
  const auto sf_file = sf_dir / "model.safetensors";
  TouchFile(sf_file);
  REQUIRE(ResolveMlxLoadPath(sf_file.string(), "safetensors") ==
          sf_dir.string());

  fs::remove_all(home);
  fs::remove_all(sf_dir);
}

TEST_CASE("ResolveLlamaLoadPath returns empty when no GGUF exists",
          "[model_format]") {
  const auto dir = MakeTempDir("nogguf");
  TouchFile(dir / "model.safetensors");

  REQUIRE(ResolveLlamaLoadPath(dir.string(), "safetensors").empty());

  fs::remove_all(dir);
}
