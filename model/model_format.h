#pragma once

#include <string>

namespace inferflux {

// Canonical model formats.
// Supported values: auto, gguf, safetensors, hf.
std::string NormalizeModelFormat(const std::string &format);
bool IsModelFormatValue(const std::string &format);

// Best-effort format detection from filesystem path.
// Returns one of: gguf, safetensors, hf, unknown.
std::string DetectModelFormat(const std::string &path);

// Resolve effective format for model loading.
// requested_format may be empty or "auto".
// Falls back to "gguf" when auto-detection is inconclusive for compatibility.
std::string ResolveModelFormat(const std::string &path,
                               const std::string &requested_format);

// Resolve HuggingFace URI-style references (hf://org/repo) to the local
// InferFlux cache directory:
// ${INFERFLUX_HOME:-$HOME/.inferflux}/models/org/repo. Returns the original
// path when it is not an hf:// reference.
std::string ResolveHfReferenceToCachePath(const std::string &path);

// Resolve an MLX-compatible load path for hf/safetensors formats.
// - hf://org/repo => local cache directory
// - *.safetensors file => parent directory
// Returns the original path for other cases/formats.
std::string ResolveMlxLoadPath(const std::string &path,
                               const std::string &resolved_format);

// Resolve a GGUF artifact path consumable by llama.cpp-style backends.
// For hf/safetensors inputs this attempts to locate a *.gguf sidecar in:
//   1) local HF cache directory (when path uses hf://...)
//   2) the provided model directory path.
// Returns empty string when no compatible GGUF artifact is found.
std::string ResolveLlamaLoadPath(const std::string &path,
                                 const std::string &resolved_format);

// Resolve a concrete GGUF artifact for gguf-format inputs: returns the path
// itself when it is a *.gguf file, the best-scoring *.gguf inside it when it
// is a directory, and empty otherwise. Model format auto-detection can hand
// routers a mixed directory (safetensors + GGUF sidecar); single-file loaders
// need the file.
std::string ResolveGgufArtifactPath(const std::string &path);

// True when the current llama.cpp-backed loader can consume the format.
bool IsLoadableByLlamaBackend(const std::string &resolved_format);

} // namespace inferflux
