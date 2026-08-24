#pragma once

#include <cstdint>
#include <string>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <unordered_map>
#include <vector>

namespace inferflux {

/**
 * Safetensors binary format parser
 *
 * Implements the safetensors format specification:
 * https://huggingface.co/docs/safetensors/index.html
 *
 * Format structure:
 * - Header (8 bytes): magic + header_size
 * - Metadata: JSON (length from header)
 * - Tensor data: binary tensors
 */
class SafetensorsParser {
public:
  struct TensorInfo {
    std::string name;
    std::vector<size_t> shape;
    std::string dtype; // "F16", "F32", "I32", etc.
    size_t offset;     // Offset in file
    size_t byte_size;  // Total bytes

    // Mapped data (after Map())
    void *data_ptr{nullptr};
    bool is_mapped{false};

    ~TensorInfo() { Unmap(); }

    void Unmap() {
      if (is_mapped && data_ptr) {
#ifdef _WIN32
        UnmapViewOfFile(data_ptr);
#else
        if (data_ptr != MAP_FAILED) {
          munmap(data_ptr, byte_size);
        }
#endif
        data_ptr = nullptr;
        is_mapped = false;
      }
    }
  };

  SafetensorsParser(const std::string &file_path);
  ~SafetensorsParser();

  /**
   * Parse the safetensors file
   * @return true on success
   */
  bool Parse();

  /**
   * Get tensor info by name
   */
  const TensorInfo *GetTensor(const std::string &name) const;

  /**
   * Get all tensor names
   */
  std::vector<std::string> GetTensorNames() const;

  /**
   * Get file size
   */
  size_t GetFileSize() const { return file_size_; }

  /**
   * Get metadata JSON
   */
  const std::string &GetMetadata() const { return metadata_; }

private:
  bool ParseHeader();
  bool ParseMetadata();
  bool ParseTensorData();

  // File handling
  std::string file_path_;
#ifdef _WIN32
  HANDLE file_handle_{INVALID_HANDLE_VALUE};
  HANDLE mapping_handle_{nullptr};
#else
  int fd_{-1};
#endif
  size_t file_size_{0};
  void *file_data_{nullptr}; // Memory-mapped file
  bool is_mapped_{false};

  // Parsed data
  std::string metadata_;
  std::unordered_map<std::string, TensorInfo> tensors_;

  // Header
  struct {
    uint8_t magic[8];
    uint8_t header_length[8];
  } header_;
};

/**
 * Safetensors data type conversions
 */
namespace SafetensorsDtype {
inline size_t GetDtypeSize(const std::string &dtype) {
  // Float types
  if (dtype == "F16" || dtype == "F16_LE")
    return 2;
  if (dtype == "BF16")
    return 2; // BFloat16
  if (dtype == "F32" || dtype == "F32_LE")
    return 4;
  if (dtype == "F64" || dtype == "F64_LE")
    return 8;

  // Integer types
  if (dtype == "I8" || dtype == "I8_LE")
    return 1;
  if (dtype == "I16" || dtype == "I16_LE")
    return 2;
  if (dtype == "I32" || dtype == "I32_LE")
    return 4;
  if (dtype == "I64" || dtype == "I64_LE")
    return 8;

  // Unsigned integer types
  if (dtype == "U8" || dtype == "U8_LE")
    return 1;
  if (dtype == "U16" || dtype == "U16_LE")
    return 2;
  if (dtype == "U32" || dtype == "U32_LE")
    return 4;
  if (dtype == "U64" || dtype == "U64_LE")
    return 8;

  // Boolean
  if (dtype == "BOOL" || dtype == "BOOL8")
    return 1;

  return 1; // Default fallback
}

inline bool IsFloatDtype(const std::string &dtype) {
  return dtype.find("F") != std::string::npos ||
         dtype.find("BF") != std::string::npos;
}

inline bool IsIntDtype(const std::string &dtype) {
  return dtype.find("I") != std::string::npos ||
         dtype.find("U") != std::string::npos;
}
} // namespace SafetensorsDtype

} // namespace inferflux
