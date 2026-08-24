#include "runtime/backends/cuda/native/safetensors_parser.h"
#include "server/logging/logger.h"

#include <cstring>
#include <fstream>
#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <nlohmann/json.hpp>

namespace inferflux {

using json = nlohmann::json;

//==============================================================================
// SafetensorsParser Implementation
//==============================================================================

SafetensorsParser::SafetensorsParser(const std::string &file_path)
    : file_path_(file_path) {}

SafetensorsParser::~SafetensorsParser() {
#ifdef _WIN32
  if (is_mapped_ && file_data_) {
    UnmapViewOfFile(file_data_);
    file_data_ = nullptr;
    is_mapped_ = false;
  }
  if (mapping_handle_) {
    CloseHandle(mapping_handle_);
    mapping_handle_ = nullptr;
  }
  if (file_handle_ != INVALID_HANDLE_VALUE) {
    CloseHandle(file_handle_);
    file_handle_ = INVALID_HANDLE_VALUE;
  }
#else
  if (is_mapped_ && file_data_ && file_data_ != MAP_FAILED) {
    munmap(file_data_, file_size_);
    file_data_ = nullptr;
    is_mapped_ = false;
  }
  if (fd_ >= 0) {
    close(fd_);
  }
#endif
}

bool SafetensorsParser::Parse() {
#ifdef _WIN32
  // Open file
  file_handle_ =
      CreateFileA(file_path_.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file_handle_ == INVALID_HANDLE_VALUE) {
    log::Error("safetensors_parser", "Cannot open file: " + file_path_);
    return false;
  }

  // Get file size
  LARGE_INTEGER li;
  if (!GetFileSizeEx(file_handle_, &li)) {
    log::Error("safetensors_parser", "Cannot get file size: " + file_path_);
    return false;
  }
  file_size_ = static_cast<size_t>(li.QuadPart);

  log::Info("safetensors_parser",
            "Parsing file: " + file_path_ + " (size: " +
                std::to_string(file_size_ / (1024 * 1024)) + " MB)");

  // Memory map the file
  mapping_handle_ =
      CreateFileMappingA(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!mapping_handle_) {
    log::Error("safetensors_parser",
               "Failed to create file mapping: " + file_path_);
    return false;
  }
  file_data_ = MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0);
  if (!file_data_) {
    log::Error("safetensors_parser",
               "Failed to map view of file: " + file_path_);
    return false;
  }
  is_mapped_ = true;
#else
  // Open file
  fd_ = open(file_path_.c_str(), O_RDONLY);
  if (fd_ < 0) {
    log::Error("safetensors_parser", "Cannot open file: " + file_path_);
    return false;
  }

  // Get file size
  struct stat st;
  if (fstat(fd_, &st) != 0) {
    log::Error("safetensors_parser", "Cannot stat file: " + file_path_);
    return false;
  }
  file_size_ = st.st_size;

  log::Info("safetensors_parser",
            "Parsing file: " + file_path_ + " (size: " +
                std::to_string(file_size_ / (1024 * 1024)) + " MB)");

  // Memory map the file
  file_data_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (file_data_ == MAP_FAILED) {
    log::Error("safetensors_parser", "Failed to mmap file: " + file_path_);
    return false;
  }
  is_mapped_ = true;
#endif

  // Parse components
  if (!ParseHeader()) {
    log::Error("safetensors_parser", "Failed to parse header");
    return false;
  }

  if (!ParseMetadata()) {
    log::Error("safetensors_parser", "Failed to parse metadata");
    return false;
  }

  if (!ParseTensorData()) {
    log::Error("safetensors_parser", "Failed to parse tensor data");
    return false;
  }

  log::Info("safetensors_parser", "Successfully parsed " +
                                      std::to_string(tensors_.size()) +
                                      " tensors");
  return true;
}

bool SafetensorsParser::ParseHeader() {
  if (file_size_ < 8) {
    log::Error("safetensors_parser", "File too small for header");
    return false;
  }

  uint8_t *data = static_cast<uint8_t *>(file_data_);

  // Safetensors format: [8 bytes: length of JSON metadata (LE)] [JSON metadata]
  // [tensor data] Extract metadata length (little-endian uint64)
  uint64_t metadata_len;
  memcpy(&metadata_len, data, 8);

  // Sanity check: metadata should be reasonable size (< 100 MB)
  if (metadata_len > 100 * 1024 * 1024) {
    log::Error("safetensors_parser",
               "Metadata size too large: " + std::to_string(metadata_len) +
                   " bytes");
    return false;
  }

  if (metadata_len + 8 > file_size_) {
    log::Error("safetensors_parser", "Metadata extends beyond file");
    return false;
  }

  // Store header length for later use
  memcpy(header_.header_length, data, 8);

  log::Info("safetensors_parser",
            "Header: metadata_size=" + std::to_string(metadata_len) + " bytes");

  return true;
}

bool SafetensorsParser::ParseMetadata() {
  uint8_t *data = static_cast<uint8_t *>(file_data_);

  // Extract header length
  uint64_t metadata_len;
  memcpy(&metadata_len, header_.header_length, 8);

  // Metadata starts at offset 8 (after the length field)
  size_t metadata_offset = 8;

  if (metadata_offset + metadata_len > file_size_) {
    log::Error("safetensors_parser", "Metadata extends beyond file");
    return false;
  }

  // Extract JSON metadata
  metadata_ = std::string(reinterpret_cast<char *>(data + metadata_offset),
                          metadata_len);

  log::Info("safetensors_parser",
            "Metadata size: " + std::to_string(metadata_len) + " bytes");

  // Parse JSON
  try {
    json metadata = json::parse(metadata_);

    // Safetensors format: flat JSON where tensor keys don't start with "__"
    // Keys starting with "__" are metadata (e.g., "__metadata__")
    size_t tensor_count = 0;
    for (auto &[name, tensor_data] : metadata.items()) {
      // Skip metadata keys (start with "__")
      if (name.find("__") == 0) {
        continue;
      }

      if (!tensor_data.is_object()) {
        continue;
      }

      TensorInfo info;
      info.name = name;

      // Parse shape
      if (tensor_data.contains("shape")) {
        auto shape = tensor_data["shape"];
        if (shape.is_array()) {
          for (auto &dim : shape) {
            info.shape.push_back(dim.get<size_t>());
          }
        }
      }

      // Parse dtype
      if (tensor_data.contains("dtype")) {
        info.dtype = tensor_data["dtype"];
      }

      // Parse offsets (safetensors stores start/offset)
      if (tensor_data.contains("data_offsets")) {
        auto offsets = tensor_data["data_offsets"];
        if (offsets.is_array() && offsets.size() >= 2) {
          info.offset = offsets[0].get<size_t>();
          size_t end_offset = offsets[1].get<size_t>();
          info.byte_size = end_offset - info.offset;
        }
      }

      tensors_[name] = info;
      tensor_count++;
    }

    log::Info("safetensors_parser", "Parsed " + std::to_string(tensor_count) +
                                        " tensors from metadata");
    return true;

  } catch (const std::exception &e) {
    log::Error("safetensors_parser",
               "JSON parse error: " + std::string(e.what()));
    return false;
  }
}

bool SafetensorsParser::ParseTensorData() {
  uint8_t *data = static_cast<uint8_t *>(file_data_);

  // Calculate where tensor data starts
  uint64_t metadata_len;
  memcpy(&metadata_len, header_.header_length, 8);
  size_t data_start = 8 + metadata_len;

  // Align to 8 bytes (safetensors alignment)
  data_start = (data_start + 7) & ~7ULL;

  log::Info("safetensors_parser",
            "Tensor data starts at offset: " + std::to_string(data_start));

  // Map each tensor to its data location
  for (auto &[name, info] : tensors_) {
    info.data_ptr = data + data_start + info.offset;

    // Validate offset is within file
    if (data_start + info.offset + info.byte_size > file_size_) {
      log::Error("safetensors_parser",
                 "Tensor " + name + " extends beyond file bounds");
      return false;
    }

    // Calculate element count
    size_t elem_count = 1;
    for (auto dim : info.shape) {
      elem_count *= dim;
    }

    size_t expected_size =
        elem_count * SafetensorsDtype::GetDtypeSize(info.dtype);
    if (expected_size != info.byte_size) {
      log::Warn("safetensors_parser",
                "Tensor " + name + " size mismatch: expected " +
                    std::to_string(expected_size) + ", got " +
                    std::to_string(info.byte_size));
    }
  }

  return true;
}

const SafetensorsParser::TensorInfo *
SafetensorsParser::GetTensor(const std::string &name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    return nullptr;
  }
  return &it->second;
}

std::vector<std::string> SafetensorsParser::GetTensorNames() const {
  std::vector<std::string> names;
  names.reserve(tensors_.size());
  for (const auto &[name, _] : tensors_) {
    names.push_back(name);
  }
  return names;
}

} // namespace inferflux
