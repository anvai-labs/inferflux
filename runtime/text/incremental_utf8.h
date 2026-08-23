#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace inferflux {

struct Utf8AssemblyResult {
  std::string text;
  std::size_t replacements{0};
};

// Incrementally assembles tokenizer byte pieces into valid UTF-8. Incomplete
// trailing sequences remain buffered until Append() receives more bytes or
// Finish() closes the stream. Malformed input is replaced with U+FFFD.
class IncrementalUtf8Assembler {
public:
  Utf8AssemblyResult Append(std::string_view bytes);
  Utf8AssemblyResult Finish();

  bool HasPendingBytes() const { return !pending_.empty(); }

private:
  Utf8AssemblyResult Consume(bool finish);

  std::string pending_;
};

} // namespace inferflux
