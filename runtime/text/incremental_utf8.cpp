#include "runtime/text/incremental_utf8.h"

#include <algorithm>

namespace inferflux {
namespace {

constexpr std::string_view kReplacementCharacter{"\xEF\xBF\xBD", 3};

bool IsContinuation(unsigned char byte) { return byte >= 0x80 && byte <= 0xBF; }

int SequenceLength(unsigned char lead) {
  if (lead <= 0x7F) {
    return 1;
  }
  if (lead >= 0xC2 && lead <= 0xDF) {
    return 2;
  }
  if (lead >= 0xE0 && lead <= 0xEF) {
    return 3;
  }
  if (lead >= 0xF0 && lead <= 0xF4) {
    return 4;
  }
  return 0;
}

bool IsValidSecondByte(unsigned char lead, unsigned char second) {
  if (!IsContinuation(second)) {
    return false;
  }
  if (lead == 0xE0) {
    return second >= 0xA0;
  }
  if (lead == 0xED) {
    return second <= 0x9F;
  }
  if (lead == 0xF0) {
    return second >= 0x90;
  }
  if (lead == 0xF4) {
    return second <= 0x8F;
  }
  return true;
}

} // namespace

Utf8AssemblyResult IncrementalUtf8Assembler::Append(std::string_view bytes) {
  pending_.append(bytes);
  return Consume(false);
}

Utf8AssemblyResult IncrementalUtf8Assembler::Finish() { return Consume(true); }

Utf8AssemblyResult IncrementalUtf8Assembler::Consume(bool finish) {
  Utf8AssemblyResult result;
  std::size_t offset = 0;

  const auto replace_invalid = [&]() {
    result.text.append(kReplacementCharacter);
    ++result.replacements;
  };

  while (offset < pending_.size()) {
    const auto lead = static_cast<unsigned char>(pending_[offset]);
    const int length = SequenceLength(lead);
    if (length == 1) {
      result.text.push_back(pending_[offset++]);
      continue;
    }
    if (length == 0) {
      replace_invalid();
      ++offset;
      continue;
    }

    const std::size_t available = pending_.size() - offset;
    const std::size_t available_tail =
        std::min<std::size_t>(available, static_cast<std::size_t>(length));
    bool valid_available_prefix =
        available_tail < 2 ||
        IsValidSecondByte(lead,
                          static_cast<unsigned char>(pending_[offset + 1]));
    for (std::size_t i = 2; valid_available_prefix && i < available_tail; ++i) {
      valid_available_prefix =
          IsContinuation(static_cast<unsigned char>(pending_[offset + i]));
    }
    if (!valid_available_prefix) {
      replace_invalid();
      ++offset;
      continue;
    }

    if (available < static_cast<std::size_t>(length)) {
      if (!finish) {
        break;
      }
      replace_invalid();
      offset = pending_.size();
      break;
    }

    result.text.append(pending_, offset, static_cast<std::size_t>(length));
    offset += static_cast<std::size_t>(length);
  }

  pending_.erase(0, offset);
  return result;
}

} // namespace inferflux
