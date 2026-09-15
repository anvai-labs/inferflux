#pragma once

#include "model/chat_template_renderer.h"
#include "runtime/text/harmony_splitter.h"
#include "runtime/text/reasoning_splitter.h"

#include <string>
#include <string_view>
#include <variant>

namespace inferflux {

// Selects and owns the response-format-appropriate splitter — ReasoningSplitter
// for the ChatML/Llama/Mistral/Gemma default (<think> tags), HarmonySplitter
// for gpt-oss's channel markers — based on a request's detected chat
// template family, and forwards the shared Feed/Finish/Drain/in_reasoning
// API to whichever one is active. Both splitters share this exact method
// shape by design; this wrapper exists so call sites don't repeat the same
// family branch five times over. One instance per request.
class ResponseSplitter {
public:
  struct Parts {
    std::string reasoning;
    std::string content;
  };

  explicit ResponseSplitter(ChatTemplateFamily family)
      : impl_(family == ChatTemplateFamily::kHarmony
                  ? Impl(std::in_place_type<HarmonySplitter>)
                  : Impl(std::in_place_type<ReasoningSplitter>)) {}

  void Feed(std::string_view piece) {
    std::visit([&](auto &s) { s.Feed(piece); }, impl_);
  }

  Parts Finish() {
    return std::visit(
        [](auto &s) -> Parts {
          auto p = s.Finish();
          return {std::move(p.reasoning), std::move(p.content)};
        },
        impl_);
  }

  Parts Drain() {
    return std::visit(
        [](auto &s) -> Parts {
          auto p = s.Drain();
          return {std::move(p.reasoning), std::move(p.content)};
        },
        impl_);
  }

  bool in_reasoning() const {
    return std::visit([](const auto &s) { return s.in_reasoning(); }, impl_);
  }

  // Whole-string convenience path (matches each splitter's own Split()).
  static Parts Split(ChatTemplateFamily family, std::string_view output) {
    if (family == ChatTemplateFamily::kHarmony) {
      auto p = HarmonySplitter::Split(output);
      return {std::move(p.reasoning), std::move(p.content)};
    }
    auto p = ReasoningSplitter::Split(output);
    return {std::move(p.reasoning), std::move(p.content)};
  }

private:
  using Impl = std::variant<ReasoningSplitter, HarmonySplitter>;
  Impl impl_;
};

} // namespace inferflux
