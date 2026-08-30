#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace inferflux {

class SimpleTokenizer {
public:
  SimpleTokenizer();

  std::vector<int> Encode(const std::string &text);
  std::string Decode(const std::vector<int> &tokens) const;

private:
  // Callers must hold mutex_: Encode/Decode serialize all vocab access
  // because Scheduler::Generate invokes Encode concurrently from HTTP
  // worker threads and the lazy AddToken append mutates both containers
  // (the unlocked version was an intermittent heap-corruption crash at
  // concurrency >= 4).
  int AddToken(const std::string &token);
  std::vector<std::string> Tokenize(const std::string &text) const;

  mutable std::mutex mutex_;
  std::unordered_map<std::string, int> vocab_;
  std::vector<std::string> reverse_;
};

} // namespace inferflux
