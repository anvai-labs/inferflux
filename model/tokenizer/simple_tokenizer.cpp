#include "model/tokenizer/simple_tokenizer.h"

#include <cctype>
#include <sstream>

namespace inferflux {

SimpleTokenizer::SimpleTokenizer() {
  reverse_.resize(3);
  vocab_["<bos>"] = 1;
  vocab_["<eos>"] = 2;
  reverse_[1] = "<bos>";
  reverse_[2] = "<eos>";
}

int SimpleTokenizer::AddToken(const std::string &token) {
  auto it = vocab_.find(token);
  if (it != vocab_.end()) {
    return it->second;
  }
  int id = static_cast<int>(reverse_.size());
  vocab_[token] = id;
  reverse_.push_back(token);
  return id;
}

std::vector<std::string>
SimpleTokenizer::Tokenize(const std::string &text) const {
  std::vector<std::string> words;
  std::string current;
  for (char c : text) {
    if (std::isalnum(static_cast<unsigned char>(c))) {
      current.push_back(
          static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    } else {
      if (!current.empty()) {
        words.push_back(current);
        current.clear();
      }
      if (!std::isspace(static_cast<unsigned char>(c))) {
        std::string punct(1, c);
        words.push_back(punct);
      }
    }
  }
  if (!current.empty()) {
    words.push_back(current);
  }
  return words;
}

std::vector<int> SimpleTokenizer::Encode(const std::string &text) {
  auto words = Tokenize(text);
  if (words.empty()) {
    return {};
  }
  std::vector<int> tokens;
  tokens.push_back(1); // <bos>
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto &word : words) {
    tokens.push_back(AddToken(word));
  }
  return tokens;
}

std::string SimpleTokenizer::Decode(const std::vector<int> &tokens) const {
  // Lock against concurrent Encode's lazy reverse_ growth: a push_back
  // realloc mid-Decode would dangle the reverse_[token] references below.
  std::lock_guard<std::mutex> lock(mutex_);
  std::ostringstream stream;
  bool first_word = true;
  for (int token : tokens) {
    if (token <= 1 || token >= static_cast<int>(reverse_.size())) {
      continue;
    }
    const std::string &word = reverse_[token];
    if (!first_word && word != "." && word != "," && word != "!" &&
        word != "?") {
      stream << ' ';
    }
    stream << word;
    first_word = false;
  }
  return stream.str();
}

} // namespace inferflux
