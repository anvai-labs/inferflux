#include "runtime/structured_output/structured_output_adapter.h"

#include <nlohmann/json.hpp>

#include "json-schema-to-grammar.h"

namespace inferflux {
namespace {

bool ParseSchema(const std::string &schema, nlohmann::ordered_json *out,
                 std::string *error) {
  if (!out) {
    if (error) {
      *error = "internal error: null schema target";
    }
    return false;
  }
  if (schema.empty()) {
    if (error) {
      *error = "response_format schema is empty";
    }
    return false;
  }
  try {
    *out = nlohmann::ordered_json::parse(schema);
    return true;
  } catch (const nlohmann::json::exception &ex) {
    if (error) {
      *error = std::string("invalid response_format schema: ") + ex.what();
    }
    return false;
  }
}

bool ConvertSchemaToGrammar(const std::string &schema_text,
                            StructuredConstraint *out, std::string *error) {
  if (!out) {
    if (error) {
      *error = "internal error: null constraint";
    }
    return false;
  }
  try {
    // llama.cpp 0.7 exposes json_schema_to_grammar over common_json (its
    // nlohmann-independent value type) — parse the already-validated schema
    // text directly into that type instead of converting the ordered_json.
    const common_json schema = common_json::parse(schema_text);
    std::string grammar = json_schema_to_grammar(schema, /*force_gbnf=*/true);
    if (grammar.empty()) {
      if (error) {
        *error = "response_format schema could not produce grammar";
      }
      return false;
    }
    out->has_grammar = true;
    out->grammar = std::move(grammar);
    out->root = "root";
    return true;
  } catch (const std::exception &ex) {
    if (error) {
      *error =
          std::string("response_format schema conversion failed: ") + ex.what();
    }
  }
  return false;
}

bool BuildJsonObjectConstraint(const std::string &schema_text,
                               StructuredConstraint *out, std::string *error) {
  out->require_json_object = true;
  if (schema_text.empty()) {
    static const char *kDefaultObjectSchema = R"({"type":"object"})";
    nlohmann::ordered_json parsed =
        nlohmann::ordered_json::parse(kDefaultObjectSchema);
    out->has_schema = true;
    out->schema = parsed.dump();
    return ConvertSchemaToGrammar(kDefaultObjectSchema, out, error);
  }
  nlohmann::ordered_json parsed;
  if (!ParseSchema(schema_text, &parsed, error)) {
    return false;
  }
  out->has_schema = true;
  out->schema = parsed.dump();
  return ConvertSchemaToGrammar(schema_text, out, error);
}

} // namespace

bool StructuredOutputAdapter::BuildConstraint(const std::string &type,
                                              const std::string &schema,
                                              const std::string &grammar,
                                              const std::string &root,
                                              StructuredConstraint *out,
                                              std::string *error) {
  if (!out) {
    if (error) {
      *error = "internal error: null constraint";
    }
    return false;
  }
  *out = StructuredConstraint{};
  if (type.empty()) {
    return true;
  }

  if (type == "json_object") {
    return BuildJsonObjectConstraint(schema, out, error);
  }

  if (type == "json_schema") {
    if (schema.empty()) {
      if (error) {
        *error = "response_format json_schema requires a schema definition";
      }
      return false;
    }
    nlohmann::ordered_json parsed;
    if (!ParseSchema(schema, &parsed, error)) {
      return false;
    }
    out->require_json_object = true;
    out->has_schema = true;
    out->schema = parsed.dump();
    return ConvertSchemaToGrammar(schema, out, error);
  }

  if (type == "grammar") {
    if (grammar.empty()) {
      if (error) {
        *error = "response_format grammar string is empty";
      }
      return false;
    }
    out->has_grammar = true;
    out->grammar = grammar;
    out->root = root.empty() ? "root" : root;
    return true;
  }

  if (type == "text") {
    return true;
  }

  if (error) {
    *error = "Unsupported response_format type: " + type;
  }
  return false;
}

} // namespace inferflux
