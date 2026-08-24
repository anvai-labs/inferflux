#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path


EMPTY_SENTINEL = "[backend returned empty response]"


def classify_response(payload: dict) -> tuple[bool, str]:
  text = str(payload.get("text", "")).strip()
  try:
    tokens = int(payload.get("tokens", 0))
  except (TypeError, ValueError):
    return False, "invalid_tokens"

  if not text:
    return False, "empty_text"
  if text == EMPTY_SENTINEL:
    return False, "backend_empty_response"
  if tokens <= 0:
    return False, "nonpositive_tokens"
  return True, "ok"


def main() -> int:
  parser = argparse.ArgumentParser(
      description="Classify one GGUF benchmark response artifact.")
  parser.add_argument("response_file", type=Path)
  parser.add_argument("--json", action="store_true", dest="emit_json")
  args = parser.parse_args()

  try:
    payload = json.loads(args.response_file.read_text(encoding="utf-8"))
  except FileNotFoundError:
    result = {"ok": False, "reason": "missing_file"}
  except json.JSONDecodeError:
    result = {"ok": False, "reason": "invalid_json"}
  else:
    ok, reason = classify_response(payload)
    result = {"ok": ok, "reason": reason}

  if args.emit_json:
    print(json.dumps(result))
  else:
    print(result["reason"])
  return 0 if result["ok"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
