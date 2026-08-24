#!/usr/bin/env python3

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT_DIR / "scripts" / "classify_benchmark_response.py"

spec = importlib.util.spec_from_file_location(
    "classify_benchmark_response", SCRIPT_PATH)
if spec is None or spec.loader is None:
  raise RuntimeError("failed to load scripts/classify_benchmark_response.py")
classifier = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = classifier
spec.loader.exec_module(classifier)


class BenchmarkResponseClassifierTests(unittest.TestCase):

  def write_payload(self, payload) -> Path:
    fd, path = tempfile.mkstemp(prefix="inferflux-bench-", suffix=".json")
    os.close(fd)
    p = Path(path)
    p.write_text(json.dumps(payload), encoding="utf-8")
    self.addCleanup(lambda: p.unlink(missing_ok=True))
    return p

  def test_classify_accepts_nonempty_positive_token_response(self):
    ok, reason = classifier.classify_response(
        {"text": "Paris", "tokens": 1, "latency_ms": 12})
    self.assertTrue(ok)
    self.assertEqual(reason, "ok")

  def test_classify_rejects_backend_empty_sentinel(self):
    ok, reason = classifier.classify_response(
        {"text": "[backend returned empty response]", "tokens": 4})
    self.assertFalse(ok)
    self.assertEqual(reason, "backend_empty_response")

  def test_classify_rejects_zero_token_payload(self):
    ok, reason = classifier.classify_response(
        {"text": "non-empty but invalid", "tokens": 0})
    self.assertFalse(ok)
    self.assertEqual(reason, "nonpositive_tokens")

  def test_main_rejects_invalid_json_file(self):
    fd, path = tempfile.mkstemp(prefix="inferflux-bench-invalid-", suffix=".json")
    os.close(fd)
    payload_path = Path(path)
    payload_path.write_text("not json", encoding="utf-8")
    self.addCleanup(lambda: payload_path.unlink(missing_ok=True))

    old_argv = sys.argv
    sys.argv = ["classify_benchmark_response.py", str(payload_path), "--json"]
    try:
      exit_code = classifier.main()
    finally:
      sys.argv = old_argv
    self.assertEqual(exit_code, 1)

  def test_main_returns_success_exit_for_valid_payload(self):
    payload_path = self.write_payload(
        {"text": "Hello", "tokens": 2, "latency_ms": 10})
    old_argv = sys.argv
    sys.argv = ["classify_benchmark_response.py", str(payload_path)]
    try:
      exit_code = classifier.main()
    finally:
      sys.argv = old_argv
    self.assertEqual(exit_code, 0)


if __name__ == "__main__":
  unittest.main()
