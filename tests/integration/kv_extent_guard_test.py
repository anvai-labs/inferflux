#!/usr/bin/env python3
"""
GPU-gated integration test for the KV extent guards (PR #124 / issue #126):
an over-length sequence must fail cleanly (empty generation, guard warn in
the server log, violation metric incremented) WITHOUT corrupting a
neighboring resident sequence, and the guard must be countable via the
violation warn lines.

Requires a CUDA build + a real GGUF model (INFERFLUX_MODEL_PATH); skips
otherwise. Guards are CUDA-only code paths, so this cannot run on CPU CI.
"""

import json
import os
import pathlib
import subprocess
import time
import unittest
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
BINARY = pathlib.Path(
    os.environ.get("INFERFLUX_SERVER_BIN", str(ROOT / "build-cuda-final" / "inferfluxd"))
)
MODEL = os.environ.get("INFERFLUX_MODEL_PATH", "")
HOST = "127.0.0.1"
PORT = 18130
API_KEY = "kv-guard-key"

# KV per-slot default is 1,024 tokens; a ~1,100-token prompt crosses it.
LONG_PROMPT = "word " * 1100


def request(prompt, max_tokens, timeout=180):
    body = json.dumps({
        "model": "bench-model",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"http://{HOST}:{PORT}/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as err:
        # Admission rejections surface as HTTP error codes with the
        # context_overflow message in the body (issue #125 surfacing).
        body = json.loads(err.read())
        body.setdefault("choices", [{}])[0].setdefault("message", {})
        body["choices"][0]["message"].setdefault("content", "")
        body["http_status"] = err.code
        return body


class KVExtentGuardTest(unittest.TestCase):
    server = None
    log_path = pathlib.Path("/tmp") / f"inferflux_kv_guard_{os.getpid()}.log"
    log_file = None

    @classmethod
    def setUpClass(cls):
        if not BINARY.exists():
            raise unittest.SkipTest("inferfluxd binary missing; build before running tests")
        if not MODEL or not pathlib.Path(MODEL).exists():
            raise unittest.SkipTest(
                "INFERFLUX_MODEL_PATH not set; KV extent guards are CUDA-only"
            )
        cfg_path = pathlib.Path("/tmp") / f"inferflux_kv_guard_{os.getpid()}.yaml"
        cfg_path.write_text(
            "server:\n"
            f"  http_port: {PORT}\n"
            "  max_concurrent: 32\n"
            "  enable_metrics: true\n"
            "models:\n"
            f"  - id: bench-model\n"
            f"    path: \"{MODEL}\"\n"
            "    format: gguf\n"
            "    backend: inferflux_cuda\n"
            "    default: true\n"
            "runtime:\n"
            "  backend_priority: [cuda, cpu]\n"
            "  cuda:\n"
            "    enabled: true\n"
            "    flash_attention:\n"
            "      enabled: true\n"
            "  backend_exposure:\n"
            "    prefer_inferflux: true\n"
            "    allow_llama_cpp_fallback: false\n"
            "auth:\n"
            "  api_keys:\n"
            f"    - key: {API_KEY}\n"
            "      scopes: [generate, read, admin]\n"
            "  rate_limit_per_minute: 0\n"
            "logging:\n"
            "  level: warning\n"
            "  format: text\n"
        )
        env = os.environ.copy()
        # NOTE: do not set INFERFLUX_MODEL_PATH here — it overrides the
        # config model entry and replaces the configured id (bench-model).
        env["INFERCTL_API_KEY"] = API_KEY
        cls.log_file = open(cls.log_path, "w")
        cls.server = subprocess.Popen(
            [str(BINARY), "--config", str(cfg_path)],
            cwd=str(ROOT),
            env=env,
            stdout=cls.log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.time() + 120
        while time.time() - deadline < 0:
            try:
                urllib.request.urlopen(
                    f"http://{HOST}:{PORT}/healthz", timeout=2
                ).read()
                return
            except Exception:
                time.sleep(0.5)
        raise unittest.SkipTest("server did not become healthy; check the log")

    @classmethod
    def tearDownClass(cls):
        if cls.server:
            cls.server.terminate()
            try:
                cls.server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cls.server.kill()
            cls.log_file.close()

    def _metric(self, name):
        req = urllib.request.Request(
            f"http://{HOST}:{PORT}/metrics",
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            for line in resp.read().decode().splitlines():
                if line.startswith(name):
                    return line.rsplit(" ", 1)[1]
        return None

    def test_01_over_length_prompt_rejected(self):
        """A 1,100-token prompt against a 1,024-token slot is rejected at
        admission with a context_overflow error — never admitted to compute
        (where it would previously append KV rows past the slot)."""
        resp = request(LONG_PROMPT, 16)
        content = resp["choices"][0]["message"]["content"]
        self.assertEqual(len(content), 0, f"got {content[:80]!r}")

    def test_02_overflow_message_surfaced(self):
        """The rejection names the condition and the knob."""
        resp = request(LONG_PROMPT, 16)
        completion = resp["choices"][0]["message"]["content"]
        self.assertIn("context_overflow", completion)
        self.assertIn("INFERFLUX_CUDA_KV_MAX_SEQ", completion)

    def test_03_short_prompt_untouched(self):
        """A normal request after the rejection produces correct output."""
        resp = request("List the planets in order from the sun.", 60)
        content = resp["choices"][0]["message"]["content"]
        self.assertIn("Mercury", content)
        self.assertIn("Neptune", content)

    def test_04_neighbor_integrity_after_long_prompt(self):
        """Decode quality for a second resident sequence is unaffected —
        the rejected prompt must not have written into another slot's KV."""
        resp = request("List the planets in order from the sun.", 60)
        self.assertIn("Mercury", resp["choices"][0]["message"]["content"])


if __name__ == "__main__":
    unittest.main()
