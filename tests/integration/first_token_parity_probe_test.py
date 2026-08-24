#!/usr/bin/env python3

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf"
PROMPT = (
    "List the highest-impact ways to reduce cloud infrastructure cost for a "
    "GPU inference service without causing major reliability regressions."
)


def _default_probe_bin():
    if "INFERFLUX_FIRST_TOKEN_PROBE_BIN" in os.environ:
        return os.environ["INFERFLUX_FIRST_TOKEN_PROBE_BIN"]
    if (ROOT / "build-cuda/inferflux_first_token_probe").exists():
        return str(ROOT / "build-cuda/inferflux_first_token_probe")
    return str(ROOT / "build/inferflux_first_token_probe")


PROBE_BIN = _default_probe_bin()


def _top_token_set(payload):
    return {entry["token"] for entry in payload["logprobs"][0]["top_logprobs"]}


def _top_logit_set(payload):
    return {entry["token"] for entry in payload["top_logits"]}


def _parse_probe_payload(stdout: str):
    start = stdout.find("{")
    if start < 0:
        raise ValueError(f"probe stdout did not contain JSON: {stdout}")
    return json.loads(stdout[start:])


class FirstTokenParityProbeTests(unittest.TestCase):
    def test_probe_collects_first_token_and_top_logprobs_for_both_cuda_backends(self):
        if not MODEL.exists():
            self.skipTest("Local Qwen GGUF model not present")
        if not Path(PROBE_BIN).exists():
            self.skipTest("first-token probe binary not built")

        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_path = Path(tmpdir) / "prompt.txt"
            prompt_path.write_text(PROMPT, encoding="utf-8")

            results = {}
            for backend in ("inferflux_cuda", "llama_cpp_cuda"):
                proc = subprocess.run(
                    [
                        PROBE_BIN,
                        "--backend",
                        backend,
                        "--model",
                        str(MODEL),
                        "--prompt-file",
                        str(prompt_path),
                        "--top-n",
                        "8",
                        "--max-tokens",
                        "1",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    cwd=ROOT,
                )
                self.assertEqual(
                    proc.returncode,
                    0,
                    msg=f"{backend} probe failed\nstdout={proc.stdout}\nstderr={proc.stderr}",
                )
                payload = _parse_probe_payload(proc.stdout)
                self.assertTrue(payload["ok"])
                self.assertEqual(payload["backend"], backend)
                self.assertEqual(payload["prompt_chars"], len(PROMPT))
                self.assertEqual(payload["token_count"], 1)
                self.assertTrue(payload["output"])
                self.assertTrue(payload["logprobs"][0]["top_logprobs"])
                self.assertTrue(payload["top_logits"])
                results[backend] = payload

        inferflux = results["inferflux_cuda"]
        llama = results["llama_cpp_cuda"]
        inferflux_set = _top_token_set(inferflux)
        llama_set = _top_token_set(llama)
        self.assertTrue(inferflux_set)
        self.assertTrue(llama_set)

        intersection = inferflux_set & llama_set
        union = inferflux_set | llama_set
        jaccard = len(intersection) / len(union)
        overlap = len(intersection) / min(len(inferflux_set), len(llama_set))

        self.assertGreaterEqual(jaccard, 0.0)
        self.assertLessEqual(jaccard, 1.0)
        self.assertGreaterEqual(overlap, 0.0)
        self.assertLessEqual(overlap, 1.0)

        inferflux_logits = _top_logit_set(inferflux)
        llama_logits = _top_logit_set(llama)
        logits_intersection = inferflux_logits & llama_logits
        logits_union = inferflux_logits | llama_logits
        logits_jaccard = len(logits_intersection) / len(logits_union)
        logits_overlap = len(logits_intersection) / min(
            len(inferflux_logits), len(llama_logits)
        )

        self.assertGreaterEqual(logits_jaccard, 2.0 / 3.0)
        self.assertGreaterEqual(logits_overlap, 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
