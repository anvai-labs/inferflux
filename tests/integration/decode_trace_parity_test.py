#!/usr/bin/env python3

import json
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "compare_decode_traces.py"


class DecodeTraceParityTests(unittest.TestCase):
    def test_reports_first_divergence(self):
        native_log = textwrap.dedent(
            """\
            [INFO] native_kernel_executor: debug_logits[primary]: request_id=2, sequence_id=0, sequence_generation=3, n_past=38 top-5: [323]=34.8125 [476]=23.8125
            [INFO] native_kernel_executor: decode_mapping[primary]: input_idx=0, sequence_id=0, request_id=2, sequence_generation=3, n_past=37, sampled_token=323, piece= and
            [INFO] native_kernel_executor: debug_logits[primary]: request_id=2, sequence_id=0, sequence_generation=3, n_past=39 top-5: [2711]=31.2188 [56370]=31.1562
            [INFO] native_kernel_executor: decode_mapping[primary]: input_idx=0, sequence_id=0, request_id=2, sequence_generation=3, n_past=38, sampled_token=2711, piece= search
            """
        )
        llama_log = textwrap.dedent(
            """\
            [INFO] llama_backend: debug_logits[batch_decode]: request_id=2, sequence_id=0, sequence_generation=3, n_past=38 top-5: [323]=34.812500 [476]=23.812500
            [INFO] llama_backend: token_trace[batch_decode]: request_id=2, sequence_id=0, sequence_generation=3, n_past=38, sampled_token=323, piece= and
            [INFO] llama_backend: debug_logits[batch_decode]: request_id=2, sequence_id=0, sequence_generation=3, n_past=39 top-5: [56370]=31.187914 [2711]=31.187233
            [INFO] llama_backend: token_trace[batch_decode]: request_id=2, sequence_id=0, sequence_generation=3, n_past=39, sampled_token=56370, piece= retrieval
            """
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            native_path = Path(tmpdir) / "native.log"
            llama_path = Path(tmpdir) / "llama.log"
            native_path.write_text(native_log, encoding="utf-8")
            llama_path.write_text(llama_log, encoding="utf-8")
            proc = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    str(native_path),
                    str(llama_path),
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            )

        payload = json.loads(proc.stdout)
        self.assertFalse(payload["parity"])
        self.assertEqual(payload["shared_steps"], 2)
        self.assertEqual(payload["first_divergence"]["request_key"], "request:2")
        self.assertEqual(payload["first_divergence"]["n_past"], 39)
        self.assertEqual(
            payload["first_divergence"]["native"]["sampled_token"], 2711
        )
        self.assertEqual(
            payload["first_divergence"]["llama"]["sampled_token"], 56370
        )

    def test_require_parity_exits_nonzero_on_drift(self):
        native_log = textwrap.dedent(
            """\
            [INFO] native_kernel_executor: debug_logits[primary]: request_id=0, sequence_id=0, sequence_generation=1, n_past=1 top-5: [1]=2.0
            [INFO] native_kernel_executor: decode_mapping[primary]: input_idx=0, sequence_id=0, request_id=0, sequence_generation=1, n_past=1, sampled_token=1, piece=a
            """
        )
        llama_log = textwrap.dedent(
            """\
            [INFO] llama_backend: debug_logits[batch_decode]: request_id=0, sequence_id=0, sequence_generation=1, n_past=1 top-5: [2]=2.1
            [INFO] llama_backend: token_trace[batch_decode]: request_id=0, sequence_id=0, sequence_generation=1, n_past=1, sampled_token=2, piece=b
            """
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            native_path = Path(tmpdir) / "native.log"
            llama_path = Path(tmpdir) / "llama.log"
            native_path.write_text(native_log, encoding="utf-8")
            llama_path.write_text(llama_log, encoding="utf-8")
            proc = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    str(native_path),
                    str(llama_path),
                    "--require-parity",
                ],
                check=False,
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            )

        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("first_divergence", proc.stdout)

    def test_normalizes_native_prefill_sample_mapping_to_effective_n_past(self):
        native_log = textwrap.dedent(
            """\
            [INFO] native_kernel_executor: sample_mapping[primary_prefill]: input_idx=0, request_id=0, sequence_id=0, sequence_generation=1, n_past=0, token_count=11, sampled_token=362, piece= A
            [INFO] native_kernel_executor: debug_logits[primary]: request_id=0, sequence_id=0, sequence_generation=1, n_past=11 top-5: [5175]=35.968750 [6531]=27.312500
            [INFO] native_kernel_executor: decode_mapping[primary]: input_idx=0, sequence_id=0, request_id=0, sequence_generation=1, n_past=11, sampled_token=5175, piece= hash
            """
        )
        llama_log = textwrap.dedent(
            """\
            [INFO] llama_backend: token_trace[unified_batch]: request_id=0, sequence_id=0, sequence_generation=1, n_past=11, sampled_token=362, piece= A
            [INFO] llama_backend: debug_logits[unified_batch]: request_id=0, sequence_id=0, sequence_generation=1, n_past=11 top-5: [5175]=36.244041 [6531]=27.433159
            [INFO] llama_backend: token_trace[unified_batch]: request_id=0, sequence_id=0, sequence_generation=1, n_past=12, sampled_token=5175, piece= hash
            """
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            native_path = Path(tmpdir) / "native.log"
            llama_path = Path(tmpdir) / "llama.log"
            native_path.write_text(native_log, encoding="utf-8")
            llama_path.write_text(llama_log, encoding="utf-8")
            proc = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    str(native_path),
                    str(llama_path),
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            )

        payload = json.loads(proc.stdout)
        self.assertTrue(payload["parity"])
        self.assertEqual(payload["shared_steps"], 2)

    def test_prefers_client_request_id_over_internal_request_id(self):
        native_log = textwrap.dedent(
            """\
            [INFO] native_kernel_executor: debug_logits[primary]: client_request_id=bench-5, request_id=9, sequence_id=0, sequence_generation=1, n_past=11 top-5: [2711]=31.2188 [56370]=31.1562
            [INFO] native_kernel_executor: decode_mapping[primary]: input_idx=0, client_request_id=bench-5, request_id=9, sequence_id=0, sequence_generation=1, n_past=11, sampled_token=2711, piece= search
            """
        )
        llama_log = textwrap.dedent(
            """\
            [INFO] llama_backend: debug_logits[unified_batch]: client_request_id=bench-5, request_id=3, sequence_id=1, sequence_generation=1, n_past=11 top-5: [56370]=31.187914 [2711]=31.187233
            [INFO] llama_backend: token_trace[unified_batch]: client_request_id=bench-5, request_id=3, sequence_id=1, sequence_generation=1, n_past=12, sampled_token=56370, piece= retrieval
            """
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            native_path = Path(tmpdir) / "native.log"
            llama_path = Path(tmpdir) / "llama.log"
            native_path.write_text(native_log, encoding="utf-8")
            llama_path.write_text(llama_log, encoding="utf-8")
            proc = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    str(native_path),
                    str(llama_path),
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            )

        payload = json.loads(proc.stdout)
        self.assertFalse(payload["parity"])
        self.assertEqual(payload["first_divergence"]["request_key"], "client:bench-5")

    def test_parses_raw_native_prefill_logits_against_generate_trace(self):
        native_log = textwrap.dedent(
            """\
            [DEBUG_LOGITS] tokens=[10,11,12] n_past=0 top-10: [42]=9.5 [7]=1.0 (nan=0 inf=0 zero=0/100)
            [INFO] inferflux_cuda_executor: sample_mapping[primary_prefill]: input_idx=0, request_id=-1, sequence_id=0, sequence_generation=0, n_past=0, token_count=3, sampled_token=42, piece= answer
            """
        )
        llama_log = textwrap.dedent(
            """\
            [INFO] llama_backend: debug_logits[generate]: request_id=-1, sequence_id=0, sequence_generation=0, n_past=3 top-5: [42]=9.500000 [7]=1.000000
            [INFO] llama_backend: token_trace[generate]: request_id=-1, sequence_id=0, sequence_generation=0, n_past=3, sampled_token=42, piece= answer
            """
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            native_path = Path(tmpdir) / "native.log"
            llama_path = Path(tmpdir) / "llama.log"
            native_path.write_text(native_log, encoding="utf-8")
            llama_path.write_text(llama_log, encoding="utf-8")
            proc = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    str(native_path),
                    str(llama_path),
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            )

        payload = json.loads(proc.stdout)
        self.assertTrue(payload["parity"])
        self.assertEqual(payload["shared_steps"], 1)


if __name__ == "__main__":
    unittest.main()
