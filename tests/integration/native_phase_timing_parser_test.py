#!/usr/bin/env python3

import json
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "parse_native_phase_timing.py"


class NativePhaseTimingParserTests(unittest.TestCase):
    def test_summarizes_decode_and_prefill_samples(self):
        sample_log = textwrap.dedent(
            """\
            [phase_timing] #1 L=36 tokens=5 embed=0.10 qkv=1.20 rope=0.30 kv=0.40 attn=1.50 o_proj=0.60 ffn_proj=1.10 ffn_silu=0.20 ffn_down=0.70 ffn=2.00 lm_head=0.50 total=6.60 ms
            [phase_timing] #2 L=36 tokens=1 embed=0.02 qkv=0.70 rope=0.05 kv=0.08 attn=0.90 o_proj=0.25 ffn_proj=0.45 ffn_silu=0.10 ffn_down=0.25 ffn=0.80 lm_head=0.20 total=3.00 ms
            [phase_timing] #3 L=36 tokens=1 embed=0.03 qkv=0.90 rope=0.04 kv=0.07 attn=1.10 o_proj=0.22 ffn_proj=0.50 ffn_silu=0.10 ffn_down=0.25 ffn=0.85 lm_head=0.19 total=3.40 ms
            """
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "native.log"
            log_file.write_text(sample_log, encoding="utf-8")
            proc = subprocess.run(
                ["python3", str(SCRIPT), str(log_file), "--json"],
                check=True,
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            )

        summary = json.loads(proc.stdout)
        self.assertEqual(summary["prefill"]["count"], 1)
        self.assertEqual(summary["decode"]["count"], 2)
        self.assertAlmostEqual(summary["decode"]["total_mean_ms"], 3.2)
        self.assertAlmostEqual(summary["decode"]["attn_mean_ms"], 1.0)
        self.assertAlmostEqual(summary["prefill"]["ffn_mean_ms"], 2.0)
        self.assertAlmostEqual(summary["decode"]["ffn_proj_mean_ms"], 0.475)
        self.assertAlmostEqual(summary["decode"]["ffn_down_mean_ms"], 0.25)

    def test_fails_when_no_phase_lines_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "empty.log"
            log_file.write_text("no phase timings here\n", encoding="utf-8")
            proc = subprocess.run(
                ["python3", str(SCRIPT), str(log_file), "--json"],
                check=False,
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            )

        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("No phase timing lines found", proc.stderr)


if __name__ == "__main__":
    unittest.main()
