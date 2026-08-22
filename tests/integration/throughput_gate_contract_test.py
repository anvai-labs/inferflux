#!/usr/bin/env python3

import argparse
import importlib.util
import os
import sys
import unittest
from unittest import mock


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
SCRIPT_PATH = os.path.join(ROOT_DIR, "scripts", "run_throughput_gate.py")

spec = importlib.util.spec_from_file_location("run_throughput_gate", SCRIPT_PATH)
if spec is None or spec.loader is None:
  raise RuntimeError("failed to load scripts/run_throughput_gate.py")
run_throughput_gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = run_throughput_gate
spec.loader.exec_module(run_throughput_gate)


class ThroughputGateContractTests(unittest.TestCase):

  def test_http_post_json_reports_response_header_timeout_and_closes(self):
    conn = mock.Mock()
    conn.getresponse.side_effect = TimeoutError("timed out")
    with mock.patch.object(run_throughput_gate.http.client, "HTTPConnection",
                           return_value=conn):
      with self.assertRaisesRegex(
          RuntimeError, "failed awaiting response headers: TimeoutError"):
        run_throughput_gate.http_post_json(
            "127.0.0.1", 8080, "/v1/completions", {"prompt": "hi"},
            timeout_s=1.0, api_key="")
    conn.close.assert_called_once_with()

  def test_http_post_json_reports_body_timeout_with_framing_and_closes(self):
    response = mock.Mock(status=200)
    response.getheader.return_value = "321"
    response.read.side_effect = TimeoutError("timed out")
    conn = mock.Mock()
    conn.getresponse.return_value = response
    with mock.patch.object(run_throughput_gate.http.client, "HTTPConnection",
                           return_value=conn):
      with self.assertRaisesRegex(
          RuntimeError, "reading response body .*content_length=321.*TimeoutError"):
        run_throughput_gate.http_post_json(
            "127.0.0.1", 8080, "/v1/completions", {"prompt": "hi"},
            timeout_s=1.0, api_key="")
    conn.close.assert_called_once_with()

  def test_read_metric_max_picks_max_series_value(self):
    metrics_text = """
inferflux_batch_size_max{backend="cpu"} 3
inferflux_batch_size_max{backend="cuda"} 7
"""
    self.assertEqual(
        run_throughput_gate.read_metric_max(
            metrics_text, "inferflux_batch_size_max", {"backend": "cuda"}),
        7.0)
    self.assertEqual(
        run_throughput_gate.read_metric_max(
            metrics_text, "inferflux_batch_size_max"),
        7.0)

  def test_parse_args_accepts_batching_threshold_flags(self):
    with mock.patch.object(
        sys, "argv",
        [
            "run_throughput_gate.py",
            "--gpu-profile",
            "none",
            "--min-batch-size-max",
            "3",
            "--max-batch-token-budget-skips",
            "9",
            "--min-batch-size-utilization",
            "0.2",
            "--max-batch-token-budget-skip-ratio",
            "0.3",
            "--require-mixed-scheduler-iterations",
        ]):
      args = run_throughput_gate.parse_args()
    self.assertEqual(args.min_batch_size_max, 3.0)
    self.assertEqual(args.max_batch_token_budget_skips, 9.0)
    self.assertEqual(args.min_batch_size_utilization, 0.2)
    self.assertEqual(args.max_batch_token_budget_skip_ratio, 0.3)
    self.assertTrue(args.require_mixed_scheduler_iterations)

  def test_gpu_profile_applies_default_min_batch_size_floor(self):
    with mock.patch.object(
        sys, "argv",
        [
            "run_throughput_gate.py",
            "--gpu-profile",
            "ada_rtx_4000",
        ]):
      args = run_throughput_gate.parse_args()
    self.assertEqual(args.min_batch_size_max, 2.0)
    self.assertEqual(args.min_batch_size_utilization, 0.06)
    self.assertTrue(args.require_mixed_scheduler_iterations)
    self.assertEqual(args.backend, "cuda")

  def test_gpu_profile_non_cuda_sets_backend_without_cuda_requirements(self):
    with mock.patch.object(
        sys, "argv",
        [
            "run_throughput_gate.py",
            "--gpu-profile",
            "apple_m3_max",
        ]):
      args = run_throughput_gate.parse_args()
    self.assertEqual(args.backend, "mps")
    self.assertFalse(args.require_cuda_lanes)
    self.assertFalse(args.require_cuda_overlap)

  def test_parse_args_normalizes_universal_provider_alias(self):
    with mock.patch.object(
        sys, "argv",
        [
            "run_throughput_gate.py",
            "--gpu-profile",
            "none",
            "--require-backend-provider",
            "universal",
        ]):
      args = run_throughput_gate.parse_args()
    self.assertEqual(args.require_backend_provider, "llama_cpp")

  def test_parse_args_normalizes_native_provider_alias(self):
    with mock.patch.object(
        sys, "argv",
        [
            "run_throughput_gate.py",
            "--gpu-profile",
            "none",
            "--require-backend-provider",
            "native",
        ]):
      args = run_throughput_gate.parse_args()
    self.assertEqual(args.require_backend_provider, "inferflux")

  def test_fetch_metrics_snapshot_reads_batching_metrics(self):
    metrics_text = """
inferflux_completion_tokens_total{backend="cuda"} 10
inferflux_batch_size_max{backend="cuda"} 5
inferflux_scheduler_batch_limit_size 32
inferflux_scheduler_batch_token_budget_skips_total{backend="cuda"} 2
inferflux_scheduler_iterations_total{backend="cuda",phase="prefill"} 3
inferflux_scheduler_iterations_total{backend="cuda",phase="decode"} 4
inferflux_scheduler_iterations_total{backend="cuda",phase="mixed"} 5
"""
    args = argparse.Namespace(
        host="127.0.0.1",
        port=18081,
        metrics_path="/metrics",
        api_key="",
        backend="cuda",
        model="",
    )
    with mock.patch.object(run_throughput_gate, "http_get",
                           return_value=(200, metrics_text)):
      snapshot = run_throughput_gate.fetch_metrics_snapshot(args)
    self.assertEqual(snapshot.batch_size_max, 5.0)
    self.assertEqual(snapshot.scheduler_batch_limit_size, 32.0)
    self.assertEqual(snapshot.scheduler_batch_token_budget_skips, 2.0)
    self.assertEqual(snapshot.scheduler_iterations_prefill, 3.0)
    self.assertEqual(snapshot.scheduler_iterations_decode, 4.0)
    self.assertEqual(snapshot.scheduler_iterations_mixed, 5.0)


if __name__ == "__main__":
  unittest.main()
