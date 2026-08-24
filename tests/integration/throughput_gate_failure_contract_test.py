#!/usr/bin/env python3

import argparse
import contextlib
import importlib.util
import io
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


def make_args(**overrides):
  args = argparse.Namespace(
      gpu_profile="none",
      host="127.0.0.1",
      port=18081,
      api_key="",
      endpoint="/v1/completions",
      model="tinyllama",
      prompt="test",
      max_tokens=8,
      concurrency=2,
      requests=4,
      warmup_requests=0,
      mixed_prompt_workload=False,
      short_prompt_repeat=1,
      long_prompt_repeat=1,
      target_total_prefill_tokens=320,
      unique_prompts=False,
      request_timeout_sec=10.0,
      startup_timeout_sec=10.0,
      metrics_path="/metrics",
      backend="cuda",
      min_completion_tok_per_sec=0.0,
      min_req_per_sec=0.0,
      min_success_rate=0.0,
      require_cuda_lanes=False,
      require_cuda_overlap=False,
      require_backend_provider="any",
      require_no_backend_fallback=False,
      min_cuda_overlap_duration_ms=-1.0,
      max_cuda_attention_fallbacks=-1.0,
      min_batch_size_max=-1.0,
      min_batch_size_utilization=-1.0,
      max_batch_token_budget_skips=-1.0,
      max_batch_token_budget_skip_ratio=-1.0,
      require_mixed_scheduler_iterations=False,
      expect_cuda_attention_kernel="",
      require_native_forward_passes=False,
      require_metrics=True,
      server_bin="",
      config="",
      server_env=[],
      server_log_path="",
  )
  for key, value in overrides.items():
    setattr(args, key, value)
  return args


def make_metrics_snapshot(batch_size_max, batch_limit_size, skip_total):
  return run_throughput_gate.MetricsSnapshot(
      completion_tokens_global=100.0,
      completion_tokens_model=50.0,
      batch_size_max=batch_size_max,
      scheduler_batch_limit_size=batch_limit_size,
      scheduler_batch_token_budget_skips=skip_total,
      scheduler_iterations_prefill=0.0,
      scheduler_iterations_decode=0.0,
      scheduler_iterations_mixed=0.0,
      decode_lane_submissions=0.0,
      prefill_lane_submissions=0.0,
      cuda_lane_overlap_events=0.0,
      cuda_lane_overlap_duration_ms=0.0,
      cuda_attention_fallback_events=0.0,
      selected_attention_kernel="fa2",
      native_forward_prefill=0.0,
      native_forward_decode=0.0,
      native_forward_batch_tokens=0.0,
  )


class ThroughputGateFailureContractTests(unittest.TestCase):

  def _run_main_with_thresholds(self, args):
    before = make_metrics_snapshot(batch_size_max=2.0, batch_limit_size=32.0,
                                   skip_total=0.0)
    after = make_metrics_snapshot(batch_size_max=2.0, batch_limit_size=32.0,
                                  skip_total=1.0)
    after.completion_tokens_global = 140.0
    after.completion_tokens_model = 70.0
    results = [
        run_throughput_gate.RequestResult(success=True, latency_s=0.01,
                                          completion_tokens=5),
        run_throughput_gate.RequestResult(success=True, latency_s=0.02,
                                          completion_tokens=5),
        run_throughput_gate.RequestResult(success=True, latency_s=0.03,
                                          completion_tokens=5),
        run_throughput_gate.RequestResult(success=True, latency_s=0.04,
                                          completion_tokens=5),
    ]
    stderr = io.StringIO()
    stdout = io.StringIO()
    with mock.patch.object(run_throughput_gate, "parse_args", return_value=args), \
        mock.patch.object(run_throughput_gate, "wait_for_server"), \
        mock.patch.object(run_throughput_gate, "fetch_backend_exposure",
                          return_value=run_throughput_gate.BackendExposureSnapshot(
                              model_id="tinyllama",
                              requested_backend="cuda",
                              exposed_backend="cuda",
                              provider="inferflux",
                              fallback=False,
                              fallback_reason="",
                          )), \
        mock.patch.object(run_throughput_gate, "fetch_metrics_snapshot",
                          side_effect=[before, after]), \
        mock.patch.object(run_throughput_gate, "run_workload",
                          return_value=(results, 1.0)), \
        contextlib.redirect_stderr(stderr), \
        contextlib.redirect_stdout(stdout):
      exit_code = run_throughput_gate.main()
    return exit_code, stderr.getvalue(), stdout.getvalue()

  def test_failure_message_batch_size_max_floor(self):
    args = make_args(min_batch_size_max=3.0)
    exit_code, stderr, _ = self._run_main_with_thresholds(args)
    self.assertEqual(exit_code, 1)
    self.assertIn("batch size max 2.000 below min 3.000", stderr)

  def test_failure_message_batch_size_utilization_floor(self):
    args = make_args(min_batch_size_utilization=0.1)
    exit_code, stderr, _ = self._run_main_with_thresholds(args)
    self.assertEqual(exit_code, 1)
    self.assertIn("batch size utilization 0.0625 below min 0.1000", stderr)

  def test_failure_message_batch_skip_count_ceiling(self):
    args = make_args(max_batch_token_budget_skips=0.0)
    exit_code, stderr, _ = self._run_main_with_thresholds(args)
    self.assertEqual(exit_code, 1)
    self.assertIn("batch token-budget skips 1.000 exceed max 0.000", stderr)

  def test_failure_message_batch_skip_ratio_ceiling(self):
    args = make_args(max_batch_token_budget_skip_ratio=0.2)
    exit_code, stderr, _ = self._run_main_with_thresholds(args)
    self.assertEqual(exit_code, 1)
    self.assertIn("batch token-budget skip ratio 0.2500 exceed max 0.2000",
                  stderr)

  def test_failure_message_requires_mixed_scheduler_iterations(self):
    args = make_args(require_mixed_scheduler_iterations=True)
    exit_code, stderr, _ = self._run_main_with_thresholds(args)
    self.assertEqual(exit_code, 1)
    self.assertIn(
        "mixed scheduler iterations missing "
        "(inferflux_scheduler_iterations_total phase=\"mixed\" delta must be > 0)",
        stderr)

  def test_harness_exception_is_reported_without_traceback(self):
    args = make_args()
    stderr = io.StringIO()
    stdout = io.StringIO()
    with mock.patch.object(run_throughput_gate, "parse_args",
                           return_value=args), \
        mock.patch.object(run_throughput_gate, "wait_for_server",
                          side_effect=RuntimeError(
                              "[Errno 1] Operation not permitted")), \
        contextlib.redirect_stderr(stderr), \
        contextlib.redirect_stdout(stdout):
      exit_code = run_throughput_gate.main()

    self.assertEqual(exit_code, 1)
    self.assertIn("[throughput-gate] FAILED", stderr.getvalue())
    self.assertIn("Operation not permitted", stderr.getvalue())
    self.assertIn("socket operations were blocked", stderr.getvalue())
    self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
  unittest.main()
