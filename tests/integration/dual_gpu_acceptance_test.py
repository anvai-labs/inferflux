#!/usr/bin/env python3
"""Model-free rejection tests for the trusted same-process setup gate."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import dual_gpu_acceptance as gate


class DualGpuAcceptanceTests(unittest.TestCase):
    def test_monitor_timeout_still_stops_owned_children_and_preserves_failure(self):
        for execution_fails in (False, True):
            with self.subTest(
                execution_fails=execution_fails
            ), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                child = mock.Mock()
                child.process.poll.return_value = None
                child.stop.side_effect = lambda: setattr(
                    child.process.poll, "return_value", 0
                )
                watcher = mock.Mock(ident=1)
                watcher.is_alive.return_value = True

                def execute(repo, private, report, children):
                    children.append(child)
                    if execution_fails:
                        raise gate.AcceptanceError("original_failure")
                    report["supplemental_gate_passed"] = True

                with mock.patch.object(
                    gate, "__file__", str(root / "scripts/gate.py")
                ), mock.patch.object(
                    gate.threading, "Thread", return_value=watcher
                ), mock.patch.object(
                    gate, "execute", side_effect=execute
                ), mock.patch.object(
                    gate, "shared_health"
                ), mock.patch.object(
                    gate.signal, "signal"
                ), mock.patch.object(
                    gate.signal, "alarm"
                ):
                    self.assertEqual(gate.main(), 1)
                child.stop.assert_called_once()
                report = json.loads(
                    (root / "build-ci-dual/dual-gpu-acceptance.json").read_text()
                )
                self.assertTrue(report["owned_processes_stopped"])
                self.assertEqual(report["cleanup_failures"], ["health_monitor_cleanup"])
                self.assertEqual(
                    report["failure"],
                    "original_failure" if execution_fails else "cleanup_failed",
                )

    def test_chat_wire_identity_and_direct_correlation_are_enforced(self):
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 0},
        }
        for streaming in (False, True):
            for mutation in (None, "model", "missing_model", "correlation"):
                with self.subTest(streaming=streaming, mutation=mutation):
                    payload = {"model": "dual-amd", "usage": usage}
                    if mutation == "model":
                        payload["model"] = "dual-nvidia"
                    elif mutation == "missing_model":
                        del payload["model"]
                    if streaming:
                        # A correct terminal frame must not hide a wrong earlier frame.
                        raw = (
                            "data: "
                            + json.dumps(dict(payload, usage=None))
                            + "\n\n"
                            + "data: "
                            + json.dumps({"model": "dual-amd", "usage": usage})
                            + "\n\ndata: [DONE]\n\n"
                        ).encode()
                    else:
                        raw = json.dumps(payload).encode()
                    response = mock.Mock(status=200)
                    response.read.return_value = raw
                    response.getheader.return_value = (
                        "wrong-request"
                        if mutation == "correlation"
                        else "expected-request"
                    )
                    with mock.patch.object(
                        gate.http.client, "HTTPConnection"
                    ) as connection:
                        connection.return_value.getresponse.return_value = response
                        args = (
                            gate.ORIGIN,
                            "/v1/chat/completions",
                            "test-key",
                            {"model": "dual-amd"},
                            {"x-inferflux-client-request-id": "expected-request"},
                            streaming,
                        )
                        if mutation:
                            with self.assertRaises(gate.AcceptanceError):
                                gate.request(*args)
                        else:
                            result, correlation = gate.request(*args)
                            self.assertEqual(result["model"], "dual-amd")
                            self.assertEqual(correlation, "expected-request")
                        connection.return_value.close.assert_called_once()

    def test_pinned_llama_legacy_cuda_cache_is_validated(self):
        base = "ENABLE_CUDA:BOOL=ON\nENABLE_ROCM:BOOL=ON\nGGML_HIP:BOOL=ON\nGGML_BACKEND_DL:BOOL=ON\n"
        for cuda in ("LLAMA_CUDA:BOOL=ON", "GGML_CUDA:BOOL=ON"):
            gate.verify_mixed_flags(base + cuda)
        with self.assertRaises(gate.AcceptanceError):
            gate.verify_mixed_flags(base + "LLAMA_CUDA:BOOL=OFF")
        with self.assertRaises(gate.AcceptanceError):
            gate.verify_mixed_flags(
                base.replace("GGML_HIP:BOOL=ON", "GGML_HIP:BOOL=OFF")
                + "LLAMA_CUDA:BOOL=ON"
            )

    def test_feature_branch_cannot_run_trusted_gate(self):
        with mock.patch.dict(
            gate.os.environ,
            {"GITHUB_ACTIONS": "true", "GITHUB_REF": "refs/heads/feature"},
            clear=True,
        ):
            with self.assertRaisesRegex(gate.AcceptanceError, "trusted_main_gate_only"):
                gate.trusted_source(ROOT)

    def test_config_uses_one_origin_two_qualified_devices(self):
        config = gate.model_config(
            {name: Path("/models/" + name + ".gguf") for name in gate.MODELS},
            "test-key",
        )
        self.assertEqual(config["server"]["host"], "127.0.0.1")
        self.assertEqual({m["device"] for m in config["models"]}, {"cuda:0", "rocm:0"})
        self.assertFalse(
            config["runtime"]["backend_exposure"]["allow_llama_cpp_fallback"]
        )
        self.assertTrue(all(m["max_parallel_sequences"] == 2 for m in config["models"]))

    def test_labels_and_discovery_do_not_prove_residency(self):
        payload = {
            "data": [
                {
                    "id": model,
                    "ready": True,
                    "backend_exposure": {"fallback": False},
                    "placement": {
                        "requested": device,
                        "effective": device,
                        "vendor": device.split(":")[0],
                        "state": "verified_weights",
                        "gpu_weight_bytes": 1024,
                        "gpu_layer_count": 8,
                        "stable_id": device,
                        "max_parallel_sequences": 2,
                        "kv_cache_type": "f16",
                    },
                }
                for model, device in gate.MODELS.items()
            ]
        }
        self.assertEqual(len(gate.verify_models(payload)), 2)
        for field, invalid in (
            ("state", "enumerated"),
            ("gpu_weight_bytes", 0),
            ("effective", "cuda:1"),
            ("stable_id", None),
        ):
            with self.subTest(field=field):
                original = payload["data"][0]["placement"][field]
                payload["data"][0]["placement"][field] = invalid
                with self.assertRaises(gate.AcceptanceError):
                    gate.verify_models(payload)
                payload["data"][0]["placement"][field] = original

    def test_overlap_requires_distinct_devices_and_intersecting_execution(self):
        a = {
            "device": "rocm:0",
            "scope": "host_backend_call",
            "start_ns": 10,
            "end_ns": 20,
        }
        b = {
            "device": "cuda:0",
            "scope": "host_backend_call",
            "start_ns": 19,
            "end_ns": 30,
        }
        self.assertTrue(gate.host_overlap([a, b]))
        self.assertFalse(gate.host_overlap([a, dict(b, start_ns=20)]))
        self.assertFalse(gate.host_overlap([a, dict(b, device="rocm:0")]))
        self.assertFalse(gate.host_overlap([a, dict(b, scope="http_request")]))

    def test_usage_requires_explicit_cache_reporting_and_conservation(self):
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 3},
        }
        self.assertEqual(gate.usage_counts(usage)["cached_tokens"], 3)
        for changed in (
            dict(usage, total_tokens=15),
            dict(usage, prompt_tokens_details={}),
            dict(usage, prompt_tokens_details={"cached_tokens": 11}),
        ):
            with self.assertRaises(gate.AcceptanceError):
                gate.usage_counts(changed)

    def test_ledger_checks_model_and_session_even_when_totals_match(self):
        call = {
            "request_id": "r",
            "session_id": "s",
            "model": "dual-amd",
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "cached_tokens": 3,
        }
        row = {
            "request_id": "r",
            "session_id": "s",
            "provider": "inferflux",
            "model": "dual-amd",
            "tokens_in": 7,
            "tokens_out": 2,
            "cache_read_tokens": 3,
            "cache_creation_tokens": 0,
        }
        gate.reconcile_rows([row], [call])
        for changed in (
            dict(row, model="dual-nvidia"),
            dict(row, session_id="different"),
        ):
            with self.assertRaises(gate.AcceptanceError):
                gate.reconcile_rows([changed], [call])


if __name__ == "__main__":
    unittest.main()
