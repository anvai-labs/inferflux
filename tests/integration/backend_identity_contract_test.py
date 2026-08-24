#!/usr/bin/env python3

import importlib.util
import pathlib
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_backend_identity.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_backend_identity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


identity = _load_module()


class BackendIdentityContractTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.log_file = pathlib.Path(self.tempdir.name) / "server.log"
        self.log_file.write_text("[server] native runtime ready\n", encoding="utf-8")
        self.model = {
            "id": "bench-model",
            "backend": "inferflux_cuda",
            "backend_exposure": {
                "requested_backend": "inferflux_cuda",
                "exposed_backend": "inferflux_cuda",
                "provider": "inferflux",
                "fallback": False,
                "fallback_reason": "",
            },
        }

    def tearDown(self):
        self.tempdir.cleanup()

    def run_main(self):
        with mock.patch.object(identity, "fetch_model", return_value=self.model):
            return identity.main(
                [
                    "--base-url",
                    "http://127.0.0.1:18091",
                    "--model-id",
                    "bench-model",
                    "--expected-provider",
                    "inferflux",
                    "--expected-backend",
                    "inferflux_cuda",
                    "--log-file",
                    str(self.log_file),
                    "--forbid-log-pattern",
                    r"^ggml_cuda_init:",
                ]
            )

    def test_backend_identity_contract_accepts_native_model(self):
        self.assertEqual(self.run_main(), 0)

    def test_backend_identity_contract_rejects_provider_mismatch(self):
        self.model["backend_exposure"]["provider"] = "llama_cpp"
        with self.assertRaisesRegex(RuntimeError, "provider mismatch"):
            self.run_main()

    def test_backend_identity_contract_rejects_forbidden_log_patterns(self):
        self.log_file.write_text(
            "ggml_cuda_init: found 1 CUDA devices\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(RuntimeError, "forbidden backend pattern"):
            self.run_main()


if __name__ == "__main__":
    unittest.main()
