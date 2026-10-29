#!/usr/bin/env python3
"""Real CPU-model bootstrap lease regression; opt in with a small local GGUF."""
import http.client
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from process_helper import start_server_process, stop_server_process

MODEL = os.environ.get("INFERFLUX_LIFECYCLE_MODEL_PATH", "")
BINARY = os.environ.get("INFERFLUX_SERVER_BIN", "./build/inferfluxd")


@unittest.skipUnless(MODEL, "set INFERFLUX_LIFECYCLE_MODEL_PATH to a small GGUF")
class StartupModelUnloadTests(unittest.TestCase):
    def exercise(self, speculative):
        with tempfile.TemporaryDirectory(prefix="inferflux-bootstrap-lease-") as directory:
            root = Path(directory)
            replica = root / "secondary.gguf"
            shutil.copyfile(MODEL, replica)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            key = secrets.token_hex(32)
            models = []
            for identifier, path in (("primary", str(Path(MODEL).resolve())), ("secondary", str(replica))):
                models.append({"id": identifier, "path": path, "backend": "cpu",
                               "format": "gguf", "gpu_layers": 0, "context_size": 512,
                               "max_parallel_sequences": 1, "default": identifier == "primary"})
            config = {"server": {"host": "127.0.0.1", "http_port": port}, "models": models,
                      "runtime": {"cuda": {"enabled": False}, "speculative_decoding": {"enabled": speculative},
                                  "backend_exposure": {"prefer_inferflux": False, "allow_llama_cpp_fallback": True}},
                      "auth": {"api_keys": [{"key": key, "scopes": ["admin", "generate"]}]}}
            path = root / "config.json"
            path.write_text(json.dumps(config))
            path.chmod(0o600)
            env = {name: value for name, value in os.environ.items() if not name.startswith("INFERFLUX_")}
            env.update(INFERFLUX_HOST_OVERRIDE="127.0.0.1", INFERFLUX_PORT_OVERRIDE=str(port),
                       INFERFLUX_MODEL_PATH="", INFERFLUX_POLICY_STORE=str(root / "policy.enc"),
                       INFERFLUX_POLICY_PASSPHRASE="disposable-lifecycle-fixture",
                       INFERFLUX_SPECULATIVE_ENABLED="true" if speculative else "false",
                       CUDA_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="", OMP_NUM_THREADS="2")
            def request(method, target, body=None):
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                try:
                    connection.request(method, target, body=json.dumps(body) if body is not None else None,
                                       headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
                    response = connection.getresponse()
                    return response.status, json.loads(response.read())
                finally:
                    connection.close()
            with (root / "server.log").open("wb") as log:
                process = start_server_process([BINARY, "--config", str(path)], env=env, merge_stderr=True)
                drain = threading.Thread(target=shutil.copyfileobj, args=(process.stdout, log), daemon=True)
                drain.start()
                try:
                    deadline = time.monotonic() + 45
                    before = None
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            drain.join(2)
                            log.flush()
                            detail=(root / "server.log").read_text(errors="replace")[-2500:].replace(key,"[redacted]")
                            self.fail("CPU lifecycle server exited before readiness: " + detail)
                        try:
                            status, before = request("GET", "/v1/admin/models")
                            if status == 200 and len(before.get("models", [])) == 2:
                                break
                        except (OSError, ValueError):
                            pass
                        time.sleep(0.1)
                    self.assertIsNotNone(before)
                    self.assertEqual(status, 200)
                    self.assertEqual(len(before.get("models", [])), 2)
                    self.assertEqual(before["default_model"], "primary")
                    self.assertTrue(all(model["ready"] for model in before["models"]))
                    status, _ = request("DELETE", "/v1/admin/models", {"id": "primary"})
                    self.assertEqual(status, 409 if speculative else 200)
                    status, after = request("GET", "/v1/admin/models")
                    self.assertEqual(status, 200)
                    expected = {"primary", "secondary"} if speculative else {"secondary"}
                    self.assertEqual({model["id"] for model in after["models"]}, expected)
                    self.assertTrue(all(model["ready"] for model in after["models"]))
                    self.assertIsNone(process.poll(), "unloading must not restart or stop the server")
                    if not speculative:
                        self.assertEqual(after["default_model"], "secondary")
                finally:
                    stop_server_process(process)
                    drain.join(5)
                    process.stdout.close()

    def test_idle_default_unloads_and_secondary_remains_ready(self):
        self.exercise(False)

    def test_speculative_decoder_retains_its_required_backend_lease(self):
        self.exercise(True)


if __name__ == "__main__":
    unittest.main()
