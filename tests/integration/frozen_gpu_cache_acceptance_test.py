#!/usr/bin/env python3
"""CPU-only contracts for the optional trusted-main replay harness."""

import faulthandler
import sys

if __name__ == "__main__":
    print("Loading frozen GPU acceptance CPU contracts", file=sys.stderr, flush=True)
    faulthandler.enable()
    faulthandler.dump_traceback_later(10, repeat=True)

import importlib.util
import http.server
import json
import os
from pathlib import Path
import sqlite3
import socketserver
import subprocess
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "frozen_gpu", ROOT / "scripts/frozen_gpu_cache_acceptance.py"
)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        (self.root / "requests").mkdir()
        self.payloads = [
            dict(
                model=gate.MODEL,
                stream=False,
                max_tokens=4096,
                messages=[{"role": "user", "content": str(i)}],
            )
            for i in range(5)
        ]

    def write_bundle(self):
        entries = []
        for i, payload in enumerate(self.payloads):
            relative = f"requests/{i:03d}.json"
            raw = json.dumps(payload).encode()
            (self.root / relative).write_bytes(raw + b"\n")
            entries.append(
                dict(ordinal=i, file=relative, payload_sha256=gate.digest(raw))
            )
        manifest = dict(
            schema_version=1,
            model=gate.MODEL,
            evidence_kind="new_actual_member_capture",
            historical_reconstruction=False,
            requests=entries,
        )
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        (self.root / "replay.py").write_text(
            "raise RuntimeError('MUST NOT EXECUTE BUNDLE CODE')\n"
        )
        sums = "".join(
            f"{gate.file_digest(p)}  {p.relative_to(self.root)}\n"
            for p in sorted(self.root.rglob("*"))
            if p.is_file() and p.name != "SHA256SUMS"
        )
        (self.root / "SHA256SUMS").write_text(sums)
        return gate.digest(sums.encode()), gate.file_digest(self.root / "manifest.json")

    def test_exact_five_payload_values_are_preserved_without_executing_code(self):
        pins = self.write_bundle()
        actual = gate.load_bundle(self.root, *pins)
        self.assertEqual([json.loads(raw) for raw in actual], self.payloads)
        self.assertEqual(len(actual) * 2, 10)

    def test_bundle_fixture_handles_symlinked_temporary_parent(self):
        pins = self.write_bundle()
        with tempfile.TemporaryDirectory() as parent:
            alias = Path(parent) / "tmp-alias"
            alias.symlink_to(self.root, target_is_directory=True)
            # macOS /var commonly aliases /private/var. Canonicalize the test
            # fixture; the production fixed-root rejection remains unchanged.
            actual = gate.load_bundle(alias.resolve(), *pins)
            self.assertEqual(len(actual), 5)
            with self.assertRaises(gate.AcceptanceError):
                gate.load_bundle(alias, *pins)

    def test_approval_pin_and_later_payload_tampering_fail(self):
        pins = self.write_bundle()
        with self.assertRaises(gate.AcceptanceError):
            gate.load_bundle(self.root, "0" * 64, pins[1])
        (self.root / "requests/000.json").write_text("{}")
        with self.assertRaises(gate.AcceptanceError):
            gate.load_bundle(self.root, *pins)

    def test_payload_shape_and_resource_bounds_fail_even_when_checksums_match(self):
        cases = [
            dict(max_tokens=4097),
            dict(max_tokens=True),
            dict(max_tokens=0),
            dict(max_completion_tokens=8192),
            dict(stream=True),
            dict(model="other"),
            dict(session_id="override"),
            dict(client_request_id="override"),
            dict(messages=[{"role": "user", "content": "x" * gate.MAX_PAYLOAD}]),
        ]
        for change in cases:
            with self.subTest(change=list(change)):
                original = self.payloads[0].copy()
                self.payloads[0].update(change)
                pins = self.write_bundle()
                with self.assertRaises(gate.AcceptanceError):
                    gate.load_bundle(self.root, *pins)
                self.payloads[0] = original
        self.payloads.append(self.payloads[0])
        pins = self.write_bundle()
        with self.assertRaises(gate.AcceptanceError):
            gate.load_bundle(self.root, *pins)

    def test_checksum_path_escape_is_rejected(self):
        self.write_bundle()
        raw = "0" * 64 + "  ../outside\n"
        (self.root / "SHA256SUMS").write_text(raw)
        with self.assertRaises(gate.AcceptanceError):
            gate.load_bundle(self.root, gate.digest(raw.encode()), "0" * 64)

    def test_symlink_asset_is_rejected(self):
        pins = self.write_bundle()
        original = self.root / "requests/000.json"
        backup = self.root / "backup.json"
        original.rename(backup)
        original.symlink_to(backup)
        with self.assertRaises(gate.AcceptanceError):
            gate.load_bundle(self.root, *pins)


class AccountingTests(unittest.TestCase):
    def test_usage_retains_reported_zero_and_rejects_invalid_conservation(self):
        usage = dict(
            prompt_tokens=40,
            completion_tokens=2,
            total_tokens=42,
            prompt_tokens_details=dict(cached_tokens=0),
        )
        self.assertEqual(gate.verify_usage(usage)["cached_tokens"], 0)
        for change in (
            dict(prompt_tokens=True),
            dict(total_tokens=41),
            dict(prompt_tokens_details=dict(cached_tokens=41)),
        ):
            with self.assertRaises(gate.AcceptanceError):
                gate.verify_usage({**usage, **change})

    def test_actual_sqlite_read_and_cache_split_reconcile(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "usage.db"
            with sqlite3.connect(path) as db:
                db.execute(
                    "CREATE TABLE usage_events(request_id,session_id,run_id,provider,model,tokens_in,tokens_out,cache_creation_tokens,cache_read_tokens)"
                )
                db.execute(
                    "INSERT INTO usage_events VALUES(?,?,?,?,?,?,?,?,?)",
                    ("req_1", "s", "s", "inferflux", gate.MODEL, 1, 2, 0, 39),
                )
                db.execute(
                    "INSERT INTO usage_events VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        "req_other",
                        "other",
                        "other",
                        "inferflux",
                        gate.MODEL,
                        99,
                        2,
                        0,
                        0,
                    ),
                )
            rows = gate.ledger_rows(path, "s")
            self.assertEqual(len(rows), 1)
            call = dict(
                request_id="req_1",
                prompt_tokens=40,
                completion_tokens=2,
                cached_tokens=39,
            )
            gate.reconcile_row(rows[0], call, "s")
            for change in (
                dict(cache_read_tokens=0),
                dict(run_id="other"),
                dict(tokens_in=None),
            ):
                with self.assertRaises(gate.AcceptanceError):
                    gate.reconcile_row({**rows[0], **change}, call, "s")

    def events(self):
        calls, events = [], []
        sessions = {"direct": "session-direct", "gateway": "session-gateway"}
        for ordinal in range(5):
            for arm in sessions:
                request_id = f"req_{ordinal}_{arm}"
                call = dict(
                    ordinal=ordinal,
                    arm=arm,
                    request_id=request_id,
                    prompt_tokens=40,
                    completion_tokens=2,
                    cached_tokens=39 if ordinal == 1 else 0,
                )
                calls.append(call)
                event = dict(
                    client_request_id=request_id,
                    backend="llama_cuda",
                    model=gate.MODEL,
                    session_sha256=gate.digest(sessions[arm].encode()),
                    prompt_tokens=40,
                    reused_tokens=call["cached_tokens"],
                    tokens_sha256=gate.digest(str(ordinal).encode()),
                    path="phased",
                )
                events.extend(
                    [
                        {**event, "stage": "lookup", "reused_tokens": 40},
                        {**event, "stage": "prefill_accepted"},
                        {**event, "stage": "completed"},
                    ]
                )
        return events, calls, sessions

    def test_backend_finalization_uses_accepted_tokens_not_lookup(self):
        events, calls, sessions = self.events()
        actual = gate.reconcile_diagnostics(events, calls, sessions)
        self.assertEqual(actual[0]["cached_tokens"], 0)
        self.assertEqual(actual[2]["cached_tokens"], 39)
        self.assertFalse(any("session_id" in row for row in actual))
        for change in (
            dict(reused_tokens=40),
            dict(prompt_tokens=3),
            dict(backend="llama_cpu"),
            dict(tokens_sha256="unknown"),
        ):
            changed = list(events)
            changed[2] = {**changed[2], **change}
            with self.assertRaises(gate.AcceptanceError):
                gate.reconcile_diagnostics(changed, calls, sessions)
        with self.assertRaises(gate.AcceptanceError):
            gate.reconcile_diagnostics(events + [events[2]], calls, sessions)


class IsolationTests(unittest.TestCase):
    def test_non_main_and_wrong_workflow_fail_before_invoking_commands(self):
        base = dict(
            GITHUB_ACTIONS="true",
            GITHUB_REF="refs/heads/main",
            GITHUB_EVENT_NAME="workflow_dispatch",
            RUNNER_NAME="aiserver1-dual-gpu",
            GITHUB_WORKFLOW_REF="anvai-labs/inferflux/.github/workflows/gpu-gates.yml@refs/heads/main",
            GITHUB_SHA="1" * 40,
        )
        for field, value in [
            ("GITHUB_REF", "refs/heads/feature"),
            ("GITHUB_EVENT_NAME", "pull_request"),
            ("RUNNER_NAME", "other"),
            ("GITHUB_WORKFLOW_REF", "another.yml@refs/heads/main"),
        ]:
            with self.subTest(field=field), mock.patch.dict(
                os.environ, {**base, field: value}, clear=True
            ), mock.patch.object(subprocess, "check_output") as run:
                with self.assertRaises(gate.AcceptanceError):
                    gate.trusted_provenance(ROOT)
                run.assert_not_called()

    def test_ambient_origin_gateway_secrets_are_not_inherited(self):
        with mock.patch.dict(
            os.environ,
            {
                "PATH": "/bin",
                "INFERFLUX_MODELS": "wrong",
                "SANDHI_STORE": "shared",
                "OPENAI_API_KEY": "secret",
            },
            clear=True,
        ):
            self.assertEqual(gate.clean_environment(), {"PATH": "/bin"})
        config = gate.cuda_config(Path("/models/qwen.gguf"), "temporary-key")
        self.assertEqual(config["server"]["http_port"], 28084)
        self.assertEqual(config["runtime"]["mps_layers"], 8)
        self.assertEqual(config["models"][0]["backend"], "llama_cpp_cuda")

    def test_http_response_is_bounded_and_redirect_is_not_followed(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                self.server.paths.append(self.path)
                self.send_response(302 if self.path == "/redirect" else 200)
                self.send_header("Location", "/must-not-follow")
                self.end_headers()
                self.wfile.write(b"x" * 33 if self.path == "/oversize" else b"{}")

        class LocalHTTPServer(http.server.HTTPServer):
            def server_bind(self):
                # A fixed loopback fixture must not depend on host reverse DNS.
                socketserver.TCPServer.server_bind(self)
                self.server_name = "127.0.0.1"
                self.server_port = self.server_address[1]

        with mock.patch(
            "socket.getfqdn", side_effect=AssertionError("No DNS in loopback fixture")
        ) as reverse_dns:
            server = LocalHTTPServer(("127.0.0.1", 0), Handler)
            reverse_dns.assert_not_called()
        server.paths = []
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            port = server.server_address[1]
            with mock.patch.object(gate, "ORIGIN_PORT", port), mock.patch.object(
                gate, "MAX_RESPONSE", 32
            ):
                client = gate.Http(time.monotonic() + 10)
                self.assertEqual(client.request(port, "/ok")[0], {})
                for path in ("/redirect", "/oversize"):
                    with self.assertRaises(gate.AcceptanceError):
                        client.request(port, path)
            self.assertEqual(server.paths, ["/ok", "/redirect", "/oversize"])
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)

    def test_http_rejects_other_ports_before_connecting(self):
        with self.assertRaises(gate.AcceptanceError):
            gate.Http(float("inf")).request(12345, "/")

    def test_live_child_short_output_is_visible_before_exit(self):
        with tempfile.TemporaryDirectory() as root:
            child = gate.Child(
                [
                    sys.executable,
                    "-c",
                    "print('short-marker', flush=True); import time; time.sleep(30)",
                ],
                gate.clean_environment(),
                root,
            )
            try:
                deadline = time.monotonic() + 2
                while b"short-marker" not in child.raw and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertIsNone(child.process.poll())
                self.assertIn(b"short-marker", child.raw)
            finally:
                child.stop()

    def test_owned_child_cleanup_and_output_bound(self):
        with tempfile.TemporaryDirectory() as root:
            child = gate.Child(
                [
                    sys.executable,
                    "-c",
                    "print('ready', flush=True); import time; time.sleep(30)",
                ],
                gate.clean_environment(),
                root,
            )
            child.stop()
            self.assertIsNotNone(child.process.poll())
            self.assertFalse(child.reader.is_alive())
            with mock.patch.object(gate, "MAX_LOG", 16):
                child = gate.Child(
                    [sys.executable, "-c", "print('x' * 8192, flush=True)"],
                    gate.clean_environment(),
                    root,
                )
                child.process.wait(timeout=5)
                child.stop()
                self.assertTrue(child.overflow)
                self.assertLessEqual(len(child.raw), 16)


if __name__ == "__main__":
    try:
        unittest.main()
    finally:
        faulthandler.cancel_dump_traceback_later()
