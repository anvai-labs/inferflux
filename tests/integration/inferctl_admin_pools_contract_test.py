#!/usr/bin/env python3
import http.server
import json
import os
import socketserver
import subprocess
import threading
import unittest


INFERCTL_BIN = os.environ.get("INFERCTL_BIN", "./build/inferctl")


class _PoolsContractHandler(http.server.BaseHTTPRequestHandler):
    admin_pools_status = 200
    admin_pools_payload = {}
    ready_status = 503
    ready_payload = {}
    metrics_body = ""

    def do_GET(self):
        if self.path == "/v1/admin/pools":
            if type(self).admin_pools_status == 404:
                self.send_response(404)
                self.end_headers()
                return
            body = json.dumps(type(self).admin_pools_payload).encode()
            self.send_response(type(self).admin_pools_status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/readyz":
            body = json.dumps(type(self).ready_payload).encode()
            self.send_response(type(self).ready_status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/metrics":
            body = type(self).metrics_body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, fmt, *args):
        pass


class _ThreadedTcpServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


class InferctlAdminPoolsContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _PoolsContractHandler.admin_pools_status = 200
        _PoolsContractHandler.admin_pools_payload = {
            "status": "ok",
            "pool_health": {
                "ready": False,
                "http_status": 503,
                "role": "decode",
                "reason": "distributed kv transport degraded",
                "model_loaded": True,
                "decode_pool_warm": True,
                "disagg_transport_degraded": True,
                "disagg_timeout_debt": 2,
                "disagg_timeout_debt_threshold": 6,
                "disagg_timeout_streak": 4,
                "disagg_timeout_streak_threshold": 3,
            },
            "scheduler": {
                "queue_depth": 7,
                "prefill_queue_depth": 2,
                "decode_queue_depth": 5,
                "batch_limit_size": 4,
                "batch_limit_tokens": 8192,
            },
            "distributed_kv": {
                "enqueue_rejections_total": 2,
                "enqueue_exhausted_total": 1,
                "tickets_enqueued_total": 5,
                "tickets_acknowledged_total": 4,
                "tickets_committed_total": 3,
                "tickets_timed_out_total": 1,
            },
        }
        _PoolsContractHandler.ready_status = 503
        _PoolsContractHandler.ready_payload = {
            "status": "not_ready",
            "role": "decode",
            "reason": "distributed kv transport degraded",
            "model_loaded": True,
            "decode_pool_warm": True,
            "disagg_transport_degraded": True,
            "disagg_timeout_debt": 2,
            "disagg_timeout_debt_threshold": 6,
            "disagg_timeout_streak": 4,
            "disagg_timeout_streak_threshold": 3,
        }
        _PoolsContractHandler.metrics_body = "\n".join(
            [
                'inferflux_scheduler_queue_depth 7',
                'inferflux_prefill_queue_depth 2',
                'inferflux_decode_queue_depth 5',
                'inferflux_scheduler_batch_limit_size 4',
                'inferflux_scheduler_batch_limit_tokens 8192',
                'inferflux_disagg_kv_enqueue_rejections_total{backend="cpu"} 2',
                'inferflux_disagg_kv_enqueue_exhausted_total{backend="cpu"} 1',
                'inferflux_disagg_kv_tickets_total{backend="cpu",stage="enqueued"} 5',
                'inferflux_disagg_kv_tickets_total{backend="cpu",stage="acknowledged"} 4',
                'inferflux_disagg_kv_tickets_total{backend="cpu",stage="committed"} 3',
                'inferflux_disagg_kv_tickets_total{backend="cpu",stage="timed_out"} 1',
                'inferflux_disagg_kv_timeout_debt{backend="cpu"} 2',
                "",
            ]
        )

        cls.httpd = _ThreadedTcpServer(("127.0.0.1", 0), _PoolsContractHandler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)

    def _run_inferctl(self, extra_args=None):
        extra_args = extra_args or []
        result = subprocess.run(
            [
                INFERCTL_BIN,
                "admin",
                "pools",
                "--get",
                *extra_args,
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--api-key",
                "dev-key-123",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        return result

    def test_admin_pools_prefers_server_endpoint_payload(self):
        _PoolsContractHandler.admin_pools_status = 200
        _PoolsContractHandler.ready_status = 500
        _PoolsContractHandler.metrics_body = ""

        result = self._run_inferctl()
        self.assertEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload, _PoolsContractHandler.admin_pools_payload)

    def test_admin_pools_falls_back_to_readyz_and_metrics_when_endpoint_missing(self):
        _PoolsContractHandler.admin_pools_status = 404
        _PoolsContractHandler.ready_status = 503
        _PoolsContractHandler.metrics_body = "\n".join(
            [
                'inferflux_scheduler_queue_depth 7',
                'inferflux_prefill_queue_depth 2',
                'inferflux_decode_queue_depth 5',
                'inferflux_scheduler_batch_limit_size 4',
                'inferflux_scheduler_batch_limit_tokens 8192',
                'inferflux_disagg_kv_enqueue_rejections_total{backend="cpu"} 2',
                'inferflux_disagg_kv_enqueue_exhausted_total{backend="cpu"} 1',
                'inferflux_disagg_kv_tickets_total{backend="cpu",stage="enqueued"} 5',
                'inferflux_disagg_kv_tickets_total{backend="cpu",stage="acknowledged"} 4',
                'inferflux_disagg_kv_tickets_total{backend="cpu",stage="committed"} 3',
                'inferflux_disagg_kv_tickets_total{backend="cpu",stage="timed_out"} 1',
                'inferflux_disagg_kv_timeout_debt{backend="cpu"} 2',
                "",
            ]
        )

        result = self._run_inferctl()
        self.assertEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload.get("status"), "ok")
        self.assertEqual(payload["pool_health"]["http_status"], 503)
        self.assertFalse(payload["pool_health"]["ready"])
        self.assertEqual(payload["pool_health"]["role"], "decode")
        self.assertEqual(
            payload["pool_health"]["reason"], "distributed kv transport degraded"
        )
        self.assertTrue(payload["pool_health"]["model_loaded"])
        self.assertTrue(payload["pool_health"]["decode_pool_warm"])
        self.assertTrue(payload["pool_health"]["disagg_transport_degraded"])
        self.assertEqual(payload["pool_health"]["disagg_timeout_debt"], 2)
        self.assertEqual(payload["pool_health"]["disagg_timeout_debt_threshold"], 6)
        self.assertEqual(payload["pool_health"]["disagg_timeout_streak"], 4)
        self.assertEqual(
            payload["pool_health"]["disagg_timeout_streak_threshold"], 3
        )
        self.assertEqual(payload["scheduler"]["queue_depth"], 7)
        self.assertEqual(payload["scheduler"]["prefill_queue_depth"], 2)
        self.assertEqual(payload["scheduler"]["decode_queue_depth"], 5)
        self.assertEqual(payload["distributed_kv"]["enqueue_rejections_total"], 2)
        self.assertEqual(payload["distributed_kv"]["enqueue_exhausted_total"], 1)
        self.assertEqual(payload["distributed_kv"]["tickets_enqueued_total"], 5)
        self.assertEqual(payload["distributed_kv"]["tickets_acknowledged_total"], 4)
        self.assertEqual(payload["distributed_kv"]["tickets_committed_total"], 3)
        self.assertEqual(payload["distributed_kv"]["tickets_timed_out_total"], 1)

    def test_admin_pools_table_output_contains_sections_and_values(self):
        _PoolsContractHandler.admin_pools_status = 200
        result = self._run_inferctl(["--table"])
        self.assertEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("POOL HEALTH", result.stdout)
        self.assertIn("SCHEDULER", result.stdout)
        self.assertIn("DISTRIBUTED KV", result.stdout)
        self.assertIn("Timeout debt", result.stdout)
        self.assertIn("2 / 6", result.stdout)
        self.assertIn("Tickets committed", result.stdout)
        self.assertIn("3", result.stdout)

    def test_admin_pools_rejects_multiple_output_modes(self):
        result = self._run_inferctl(["--json", "--table"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("choose at most one of --json, --table", result.stderr)


if __name__ == "__main__":
    unittest.main()
