#!/usr/bin/env python3
import json
import os
import signal
import subprocess
import time
import unittest
import http.client

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 18081
SERVER_FAIL_PORT = 18082
SERVER_STRICT_PORT = 18083
SERVER_BIN = os.environ.get("INFERFLUX_SERVER_BIN", "./build/inferfluxd")
INFERCTL_BIN = os.environ.get("INFERCTL_BIN", "./build/inferctl")

class StubIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = os.environ.copy()
        env["INFERFLUX_HOST_OVERRIDE"] = SERVER_HOST
        env["INFERFLUX_PORT_OVERRIDE"] = str(SERVER_PORT)
        env["INFERFLUX_MODEL_PATH"] = ""
        # We assume the binary is in the build directory and we are in the project root.
        cls.server_proc = subprocess.Popen(
            [SERVER_BIN, "--config", "config/server.yaml"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        deadline = time.time() + 20.0
        ready = False
        while time.time() < deadline:
            if cls.server_proc.poll() is not None:
                break
            try:
                conn = http.client.HTTPConnection(SERVER_HOST, SERVER_PORT, timeout=1)
                conn.request("GET", "/livez", headers={"Authorization": "Bearer dev-key-123"})
                resp = conn.getresponse()
                resp.read()
                conn.close()
                if resp.status in (200, 401):
                    ready = True
                    break
            except Exception:
                time.sleep(0.1)
        if not ready:
            try:
                out, err = cls.server_proc.communicate(timeout=1)
            except Exception:
                out, err = ("", "")
            raise RuntimeError(
                "inferfluxd did not become ready in time; "
                f"stdout={out[-400:] if out else ''} stderr={err[-400:] if err else ''}"
            )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'server_proc') and cls.server_proc:
            try:
                os.killpg(os.getpgid(cls.server_proc.pid), signal.SIGTERM)
                cls.server_proc.wait(timeout=5)
            except:
                pass

    def _post(self, path, data):
        conn = http.client.HTTPConnection(SERVER_HOST, SERVER_PORT, timeout=10)
        headers = {"Content-Type": "application/json", "Authorization": "Bearer dev-key-123"}
        conn.request("POST", path, body=json.dumps(data), headers=headers)
        resp = conn.getresponse()
        body = resp.read().decode()
        conn.close()
        return resp, body

    def _put(self, path, data):
        conn = http.client.HTTPConnection(SERVER_HOST, SERVER_PORT, timeout=10)
        headers = {"Content-Type": "application/json", "Authorization": "Bearer dev-key-123"}
        conn.request("PUT", path, body=json.dumps(data), headers=headers)
        resp = conn.getresponse()
        body = resp.read().decode()
        conn.close()
        return resp, body

    def _delete(self, path, data):
        conn = http.client.HTTPConnection(SERVER_HOST, SERVER_PORT, timeout=10)
        headers = {"Content-Type": "application/json", "Authorization": "Bearer dev-key-123"}
        conn.request("DELETE", path, body=json.dumps(data), headers=headers)
        resp = conn.getresponse()
        body = resp.read().decode()
        conn.close()
        return resp, body

    def _get(self, path):
        conn = http.client.HTTPConnection(SERVER_HOST, SERVER_PORT, timeout=10)
        headers = {"Authorization": "Bearer dev-key-123"}
        conn.request("GET", path, headers=headers)
        resp = conn.getresponse()
        body = resp.read().decode()
        conn.close()
        return resp, body

    def _run_inferctl(self, args, api_key="dev-key-123"):
        cmd = [
            INFERCTL_BIN,
            *args,
            "--host",
            SERVER_HOST,
            "--port",
            str(SERVER_PORT),
            "--api-key",
            api_key,
        ]
        return subprocess.run(cmd, capture_output=True, text=True, check=False)

    def _get_routing_policy_via_inferctl(self):
        result = self._run_inferctl(["admin", "routing", "--get"])
        self.assertEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        return json.loads(result.stdout)

    def test_health(self):
        # /healthz might require auth depending on config
        conn = http.client.HTTPConnection(SERVER_HOST, SERVER_PORT, timeout=5)
        headers = {"Authorization": "Bearer dev-key-123"}
        conn.request("GET", "/healthz", headers=headers)
        resp = conn.getresponse()
        body = resp.read().decode()
        self.assertEqual(resp.status, 200, msg=f"Status: {resp.status}, Body: {body}")
        conn.close()

    def test_stub_completion(self):
        resp, body = self._post("/v1/completions", {"model": "default", "prompt": "hi"})
        self.assertEqual(resp.status, 200, msg=f"Status: {resp.status}, Body: {body}")

    def test_cache_warm(self):
        resp, body = self._post("/v1/admin/cache/warm", {"tokens": [1, 2, 3], "block_table": [100]})
        self.assertEqual(resp.status, 200, msg=f"Status: {resp.status}, Body: {body}")

    def test_embeddings_requires_input(self):
        resp, body = self._post("/v1/embeddings", {})
        self.assertEqual(resp.status, 400, msg=f"Status: {resp.status}, Body: {body}")

    def test_embeddings_backend_unavailable_default_model(self):
        resp, body = self._post("/v1/embeddings", {"input": "hello"})
        self.assertEqual(resp.status, 503, msg=f"Status: {resp.status}, Body: {body}")
        payload = json.loads(body)
        self.assertEqual(payload.get("error"), "no_backend")

    def test_embeddings_explicit_model_not_found(self):
        resp, body = self._post("/v1/embeddings", {"model": "explicit-model", "input": "hello"})
        self.assertEqual(resp.status, 404, msg=f"Status: {resp.status}, Body: {body}")
        payload = json.loads(body)
        self.assertEqual(payload.get("error"), "model_not_found")

    def test_completion_explicit_model_not_found(self):
        resp, body = self._post("/v1/completions", {"model": "explicit-model", "prompt": "hi"})
        self.assertEqual(resp.status, 404, msg=f"Status: {resp.status}, Body: {body}")
        payload = json.loads(body)
        self.assertEqual(payload.get("error"), "model_not_found")

    def test_models_get_by_id_not_found(self):
        resp, body = self._get("/v1/models/explicit-model")
        self.assertEqual(resp.status, 404, msg=f"Status: {resp.status}, Body: {body}")
        payload = json.loads(body)
        self.assertEqual(payload.get("error"), "model_not_found")

    def test_admin_models_load_requires_path(self):
        resp, body = self._post("/v1/admin/models", {"id": "explicit-model"})
        self.assertEqual(resp.status, 400, msg=f"Status: {resp.status}, Body: {body}")
        payload = json.loads(body)
        self.assertEqual(payload.get("error"), "path is required")

    def test_admin_models_unload_requires_id(self):
        resp, body = self._delete("/v1/admin/models", {})
        self.assertEqual(resp.status, 400, msg=f"Status: {resp.status}, Body: {body}")
        payload = json.loads(body)
        self.assertEqual(payload.get("error"), "id is required")

    def test_admin_models_unload_not_found(self):
        resp, body = self._delete("/v1/admin/models", {"id": "explicit-model"})
        self.assertEqual(resp.status, 404, msg=f"Status: {resp.status}, Body: {body}")
        payload = json.loads(body)
        self.assertEqual(payload.get("error"), "model_not_found")

    def test_admin_models_set_default_requires_id(self):
        resp, body = self._put("/v1/admin/models/default", {})
        self.assertEqual(resp.status, 400, msg=f"Status: {resp.status}, Body: {body}")
        payload = json.loads(body)
        self.assertEqual(payload.get("error"), "id is required")

    def test_admin_models_set_default_not_found(self):
        resp, body = self._put("/v1/admin/models/default", {"id": "explicit-model"})
        self.assertEqual(resp.status, 404, msg=f"Status: {resp.status}, Body: {body}")
        payload = json.loads(body)
        self.assertEqual(payload.get("error"), "model_not_found")

    def test_inferctl_admin_models_unload_not_found(self):
        result = self._run_inferctl(["admin", "models", "--unload", "explicit-model"])
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload.get("error"), "model_not_found")

    def test_inferctl_admin_models_set_default_not_found(self):
        result = self._run_inferctl(
            ["admin", "models", "--set-default", "explicit-model"]
        )
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload.get("error"), "model_not_found")

    def test_inferctl_models_json(self):
        result = self._run_inferctl(["models", "--json"])
        self.assertEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        payload = json.loads(result.stdout)
        self.assertIn("data", payload)
        self.assertIsInstance(payload["data"], list)
        for model in payload["data"]:
            self.assertIn("backend_exposure", model)
            self.assertIsInstance(model["backend_exposure"], dict)
            self.assertIn("requested_backend", model["backend_exposure"])
            self.assertIn("exposed_backend", model["backend_exposure"])
            self.assertIn("provider", model["backend_exposure"])
            self.assertIn("fallback", model["backend_exposure"])

    def test_inferctl_models_id_not_found_json(self):
        result = self._run_inferctl(["models", "--id", "explicit-model", "--json"])
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload.get("error"), "model_not_found")

    def test_inferctl_models_id_not_found_default_output(self):
        result = self._run_inferctl(["models", "--id", "explicit-model"])
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload.get("error"), "model_not_found")

    def test_inferctl_models_id_requires_value(self):
        result = self._run_inferctl(["models", "--id"])
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("--id requires MODEL_ID", result.stderr)

    def test_inferctl_admin_models_load_requires_value(self):
        result = self._run_inferctl(["admin", "models", "--load"])
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("--load requires PATH", result.stderr)

    def test_inferctl_admin_models_unload_requires_value(self):
        result = self._run_inferctl(["admin", "models", "--unload"])
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("--unload requires ID", result.stderr)

    def test_inferctl_admin_models_set_default_requires_value(self):
        result = self._run_inferctl(["admin", "models", "--set-default"])
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("--set-default requires ID", result.stderr)

    def test_inferctl_admin_models_rejects_multiple_operations(self):
        result = self._run_inferctl(
            ["admin", "models", "--list", "--unload", "explicit-model"]
        )
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("choose exactly one", result.stderr)

    def test_inferctl_admin_models_default_requires_load(self):
        result = self._run_inferctl(["admin", "models", "--default"])
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("--default requires --load PATH", result.stderr)

    def test_inferctl_admin_cache_rejects_multiple_operations(self):
        result = self._run_inferctl(["admin", "cache", "--status", "--warm"])
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("choose exactly one of --status or --warm", result.stderr)

    def test_inferctl_admin_cache_tokens_requires_warm(self):
        result = self._run_inferctl(["admin", "cache", "--tokens", "1,2,3"])
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("--tokens requires --warm", result.stderr)

    def test_inferctl_admin_cache_warm_requires_tokens(self):
        result = self._run_inferctl(["admin", "cache", "--warm", "--completion", "ok"])
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("--tokens is required with --warm", result.stderr)

    def test_inferctl_admin_cache_warm_requires_completion(self):
        result = self._run_inferctl(["admin", "cache", "--warm", "--tokens", "1,2,3"])
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("--completion is required with --warm", result.stderr)

    def test_inferctl_admin_cache_completion_tokens_requires_integer(self):
        result = self._run_inferctl(
            [
                "admin",
                "cache",
                "--warm",
                "--tokens",
                "1,2,3",
                "--completion",
                "ok",
                "--completion-tokens",
                "abc",
            ]
        )
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("--completion-tokens must be an integer", result.stderr)

    def test_inferctl_admin_cache_status_auth_failure(self):
        result = self._run_inferctl(["admin", "cache", "--status"], api_key="invalid-key")
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("authentication required", result.stderr)

    def test_inferctl_admin_api_keys_rejects_multiple_operations(self):
        result = self._run_inferctl(
            ["admin", "api-keys", "--list", "--remove", "explicit-key"]
        )
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("choose exactly one of --list, --add, --remove", result.stderr)

    def test_inferctl_admin_api_keys_scopes_requires_add(self):
        result = self._run_inferctl(["admin", "api-keys", "--scopes", "read,admin"])
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("--scopes requires --add KEY", result.stderr)

    def test_inferctl_admin_api_keys_add_requires_scopes(self):
        result = self._run_inferctl(["admin", "api-keys", "--add", "explicit-key"])
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("--add requires --scopes read,admin", result.stderr)

    def test_inferctl_admin_api_keys_add_requires_value(self):
        result = self._run_inferctl(["admin", "api-keys", "--add"])
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("--add requires KEY", result.stderr)

    def test_inferctl_admin_api_keys_remove_requires_value(self):
        result = self._run_inferctl(["admin", "api-keys", "--remove"])
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("--remove requires KEY", result.stderr)

    def test_inferctl_admin_api_keys_list_auth_failure(self):
        result = self._run_inferctl(["admin", "api-keys", "--list"], api_key="invalid-key")
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("authentication required", result.stderr)

    def test_inferctl_admin_guardrails_rejects_multiple_operations(self):
        result = self._run_inferctl(
            ["admin", "guardrails", "--list", "--set", "secret,pii"]
        )
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("choose exactly one of --list, --set", result.stderr)

    def test_inferctl_admin_guardrails_set_requires_value(self):
        result = self._run_inferctl(["admin", "guardrails", "--set"])
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("--set requires word1,word2", result.stderr)

    def test_inferctl_admin_guardrails_list_auth_failure(self):
        result = self._run_inferctl(
            ["admin", "guardrails", "--list"], api_key="invalid-key"
        )
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("authentication required", result.stderr)

    def test_inferctl_admin_rate_limit_rejects_multiple_operations(self):
        result = self._run_inferctl(["admin", "rate-limit", "--get", "--set", "120"])
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("choose exactly one of --get, --set", result.stderr)

    def test_inferctl_admin_rate_limit_set_requires_value(self):
        result = self._run_inferctl(["admin", "rate-limit", "--set"])
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("--set requires N", result.stderr)

    def test_inferctl_admin_rate_limit_set_requires_integer(self):
        result = self._run_inferctl(["admin", "rate-limit", "--set", "abc"])
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("--set must be an integer", result.stderr)

    def test_inferctl_admin_rate_limit_get_auth_failure(self):
        result = self._run_inferctl(
            ["admin", "rate-limit", "--get"], api_key="invalid-key"
        )
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("authentication required", result.stderr)

    def test_inferctl_admin_pools_requires_get(self):
        result = self._run_inferctl(["admin", "pools"])
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("--get is required", result.stderr)

    def test_inferctl_admin_pools_get_auth_failure(self):
        result = self._run_inferctl(
            ["admin", "pools", "--get"], api_key="invalid-key"
        )
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("authentication required", result.stderr)

    def test_inferctl_admin_pools_get(self):
        result = self._run_inferctl(["admin", "pools", "--get"])
        self.assertEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload.get("status"), "ok")
        self.assertIn("pool_health", payload)
        self.assertIn("scheduler", payload)
        self.assertIn("distributed_kv", payload)
        self.assertIn("ready", payload["pool_health"])
        self.assertIn("role", payload["pool_health"])
        self.assertIn("reason", payload["pool_health"])
        self.assertIn("model_loaded", payload["pool_health"])
        self.assertIn("decode_pool_warm", payload["pool_health"])
        self.assertIn("disagg_transport_degraded", payload["pool_health"])
        self.assertIn("disagg_timeout_debt", payload["pool_health"])
        self.assertIn("disagg_timeout_debt_threshold", payload["pool_health"])
        self.assertIn("disagg_timeout_streak", payload["pool_health"])
        self.assertIn("disagg_timeout_streak_threshold", payload["pool_health"])
        self.assertIn("queue_depth", payload["scheduler"])
        self.assertIn("prefill_queue_depth", payload["scheduler"])
        self.assertIn("decode_queue_depth", payload["scheduler"])
        self.assertIn("enqueue_rejections_total", payload["distributed_kv"])
        self.assertIn("enqueue_exhausted_total", payload["distributed_kv"])
        self.assertIn("tickets_enqueued_total", payload["distributed_kv"])
        self.assertIn("tickets_acknowledged_total", payload["distributed_kv"])
        self.assertIn("tickets_committed_total", payload["distributed_kv"])
        self.assertIn("tickets_timed_out_total", payload["distributed_kv"])

    def test_inferctl_admin_routing_rejects_multiple_operations(self):
        result = self._run_inferctl(["admin", "routing", "--get", "--set"])
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("choose exactly one of --get, --set", result.stderr)

    def test_inferctl_admin_routing_set_requires_fields(self):
        result = self._run_inferctl(["admin", "routing", "--set"])
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("--set requires at least one of", result.stderr)

    def test_inferctl_admin_routing_allow_default_requires_set(self):
        result = self._run_inferctl(
            ["admin", "routing", "--allow-default-fallback", "true"]
        )
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("--allow-default-fallback requires --set", result.stderr)

    def test_inferctl_admin_routing_allow_default_requires_value(self):
        result = self._run_inferctl(
            ["admin", "routing", "--set", "--allow-default-fallback"]
        )
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("--allow-default-fallback requires true|false", result.stderr)

    def test_inferctl_admin_routing_allow_default_requires_boolean(self):
        result = self._run_inferctl(
            ["admin", "routing", "--set", "--allow-default-fallback", "maybe"]
        )
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("--allow-default-fallback must be true or false", result.stderr)

    def test_inferctl_admin_routing_require_ready_requires_boolean(self):
        result = self._run_inferctl(
            ["admin", "routing", "--set", "--require-ready-backend", "maybe"]
        )
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("--require-ready-backend must be true or false", result.stderr)

    def test_inferctl_admin_routing_fallback_scope_requires_value(self):
        result = self._run_inferctl(
            ["admin", "routing", "--set", "--fallback-scope"]
        )
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn(
            "--fallback-scope requires any_compatible|same_path_only", result.stderr
        )

    def test_inferctl_admin_routing_fallback_scope_validates_value(self):
        result = self._run_inferctl(
            ["admin", "routing", "--set", "--fallback-scope", "invalid-scope"]
        )
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn(
            "--fallback-scope must be any_compatible or same_path_only", result.stderr
        )

    def test_inferctl_admin_routing_get_auth_failure(self):
        result = self._run_inferctl(
            ["admin", "routing", "--get"], api_key="invalid-key"
        )
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("authentication required", result.stderr)

    def test_inferctl_admin_routing_get(self):
        result = self._run_inferctl(["admin", "routing", "--get"])
        self.assertEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        payload = json.loads(result.stdout)
        self.assertIn("allow_default_fallback", payload)
        self.assertIn("require_ready_backend", payload)
        self.assertIn("fallback_scope", payload)

    def test_inferctl_admin_routing_set_scope_success(self):
        before = self._get_routing_policy_via_inferctl()
        original_scope = before.get("fallback_scope", "any_compatible")
        target_scope = (
            "same_path_only" if original_scope != "same_path_only" else "any_compatible"
        )
        try:
            result = self._run_inferctl(
                ["admin", "routing", "--set", "--fallback-scope", target_scope]
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
            )
            payload = json.loads(result.stdout)
            self.assertEqual(payload.get("status"), "ok")
            self.assertEqual(payload.get("fallback_scope"), target_scope)

            after = self._get_routing_policy_via_inferctl()
            self.assertEqual(after.get("fallback_scope"), target_scope)
        finally:
            self._run_inferctl(
                ["admin", "routing", "--set", "--fallback-scope", original_scope]
            )

    def test_inferctl_admin_routing_set_booleans_success(self):
        before = self._get_routing_policy_via_inferctl()
        original_allow = bool(before.get("allow_default_fallback", True))
        original_ready = bool(before.get("require_ready_backend", True))
        target_allow = not original_allow
        target_ready = not original_ready
        try:
            result = self._run_inferctl(
                [
                    "admin",
                    "routing",
                    "--set",
                    "--allow-default-fallback",
                    "true" if target_allow else "false",
                    "--require-ready-backend",
                    "true" if target_ready else "false",
                ]
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
            )
            payload = json.loads(result.stdout)
            self.assertEqual(payload.get("status"), "ok")
            self.assertEqual(payload.get("allow_default_fallback"), target_allow)
            self.assertEqual(payload.get("require_ready_backend"), target_ready)

            after = self._get_routing_policy_via_inferctl()
            self.assertEqual(after.get("allow_default_fallback"), target_allow)
            self.assertEqual(after.get("require_ready_backend"), target_ready)
        finally:
            self._run_inferctl(
                [
                    "admin",
                    "routing",
                    "--set",
                    "--allow-default-fallback",
                    "true" if original_allow else "false",
                    "--require-ready-backend",
                    "true" if original_ready else "false",
                ]
            )

    def test_inferctl_models_default_output(self):
        result = self._run_inferctl(["models"])
        self.assertEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        self.assertTrue(lines, msg=f"stdout was empty: {result.stdout!r}")
        self.assertTrue(
            lines[0] == "(no models loaded)" or lines[0].startswith("ID"),
            msg=f"unexpected inferctl models output: {result.stdout!r}",
        )
        if lines[0] != "(no models loaded)":
            self.assertIn("EXPOSED-BE", lines[0])
            self.assertIn("REQ-BE", lines[0])
            self.assertIn("PROVIDER", lines[0])
            self.assertIn("FALLBACK", lines[0])

    def test_inferctl_admin_models_list_json(self):
        result = self._run_inferctl(["admin", "models", "--list", "--json"])
        self.assertEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        payload = json.loads(result.stdout)
        self.assertIn("models", payload)
        self.assertIsInstance(payload["models"], list)
        for model in payload["models"]:
            self.assertIn("backend_exposure", model)
            self.assertIsInstance(model["backend_exposure"], dict)
            self.assertIn("requested_backend", model["backend_exposure"])
            self.assertIn("exposed_backend", model["backend_exposure"])
            self.assertIn("provider", model["backend_exposure"])
            self.assertIn("fallback", model["backend_exposure"])

    def test_inferctl_admin_models_list_default_output(self):
        result = self._run_inferctl(["admin", "models", "--list"])
        self.assertEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        self.assertTrue(lines, msg=f"stdout was empty: {result.stdout!r}")
        self.assertTrue(
            lines[0] == "(no models loaded)" or lines[0].startswith("DEF"),
            msg=f"unexpected inferctl admin models output: {result.stdout!r}",
        )
        if lines[0] != "(no models loaded)":
            self.assertIn("EXPOSED-BE", lines[0])
            self.assertIn("REQ-BE", lines[0])
            self.assertIn("PROVIDER", lines[0])
            self.assertIn("FALLBACK", lines[0])

    def test_inferctl_models_json_auth_failure(self):
        result = self._run_inferctl(["models", "--json"], api_key="invalid-key")
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("authentication required", result.stderr)

    def test_inferctl_admin_models_list_json_auth_failure(self):
        result = self._run_inferctl(
            ["admin", "models", "--list", "--json"], api_key="invalid-key"
        )
        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"rc={result.returncode} stdout={result.stdout} stderr={result.stderr}",
        )
        self.assertIn("authentication required", result.stderr)

    def test_admin_routing_policy(self):
        resp, body = self._get("/v1/admin/routing")
        self.assertEqual(resp.status, 200, msg=f"Status: {resp.status}, Body: {body}")
        payload = json.loads(body)
        self.assertIn("allow_default_fallback", payload)
        self.assertIn("require_ready_backend", payload)
        self.assertIn("fallback_scope", payload)

        resp, body = self._put("/v1/admin/routing", {"fallback_scope": "same_path_only"})
        self.assertEqual(resp.status, 200, msg=f"Status: {resp.status}, Body: {body}")

        resp, body = self._get("/v1/admin/routing")
        self.assertEqual(resp.status, 200, msg=f"Status: {resp.status}, Body: {body}")
        payload = json.loads(body)
        self.assertEqual(payload.get("fallback_scope"), "same_path_only")

        # Restore default scope so this test does not affect other flows.
        resp, body = self._put("/v1/admin/routing", {"fallback_scope": "any_compatible"})
        self.assertEqual(resp.status, 200, msg=f"Status: {resp.status}, Body: {body}")

class StubIntegrationPolicyPersistenceFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = os.environ.copy()
        env["INFERFLUX_HOST_OVERRIDE"] = SERVER_HOST
        env["INFERFLUX_PORT_OVERRIDE"] = str(SERVER_FAIL_PORT)
        env["INFERFLUX_MODEL_PATH"] = ""
        env["INFERFLUX_POLICY_STORE"] = "/proc/inferflux_policy_unwritable.conf"
        cls.server_proc = subprocess.Popen(
            [SERVER_BIN, "--config", "config/server.yaml"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        deadline = time.time() + 20.0
        ready = False
        while time.time() < deadline:
            if cls.server_proc.poll() is not None:
                break
            try:
                conn = http.client.HTTPConnection(SERVER_HOST, SERVER_FAIL_PORT, timeout=1)
                conn.request("GET", "/livez", headers={"Authorization": "Bearer dev-key-123"})
                resp = conn.getresponse()
                resp.read()
                conn.close()
                if resp.status in (200, 401):
                    ready = True
                    break
            except Exception:
                time.sleep(0.1)
        if not ready:
            try:
                out, err = cls.server_proc.communicate(timeout=1)
            except Exception:
                out, err = ("", "")
            raise RuntimeError(
                "inferfluxd did not become ready in time; "
                f"stdout={out[-400:] if out else ''} stderr={err[-400:] if err else ''}"
            )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'server_proc') and cls.server_proc:
            try:
                os.killpg(os.getpgid(cls.server_proc.pid), signal.SIGTERM)
                cls.server_proc.wait(timeout=5)
            except:
                pass

    def _put(self, path, data):
        conn = http.client.HTTPConnection(SERVER_HOST, SERVER_FAIL_PORT, timeout=10)
        headers = {"Content-Type": "application/json", "Authorization": "Bearer dev-key-123"}
        conn.request("PUT", path, body=json.dumps(data), headers=headers)
        resp = conn.getresponse()
        body = resp.read().decode()
        conn.close()
        return resp, body

    def _post(self, path, data):
        conn = http.client.HTTPConnection(SERVER_HOST, SERVER_FAIL_PORT, timeout=10)
        headers = {"Content-Type": "application/json", "Authorization": "Bearer dev-key-123"}
        conn.request("POST", path, body=json.dumps(data), headers=headers)
        resp = conn.getresponse()
        body = resp.read().decode()
        conn.close()
        return resp, body

    def _delete(self, path, data):
        conn = http.client.HTTPConnection(SERVER_HOST, SERVER_FAIL_PORT, timeout=10)
        headers = {"Content-Type": "application/json", "Authorization": "Bearer dev-key-123"}
        conn.request("DELETE", path, body=json.dumps(data), headers=headers)
        resp = conn.getresponse()
        body = resp.read().decode()
        conn.close()
        return resp, body

    def _get(self, path):
        return self._get_with_key(path, "dev-key-123")

    def _get_with_key(self, path, key):
        conn = http.client.HTTPConnection(SERVER_HOST, SERVER_FAIL_PORT, timeout=10)
        headers = {"Authorization": f"Bearer {key}"}
        conn.request("GET", path, headers=headers)
        resp = conn.getresponse()
        body = resp.read().decode()
        conn.close()
        return resp, body

    def test_admin_routing_returns_500_when_policy_store_unwritable(self):
        resp, body = self._get("/v1/admin/routing")
        self.assertEqual(resp.status, 200, msg=f"Status: {resp.status}, Body: {body}")
        before = json.loads(body)

        resp, body = self._put("/v1/admin/routing", {"fallback_scope": "same_path_only"})
        self.assertEqual(resp.status, 500, msg=f"Status: {resp.status}, Body: {body}")
        payload = json.loads(body)
        self.assertEqual(payload.get("error"), "policy_persist_failed")

        resp, body = self._get("/v1/admin/routing")
        self.assertEqual(resp.status, 200, msg=f"Status: {resp.status}, Body: {body}")
        after = json.loads(body)
        self.assertEqual(after.get("fallback_scope"), before.get("fallback_scope"))

    def test_admin_guardrails_returns_500_and_rolls_back(self):
        resp, body = self._get("/v1/admin/guardrails")
        self.assertEqual(resp.status, 200, msg=f"Status: {resp.status}, Body: {body}")
        before = json.loads(body)

        resp, body = self._put("/v1/admin/guardrails", {"blocklist": ["only-new-value"]})
        self.assertEqual(resp.status, 500, msg=f"Status: {resp.status}, Body: {body}")
        payload = json.loads(body)
        self.assertEqual(payload.get("error"), "policy_persist_failed")

        resp, body = self._get("/v1/admin/guardrails")
        self.assertEqual(resp.status, 200, msg=f"Status: {resp.status}, Body: {body}")
        after = json.loads(body)
        self.assertEqual(after.get("blocklist"), before.get("blocklist"))

    def test_admin_rate_limit_returns_500_and_rolls_back(self):
        resp, body = self._get("/v1/admin/rate_limit")
        self.assertEqual(resp.status, 200, msg=f"Status: {resp.status}, Body: {body}")
        before = json.loads(body)

        resp, body = self._put("/v1/admin/rate_limit", {"tokens_per_minute": 999})
        self.assertEqual(resp.status, 500, msg=f"Status: {resp.status}, Body: {body}")
        payload = json.loads(body)
        self.assertEqual(payload.get("error"), "policy_persist_failed")

        resp, body = self._get("/v1/admin/rate_limit")
        self.assertEqual(resp.status, 200, msg=f"Status: {resp.status}, Body: {body}")
        after = json.loads(body)
        self.assertEqual(after.get("tokens_per_minute"), before.get("tokens_per_minute"))

    def test_admin_api_key_upsert_returns_500_and_rolls_back(self):
        key = "persist-fail-temp-key"

        resp, body = self._post("/v1/admin/api_keys", {"key": key, "scopes": ["admin"]})
        self.assertEqual(resp.status, 500, msg=f"Status: {resp.status}, Body: {body}")
        payload = json.loads(body)
        self.assertEqual(payload.get("error"), "policy_persist_failed")

        # Ensure key was not left active in memory.
        resp, body = self._get_with_key("/v1/admin/rate_limit", key)
        self.assertEqual(resp.status, 401, msg=f"Status: {resp.status}, Body: {body}")

    def test_z_admin_api_key_delete_returns_500_and_rolls_back(self):
        # Deleting the active admin key should fail and keep it usable.
        resp, body = self._delete("/v1/admin/api_keys", {"key": "dev-key-123"})
        self.assertEqual(resp.status, 500, msg=f"Status: {resp.status}, Body: {body}")
        payload = json.loads(body)
        self.assertEqual(payload.get("error"), "policy_persist_failed")

        resp, body = self._get("/v1/admin/rate_limit")
        self.assertEqual(resp.status, 200, msg=f"Status: {resp.status}, Body: {body}")


class StubIntegrationStrictNativePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = os.environ.copy()
        env["INFERFLUX_HOST_OVERRIDE"] = SERVER_HOST
        env["INFERFLUX_PORT_OVERRIDE"] = str(SERVER_STRICT_PORT)
        env["INFERFLUX_MODEL_PATH"] = ""
        env["INFERFLUX_BACKEND_STRICT_INFERFLUX_REQUEST"] = "true"
        cls.server_proc = subprocess.Popen(
            [SERVER_BIN, "--config", "config/server.yaml"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        deadline = time.time() + 20.0
        ready = False
        while time.time() < deadline:
            if cls.server_proc.poll() is not None:
                break
            try:
                conn = http.client.HTTPConnection(SERVER_HOST, SERVER_STRICT_PORT, timeout=1)
                conn.request("GET", "/livez", headers={"Authorization": "Bearer dev-key-123"})
                resp = conn.getresponse()
                resp.read()
                conn.close()
                if resp.status in (200, 401):
                    ready = True
                    break
            except Exception:
                time.sleep(0.1)
        if not ready:
            try:
                out, err = cls.server_proc.communicate(timeout=1)
            except Exception:
                out, err = ("", "")
            raise RuntimeError(
                "inferfluxd did not become ready in time; "
                f"stdout={out[-400:] if out else ''} stderr={err[-400:] if err else ''}"
            )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'server_proc') and cls.server_proc:
            try:
                os.killpg(os.getpgid(cls.server_proc.pid), signal.SIGTERM)
                cls.server_proc.wait(timeout=5)
            except:
                pass

    def _post(self, path, data):
        conn = http.client.HTTPConnection(SERVER_HOST, SERVER_STRICT_PORT, timeout=10)
        headers = {"Content-Type": "application/json", "Authorization": "Bearer dev-key-123"}
        conn.request("POST", path, body=json.dumps(data), headers=headers)
        resp = conn.getresponse()
        body = resp.read().decode()
        conn.close()
        return resp, body

    def test_admin_models_load_explicit_native_rejected_in_strict_mode(self):
        resp, body = self._post(
            "/v1/admin/models",
            {
                "id": "strict-native-test",
                "path": "/tmp/strict-native-test.gguf",
                "backend": "inferflux_cuda",
                "format": "gguf",
            },
        )
        self.assertEqual(resp.status, 422, msg=f"Status: {resp.status}, Body: {body}")
        payload = json.loads(body)
        self.assertEqual(payload.get("error"), "backend_policy_violation")
        reason = payload.get("reason", "")
        self.assertIn("strict_inferflux_request", reason)

if __name__ == "__main__":
    unittest.main()
