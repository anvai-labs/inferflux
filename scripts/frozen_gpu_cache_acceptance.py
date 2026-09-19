#!/usr/bin/env python3
"""Replay approved request DATA on an owned CUDA origin and Sandhi gateway.

Only gpu-gates.yml dispatched from trusted main may execute this harness. Bundle
Python/tools are never imported or executed. Evidence contains hashes and counts,
never model responses, prompts, configuration, credentials or raw process logs.
"""

import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import socket
import sqlite3
import subprocess
import tempfile
import threading
import time

BUNDLE = Path("/tmp/victor-member-replay-63ad80a3f502")
MODEL = "qwen3-coder-30b"
ORIGIN_PORT = 28084
GATEWAY_PORT = 18793
SHARED_PORT = 8080
MAX_PAYLOAD = 128 * 1024
MAX_RESPONSE = 1024 * 1024
MAX_LOG = 8 * 1024 * 1024
DEADLINE_SECONDS = 1800
HEX256 = re.compile(r"[0-9a-f]{64}\Z")
HEX160 = re.compile(r"[0-9a-f]{40}\Z")
CORRELATION = re.compile(r"req_[A-Za-z0-9_-]{1,120}\Z")


class AcceptanceError(Exception):
    """Carries a static, non-sensitive failure code only."""


def require(condition, code):
    if not condition:
        raise AcceptanceError(code)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_bounded(path, limit):
    require(not path.is_symlink() and path.is_file(), "invalid_asset_file")
    with path.open("rb") as source:
        raw = source.read(limit + 1)
    require(len(raw) <= limit, "asset_size_bound")
    return raw


def load_bundle(root, sums_sha, manifest_sha):
    """Verify the external approval pins before examining inert capture data."""
    require(
        HEX256.fullmatch(sums_sha) and HEX256.fullmatch(manifest_sha),
        "invalid_bundle_pin",
    )
    require(root.resolve() == root and not root.is_symlink(), "invalid_bundle_root")
    sums = read_bounded(root / "SHA256SUMS", 16 * 1024)
    require(digest(sums) == sums_sha, "bundle_approval_mismatch")
    checked = {}
    total_bytes = 0
    for line in sums.decode("ascii").splitlines():
        parts = line.split("  ")
        require(len(parts) == 2 and HEX256.fullmatch(parts[0]), "invalid_checksum_line")
        expected, relative = parts
        require(
            len(checked) < 32
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts,
            "bundle_file_bound",
        )
        path = root / relative
        require(
            relative not in checked and path.resolve().is_relative_to(root),
            "invalid_bundle_path",
        )
        raw = read_bounded(path, 2 * 1024 * 1024)
        total_bytes += len(raw)
        require(total_bytes <= 4 * 1024 * 1024, "bundle_total_bound")
        require(digest(raw) == expected, "bundle_checksum_mismatch")
        checked[relative] = expected
    manifest_raw = read_bounded(root / "manifest.json", MAX_PAYLOAD)
    require(
        checked.get("manifest.json") == manifest_sha == digest(manifest_raw),
        "manifest_approval_mismatch",
    )
    manifest = json.loads(manifest_raw)
    require(
        manifest.get("schema_version") == 1 and manifest.get("model") == MODEL,
        "manifest_contract",
    )
    require(
        manifest.get("evidence_kind") == "new_actual_member_capture"
        and manifest.get("historical_reconstruction") is False,
        "manifest_scope",
    )
    entries = manifest.get("requests")
    require(isinstance(entries, list) and len(entries) == 5, "request_budget")
    payloads = []
    for index, entry in enumerate(entries):
        relative = f"requests/{index:03d}.json"
        require(
            entry.get("ordinal") == index and entry.get("file") == relative,
            "request_order",
        )
        require(relative in checked, "unchecked_payload")
        raw = read_bounded(root / relative, MAX_PAYLOAD + 1).removesuffix(b"\n")
        require(
            len(raw) <= MAX_PAYLOAD and digest(raw) == entry.get("payload_sha256"),
            "payload_integrity",
        )
        payload = json.loads(raw)
        require(
            payload.get("model") == MODEL and payload.get("stream") is False,
            "payload_shape",
        )
        require(
            type(payload.get("max_tokens")) is int
            and 1 <= payload["max_tokens"] <= 4096,
            "output_token_bound",
        )
        require("max_completion_tokens" not in payload, "ambiguous_output_budget")
        require(
            not any(name in payload for name in ("session_id", "client_request_id")),
            "body_identity_override",
        )
        require(isinstance(payload.get("messages"), list), "payload_messages")
        payloads.append(raw)
    return payloads


def verify_usage(value):
    prompt = value["prompt_tokens"]
    completion = value["completion_tokens"]
    cached = value["prompt_tokens_details"]["cached_tokens"]
    total = value["total_tokens"]
    require(
        all(
            type(item) is int and item >= 0
            for item in (prompt, completion, cached, total)
        ),
        "invalid_usage",
    )
    require(cached <= prompt and prompt + completion == total, "usage_conservation")
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cached_tokens": cached,
    }


def reconcile_row(row, call, session):
    require(
        row.get("request_id") == call["request_id"]
        and row.get("session_id") == session
        and row.get("run_id") == session,
        "ledger_identity",
    )
    require(
        row.get("model") == MODEL and row.get("provider") == "inferflux", "ledger_model"
    )
    counts = [
        row.get(field)
        for field in (
            "tokens_in",
            "tokens_out",
            "cache_creation_tokens",
            "cache_read_tokens",
        )
    ]
    require(all(type(value) is int and value >= 0 for value in counts), "ledger_counts")
    fresh, output, creation, cached = counts
    require(
        fresh + creation + cached == call["prompt_tokens"]
        and output == call["completion_tokens"]
        and cached == call["cached_tokens"],
        "ledger_conservation",
    )


def reconcile_diagnostics(events, calls, sessions):
    result = []
    for call in calls:
        correlated = [
            event
            for event in events
            if event.get("client_request_id") == call["request_id"]
        ]
        completed = [event for event in correlated if event.get("stage") == "completed"]
        require(len(completed) == 1, "backend_finalization_count")
        final = completed[0]
        require(
            final.get("backend") == "llama_cpp_cuda" and final.get("model") == MODEL,
            "backend_execution_identity",
        )
        require(
            final.get("session_sha256") == digest(sessions[call["arm"]].encode()),
            "backend_session_identity",
        )
        require(
            final.get("prompt_tokens") == call["prompt_tokens"]
            and final.get("reused_tokens") == call["cached_tokens"],
            "backend_usage_conservation",
        )
        token_hash = final.get("tokens_sha256", "")
        require(HEX256.fullmatch(token_hash), "backend_token_hash")
        path = final.get("path")
        require(path in ("phased", "full_generate"), "backend_execution_path")
        if path == "phased":
            accepted = [
                event
                for event in correlated
                if event.get("stage") == "prefill_accepted"
            ]
            require(
                len(accepted) == 1
                and accepted[0].get("reused_tokens") == call["cached_tokens"],
                "backend_accepted_reuse",
            )
        else:
            require(call["cached_tokens"] == 0, "full_generate_reuse")
        result.append(
            {**call, "tokenized_prompt_sha256": token_hash, "execution_path": path}
        )
    for ordinal in range(5):
        pair = [call for call in result if call["ordinal"] == ordinal]
        require(
            len(pair) == 2
            and pair[0]["tokenized_prompt_sha256"]
            == pair[1]["tokenized_prompt_sha256"],
            "paired_tokenization",
        )
    return result


def diagnostic_evidence(events, calls):
    """Project only bounded known identities, hashes and counts, even on failure."""
    request_ids = {call["request_id"] for call in calls}
    records = []
    matched = 0
    for event in events:
        if event.get("client_request_id") not in request_ids:
            continue
        matched += 1
        if len(records) == 64:
            continue
        row = {"client_request_id": event["client_request_id"]}
        for field, allowed in (
            (
                "stage",
                (
                    "lookup",
                    "policy_bypass",
                    "prefill_accepted",
                    "completed",
                    "capacity_eviction",
                    "copy_failed",
                    "partial_prefill_failed",
                    "prefill_deferred",
                    "admission_failed_no_sequence",
                    "admission_failed_no_blocks",
                    "session_warm",
                    "session_cold",
                    "session_incompatible",
                    "session_busy",
                ),
            ),
            (
                "backend",
                (
                    "llama_cpp_cuda",
                    "llama_cpp_rocm",
                    "llama_cpu",
                    "inferflux_cuda",
                    "unresolved",
                ),
            ),
            ("model", (MODEL,)),
            ("path", ("phased", "full_generate")),
        ):
            value = event.get(field)
            row[field] = value if value in allowed else None
        for field in ("session_sha256", "tokens_sha256"):
            value = event.get(field)
            row[field] = (
                value if isinstance(value, str) and HEX256.fullmatch(value) else None
            )
        for field in (
            "prompt_tokens",
            "matched_tokens",
            "reused_tokens",
            "previous_request_common_prefix_tokens",
            "sequence_generation",
            "evicted_sequences",
        ):
            value = event.get(field)
            row[field] = value if type(value) is int and 0 <= value < 2**64 else None
        for field in ("source_sequence", "sequence_id"):
            value = event.get(field)
            row[field] = value if type(value) is int and -1 <= value < 2**63 else None
        for field in ("session_handles_enabled", "session_lease_acquired"):
            value = event.get(field)
            row[field] = value if type(value) is bool else None
        records.append(row)
    return {"records": records, "truncated": matched > len(records)}


def finalize_diagnostics(report, events, calls, sessions):
    # Preserve completed wire calls and the safe projection before validation can
    # fail; retaining these observations does not make a failed run accepted.
    report["calls"] = calls
    report["backend_diagnostics"] = diagnostic_evidence(events, calls)
    report["calls"] = reconcile_diagnostics(events, calls, sessions)
    report["passed"] = True


class Http:
    """Fixed loopback HTTP only; no redirects, proxies, or unbounded bodies."""

    def __init__(self, deadline):
        self.deadline = deadline

    def request(self, port, path, key="", body=None, headers=None, timeout=180):
        require(port in (ORIGIN_PORT, GATEWAY_PORT, SHARED_PORT), "invalid_endpoint")
        remaining = self.deadline - time.monotonic()
        require(remaining > 0, "deadline")
        connection = http.client.HTTPConnection(
            "127.0.0.1", port, timeout=min(timeout, remaining)
        )
        outgoing = {"Content-Type": "application/json", "Accept-Encoding": "identity"}
        if key:
            outgoing["Authorization"] = "Bearer " + key
        outgoing.update(headers or {})
        try:
            connection.request(
                "POST" if body is not None else "GET", path, body=body, headers=outgoing
            )
            response = connection.getresponse()
            raw = response.read(MAX_RESPONSE + 1)
            require(
                response.status == 200 and len(raw) <= MAX_RESPONSE, "http_response"
            )
            return json.loads(raw), response.getheader("x-inferflux-client-request-id")
        finally:
            connection.close()


class Child:
    """Drain output into bounded private memory; terminate only the owned process."""

    def __init__(self, command, env, cwd):
        self.raw = bytearray()
        self.overflow = False
        self.process = subprocess.Popen(
            command, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        self.reader = threading.Thread(target=self._drain, daemon=True)
        self.reader.start()

    def _drain(self):
        while chunk := self.process.stdout.read1(4096):
            if len(self.raw) + len(chunk) <= MAX_LOG:
                self.raw.extend(chunk)
            else:
                self.overflow = True
                self.process.terminate()
                break

    def check(self):
        require(
            not self.overflow and self.process.poll() is None, "owned_process_unhealthy"
        )

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.reader.join(timeout=5)
        require(not self.reader.is_alive(), "process_output_cleanup")
        self.process.stdout.close()


def clean_environment():
    # Deliberately exclude ambient InferFlux/Sandhi config and credentials.
    return {
        name: os.environ[name]
        for name in (
            "PATH",
            "HOME",
            "LD_LIBRARY_PATH",
            "SYSTEMROOT",
            "RUNNER_TRACKING_ID",
        )
        if name in os.environ
    }


def cuda_config(model_path, key):
    return {
        "server": {
            "host": "127.0.0.1",
            "http_port": ORIGIN_PORT,
            "max_concurrent": 2,
            "enable_metrics": True,
        },
        "models": [
            {
                "id": MODEL,
                "path": str(model_path),
                "format": "gguf",
                "backend": "llama_cpp_cuda",
                "default": True,
            }
        ],
        "runtime": {
            "backend_priority": ["llama_cpp_cuda"],
            "cuda": {
                "enabled": True,
                "flash_attention": {"enabled": True},
                "phase_overlap": {"enabled": False},
            },
            "backend_exposure": {
                "prefer_inferflux": False,
                "allow_llama_cpp_fallback": False,
            },
            "capability_routing": {
                "allow_default_fallback": False,
                "require_ready_backend": True,
            },
            "llama": {"max_parallel_sequences": 2},
            "mps_layers": 8,
            "scheduler": {
                "max_batch_size": 2,
                "max_batch_tokens": 16384,
                "min_batch_size": 1,
                "batch_accumulation_ms": 0,
                "session_handles": {"enabled": False},
            },
            "paged_kv": {"cpu_pages": 4096, "eviction": "lru"},
        },
        "auth": {
            "api_keys": [{"key": key, "scopes": ["generate", "read", "admin"]}],
            "rate_limit_per_minute": 120,
        },
        "logging": {"level": "info", "format": "text"},
    }


def trusted_provenance(repo):
    env = os.environ
    require(
        env.get("GITHUB_ACTIONS") == "true"
        and env.get("GITHUB_REF") == "refs/heads/main",
        "trusted_main_only",
    )
    require(
        env.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
        and env.get("RUNNER_NAME") == "aiserver1-dual-gpu",
        "trusted_runner_only",
    )
    require(
        env.get("GITHUB_WORKFLOW_REF", "").endswith(
            "/.github/workflows/gpu-gates.yml@refs/heads/main"
        ),
        "trusted_workflow_only",
    )
    source = env.get("GITHUB_SHA", "")
    require(HEX160.fullmatch(source), "invalid_source_pin")
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True, timeout=10
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        text=True,
        timeout=10,
    )
    require(actual == source and not dirty, "dirty_or_wrong_source")
    binary = repo / "build-ci-cuda/inferfluxd"
    require(binary.is_file() and not binary.is_symlink(), "missing_built_origin")
    return {
        "inferflux_source": source,
        "inferflux_binary_sha256": file_digest(binary),
        "cmake_cache_sha256": file_digest(repo / "build-ci-cuda/CMakeCache.txt"),
    }


def pinned_asset(path_name, hash_name):
    path = Path(os.environ[path_name])
    expected = os.environ[hash_name]
    require(
        path.is_absolute()
        and path.is_file()
        and not path.is_symlink()
        and HEX256.fullmatch(expected),
        "invalid_binary_or_model_pin",
    )
    require(file_digest(path) == expected, "binary_or_model_checksum")
    return path, expected


def wait_ready(http, port, key, child, deadline):
    while time.monotonic() < deadline:
        child.check()
        try:
            return http.request(
                port, "/version" if port == GATEWAY_PORT else "/healthz", key
            )[0]
        except (OSError, AcceptanceError):
            time.sleep(0.2)
    raise AcceptanceError("startup_deadline")


def shared_health(http):
    health, _ = http.request(SHARED_PORT, "/healthz", timeout=5)
    require(health.get("model_ready") is True, "shared_service_unhealthy")


class SharedMonitor:
    """Stop only owned work if the preserved shared service becomes unhealthy."""

    def __init__(self, http, children):
        self.http, self.children = http, children
        self.done = threading.Event()
        self.failed = False
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _run(self):
        while not self.done.wait(5):
            try:
                shared_health(self.http)
            except Exception:
                self.failed = True
                for child in self.children:
                    if child.process.poll() is None:
                        child.process.terminate()
                return

    def stop(self):
        self.done.set()
        self.worker.join(timeout=10)
        require(not self.worker.is_alive(), "health_monitor_cleanup")


def model_ready(http, key):
    models, _ = http.request(ORIGIN_PORT, "/v1/models", key)
    matches = [model for model in models.get("data", []) if model.get("id") == MODEL]
    require(len(matches) == 1, "origin_model_identity")
    model = matches[0]
    require(
        model.get("ready") is True and model.get("backend") == "llama_cpp_cuda",
        "origin_backend_identity",
    )
    require(
        model.get("backend_exposure", {}).get("fallback") is False,
        "origin_backend_fallback",
    )
    runtime = model.get("runtime", {})
    require(
        runtime.get("sequence_capacity") == 2
        and runtime.get("context_tokens_per_sequence") == 16384
        and runtime.get("session_handles_enabled") is False,
        "origin_runtime_capacity",
    )


def ledger_rows(path, session):
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        return [
            dict(row)
            for row in connection.execute(
                "SELECT request_id,session_id,run_id,provider,model,tokens_in,tokens_out,cache_creation_tokens,cache_read_tokens FROM usage_events WHERE session_id=? ORDER BY rowid LIMIT 6",
                (session,),
            )
        ]


def execute(repo, private, report, children, monitors):
    payloads = load_bundle(
        BUNDLE, os.environ["CACHE_BUNDLE_SHA256"], os.environ["CACHE_MANIFEST_SHA256"]
    )
    report.update(trusted_provenance(repo))
    model, report["model_sha256"] = pinned_asset(
        "CACHE_MODEL_PATH", "CACHE_MODEL_SHA256"
    )
    sandhi, report["sandhi_binary_sha256"] = pinned_asset(
        "CACHE_SANDHI_BINARY", "CACHE_SANDHI_SHA256"
    )
    # Execute a private verified copy, so a concurrently rebuilt external Sandhi
    # path cannot change the pinned executable between verification and launch.
    sandhi_copy = private / "sandhi-proxy"
    shutil.copyfile(sandhi, sandhi_copy)
    require(
        file_digest(sandhi_copy) == report["sandhi_binary_sha256"],
        "sandhi_copy_checksum",
    )
    sandhi_copy.chmod(0o700)
    sandhi = sandhi_copy
    sandhi_source = os.environ["CACHE_SANDHI_SOURCE"]
    require(HEX160.fullmatch(sandhi_source), "invalid_sandhi_source")
    report.update(
        sandhi_source=sandhi_source,
        manifest_sha256=os.environ["CACHE_MANIFEST_SHA256"],
        bundle_sha256=os.environ["CACHE_BUNDLE_SHA256"],
    )
    # Partial layer placement avoids fitting the entire 16.45-GB MoE + KV into20GB.
    free = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
        .strip()
        .splitlines()
    )
    require(len(free) == 1 and int(free[0]) >= 6144, "cuda_headroom")
    mem = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    require(
        int(mem["MemAvailable"].split()[0]) >= 24 * 1024 * 1024, "host_memory_headroom"
    )
    report["cuda_free_mib_before"] = int(free[0])
    for port in (ORIGIN_PORT, GATEWAY_PORT):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    deadline = time.monotonic() + DEADLINE_SECONDS
    http = Http(deadline)
    shared_health(http)
    report["shared_health_before"] = True
    monitor = SharedMonitor(http, children)
    monitors.append(monitor)
    key, admin = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    config = private / "origin.json"
    config.write_text(json.dumps(cuda_config(model, key)))
    env = clean_environment()
    env.update(
        INFERFLUX_LLAMA_CTX_SIZE="32768",
        INFERFLUX_CACHE_DIAGNOSTIC_REQUEST_PREFIX="req_",
        INFERFLUX_POLICY_STORE=str(private / "policy.private"),
        INFERFLUX_POLICY_PASSPHRASE=secrets.token_urlsafe(32),
    )
    origin = Child(
        [str(repo / "build-ci-cuda/inferfluxd"), "--config", str(config)], env, private
    )
    children.append(origin)
    wait_ready(http, ORIGIN_PORT, key, origin, time.monotonic() + 180)
    model_ready(http, key)
    require(
        re.search(rb"offloaded 8/\d+ layers to GPU", bytes(origin.raw)),
        "cuda_layer_placement",
    )
    env = clean_environment()
    database = private / "usage.private.db"
    env.update(
        SANDHI_BIND=f"127.0.0.1:{GATEWAY_PORT}",
        SANDHI_PUBLIC_URL=f"http://127.0.0.1:{GATEWAY_PORT}",
        SANDHI_STORE=str(database),
        SANDHI_ADMIN_TOKEN=admin,
        SANDHI_INFERFLUX_KEY=key,
        SANDHI_INFERFLUX_BASE=f"http://127.0.0.1:{ORIGIN_PORT}/v1",
    )
    gateway = Child([str(sandhi)], env, private)
    children.append(gateway)
    wait_ready(http, GATEWAY_PORT, admin, gateway, time.monotonic() + 30)
    minted, _ = http.request(
        GATEWAY_PORT,
        "/admin/keys/share",
        admin,
        json.dumps(
            {"upstream": "inferflux", "models": [MODEL], "rate_limit_per_min": 10}
        ).encode(),
    )
    replay_key = minted.get("virtual_key")
    require(
        isinstance(replay_key, str)
        and replay_key.startswith("vk_")
        and len(replay_key) <= 256,
        "private_gateway_key",
    )
    run = "req_cachegpu_" + secrets.token_hex(8)
    sessions = {arm: run + "_" + arm for arm in ("direct", "gateway")}
    calls = []
    report["calls"] = calls
    for ordinal, payload in enumerate(payloads):
        for arm in ("direct", "gateway"):
            shared_health(http)
            origin.check()
            gateway.check()
            headers = (
                {
                    "x-inferflux-session-id": sessions[arm],
                    "x-inferflux-client-request-id": f"{run}_{ordinal}",
                }
                if arm == "direct"
                else {
                    "x-sandhi-session": sessions[arm],
                    "x-sandhi-run-id": sessions[arm],
                }
            )
            port, credential = (
                (ORIGIN_PORT, key) if arm == "direct" else (GATEWAY_PORT, replay_key)
            )
            response, correlation = http.request(
                port, "/v1/chat/completions", credential, payload, headers
            )
            require(
                isinstance(correlation, str) and CORRELATION.fullmatch(correlation),
                "request_correlation",
            )
            if arm == "direct":
                require(
                    correlation == headers["x-inferflux-client-request-id"],
                    "direct_correlation",
                )
            usage = verify_usage(response["usage"])
            calls.append(
                {
                    "ordinal": ordinal,
                    "arm": arm,
                    "request_id": correlation,
                    "payload_sha256": digest(payload),
                    **usage,
                }
            )
    require(
        len(calls) == 10 and len({call["request_id"] for call in calls}) == 10,
        "request_budget_or_identity",
    )
    gateway_calls = [call for call in calls if call["arm"] == "gateway"]
    session = sessions["gateway"]
    for _ in range(20):
        rows = ledger_rows(database, session)
        if len(rows) == 5:
            break
        time.sleep(0.1)
    require(len(rows) == 5, "ledger_row_count")
    by_id = {row["request_id"]: row for row in rows}
    require(len(by_id) == 5, "ledger_ambiguous_identity")
    query = json.dumps(
        {"selector": {"kind": "run", "value": session}, "limit": 10}
    ).encode()
    c4, _ = http.request(GATEWAY_PORT, "/admin/usage/diagnostics", admin, query)
    require(
        c4.get("truncated") is False
        and c4.get("returned_rows") == 5
        and len(c4.get("rows", [])) == 5,
        "diagnostic_completeness",
    )
    projected = {row["request_id"]: row for row in c4["rows"]}
    require(len(projected) == 5, "diagnostic_ambiguous_identity")
    for call in gateway_calls:
        reconcile_row(by_id.get(call["request_id"], {}), call, session)
        row = projected.get(call["request_id"], {})
        reconcile_row(row, call, session)
        require(
            not row.get("warnings")
            and row.get("cache_read_observation")
            == {"status": "reported", "source": "origin_usage"},
            "cache_reporting_availability",
        )
    dashboard, _ = http.request(GATEWAY_PORT, "/dashboard/api/usage", admin)
    total = dashboard["total"]
    require(total["calls"] == 5, "dashboard_row_count")
    for field in (
        "tokens_in",
        "tokens_out",
        "cache_creation_tokens",
        "cache_read_tokens",
    ):
        require(
            total[field] == sum(row[field] for row in rows), "dashboard_conservation"
        )
    require(
        total["cache_read_coverage"]["reported"] == 5
        and total["cache_read_coverage"]["unknown"] == 0,
        "dashboard_reporting_availability",
    )
    report["gateway_rows"] = 5
    report["gateway_accounting_reconciled"] = True
    model_ready(http, key)
    shared_health(http)
    report["shared_health_after"] = True
    require(not monitor.failed, "shared_health_during_run")
    # Stop first so the bounded reader contains every completed diagnostic line.
    for child in reversed(children):
        child.stop()
    require(not origin.overflow and not gateway.overflow, "process_log_bound")
    events = []
    for line in bytes(origin.raw).splitlines():
        if b"cache_decision" in line and b"{" in line:
            events.append(json.loads(line[line.index(b"{") :]))
    finalize_diagnostics(report, events, calls, sessions)


def main():
    os.umask(0o077)
    repo = Path(__file__).resolve().parent.parent
    output = repo / "build-ci-cuda/cache-acceptance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    children = []
    monitors = []
    report = {
        "schema_version": 1,
        "passed": False,
        "scope": "Frozen five-request member DATA; paired direct/Sandhi CUDA partial-eight-layer execution. No tool execution, cold-cache, original forty-call, completion-tokenizer, or ROCm-Qwen acceptance claim.",
        "request_budget": 10,
        "context_tokens": 32768,
        "sequences": 2,
        "gpu_layers": 8,
    }

    def interrupted(_signum, _frame):
        raise AcceptanceError("deadline_or_interrupt")

    previous = {
        sig: signal.signal(sig, interrupted)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)
    }
    signal.alarm(DEADLINE_SECONDS)
    try:
        with tempfile.TemporaryDirectory(prefix="inferflux-gpu-cache-") as private:
            try:
                execute(repo, Path(private), report, children, monitors)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                cleanup_failed = False
                for owner in [*monitors, *reversed(children)]:
                    try:
                        owner.stop()
                    except Exception:
                        cleanup_failed = True
                require(not cleanup_failed, "owned_resource_cleanup")
    except Exception as error:
        report["passed"] = False
        report["failure"] = (
            str(error) if isinstance(error, AcceptanceError) else type(error).__name__
        )
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        report["owned_processes_stopped"] = all(
            child.process.poll() is not None for child in children
        )
        if children:
            try:
                shared_health(Http(time.monotonic() + 5))
                report["shared_health_after_cleanup"] = True
            except Exception:
                report["shared_health_after_cleanup"] = False
                report["passed"] = False
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "evidence": str(output),
                "failure": report.get("failure"),
            }
        )
    )
    return 0 if report["passed"] and report["owned_processes_stopped"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
