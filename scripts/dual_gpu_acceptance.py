#!/usr/bin/env python3
"""Opt-in trusted-main, same-process placement and gateway setup gate.

Synthetic setup evidence only. Actual Victor cohorts, GPU kernel timelines and
C5 acceptance remain separate requirements. Never contacts the retained gateway.
"""

import concurrent.futures
import ctypes
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

from frozen_gpu_cache_acceptance import (
    AcceptanceError,
    Child,
    clean_environment,
    file_digest,
    pinned_asset,
    require,
)

ORIGIN = 28085
GATEWAY = 18794
SHARED = (8080, 8081, 8090)
MODELS = {"dual-amd": "rocm:0", "dual-nvidia": "cuda:0"}


def model_config(paths, key):
    return {
        "server": {"host": "127.0.0.1", "http_port": ORIGIN},
        "models": [
            {
                "id": model,
                "path": str(paths[model]),
                "backend": "llama_cpp_" + device.split(":")[0],
                "device": device,
                "context_size": 2048,
                "gpu_layers": 8,
                "max_parallel_sequences": 2,
                "kv_cache_type": "f16",
            }
            for model, device in MODELS.items()
        ],
        "runtime": {
            "backend_exposure": {
                "prefer_inferflux": False,
                "allow_llama_cpp_fallback": False,
            },
            "capability_routing": {
                "allow_default_fallback": False,
                "require_ready_backend": True,
            },
            "scheduler": {
                "max_batch_size": 2,
                "min_batch_size": 2,
                "batch_accumulation_ms": 50,
                "decode_burst_tokens": 8,
                "session_handles": {"enabled": False},
            },
            "disaggregated": {"decode_pool_size": 0},
            "paged_kv": {"cpu_pages": 256},
        },
        "auth": {
            "api_keys": [{"key": key, "scopes": ["generate", "read", "admin"]}],
            "rate_limit_per_minute": 120,
        },
    }


def verify_models(payload):
    models = {model["id"]: model for model in payload["data"]}
    require(set(models) == set(MODELS), "model_ids")
    result = {}
    for model, device in MODELS.items():
        info = models[model]
        p = info.get("placement", {})
        require(
            info.get("ready") is True and not info["backend_exposure"]["fallback"],
            "model_ready",
        )
        require(
            p.get("requested") == device
            and p.get("effective") == device
            and p.get("state") == "verified_weights"
            and p.get("vendor") == device.split(":")[0]
            and p.get("gpu_weight_bytes", 0) > 0
            and p.get("gpu_layer_count", 0) > 0,
            "physical_weight_placement",
        )
        require(p.get("stable_id"), "stable_device_identity_unavailable")
        require(
            p.get("max_parallel_sequences") == 2 and p.get("kv_cache_type") == "f16",
            "model_resources",
        )
        result[model] = p
    return result


def usage_counts(usage):
    prompt = usage.get("prompt_tokens")
    output = usage.get("completion_tokens")
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens")
    require(
        all(type(v) is int and v >= 0 for v in (prompt, output, cached)),
        "usage_reporting",
    )
    require(
        cached <= prompt and usage.get("total_tokens") == prompt + output,
        "usage_conservation",
    )
    return {
        "prompt_tokens": prompt,
        "completion_tokens": output,
        "cached_tokens": cached,
    }


def reconcile_rows(rows, calls):
    require(len(rows) == len(calls), "ledger_count")
    indexed = {row["request_id"]: row for row in rows}
    require(len(indexed) == len(calls), "ledger_duplicate_identity")
    for call in calls:
        row = indexed.get(call["request_id"], {})
        require(
            row.get("model") == call["model"]
            and row.get("provider") == "inferflux"
            and row.get("session_id") == call["session_id"],
            "ledger_model_session",
        )
        require(
            row.get("tokens_in", -1)
            + row.get("cache_read_tokens", -1)
            + row.get("cache_creation_tokens", -1)
            == call["prompt_tokens"]
            and row.get("tokens_out") == call["completion_tokens"]
            and row.get("cache_read_tokens") == call["cached_tokens"],
            "ledger_conservation",
        )


def host_overlap(events):
    # Explicitly a host-call envelope, never a GPU kernel-overlap assertion.
    events = [e for e in events if e.get("scope") == "host_backend_call"]
    return any(
        a.get("device") == "rocm:0"
        and b.get("device") == "cuda:0"
        and max(a["start_ns"], b["start_ns"]) < min(a["end_ns"], b["end_ns"])
        for a in events
        for b in events
    )


def trusted_source(repo):
    require(
        os.environ.get("GITHUB_ACTIONS") == "true"
        and os.environ.get("GITHUB_REF") == "refs/heads/main"
        and os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
        and os.environ.get("RUNNER_NAME") == "aiserver1-dual-gpu"
        and os.environ.get("GITHUB_WORKFLOW_REF", "").endswith(
            "/.github/workflows/gpu-gates.yml@refs/heads/main"
        ),
        "trusted_main_gate_only",
    )
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo
    )
    require(sha == os.environ.get("GITHUB_SHA") and not dirty, "source_provenance")
    return sha


def request(port, path, key="", payload=None, headers=None, stream=False):
    require(port in (ORIGIN, GATEWAY, *SHARED), "endpoint")
    conn = http.client.HTTPConnection(
        "127.0.0.1", port, timeout=5 if port in SHARED else 120
    )
    outgoing = {"Content-Type": "application/json", "Authorization": "Bearer " + key}
    outgoing.update(headers or {})
    try:
        conn.request(
            "POST" if payload is not None else "GET",
            path,
            body=json.dumps(payload) if payload is not None else None,
            headers=outgoing,
        )
        response = conn.getresponse()
        require(response.status == 200, "http_status")
        raw = response.read(1024 * 1024 + 1)
        require(len(raw) <= 1024 * 1024, "response_bound")
        correlation = response.getheader("x-inferflux-client-request-id")
        if stream:
            frames = [
                json.loads(line[6:])
                for line in raw.decode().splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            usages = [frame["usage"] for frame in frames if frame.get("usage")]
            require(
                len(usages) == 1 and b"data: [DONE]" in raw, "stream_terminal_usage"
            )
            return {"usage": usages[0]}, correlation
        return json.loads(raw), correlation
    finally:
        conn.close()


def shared_health():
    for port in SHARED:
        payload, _ = request(port, "/healthz")
        require(payload.get("model_ready") is True, "preserved_service_health")


def wait_ready(child, port, path, key):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        child.check()
        try:
            payload, _ = request(port, path, key)
            return payload
        except (OSError, AcceptanceError):
            time.sleep(0.2)
    raise AcceptanceError("startup_timeout")


def verify_mixed_flags(cache):
    require(
        all(
            flag in cache
            for flag in (
                "ENABLE_CUDA:BOOL=ON",
                "ENABLE_ROCM:BOOL=ON",
                "GGML_HIP:BOOL=ON",
                "GGML_BACKEND_DL:BOOL=ON",
            )
        ),
        "mixed_build_flags",
    )
    # The pinned llama revision maps the legacy option without creating a
    # GGML_CUDA cache entry. Accept either spelling, not a missing CUDA flag.
    require(
        "GGML_CUDA:BOOL=ON" in cache or "LLAMA_CUDA:BOOL=ON" in cache,
        "mixed_cuda_build_flag",
    )


def execute(repo, private, report, children):
    report["source"] = trusted_source(repo)
    report["llama_source"] = subprocess.check_output(
        ["git", "-C", "external/llama.cpp", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    build = repo / "build-ci-dual"
    cache = (build / "CMakeCache.txt").read_text()
    verify_mixed_flags(cache)
    report["build_sha256"] = {
        name: file_digest(build / name)
        for name in (
            "inferfluxd",
            "CMakeCache.txt",
            "libggml-cuda.so",
            "libggml-hip.so",
        )
    }
    paths = {}
    report["model_sha256"] = {}
    for model, prefix in (
        ("dual-amd", "DUAL_ROCM_MODEL"),
        ("dual-nvidia", "DUAL_CUDA_MODEL"),
    ):
        paths[model], report["model_sha256"][model] = pinned_asset(
            prefix + "_PATH", prefix + "_SHA256"
        )
        require(paths[model].stat().st_size <= 2 * 1024**3, "small_gate_asset_required")
    require(not os.path.samefile(*paths.values()), "distinct_artifacts_required")
    sandhi, report["sandhi_sha256"] = pinned_asset(
        "DUAL_SANDHI_BINARY", "DUAL_SANDHI_SHA256"
    )
    report["sandhi_source"] = os.environ["DUAL_SANDHI_SOURCE"]
    require(re.fullmatch("[a-f0-9]{40}", report["sandhi_source"]), "sandhi_source_pin")
    copy = private / "sandhi"
    shutil.copyfile(sandhi, copy)
    require(file_digest(copy) == report["sandhi_sha256"], "sandhi_copy_pin")
    copy.chmod(0o700)
    free = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    require(len(free.splitlines()) == 1 and int(free.strip()) >= 2048, "cuda_headroom")
    hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    hip_free, hip_total = ctypes.c_size_t(), ctypes.c_size_t()
    require(
        hip.hipSetDevice(0) == 0
        and hip.hipMemGetInfo(ctypes.byref(hip_free), ctypes.byref(hip_total)) == 0
        and hip_free.value >= 2 * 1024**3,
        "rocm_headroom",
    )
    for port in (ORIGIN, GATEWAY):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    shared_health()
    report["shared_health_before"] = True
    key, admin = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    config = private / "origin.json"
    config.write_text(json.dumps(model_config(paths, key)))
    env = clean_environment()
    env.update(
        INFERFLUX_POLICY_STORE=str(private / "policy"),
        INFERFLUX_POLICY_PASSPHRASE=secrets.token_urlsafe(32),
        INFERFLUX_PLACEMENT_DIAGNOSTICS="1",
        INFERFLUX_CACHE_DIAGNOSTIC_REQUEST_PREFIX="req_",
    )
    origin = Child([str(build / "inferfluxd"), "--config", str(config)], env, private)
    children.append(origin)
    payload = wait_ready(origin, ORIGIN, "/v1/models", key)
    report["placement"] = verify_models(payload)
    env = clean_environment()
    database = private / "usage.db"
    env.update(
        SANDHI_BIND=f"127.0.0.1:{GATEWAY}",
        SANDHI_PUBLIC_URL=f"http://127.0.0.1:{GATEWAY}",
        SANDHI_STORE=str(database),
        SANDHI_ADMIN_TOKEN=admin,
        SANDHI_INFERFLUX_KEY=key,
        SANDHI_INFERFLUX_BASE=f"http://127.0.0.1:{ORIGIN}/v1",
    )
    gateway = Child([str(copy)], env, private)
    children.append(gateway)
    wait_ready(gateway, GATEWAY, "/version", admin)
    minted, _ = request(
        GATEWAY,
        "/admin/keys/share",
        admin,
        {"upstream": "inferflux", "models": list(MODELS), "rate_limit_per_min": 30},
    )
    virtual_key = minted["virtual_key"]
    run = "req_dualgpu_" + secrets.token_hex(8)
    calls = report["calls"] = []
    # Two rounds per route: plain and SSE. Same session string across models
    # deliberately exercises cross-model cache isolation; IDs remain per call.
    for arm, port, credential in (
        ("direct", ORIGIN, key),
        ("gateway", GATEWAY, virtual_key),
    ):
        for streaming in (False, True):
            barrier = threading.Barrier(2)
            session = run + "_" + arm + "_" + str(streaming)

            def call(model):
                headers = (
                    {
                        "x-inferflux-session-id": session,
                        "x-inferflux-client-request-id": session + "_" + model,
                    }
                    if arm == "direct"
                    else {"x-sandhi-session": session, "x-sandhi-run-id": session}
                )
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Count from 1 to 40, one number per line.",
                        }
                    ],
                    "temperature": 0,
                    "max_tokens": 64,
                    "stream": streaming,
                }
                if streaming:
                    payload["stream_options"] = {"include_usage": True}
                barrier.wait(timeout=5)
                response, correlation = request(
                    port,
                    "/v1/chat/completions",
                    credential,
                    payload,
                    headers,
                    streaming,
                )
                require(
                    isinstance(correlation, str)
                    and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", correlation),
                    "request_correlation",
                )
                return {
                    "arm": arm,
                    "model": model,
                    "session_id": session,
                    "request_id": correlation,
                    "stream": streaming,
                    **usage_counts(response["usage"]),
                }

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                calls.extend(pool.map(call, MODELS))
            shared_health()
            origin.check()
            gateway.check()
    gateway_calls = [call for call in calls if call["arm"] == "gateway"]
    for _ in range(30):
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            rows = [
                dict(r)
                for r in conn.execute(
                    "SELECT request_id,session_id,run_id,provider,model,tokens_in,tokens_out,cache_creation_tokens,cache_read_tokens FROM usage_events"
                )
            ]
        if len(rows) == len(gateway_calls):
            break
        time.sleep(0.1)
    reconcile_rows(rows, gateway_calls)
    for session in {call["session_id"] for call in gateway_calls}:
        c4, _ = request(
            GATEWAY,
            "/admin/usage/diagnostics",
            admin,
            {"selector": {"kind": "session", "value": session}, "limit": 10},
        )
        require(not c4.get("truncated"), "bounded_diagnostics")
        reconcile_rows(
            c4["rows"],
            [call for call in gateway_calls if call["session_id"] == session],
        )
    dashboard, _ = request(GATEWAY, "/dashboard/api/usage", admin)
    require(dashboard["total"]["calls"] == len(rows), "dashboard_calls")
    for field in (
        "tokens_in",
        "tokens_out",
        "cache_creation_tokens",
        "cache_read_tokens",
    ):
        require(
            dashboard["total"][field] == sum(row[field] for row in rows),
            "dashboard_conservation",
        )
    report["gateway_accounting_reconciled"] = True
    origin.stop()
    events = []
    for line in origin.raw.decode(errors="replace").splitlines():
        marker = "placement_execution: "
        if marker in line:
            events.append(json.loads(line.split(marker, 1)[1]))
    report["host_backend_overlap"] = host_overlap(events)
    report["host_execution_events"] = events[:64]
    require(report["host_backend_overlap"], "host_backend_overlap_missing")
    report["supplemental_gate_passed"] = True


def main():
    repo = Path(__file__).resolve().parents[1]
    output = repo / "build-ci-dual/dual-gpu-acceptance.json"
    report = {
        "supplemental_gate_passed": False,
        "c5_accepted": False,
        "gpu_kernel_overlap_verified": False,
        "actual_victor_cohort": False,
        "cancellation_acceptance": "pending",
        "executed_cache_reuse": "pending_backend_reconciliation",
    }
    children = []
    stop_monitor = threading.Event()
    health_failed = threading.Event()

    def monitor():
        while not stop_monitor.wait(5):
            try:
                shared_health()
            except Exception:
                health_failed.set()
                for child in list(children):
                    if child.process.poll() is None:
                        child.process.terminate()
                return

    watcher = threading.Thread(target=monitor, daemon=True)

    def deadline(_signum, _frame):
        raise AcceptanceError("overall_deadline")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(900)
    try:
        with tempfile.TemporaryDirectory(prefix="inferflux-dual-gate-") as folder:
            try:
                watcher.start()
                execute(repo, Path(folder), report, children)
                require(not health_failed.is_set(), "shared_health_during_run")
            finally:
                stop_monitor.set()
                if watcher.ident is not None:
                    watcher.join(timeout=20)
                    require(not watcher.is_alive(), "health_monitor_cleanup")
                cleanup_failed = False
                for child in reversed(children):
                    try:
                        child.stop()
                    except Exception:
                        cleanup_failed = True
                require(not cleanup_failed, "owned_cleanup")
    except Exception as error:
        report["supplemental_gate_passed"] = False
        report["failure"] = (
            str(error) if isinstance(error, AcceptanceError) else type(error).__name__
        )
    finally:
        signal.alarm(0)
        report["owned_processes_stopped"] = all(
            child.process.poll() is not None for child in children
        )
        if children:
            try:
                shared_health()
                report["shared_health_after"] = True
            except Exception:
                report["shared_health_after"] = False
                report["supplemental_gate_passed"] = False
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
    return (
        0
        if report["supplemental_gate_passed"] and report["owned_processes_stopped"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
