#!/usr/bin/env python3
"""Replay the WS-E Qwen writer through a temporary local Sandhi gateway.

Run with a Victor-capable Python environment. The external repositories must
be checked out and the Sandhi proxy built beforehand. No ZAI credential is used.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import secrets
import socket
import sqlite3
import subprocess
import sys
import uuid

from aiohttp import ClientSession, web

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--victor-repo", type=Path, required=True)
parser.add_argument("--sandhi-binary", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, default=Path("/tmp"))
parser.add_argument("--base-url", default="http://127.0.0.1:8080")
parser.add_argument("--model", default="qwen3-coder-30b")
parser.add_argument("--api-key-env", default="INFERCTL_API_KEY")
parser.add_argument("--gateway-port", type=int, default=18789)
parser.add_argument("--observer-port", type=int, default=18084)
args = parser.parse_args()
SOURCE = args.victor_repo.resolve()
args.sandhi_binary = args.sandhi_binary.resolve()
API_KEY = os.environ[args.api_key_env]
GATEWAY_URL = f"http://127.0.0.1:{args.gateway_port}"
OBSERVER_URL = f"http://127.0.0.1:{args.observer_port}"
sys.path.insert(0, str(SOURCE))
ROOT = args.output_dir.resolve() / ("inferflux-member-" + uuid.uuid4().hex[:10])
ROOT.mkdir(mode=0o700)
os.chdir(ROOT)
os.environ.update(
    SANDHI_GATEWAY_URL=GATEWAY_URL,
    SANDHI_GATEWAY_VIRTUAL_KEY_INFERFLUX="vk_inferflux_demo",
    INFERFLUX_API_KEY=API_KEY,
)
import victor.config.secure_paths as secure_paths

secure_paths.get_victor_dir = lambda: ROOT / "victor-state"
import victor.config.settings as settings

settings.GLOBAL_VICTOR_DIR = ROOT / "victor-state"
from victor.framework import Agent
from victor.framework.teams import AgentTeam, TeamMemberSpec, TeamFormation

logging.disable(logging.CRITICAL)


async def main():
    rows = []
    last_messages = b""
    # Refuse to attach the test harness to an existing gateway on this port.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.gateway_port))
    async with ClientSession() as client:

        async def forward(request):
            nonlocal last_messages
            body = await request.read()
            payload = (
                json.loads(body)
                if body and request.path.endswith("/chat/completions")
                else None
            )
            headers = {
                k: v
                for k, v in request.headers.items()
                if k.lower() not in ("host", "content-length", "transfer-encoding")
            }
            headers["Accept-Encoding"] = "identity"
            if payload is not None:
                if len(rows) >= 32:
                    raise web.HTTPTooManyRequests(
                        text="Bounded capture request limit reached"
                    )
                correlation = (
                    request.headers.get("x-inferflux-client-request-id")
                    or f"{ROOT.name}-{len(rows)}"
                )
                headers["x-inferflux-client-request-id"] = correlation
                messages = json.dumps(payload.get("messages"), sort_keys=True).encode()
                common = 0
                for a, b in zip(messages, last_messages):
                    if a != b:
                        break
                    common += 1
                last_messages = messages
                row = dict(
                    client_request_id=correlation,
                    gateway_client_request_id=request.headers.get(
                        "x-inferflux-client-request-id"
                    ),
                    session_sha256=hashlib.sha256(
                        request.headers.get("x-inferflux-session-id", "").encode()
                    ).hexdigest(),
                    model=payload.get("model"),
                    stream=payload.get("stream", False),
                    tools=len(payload.get("tools", [])),
                    response_format_type=(payload.get("response_format") or {}).get(
                        "type"
                    ),
                    logprobs=payload.get("logprobs", False),
                    messages_sha256=hashlib.sha256(messages).hexdigest(),
                    common_serialized_message_prefix_bytes=common,
                )
                rows.append(row)
            async with client.request(
                request.method,
                args.base_url.rstrip("/") + request.path_qs,
                data=body,
                headers=headers,
            ) as response:
                output = await response.read()
                if payload is not None:
                    row["status"] = response.status
                    if not payload.get("stream"):
                        row["usage"] = json.loads(output).get("usage")
                    else:
                        for line in output.splitlines():
                            if line.startswith(b"data: ") and line != b"data: [DONE]":
                                chunk = json.loads(line[6:])
                                if chunk.get("usage"):
                                    row["usage"] = chunk["usage"]
                    (ROOT / "wire.json").write_text(json.dumps(rows, indent=2))
                    print(json.dumps(row), flush=True)
                return web.Response(
                    status=response.status,
                    body=output,
                    headers={
                        "Content-Type": response.headers.get(
                            "Content-Type", "application/json"
                        )
                    },
                )

        app = web.Application(client_max_size=4 * 1024 * 1024)
        app.router.add_route("*", "/{tail:.*}", forward)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", args.observer_port).start()
        env = {k: v for k, v in os.environ.items() if not k.startswith("SANDHI_")}
        env.update(
            SANDHI_BIND=f"127.0.0.1:{args.gateway_port}",
            SANDHI_PUBLIC_URL=GATEWAY_URL,
            SANDHI_STORE=str(ROOT / "usage.db"),
            SANDHI_ADMIN_TOKEN=secrets.token_urlsafe(32),
            SANDHI_INFERFLUX_KEY=API_KEY,
            SANDHI_INFERFLUX_BASE=OBSERVER_URL + "/v1",
        )
        with (ROOT / "gateway.log").open("w") as log:
            gateway = subprocess.Popen(
                [str(args.sandhi_binary.resolve())], env=env, stdout=log, stderr=log
            )
        agent = None
        try:
            for _ in range(100):
                if gateway.poll() is not None:
                    raise RuntimeError("Temporary gateway exited during startup")
                try:
                    async with client.get(GATEWAY_URL + "/version") as r:
                        if r.status == 200:
                            break
                except OSError:
                    pass
                await asyncio.sleep(0.1)
            else:
                raise RuntimeError("Temporary gateway did not become ready")
            print("REPLAY_ROOT", ROOT, flush=True)
            agent = await Agent.create(
                provider="inferflux",
                model=args.model,
                workspace=str(ROOT),
                enable_observability=False,
                session_id=ROOT.name,
            )
            name = "writer"
            python = Path(sys.executable)
            member = TeamMemberSpec(
                role="executor",
                name=name,
                provider="inferflux",
                model=args.model,
                reasoning_effort="high",
                goal=f"Assigned member: {name}. Complete these steps in order: "
                f"1. Write {ROOT}/{name}.py with function {name}(x) returning x * 2. "
                f"2. Write {ROOT}/test_{name}.py importing that function and defining "
                f"def test_{name}(): assert {name}(4) == 8. "
                f"3. Run {python} -m pytest {ROOT}/test_{name}.py -q. "
                "Both files must exist and one test must pass. Use write twice then shell. "
                "Run shell with readonly=False. Never repeat a successful write; move to the next step. "
                "Return file references and test result.",
                allowed_tools=["read", "write", "shell"],
                tool_budget=12,
                max_iterations=12,
            )
            team = await AgentTeam.create(
                agent.get_orchestrator(),
                "Gateway review",
                "Deliver assigned files",
                [member],
                formation=TeamFormation.PIPELINE,
                shared_context={
                    "thread_id": ROOT.name,
                    "parent_session_id": ROOT.name,
                    "capture_member_usage": True,
                },
                timeout_seconds=600,
            )
            result = await asyncio.wait_for(team.run(), 600)
            print("TEAM_SUCCESS", result.success, flush=True)
            assert result.success
            test = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "pytest", str(ROOT / "test_writer.py"), "-q"
            )
            assert await test.wait() == 0
            await asyncio.sleep(0.2)
            with sqlite3.connect(ROOT / "usage.db") as db:
                usage = db.execute(
                    "SELECT request_id, tokens_in, tokens_out, cache_read_tokens "
                    "FROM usage_events ORDER BY occurred_at"
                ).fetchall()
            by_id = {r["gateway_client_request_id"]: r for r in rows}
            assert len(usage) == len(rows)
            for request_id, fresh, output, cached in usage:
                wire = by_id[request_id]["usage"]
                assert fresh + cached == wire["prompt_tokens"]
                assert cached == wire["prompt_tokens_details"]["cached_tokens"]
                assert output == wire["completion_tokens"]
            (ROOT / "reconciliation.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "requests": len(usage),
                        "scope": "WS-E local writer member",
                        "victor_sha": subprocess.check_output(
                            ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
                        ).strip(),
                    },
                    indent=2,
                )
            )
            print("LEDGER_RECONCILED", len(usage), flush=True)
        finally:
            try:
                if agent:
                    await agent.close()
            finally:
                gateway.terminate()
                gateway.wait(timeout=15)
                await runner.cleanup()
                (ROOT / "wire.json").write_text(json.dumps(rows, indent=2))


asyncio.run(main())
