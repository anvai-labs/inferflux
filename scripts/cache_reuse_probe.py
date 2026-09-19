#!/usr/bin/env python3
"""Bounded cache wire probe; never clears caches or records prompt contents."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import uuid


def complete(base_url, key, payload, correlation, session):
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "x-inferflux-client-request-id": correlation,
            "x-inferflux-session-id": session,
        },
    )
    usage, content = None, ""
    with urllib.request.urlopen(request, timeout=180) as response:
        if payload["stream"]:
            for line in response:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices", []):
                    content += choice.get("delta", {}).get("content") or ""
        else:
            body = json.load(response)
            usage = body.get("usage")
            content = body["choices"][0]["message"].get("content") or ""
    if not usage:
        raise RuntimeError("Missing terminal usage for " + correlation)
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens")
    if not isinstance(cached, int) or not 0 <= cached <= usage["prompt_tokens"]:
        raise RuntimeError("Invalid cached token accounting for " + correlation)
    return usage, content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-env", default="INFERCTL_API_KEY")
    args = parser.parse_args()
    key = os.environ[args.api_key_env]
    run = "cache-probe-" + uuid.uuid4().hex[:12]
    rows = []
    # Each shape/stream pair has a unique prefix; the repeat is deliberately
    # warm. Concurrent production traffic may evict it: a miss is evidence,
    # not a test failure or permission to flush a shared server's cache.
    for stream in (False, True):
        for shape in ("plain", "tools", "json", "logprobs"):
            prefix = f"{run}-{shape}-{stream}. " + "alpha beta gamma delta. " * 80
            messages = [
                {"role": "system", "content": prefix},
                {"role": "user", "content": 'Return {"ok":true}.'},
            ]
            payload = dict(
                model=args.model,
                messages=messages,
                temperature=0,
                max_tokens=32,
                stream=stream,
            )
            if stream:
                payload["stream_options"] = {"include_usage": True}
            if shape == "tools":
                payload["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": "read",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                            },
                        },
                    }
                ]
            elif shape == "json":
                payload["response_format"] = {"type": "json_object"}
            elif shape == "logprobs":
                payload["logprobs"] = True
            for stage in ("unique_prefix", "repeat", "appended"):
                correlation = f"{run}-{shape}-{stream}-{stage}"
                usage, content = complete(args.base_url, key, payload, correlation, run)
                rows.append(
                    dict(
                        client_request_id=correlation,
                        shape=shape,
                        stream=stream,
                        stage=stage,
                        usage=usage,
                        system_sha256=hashlib.sha256(prefix.encode()).hexdigest(),
                    )
                )
                print(correlation, json.dumps(usage), flush=True)
                if stage == "repeat":
                    messages.extend(
                        [
                            {"role": "assistant", "content": content},
                            {"role": "user", "content": "Again."},
                        ]
                    )
                # Persist partial evidence if a later request fails. No key,
                # prompt, response text, or inferred internal state is stored.
                args.output.write_text(
                    json.dumps({"run": run, "rows": rows}, indent=2) + "\n"
                )


if __name__ == "__main__":
    main()
