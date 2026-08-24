#!/usr/bin/env python3

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
import urllib.error
from pathlib import Path
from typing import Dict, List


def load_prompts(prompt_suite_path: str, single_prompt: str, num_requests: int) -> List[str]:
    if single_prompt:
        return [single_prompt for _ in range(num_requests)]
    suite = json.loads(Path(prompt_suite_path).read_text(encoding="utf-8"))
    prompts = [entry.get("prompt", "") for entry in suite.get("prompts", []) if entry.get("prompt")]
    if not prompts:
        raise RuntimeError("prompt suite did not contain any prompts")
    return [prompts[i % len(prompts)] for i in range(num_requests)]


def request_openai(endpoint: str, model: str, prompt: str, max_tokens: int,
                   request_id: str, api_key: str, inferflux_style: bool) -> Dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    cmd = [
        "curl",
        "-sf",
        "-X",
        "POST",
        endpoint,
        "-H",
        "Content-Type: application/json",
    ]
    if api_key:
        cmd += ["-H", f"Authorization: Bearer {api_key}"]
    if inferflux_style:
        cmd += ["-H", f"x-inferflux-request-id: {request_id}"]
    cmd += ["-d", json.dumps(payload)]
    started = time.time_ns()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "curl request failed")
    body = json.loads(proc.stdout)
    latency_ms = (time.time_ns() - started) // 1_000_000
    text = body.get("choices", [{}])[0].get("text", "").strip()
    tokens = body.get("usage", {}).get("completion_tokens", 0)
    if not tokens and text:
        tokens = len(text.split())
    return {
        "request_id": request_id,
        "text": text,
        "tokens": tokens,
        "latency_ms": latency_ms,
    }


def request_ollama(endpoint: str, model: str, prompt: str, max_tokens: int,
                   request_id: str) -> Dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {"num_predict": max_tokens, "temperature": 0.0},
        "stream": False,
    }
    cmd = [
        "curl",
        "-sf",
        "-X",
        "POST",
        endpoint,
        "-H",
        "Content-Type: application/json",
        "-d",
        json.dumps(payload),
    ]
    started = time.time_ns()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "curl request failed")
    body = json.loads(proc.stdout)
    latency_ms = (time.time_ns() - started) // 1_000_000
    text = body.get("response", "").strip()
    tokens = len(text.split())
    return {
        "request_id": request_id,
        "text": text,
        "tokens": tokens,
        "latency_ms": latency_ms,
    }


def write_error(path: Path, marker: str) -> None:
    path.write_text(marker, encoding="utf-8")


def worker(index: int, prompt: str, args: argparse.Namespace) -> Dict:
    output_path = Path(args.output_dir) / f"req_{index}.json"
    request_id = f"bench-c{args.concurrency}-{index}"
    try:
        if args.backend_kind == "ollama":
            result = request_ollama(args.endpoint, args.model, prompt, args.max_tokens,
                                    request_id)
        else:
            result = request_openai(
                args.endpoint,
                args.model,
                prompt,
                args.max_tokens,
                request_id,
                args.api_key,
                inferflux_style=(args.backend_kind == "inferflux"),
            )
        output_path.write_text(json.dumps(result), encoding="utf-8")
        return result
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        write_error(output_path, "ERROR")
        return {}
    except Exception:
        write_error(output_path, "PARSE_ERROR")
        return {}


def run_warmup(prompts: List[str], args: argparse.Namespace) -> None:
    for i in range(min(2, len(prompts))):
        request_id = f"warmup-{i}"
        try:
            if args.backend_kind == "ollama":
                request_ollama(args.endpoint, args.model, prompts[i], args.max_tokens,
                               request_id)
            else:
                request_openai(
                    args.endpoint,
                    args.model,
                    prompts[i],
                    args.max_tokens,
                    request_id,
                    args.api_key,
                    inferflux_style=(args.backend_kind == "inferflux"),
                )
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Concurrent benchmark request driver")
    parser.add_argument("--backend-kind", choices=["inferflux", "openai", "ollama"],
                        required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--prompt-suite")
    parser.add_argument("--single-prompt", default="")
    parser.add_argument("--num-requests", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if not args.single_prompt and not args.prompt_suite:
        raise SystemExit("--prompt-suite is required when --single-prompt is not set")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args.prompt_suite, args.single_prompt, args.num_requests)
    run_warmup(prompts, args)
    time.sleep(1)

    results: List[Dict] = []
    wall_started = time.perf_counter_ns()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(worker, i, prompt, args)
                   for i, prompt in enumerate(prompts)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    wall_time_ms = (time.perf_counter_ns() - wall_started) // 1_000_000

    success = [result for result in results if result]
    summary = {
        "success_count": len(success),
        "total_tokens": sum(int(result["tokens"]) for result in success),
        "total_latency_ms": sum(int(result["latency_ms"]) for result in success),
        "latencies": [int(result["latency_ms"]) for result in success],
        "wall_time_ms": int(wall_time_ms),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
