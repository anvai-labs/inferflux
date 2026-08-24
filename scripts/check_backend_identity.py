#!/usr/bin/env python3
"""Fail-closed backend identity contract checks for startup/benchmark flows."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import urllib.error
import urllib.request


def fetch_model(base_url: str, model_id: str, api_key: str | None) -> dict:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/models/{model_id}",
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"identity check HTTP {exc.code} for {model_id}: {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"identity check failed to reach server: {exc}") from exc


def assert_model_identity(
    model: dict,
    expected_provider: str,
    expected_backend: str | None,
    expect_no_fallback: bool,
) -> None:
    exposure = model.get("backend_exposure") or {}
    provider = exposure.get("provider")
    if provider != expected_provider:
        raise RuntimeError(
            f"provider mismatch: expected {expected_provider}, got {provider!r}"
        )
    if expected_backend is not None and model.get("backend") != expected_backend:
        raise RuntimeError(
            f"backend mismatch: expected {expected_backend}, got {model.get('backend')!r}"
        )
    if expect_no_fallback and exposure.get("fallback"):
        raise RuntimeError(
            f"unexpected fallback enabled: {exposure.get('fallback_reason', '')}"
        )


def assert_log_clean(log_file: pathlib.Path, forbidden_patterns: list[str]) -> None:
    if not log_file.exists():
        raise RuntimeError(f"log file not found: {log_file}")
    text = log_file.read_text(encoding="utf-8", errors="replace")
    for pattern in forbidden_patterns:
        if re.search(pattern, text, flags=re.MULTILINE):
            raise RuntimeError(
                f"forbidden backend pattern matched in {log_file}: {pattern}"
            )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that a running model instance matches the expected backend identity."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--expected-provider", required=True)
    parser.add_argument("--expected-backend")
    parser.add_argument("--api-key")
    parser.add_argument("--log-file", type=pathlib.Path)
    parser.add_argument(
        "--forbid-log-pattern",
        action="append",
        default=[],
        help="Regex pattern that must not appear in the log file.",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Do not fail when backend_exposure.fallback is true.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    model = fetch_model(args.base_url, args.model_id, args.api_key)
    assert_model_identity(
        model,
        expected_provider=args.expected_provider,
        expected_backend=args.expected_backend,
        expect_no_fallback=not args.allow_fallback,
    )
    if args.log_file is not None and args.forbid_log_pattern:
        assert_log_clean(args.log_file, args.forbid_log_pattern)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except RuntimeError as exc:
        print(f"identity check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
