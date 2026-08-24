#!/usr/bin/env python3
"""Compare native-vs-llama decode traces and report the first divergence."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


LOGITS_RE = re.compile(
    r"debug_logits\[[^\]]+\]: "
    r"(?:(?:client_request_id=(?P<client_request_id>[^,]+), )?)"
    r"request_id=(?P<request_id>-?\d+), "
    r"sequence_id=(?P<sequence_id>-?\d+), "
    r"sequence_generation=(?P<sequence_generation>\d+), "
    r"n_past=(?P<n_past>-?\d+) top-\d+: (?P<top>.+)$"
)

RAW_NATIVE_LOGITS_RE = re.compile(
    r"\[DEBUG_LOGITS\] tokens=\[(?P<tokens>[^\]]*)\] "
    r"n_past=(?P<n_past>-?\d+) top-\d+: (?P<top>.+?)(?: \(nan=.*)?$"
)

TOKEN_RE = re.compile(
    r"(?P<kind>token_trace|decode_mapping|sample_mapping)\[[^\]]+\]: .*?"
    r"(?:(?:client_request_id=(?P<client_request_id>[^,]+), )?)"
    r"request_id=(?P<request_id>-?\d+), "
    r".*?"
    r"sequence_generation=(?P<sequence_generation>\d+), "
    r"n_past=(?P<n_past>-?\d+), "
    r"(?:(?:token_count=(?P<token_count>\d+), )?)"
    r"sampled_token=(?P<sampled_token>-?\d+), piece=(?P<piece>.*)$"
)

TOP_ENTRY_RE = re.compile(r"\[(?P<token>-?\d+)\]=(?P<value>-?\d+(?:\.\d+)?)")


def parse_top_entries(raw: str) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    for match in TOP_ENTRY_RE.finditer(raw):
        entries.append(
            {
                "token": int(match.group("token")),
                "value": float(match.group("value")),
            }
        )
    return entries


TraceKey = Tuple[str, int, int]


def trace_request_key(client_request_id: str | None, request_id: int) -> str:
    if client_request_id:
        return f"client:{client_request_id}"
    return f"request:{request_id}"


def parse_trace(path: Path) -> Dict[TraceKey, Dict[str, object]]:
    rows: Dict[TraceKey, Dict[str, object]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        logits_match = LOGITS_RE.search(line)
        if logits_match:
            request_id = int(logits_match.group("request_id"))
            key = (
                trace_request_key(logits_match.group("client_request_id"),
                                  request_id),
                int(logits_match.group("sequence_generation")),
                int(logits_match.group("n_past")),
            )
            row = rows.setdefault(key, {})
            row["request_key"] = key[0]
            row["request_id"] = request_id
            if logits_match.group("client_request_id"):
                row["client_request_id"] = logits_match.group("client_request_id")
            row["sequence_generation"] = key[1]
            row["n_past"] = key[2]
            row["top_logits"] = parse_top_entries(logits_match.group("top"))
            continue

        raw_native_logits_match = RAW_NATIVE_LOGITS_RE.search(line)
        if raw_native_logits_match:
            tokens = [
                int(token)
                for token in raw_native_logits_match.group("tokens").split(",")
                if token.strip()
            ]
            effective_n_past = int(raw_native_logits_match.group("n_past")) + len(
                tokens
            )
            key = ("request:-1", 0, effective_n_past)
            row = rows.setdefault(key, {})
            row["request_key"] = key[0]
            row["request_id"] = -1
            row["sequence_generation"] = key[1]
            row["n_past"] = key[2]
            row["prompt_tokens"] = tokens
            row["top_logits"] = parse_top_entries(raw_native_logits_match.group("top"))
            continue

        token_match = TOKEN_RE.search(line)
        if token_match:
            n_past = int(token_match.group("n_past"))
            token_count = token_match.group("token_count")
            kind = token_match.group("kind")
            request_id = int(token_match.group("request_id"))
            if kind == "sample_mapping" and token_count is not None:
                n_past += int(token_count)
            elif kind == "decode_mapping":
                n_past += 1
            key = (
                trace_request_key(token_match.group("client_request_id"),
                                  request_id),
                int(token_match.group("sequence_generation")),
                n_past,
            )
            row = rows.setdefault(key, {})
            row["request_key"] = key[0]
            row["request_id"] = request_id
            if token_match.group("client_request_id"):
                row["client_request_id"] = token_match.group("client_request_id")
            row["sequence_generation"] = key[1]
            row["n_past"] = key[2]
            row["sampled_token"] = int(token_match.group("sampled_token"))
            row["piece"] = token_match.group("piece")
    return rows


def compare(
    native_rows: Dict[TraceKey, Dict[str, object]],
    llama_rows: Dict[TraceKey, Dict[str, object]],
) -> Dict[str, object]:
    shared_keys = sorted(set(native_rows) & set(llama_rows))
    result: Dict[str, object] = {
        "shared_steps": len(shared_keys),
        "parity": True,
        "first_divergence": None,
        "native_only_steps": len(set(native_rows) - set(llama_rows)),
        "llama_only_steps": len(set(llama_rows) - set(native_rows)),
    }
    for key in shared_keys:
        native = native_rows[key]
        llama = llama_rows[key]
        native_token = native.get("sampled_token")
        llama_token = llama.get("sampled_token")
        native_top = native.get("top_logits", [])
        llama_top = llama.get("top_logits", [])
        native_top_token = native_top[0]["token"] if native_top else None
        llama_top_token = llama_top[0]["token"] if llama_top else None
        if native_token != llama_token or native_top_token != llama_top_token:
            result["parity"] = False
            result["first_divergence"] = {
                "request_key": key[0],
                "sequence_generation": key[1],
                "n_past": key[2],
                "native": native,
                "llama": llama,
            }
            break
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare native and llama decode traces and report the first divergence."
    )
    parser.add_argument("native_log", type=Path)
    parser.add_argument("llama_log", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-parity", action="store_true")
    args = parser.parse_args()

    native_rows = parse_trace(args.native_log)
    llama_rows = parse_trace(args.llama_log)
    result = compare(native_rows, llama_rows)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"shared_steps={result['shared_steps']}")
        print(f"parity={result['parity']}")
        print(f"native_only_steps={result['native_only_steps']}")
        print(f"llama_only_steps={result['llama_only_steps']}")
        if result["first_divergence"]:
            div = result["first_divergence"]
            print(
                f"first_divergence: request_key={div['request_key']} "
                f"sequence_generation={div['sequence_generation']} n_past={div['n_past']}"
            )
            print(f"  native: {json.dumps(div['native'], sort_keys=True)}")
            print(f"  llama:  {json.dumps(div['llama'], sort_keys=True)}")

    if args.require_parity and not result["parity"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
