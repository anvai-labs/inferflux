#!/usr/bin/env python3
"""Invalid selectors must fail at startup, before binding any listener."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

binary = str(Path(sys.argv[1]).resolve())
base_env = {
    key: value for key, value in os.environ.items() if not key.startswith("INFERFLUX_")
}
with tempfile.TemporaryDirectory(prefix="inferflux-load-spec-") as folder:
    for extra, overrides in (
        ({"models": [{"path": "/missing.gguf", "device": "cuda:-1"}]}, {}),
        ({"runtime": {"cuda": {"device_id": -1}}}, {}),
        ({}, {"INFERFLUX_MODELS": "path=/missing.gguf,device=rocm:-1"}),
        ({"models": [{"path": "/missing.gguf", "max_parallel_sequences": 0}]}, {}),
    ):
        config = Path(folder) / "config.json"
        config.write_text(
            json.dumps({"server": {"host": "127.0.0.1", "http_port": 0}, **extra})
        )
        env = {
            **base_env,
            **overrides,
            "INFERFLUX_POLICY_STORE": str(Path(folder) / "policy"),
        }
        result = subprocess.run(
            [binary, "--config", str(config)],
            env=env,
            cwd=folder,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        assert result.returncode != 0, "invalid model placement reached serving state"
        assert (
            b"device" in result.stdout or b"max_parallel_sequences" in result.stdout
        ), "missing validation reason"
print("4 startup rejection contracts passed")
