#!/usr/bin/env python3
"""Validate dependency evidence without accessing the network."""

import hashlib
import json
import pathlib
import re
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "dependencies.json"
REQUIRED_FIELDS = {
    "name", "version", "source_url", "source_ref", "source_digest",
    "integration", "artifacts", "purl", "license", "owner",
    "supported_toolchains", "upgrade_target", "rollback",
}


def fail(message: str) -> None:
  raise ValueError(message)


def main() -> int:
  data = json.loads(MANIFEST.read_text(encoding="utf-8"))
  if data.get("schema_version") != 1:
    fail("unsupported dependency manifest schema_version")
  dependencies = data.get("dependencies")
  if not isinstance(dependencies, list) or not dependencies:
    fail("dependencies must be a non-empty list")

  by_name = {}
  for dependency in dependencies:
    missing = REQUIRED_FIELDS - dependency.keys()
    if missing:
      fail(f"{dependency.get('name', '<unnamed>')}: missing {sorted(missing)}")
    name = dependency["name"]
    if name in by_name:
      fail(f"duplicate dependency name: {name}")
    by_name[name] = dependency
    if not dependency["supported_toolchains"]:
      fail(f"{name}: supported_toolchains must not be empty")
    for artifact in dependency["artifacts"]:
      path = ROOT / artifact["path"]
      if not path.exists():
        fail(f"{name}: artifact not found: {artifact['path']}")
      expected = artifact.get("sha256")
      if expected:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
          fail(f"{name}: sha256 drift for {artifact['path']}")

  llama = by_name["llama.cpp"]
  gitlink = subprocess.check_output(
      ["git", "ls-files", "-s", "external/llama.cpp"], cwd=ROOT,
      text=True).split()[1]
  if gitlink != llama["source_ref"]:
    fail("llama.cpp manifest ref does not match the submodule gitlink")

  cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
  mlx_match = re.search(r'set\(MLX_C_TAG\s+"([^"]+)"', cmake)
  if not mlx_match or mlx_match.group(1) != f"v{by_name['mlx-c']['version']}":
    fail("mlx-c manifest version does not match MLX_C_TAG")

  dependencies_cmake = (ROOT / "cmake" / "Dependencies.cmake").read_text(
      encoding="utf-8")
  yaml_ref = by_name["yaml-cpp"]["source_ref"]
  if not re.search(rf"GIT_TAG\s+{re.escape(yaml_ref)}(?:\s|$)",
                   dependencies_cmake, re.MULTILINE):
    fail("yaml-cpp manifest ref does not match FetchContent GIT_TAG")

  print(f"[dependency-gate] dependencies checked: {len(dependencies)}")
  print("[dependency-gate] refs and vendored digests match")
  print("[dependency-gate] PASSED")
  return 0


if __name__ == "__main__":
  try:
    sys.exit(main())
  except (OSError, ValueError, KeyError, IndexError,
          subprocess.CalledProcessError) as exc:
    print(f"[dependency-gate] FAILED: {exc}", file=sys.stderr)
    sys.exit(1)
