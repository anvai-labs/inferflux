#!/usr/bin/env python3
"""Validate planning artifact identity, metadata, indexes, and dependencies."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KINDS = {
    "ADR": (ROOT / "docs" / "adr", {"Proposed", "Accepted", "Rejected", "Superseded", "Deprecated"}),
    "FTR": (ROOT / "docs" / "features", {"Proposed", "Discovery", "In Progress", "Validated", "Released", "Withdrawn"}),
    "TD": (ROOT / "docs" / "technical-debt", {"Proposed", "In Progress", "Blocked", "Validated", "Closed", "Superseded"}),
}
ID_RE = re.compile(r"^(ADR-\d{4}|FTR-\d{3}|TD-\d{3})")
DEP_RE = re.compile(r"(?:ADR-\d{4}|FTR-\d{3}|TD-\d{3})")


@dataclass(frozen=True)
class Artifact:
  artifact_id: str
  path: Path
  status: str
  dependencies: tuple[str, ...]


def fail(message: str) -> None:
  print(f"[planning-gate] FAIL: {message}")
  raise SystemExit(1)


def field(text: str, name: str, path: Path) -> str:
  match = re.search(rf"^{re.escape(name)}:\s*(.+?)\s*$", text, re.MULTILINE)
  if not match:
    fail(f"{path.relative_to(ROOT)} missing '{name}:' metadata")
  return match.group(1)


def load_artifacts() -> dict[str, Artifact]:
  artifacts: dict[str, Artifact] = {}
  for prefix, (directory, statuses) in KINDS.items():
    index_text = (directory / "README.md").read_text(encoding="utf-8")
    for path in sorted(directory.glob(f"{prefix}-*.md")):
      text = path.read_text(encoding="utf-8")
      title = text.splitlines()[0].removeprefix("# ")
      id_match = ID_RE.match(title)
      if not id_match:
        fail(f"{path.relative_to(ROOT)} title must start with a valid artifact ID")
      artifact_id = id_match.group(1)
      if not path.name.startswith(artifact_id):
        fail(f"{path.relative_to(ROOT)} filename does not match {artifact_id}")
      if artifact_id in artifacts:
        fail(f"duplicate artifact ID: {artifact_id}")
      status = field(text, "Status", path)
      if status not in statuses:
        fail(f"{artifact_id} has invalid status '{status}'")
      field(text, "Owners", path)
      if prefix != "ADR":
        field(text, "Priority", path)
      dependencies_raw = field(text, "Dependencies", path)
      dependencies = tuple(DEP_RE.findall(dependencies_raw))
      if dependencies_raw != "None" and not dependencies:
        fail(f"{artifact_id} dependencies contain no recognized IDs")
      index_lines = [line for line in index_text.splitlines() if artifact_id in line]
      if not index_lines:
        fail(f"{artifact_id} missing from {directory.relative_to(ROOT)}/README.md")
      index_cells = {cell.strip() for cell in index_lines[0].split("|")}
      if status not in index_cells:
        fail(f"{artifact_id} status '{status}' disagrees with its index row")
      artifacts[artifact_id] = Artifact(artifact_id, path, status, dependencies)
  return artifacts


def validate_dependencies(artifacts: dict[str, Artifact]) -> None:
  for artifact in artifacts.values():
    for dependency in artifact.dependencies:
      if dependency == artifact.artifact_id:
        fail(f"{artifact.artifact_id} depends on itself")
      if dependency not in artifacts:
        fail(f"{artifact.artifact_id} has unknown dependency {dependency}")

  visiting: set[str] = set()
  visited: set[str] = set()

  def visit(artifact_id: str, path: tuple[str, ...]) -> None:
    if artifact_id in visiting:
      fail("dependency cycle: " + " -> ".join((*path, artifact_id)))
    if artifact_id in visited:
      return
    visiting.add(artifact_id)
    for dependency in artifacts[artifact_id].dependencies:
      visit(dependency, (*path, artifact_id))
    visiting.remove(artifact_id)
    visited.add(artifact_id)

  for artifact_id in artifacts:
    visit(artifact_id, ())


def main() -> int:
  artifacts = load_artifacts()
  validate_dependencies(artifacts)
  roadmap = (ROOT / "docs" / "Roadmap.md").read_text(encoding="utf-8")
  missing = sorted(artifact_id for artifact_id in artifacts if artifact_id not in roadmap)
  if missing:
    fail("roadmap does not reference: " + ", ".join(missing))
  print(f"[planning-gate] artifacts checked: {len(artifacts)}")
  print("[planning-gate] dependency graph is complete and acyclic")
  print("[planning-gate] PASSED")
  return 0


if __name__ == "__main__":
  sys.exit(main())
