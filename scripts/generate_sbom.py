#!/usr/bin/env python3
"""
InferFlux SBOM generator — produces CycloneDX 1.5 JSON and SPDX 2.3 tag-value
output from config/dependencies.json.

Usage (via CMake):
  cmake --build build --target sbom

Output files (in --out-dir):
  inferflux-sbom.cdx.json   — CycloneDX 1.5
  inferflux-sbom.spdx        — SPDX 2.3 tag-value

The manifest lists each dependency's version, immutable source evidence,
ownership, supported toolchains, and rollback instruction.
"""

import argparse
import json
import os
import sys
import time
import uuid


def load_components(source_dir: str) -> list:
    manifest_path = os.path.join(source_dir, "config", "dependencies.json")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    return manifest["dependencies"]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# CycloneDX 1.5 JSON
# ---------------------------------------------------------------------------

def build_cyclonedx(version: str, manifest_components: list) -> dict:
    components = []
    for c in manifest_components:
        components.append({
            "type": "library",
            "name": c["name"],
            "version": c["version"],
            "purl": c["purl"],
            "licenses": [{"license": {"id": c["license"]}}],
            "properties": [
                {"name": "inferflux:source_ref", "value": c["source_ref"]},
                {"name": "inferflux:source_digest", "value": c["source_digest"]},
                {"name": "inferflux:owner", "value": c["owner"]},
                {"name": "inferflux:supported_toolchains",
                 "value": ", ".join(c["supported_toolchains"])},
            ],
        })
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": now_iso(),
            "component": {
                "type": "application",
                "name": "InferFlux",
                "version": version,
            },
        },
        "components": components,
    }


def write_cyclonedx(out_dir: str, version: str, components: list) -> str:
    doc = build_cyclonedx(version, components)
    path = os.path.join(out_dir, "inferflux-sbom.cdx.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# SPDX 2.3 tag-value
# ---------------------------------------------------------------------------

def write_spdx(out_dir: str, version: str, components: list) -> str:
    path = os.path.join(out_dir, "inferflux-sbom.spdx")
    doc_namespace = f"https://inferencial.ai/sbom/inferflux-{version}-{uuid.uuid4()}"

    lines = [
        "SPDXVersion: SPDX-2.3",
        "DataLicense: CC0-1.0",
        f"SPDXID: SPDXRef-DOCUMENT",
        f"DocumentName: InferFlux-{version}",
        f"DocumentNamespace: {doc_namespace}",
        f"Creator: Tool: InferFlux-generate_sbom.py",
        f"Created: {now_iso()}",
        "",
        "# ---- Primary package ----",
        f"PackageName: InferFlux",
        f"SPDXID: SPDXRef-InferFlux",
        f"PackageVersion: {version}",
        f"PackageDownloadLocation: https://github.com/inferencial/InferFlux",
        "FilesAnalyzed: false",
        "PackageLicenseConcluded: NOASSERTION",
        "PackageLicenseDeclared: NOASSERTION",
        "PackageCopyrightText: NOASSERTION",
        "",
    ]

    for c in components:
        spdx_id = f"SPDXRef-{c['name'].replace('.', '-').replace('/', '-')}"
        lines += [
            f"# ---- {c['name']} ----",
            f"PackageName: {c['name']}",
            f"SPDXID: {spdx_id}",
            f"PackageVersion: {c['version']}",
            f"PackageDownloadLocation: {c['purl']}",
            "FilesAnalyzed: false",
            f"PackageLicenseConcluded: {c['license']}",
            f"PackageLicenseDeclared: {c['license']}",
            "PackageCopyrightText: NOASSERTION",
            f"PackageComment: source_ref={c['source_ref']}; "
            f"source_digest={c['source_digest']}; owner={c['owner']}; "
            f"toolchains={', '.join(c['supported_toolchains'])}",
            "",
            f"Relationship: SPDXRef-InferFlux DYNAMIC_LINK {spdx_id}",
            "",
        ]

    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate InferFlux SBOM")
    parser.add_argument("--source-dir", required=True, help="Repository root")
    parser.add_argument("--version", required=True, help="Product version string")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    components = load_components(args.source_dir)
    cdx = write_cyclonedx(args.out_dir, args.version, components)
    spdx = write_spdx(args.out_dir, args.version, components)

    print(f"[sbom] CycloneDX 1.5 → {cdx}")
    print(f"[sbom] SPDX 2.3      → {spdx}")
    print(f"[sbom] {len(components)} components listed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
