#!/usr/bin/env python3
"""
InferFlux SBOM generator — produces CycloneDX 1.5 JSON and SPDX 2.3 tag-value
output from a fixed vendor component manifest embedded in this script.

Usage (via CMake):
  cmake --build build --target sbom

Output files (in --out-dir):
  inferflux-sbom.cdx.json   — CycloneDX 1.5
  inferflux-sbom.spdx        — SPDX 2.3 tag-value

The manifest lists every vendored dependency with its declared version and
SPDX license expression.  Pin new versions here whenever you update a vendor.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import uuid


# ---------------------------------------------------------------------------
# Vendor manifest — update when bumping dependency versions.
# ---------------------------------------------------------------------------
COMPONENTS = [
    {
        "name": "llama.cpp",
        "version": "pinned-submodule",
        "purl": "pkg:github/ggml-org/llama.cpp",
        "license": "MIT",
        "description": "C/C++ inference library for GGUF models (submodule at external/llama.cpp)",
    },
    {
        "name": "nlohmann-json",
        "version": "3.11.3",
        "purl": "pkg:github/nlohmann/json@v3.11.3",
        "license": "MIT",
        "description": "Single-header JSON library (external/nlohmann/json.hpp)",
    },
    {
        "name": "Catch2",
        "version": "3.7.1",
        "purl": "pkg:github/catchorg/Catch2@v3.7.1",
        "license": "BSL-1.0",
        "description": "C++ unit test framework — amalgamated (external/catch2/)",
    },
    {
        "name": "yaml-cpp",
        "version": "system",
        "purl": "pkg:generic/yaml-cpp",
        "license": "MIT",
        "description": "YAML parser/emitter linked as system or submodule library",
    },
    {
        "name": "OpenSSL",
        "version": "system",
        "purl": "pkg:generic/openssl",
        "license": "Apache-2.0",
        "description": "TLS, AES-GCM policy encryption, SHA-256, RS256 JWT verification",
    },
]


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# CycloneDX 1.5 JSON
# ---------------------------------------------------------------------------

def build_cyclonedx(version: str) -> dict:
    components = []
    for c in COMPONENTS:
        components.append({
            "type": "library",
            "name": c["name"],
            "version": c["version"],
            "purl": c["purl"],
            "licenses": [{"license": {"id": c["license"]}}],
            "description": c["description"],
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


def write_cyclonedx(out_dir: str, version: str) -> str:
    doc = build_cyclonedx(version)
    path = os.path.join(out_dir, "inferflux-sbom.cdx.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# SPDX 2.3 tag-value
# ---------------------------------------------------------------------------

def write_spdx(out_dir: str, version: str) -> str:
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

    for i, c in enumerate(COMPONENTS):
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
            f"PackageComment: {c['description']}",
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

    cdx = write_cyclonedx(args.out_dir, args.version)
    spdx = write_spdx(args.out_dir, args.version)

    print(f"[sbom] CycloneDX 1.5 → {cdx}")
    print(f"[sbom] SPDX 2.3      → {spdx}")
    print(f"[sbom] {len(COMPONENTS)} components listed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
