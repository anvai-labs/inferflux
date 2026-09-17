#!/usr/bin/env python3
"""Read-only, fail-closed release eligibility and pinned package metadata.

Run eligibility from trusted main code, never from the triggering revision.
The CI provenance artifact is evidence from that exact run/attempt, not a
signature or a substitute for protected-main review.
"""

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import urllib.parse
import urllib.request
import zipfile

SHA = re.compile(r"[0-9a-f]{40}\Z")
VERSION = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\Z")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
GPU_JOBS = {
    "CUDA runtime gate (RTX 4000 Ada)": "Run CUDA model-backed gate",
    "ROCm runtime gate (Radeon AI PRO R9700)": "Run ROCm model-backed behavioral gate",
    "Dual-GPU gate result": None,
}


class EligibilityError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise EligibilityError(message)


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, url):
        redirected = super().redirect_request(request, fp, code, message, headers, url)
        if (
            urllib.parse.urlsplit(request.full_url).netloc
            != urllib.parse.urlsplit(url).netloc
        ):
            redirected.remove_header("Authorization")
        require(urllib.parse.urlsplit(url).scheme == "https", "Non-HTTPS API redirect")
        return redirected


class GitHub:
    def __init__(self, repository):
        require(REPOSITORY.fullmatch(repository), "Invalid repository")
        self.prefix = f"https://api.github.com/repos/{repository}"
        self.opener = urllib.request.build_opener(SafeRedirect())

    def read(self, path, limit=8_000_000):
        request = urllib.request.Request(
            self.prefix + path,
            headers={
                "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with self.opener.open(request, timeout=60) as response:
            data = response.read(limit + 1)
        require(len(data) <= limit, "GitHub response exceeds size bound")
        return data

    def get(self, path):
        return json.loads(self.read(path))

    def pages(self, path, key):
        separator = "&" if "?" in path else "?"
        for page in range(1, 101):
            values = self.get(f"{path}{separator}per_page=100&page={page}")[key]
            yield from values
            if len(values) < 100:
                return
        raise EligibilityError("GitHub pagination limit reached")

    def provenance(self, run):
        name = f"ci-release-source-{run['head_sha']}-{run['run_attempt']}"
        matches = [
            artifact
            for artifact in self.pages(
                f"/actions/runs/{run['id']}/artifacts", "artifacts"
            )
            if artifact["name"] == name and not artifact["expired"]
        ]
        require(len(matches) == 1, "Missing or ambiguous CI provenance artifact")
        data = self.read(f"/actions/artifacts/{matches[0]['id']}/zip", limit=100_000)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            require(
                archive.namelist() == ["source.json"], "Unexpected provenance archive"
            )
            require(
                archive.getinfo("source.json").file_size < 4096, "Oversized provenance"
            )
            return json.loads(archive.read("source.json"))


def successful_jobs(api, run):
    return list(
        api.pages(
            f"/actions/runs/{run['id']}/attempts/{run['run_attempt']}/jobs", "jobs"
        )
    )


def peeled_tag(api, tag):
    obj = api.get(f"/git/ref/tags/{tag}")["object"]
    for _ in range(10):
        if obj["type"] == "commit":
            return obj["sha"]
        require(obj["type"] == "tag", "Tag does not point to a commit")
        obj = api.get(f"/git/tags/{obj['sha']}")["object"]
    raise EligibilityError("Annotated tag nesting exceeds limit")


def verify_gpu(api, repository, sha):
    for run in api.pages(
        f"/actions/workflows/gpu-gates.yml/runs?head_sha={sha}&status=completed",
        "workflow_runs",
    ):
        if not (
            run["head_sha"] == sha
            and run["head_branch"] == "main"
            and run["event"] in ("push", "workflow_dispatch")
            and run["conclusion"] == "success"
            and run["head_repository"]["full_name"] == repository
            and run["path"].split("@")[0] == ".github/workflows/gpu-gates.yml"
        ):
            continue
        jobs = successful_jobs(api, run)
        valid = True
        for name, step_name in GPU_JOBS.items():
            matches = [job for job in jobs if job["name"] == name]
            if len(matches) != 1 or matches[0]["conclusion"] != "success":
                valid = False
                break
            if step_name and not any(
                step["name"] == step_name and step["conclusion"] == "success"
                for step in matches[0]["steps"]
            ):
                valid = False
                break
        if not valid:
            continue
        artifacts = list(api.pages(f"/actions/runs/{run['id']}/artifacts", "artifacts"))
        if all(
            sum(
                artifact["name"] == f"{backend}-gate-{sha}"
                and not artifact["expired"]
                and artifact["size_in_bytes"] > 0
                for artifact in artifacts
            )
            == 1
            for backend in ("cuda", "rocm")
        ):
            return run["id"]
    raise EligibilityError("No successful exact-SHA trusted-main dual-GPU evidence")


def evaluate(event, repository, api):
    run = event["workflow_run"]
    skipped = {"package": "false", "publish": "false"}
    if run["conclusion"] != "success" or run["event"] != "push":
        return skipped
    if run["head_repository"]["full_name"] != repository:
        return skipped
    require(
        run["path"].split("@")[0] == ".github/workflows/ci.yml",
        "Unexpected source workflow",
    )
    # Refresh run data: bind artifacts and jobs to the same successful attempt.
    current = api.get(f"/actions/runs/{run['id']}")
    for key in (
        "id",
        "head_sha",
        "head_branch",
        "event",
        "run_attempt",
        "conclusion",
        "path",
    ):
        require(current[key] == run[key], f"Source run changed: {key}")
    require(current["status"] == "completed", "Source CI has not completed")
    sha = run["head_sha"]
    require(SHA.fullmatch(sha), "Invalid source SHA")
    source = api.provenance(run)
    for key, expected in {
        "event": "push",
        "sha": sha,
        "repository": repository,
        "run_id": str(run["id"]),
        "run_attempt": str(run["run_attempt"]),
    }.items():
        require(source.get(key) == expected, f"CI provenance mismatch: {key}")
    source_jobs = successful_jobs(api, run)
    require(
        any(
            job["name"] == "Release source provenance"
            and job["conclusion"] == "success"
            for job in source_jobs
        ),
        "CI provenance job did not succeed",
    )
    ref = source.get("ref", "")
    tag = ""
    if ref == "refs/heads/main":
        require(
            run["head_branch"] == "main", "Main provenance does not match CI branch"
        )
    elif ref.startswith("refs/tags/v") and VERSION.fullmatch(ref[len("refs/tags/v") :]):
        tag = ref[len("refs/tags/") :]
        require(run["head_branch"] == tag, "Tag provenance does not match CI branch")
        require(peeled_tag(api, tag) == sha, "Tag does not resolve to tested SHA")
    else:
        raise EligibilityError(
            "CI did not run for upstream main or a stable semver tag"
        )
    main_sha = api.get("/git/ref/heads/main")["object"]["sha"]
    comparison = api.get(f"/compare/{sha}...{main_sha}")
    require(
        comparison["status"] in ("ahead", "identical"),
        "Candidate is not on main ancestry",
    )
    # Only inspect candidate-controlled metadata after ancestry validation.
    content = api.get(f"/contents/CMakeLists.txt?ref={sha}")
    cmake = base64.b64decode(content["content"], validate=False).decode("utf-8")
    match = re.search(
        r"^\s*project\s*\(\s*InferFlux\s+VERSION\s+([^\s)]+)(?=[\s)])",
        cmake,
        re.MULTILINE | re.IGNORECASE,
    )
    require(match is not None, "CMake project version is missing")
    version = match[1]
    require(VERSION.fullmatch(version), "Invalid CMake project version")
    if tag:
        require(tag == f"v{version}", "Tag and CMake project version differ")
        verify_gpu(api, repository, sha)
    return {
        "package": "true",
        "publish": str(bool(tag)).lower(),
        "sha": sha,
        "tag": tag,
        "version": version,
    }


def render_manifests(repository, sha, version, tag, packages, destination):
    require(REPOSITORY.fullmatch(repository), "Invalid repository")
    require(
        SHA.fullmatch(sha) and VERSION.fullmatch(version),
        "Invalid manifest revision/version",
    )
    require(not tag or tag == f"v{version}", "Invalid manifest tag")
    url = f"https://github.com/{repository}"
    formula_version = version if tag else f"{version}-dev-{sha[:7]}"
    homebrew = destination / "homebrew"
    homebrew.mkdir(parents=True, exist_ok=True)
    # Homebrew's GitDownloadStrategy recursively initializes pinned submodules.
    # GitHub source archives omit external/llama.cpp and cannot build InferFlux.
    (homebrew / "inferflux.rb").write_text(
        f"""class Inferflux < Formula
  desc "InferFlux inference server and CLI"
  homepage "{url}"
  url "{url}.git", using: :git, revision: "{sha}"
  version "{formula_version}"
  license "Apache-2.0"

  depends_on "cmake" => :build
  depends_on "openssl@3"
  depends_on "yaml-cpp"

  def install
    args = %w[-DBUILD_SHARED_LIBS=OFF -DENABLE_WEBUI=ON -DENABLE_CUDA=OFF -DENABLE_ROCM=OFF
              -DENABLE_BLAS=OFF -DENABLE_VULKAN=OFF -DENABLE_MTMD=OFF]
    args << "-DENABLE_MPS=#{{OS.mac? ? 'ON' : 'OFF'}}"
    system "cmake", "-S", ".", "-B", "build", *args, *std_cmake_args
    system "cmake", "--build", "build", "--target", "inferfluxd", "inferctl", "-j"
    bin.install "build/inferfluxd", "build/inferctl"
    pkgshare.install "docs/Quickstart.md"
  end

  test do
    assert_match "Usage:", shell_output("#{{bin}}/inferctl --help", 1)
    assert_match "usage: inferfluxd", shell_output("#{{bin}}/inferfluxd --help")
  end
end
""",
        encoding="utf-8",
    )
    if tag:
        matches = list(packages.glob("*.msi"))
        require(len(matches) == 1, "Expected exactly one Windows MSI")
        msi = matches[0]
        checksum = hashlib.sha256(msi.read_bytes()).hexdigest()
        winget = destination / "winget"
        winget.mkdir(parents=True, exist_ok=True)
        (winget / "inferencial.inferflux.yaml").write_text(
            f"""Id: Inferencial.InferFlux
Name: InferFlux
Publisher: Inferencial Labs
PublisherUrl: https://inferencial.ai
PackageUrl: {url}
License: Apache-2.0
ShortDescription: C++ LLM inference server and CLI.
Moniker: inferflux
Version: {version}
InstallerType: wix
Installers:
  - Architecture: x64
    InstallerUrl: {url}/releases/download/{tag}/{msi.name}
    InstallerSha256: {checksum}
Commands:
  - inferfluxd
  - inferctl
UpgradeBehavior: install
ReleaseNotesUrl: {url}/releases/tag/{tag}
""",
            encoding="utf-8",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    provenance = subparsers.add_parser("record-source")
    provenance.add_argument("--output", required=True, type=Path)
    check = subparsers.add_parser("check")
    check.add_argument("--event", required=True, type=Path)
    check.add_argument("--repository", required=True)
    check.add_argument("--output", required=True, type=Path)
    render = subparsers.add_parser("render")
    for name in ("repository", "sha", "version", "tag"):
        render.add_argument(f"--{name}", required=True)
    render.add_argument("--packages", required=True, type=Path)
    render.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "record-source":
            data = {
                name: os.environ[f"GITHUB_{name.upper()}"]
                for name in ("ref", "sha", "repository", "run_id", "run_attempt")
            }
            data["event"] = os.environ["GITHUB_EVENT_NAME"]
            args.output.write_text(json.dumps(data), encoding="utf-8")
        elif args.command == "check":
            result = evaluate(
                json.loads(args.event.read_text()),
                args.repository,
                GitHub(args.repository),
            )
            with args.output.open("a", encoding="utf-8") as output:
                for key, value in result.items():
                    output.write(f"{key}={value}\n")
            print(json.dumps(result, sort_keys=True))
        else:
            render_manifests(
                args.repository,
                args.sha,
                args.version,
                args.tag,
                args.packages,
                args.destination,
            )
    except (
        EligibilityError,
        OSError,
        ValueError,
        KeyError,
        zipfile.BadZipFile,
    ) as error:
        parser.exit(1, f"Release eligibility failed: {error}\n")


if __name__ == "__main__":
    main()
