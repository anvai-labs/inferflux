#!/usr/bin/env python3
"""Smoke every release package on its matching disposable hosted runner.

Requires Python 3.12+. Linux DEB, macOS PKG and Windows MSI execute native
installers. RPM installs into an isolated RPM root using Ubuntu's runtime
libraries; --nodeps is necessary because Ubuntu has no RPM dependency database.
Archives and DMG payloads are extracted/copied, not claimed as installer tests.
"""

import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile
import xml.etree.ElementTree as ET
import zipfile


class SmokeError(RuntimeError):
    pass


def run(*args, expected=(0,), timeout=300):
    print("+", subprocess.list2cmdline([str(arg) for arg in args]), flush=True)
    result = subprocess.run(
        [str(arg) for arg in args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    print(result.stdout, end="", flush=True)
    if result.returncode not in expected:
        raise SmokeError(f"Command exited {result.returncode}; expected {expected}")
    return result.stdout


def one(root, pattern):
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise SmokeError(f"Expected one {pattern} in {root}; found {len(matches)}")
    return matches[0].resolve()


def smoke_binaries(root):
    """Execute the package's own binaries, never an executable found on PATH."""
    suffix = ".exe" if platform.system() == "Windows" else ""
    cli = one(root, f"**/bin/inferctl{suffix}")
    server = one(root, f"**/bin/inferfluxd{suffix}")
    smoke_binary_paths(cli, server)


def smoke_binary_paths(cli, server):
    if "Usage:" not in run(cli, "--help", expected=(1,), timeout=20):
        raise SmokeError("inferctl --help did not print its usage contract")
    if "usage: inferfluxd" not in run(server, "--help", timeout=20):
        raise SmokeError("inferfluxd --help did not print its usage contract")


def smoke_deb_binaries():
    # CPack DEB defaults to /usr, unlike a plain CMake install's /usr/local.
    # Query the installed package, never search PATH or another staging tree.
    files = run("dpkg-query", "--listfiles", "inferflux").splitlines()
    binaries = []
    for name in ("inferctl", "inferfluxd"):
        matches = [Path(path) for path in files if path.endswith(f"/bin/{name}")]
        if len(matches) != 1 or not matches[0].is_absolute():
            raise SmokeError(f"Expected one package-owned bin/{name}; found {matches}")
        binaries.append(matches[0])
    smoke_binary_paths(*binaries)


def validate_pkg_distribution(expanded):
    """Reject CPack's empty monolithic distribution before native installation."""
    distribution = ET.parse(expanded / "Distribution").getroot()
    choices = {choice.get("id"): choice for choice in distribution.findall("choice")}
    referenced = set()
    for line in distribution.findall("choices-outline//line"):
        choice = choices.get(line.get("choice"))
        if choice is None:
            raise SmokeError("PKG distribution references a missing choice")
        referenced.update(ref.get("id") for ref in choice.findall("pkg-ref"))
    payloads = {
        ref.get("id")
        for ref in distribution.findall("pkg-ref")
        if ref.text and ref.text.strip()
    }
    if not referenced or None in referenced or not referenced <= payloads:
        raise SmokeError("PKG distribution has no complete installable package choices")


def smoke_archive(package, root):
    root.mkdir()
    if package.name.endswith(".tar.gz"):
        with tarfile.open(package) as archive:
            archive.extractall(root, filter="data")
    else:
        with zipfile.ZipFile(package) as archive:
            for member in archive.infolist():
                target = (root / member.filename).resolve()
                if not target.is_relative_to(root.resolve()):
                    raise SmokeError("Archive member escapes extraction root")
            archive.extractall(root)
    smoke_binaries(root)


def smoke_linux(packages, temporary):
    archive = one(packages, "inferflux-*Linux*.tar.gz")
    deb = one(packages, "inferflux-*Linux*.deb")
    rpm = one(packages, "inferflux-*Linux*.rpm")
    smoke_archive(archive, temporary / "tgz")
    run("sudo", "apt-get", "install", "-y", deb)
    try:
        smoke_deb_binaries()
    finally:
        run("sudo", "apt-get", "remove", "-y", "inferflux")

    rpm_root = temporary / "rpm-root"
    rpm_root.mkdir()
    print(
        "RPM: native installation with --nodeps; Ubuntu runtime libraries supply dependencies."
    )
    try:
        run("sudo", "rpm", "--root", rpm_root, "--initdb")
        run("sudo", "rpm", "--root", rpm_root, "--install", "--nodeps", rpm)
        smoke_binaries(rpm_root)
        run("sudo", "rpm", "--root", rpm_root, "--erase", "--nodeps", "inferflux")
    finally:
        # Return only this generated temporary root to the runner for cleanup.
        run("sudo", "chown", "-R", f"{os.getuid()}:{os.getgid()}", rpm_root)


def smoke_macos(packages, temporary):
    archive = one(packages, "inferflux-*Darwin*.tar.gz")
    pkg = one(packages, "inferflux-*.pkg")
    dmg = one(packages, "inferflux-*.dmg")
    smoke_archive(archive, temporary / "tgz")
    expanded = temporary / "pkg-expanded"
    run("pkgutil", "--expand", pkg, expanded)
    validate_pkg_distribution(expanded)
    run("sudo", "installer", "-pkg", pkg, "-target", "/")
    smoke_binaries(Path("/usr/local"))
    run("hdiutil", "verify", dmg)
    mount = temporary / "mounted"
    run("hdiutil", "attach", "-readonly", "-nobrowse", "-mountpoint", mount, dmg)
    try:
        copied = temporary / "dmg-copy"
        # Ignore DragNDrop's Applications convenience link; copy the package payload.
        shutil.copytree(mount, copied, symlinks=True)
        smoke_binaries(copied)
    finally:
        run("hdiutil", "detach", mount)


def smoke_windows(packages, temporary):
    archive = one(packages, "inferflux-*.zip")
    msi = one(packages, "inferflux-*.msi")
    smoke_archive(archive, temporary / "zip")
    installed = temporary / "msi-installed"
    log = temporary / "msi-install.log"
    try:
        # CPack WiX names its configurable target directory INSTALL_ROOT.
        run(
            "msiexec.exe",
            "/i",
            msi,
            "/qn",
            "/norestart",
            f"INSTALL_ROOT={installed}",
            "/l*v",
            log,
            expected=(0, 3010),
        )
        smoke_binaries(installed)
    finally:
        run("msiexec.exe", "/x", msi, "/qn", "/norestart", expected=(0, 3010, 1605))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", required=True, type=Path)
    args = parser.parse_args(argv)
    runners = {"Linux": smoke_linux, "Darwin": smoke_macos, "Windows": smoke_windows}
    host = platform.system()
    if host not in runners:
        parser.error(f"Unsupported package smoke host: {host}")
    try:
        with tempfile.TemporaryDirectory(
            prefix="inferflux-package-smoke-"
        ) as directory:
            runners[host](args.packages.resolve(), Path(directory))
        print(f"Package smoke passed on {host}")
        return 0
    except (
        SmokeError,
        OSError,
        subprocess.TimeoutExpired,
        tarfile.TarError,
        zipfile.BadZipFile,
        ET.ParseError,
    ) as exc:
        print(f"Package smoke FAILED: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
