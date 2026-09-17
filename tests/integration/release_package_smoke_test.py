"""Offline release installer orchestration and fail-closed smoke regressions."""

import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "package_smoke", ROOT / "scripts/smoke_release_packages.py"
)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class PackageSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.packages = self.root / "packages"
        self.packages.mkdir()
        self.scratch = self.root / "scratch"
        self.scratch.mkdir()

    def artifacts(self, names):
        for name in names:
            (self.packages / name).touch()

    def test_missing_or_duplicate_artifact_fails(self):
        with self.assertRaises(smoke.SmokeError):
            smoke.one(self.packages, "*.deb")
        self.artifacts(["a.deb", "b.deb"])
        with self.assertRaises(smoke.SmokeError):
            smoke.one(self.packages, "*.deb")

    def test_binary_contract_checks_exit_and_output(self):
        binaries = self.scratch / "bin"
        binaries.mkdir()
        (binaries / "inferctl").touch()
        (binaries / "inferfluxd").touch()
        with patch.object(smoke.platform, "system", return_value="Linux"), patch.object(
            smoke, "run", side_effect=["Usage:\n", "usage: inferfluxd\n"]
        ) as run:
            smoke.smoke_binaries(self.scratch)
        self.assertEqual(run.call_args_list[0].kwargs["expected"], (1,))
        self.assertEqual(
            run.call_args_list[0].args[0], (binaries / "inferctl").resolve()
        )
        with patch.object(smoke.platform, "system", return_value="Linux"), patch.object(
            smoke, "run", return_value="unrelated output"
        ), self.assertRaises(smoke.SmokeError):
            smoke.smoke_binaries(self.scratch)

    def test_unexpected_exit_fails(self):
        with patch.object(smoke.subprocess, "run") as run:
            run.return_value.returncode = 127
            run.return_value.stdout = "missing shared library"
            with self.assertRaises(smoke.SmokeError):
                smoke.run("inferctl", expected=(1,))

    def test_archive_extraction_reaches_binaries_and_rejects_escape(self):
        archive = self.packages / "inferflux.tar.gz"
        with tarfile.open(archive, "w:gz") as output:
            member = tarfile.TarInfo("inferflux/bin/inferctl")
            member.size = 4
            output.addfile(member, io.BytesIO(b"test"))
        extracted = self.scratch / "extracted"
        with patch.object(smoke, "smoke_binaries") as binaries:
            smoke.smoke_archive(archive, extracted)
            binaries.assert_called_once_with(extracted)
        self.assertEqual((extracted / "inferflux/bin/inferctl").read_bytes(), b"test")
        with tarfile.open(archive, "w:gz") as output:
            member = tarfile.TarInfo("../escape")
            output.addfile(member)
        with self.assertRaises(tarfile.TarError):
            smoke.smoke_archive(archive, self.scratch / "unsafe")

    def test_linux_runs_real_installers_and_isolated_rpm(self):
        self.artifacts([f"inferflux-1-Linux.{ext}" for ext in ("tar.gz", "deb", "rpm")])
        with patch.object(smoke, "run") as run, patch.object(
            smoke, "smoke_archive"
        ), patch.object(smoke, "smoke_binaries") as binaries:
            smoke.smoke_linux(self.packages, self.scratch)
        calls = [call.args for call in run.call_args_list]
        self.assertEqual(calls[0][:4], ("sudo", "apt-get", "install", "-y"))
        self.assertIn("--install", calls[3])
        self.assertIn("--nodeps", calls[3])
        self.assertEqual(binaries.call_args_list[1].args[0], self.scratch / "rpm-root")

    def test_macos_installs_pkg_and_detaches_dmg_on_smoke_failure(self):
        self.artifacts(
            ["inferflux-1-Darwin.tar.gz", "inferflux-1.pkg", "inferflux-1.dmg"]
        )
        with patch.object(smoke, "run") as run, patch.object(
            smoke, "smoke_archive"
        ), patch.object(smoke.shutil, "copytree"), patch.object(
            smoke, "smoke_binaries", side_effect=[None, smoke.SmokeError("bad")]
        ):
            with self.assertRaises(smoke.SmokeError):
                smoke.smoke_macos(self.packages, self.scratch)
        self.assertEqual(run.call_args_list[0].args[:3], ("sudo", "installer", "-pkg"))
        self.assertEqual(run.call_args_list[-1].args[:2], ("hdiutil", "detach"))

    def test_windows_executes_msi_and_uninstalls_after_binary_failure(self):
        self.artifacts(["inferflux-1.zip", "inferflux-1.msi"])
        with patch.object(smoke, "run") as run, patch.object(
            smoke, "smoke_archive"
        ), patch.object(smoke, "smoke_binaries", side_effect=smoke.SmokeError("bad")):
            with self.assertRaises(smoke.SmokeError):
                smoke.smoke_windows(self.packages, self.scratch)
        self.assertEqual(run.call_args_list[0].args[:2], ("msiexec.exe", "/i"))
        self.assertEqual(run.call_args_list[-1].args[:2], ("msiexec.exe", "/x"))

    def test_workflow_smokes_before_every_package_upload(self):
        workflow = (ROOT / ".github/workflows/release.yml").read_text()
        self.assertNotIn("runs-on: ubuntu-latest", workflow)
        for folder in ("linux", "linux-arm64", "macos", "windows"):
            command = f"python scripts/smoke_release_packages.py --packages artifacts/{folder}"
            self.assertEqual(workflow.count(command + "\n"), 1)
            upload = workflow.index(f"path: artifacts/{folder}\n")
            self.assertLess(workflow.index(command), upload)


if __name__ == "__main__":
    unittest.main()
