"""Offline release installer orchestration and fail-closed smoke regressions."""

import importlib.util
import io
from pathlib import Path
import shutil
import subprocess
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
        ), patch.object(smoke, "smoke_deb_binaries") as deb_binaries, patch.object(
            smoke, "smoke_binaries"
        ) as binaries:
            smoke.smoke_linux(self.packages, self.scratch)
        calls = [call.args for call in run.call_args_list]
        self.assertEqual(calls[0][:4], ("sudo", "apt-get", "install", "-y"))
        self.assertIn("--install", calls[3])
        self.assertIn("--nodeps", calls[3])
        deb_binaries.assert_called_once_with()
        self.assertEqual(binaries.call_args_list[0].args[0], self.scratch / "rpm-root")

    def test_deb_smokes_exact_package_owned_paths(self):
        for prefix in ("/usr", "/usr/local", "/opt/inferflux"):
            with self.subTest(prefix=prefix), patch.object(
                smoke,
                "run",
                return_value=f"{prefix}/bin/inferctl\n{prefix}/bin/inferfluxd\n",
            ) as run, patch.object(smoke, "smoke_binary_paths") as binaries:
                smoke.smoke_deb_binaries()
                run.assert_called_once_with("dpkg-query", "--listfiles", "inferflux")
                binaries.assert_called_once_with(
                    Path(prefix) / "bin/inferctl", Path(prefix) / "bin/inferfluxd"
                )

    def test_deb_missing_duplicate_or_relative_paths_fail(self):
        for listing in (
            "/usr/bin/inferctl\n",
            "/usr/bin/inferctl\n/opt/bin/inferctl\n/usr/bin/inferfluxd\n",
            "usr/bin/inferctl\n/usr/bin/inferfluxd\n",
        ):
            with self.subTest(listing=listing), patch.object(
                smoke, "run", return_value=listing
            ), patch.object(smoke, "smoke_binary_paths") as binaries:
                with self.assertRaises(smoke.SmokeError):
                    smoke.smoke_deb_binaries()
                binaries.assert_not_called()

    def test_deb_cleanup_runs_after_installed_binary_failure(self):
        self.artifacts([f"inferflux-1-Linux.{ext}" for ext in ("tar.gz", "deb", "rpm")])
        with patch.object(smoke, "run") as run, patch.object(
            smoke, "smoke_archive"
        ), patch.object(
            smoke, "smoke_deb_binaries", side_effect=smoke.SmokeError("bad")
        ):
            with self.assertRaises(smoke.SmokeError):
                smoke.smoke_linux(self.packages, self.scratch)
        self.assertEqual(
            run.call_args_list[-1].args,
            ("sudo", "apt-get", "remove", "-y", "inferflux"),
        )

    def test_pkg_distribution_requires_referenced_payload(self):
        valid = '<installer-gui-script><choices-outline><line choice="runtime"/></choices-outline><choice id="runtime"><pkg-ref id="ai.inferencial.inferflux.Unspecified"/></choice><pkg-ref id="ai.inferencial.inferflux.Unspecified">runtime.pkg</pkg-ref></installer-gui-script>'
        distribution = self.scratch / "Distribution"
        distribution.write_text(valid)
        smoke.validate_pkg_distribution(self.scratch)
        for invalid in (
            "<installer-gui-script><choices-outline/></installer-gui-script>",
            valid.replace('choice="runtime"', 'choice="missing"'),
            valid.replace(">runtime.pkg</pkg-ref>", "></pkg-ref>"),
            valid.replace(
                'id="ai.inferencial.inferflux.Unspecified"/>', 'id="missing"/>'
            ),
        ):
            distribution.write_text(invalid)
            with self.assertRaises(smoke.SmokeError):
                smoke.validate_pkg_distribution(self.scratch)

    @unittest.skipUnless(
        shutil.which("cmake"), "CMake is required for packaging config probe"
    )
    def test_productbuild_config_has_component_identifier_and_cli_prefix(self):
        probe = self.scratch / "probe.cmake"
        probe.write_text(f"""
include("{ROOT / 'cmake/CPackMacOS.cmake'}")
if(CPACK_GENERATOR STREQUAL "productbuild")
  if(NOT CPACK_COMPONENTS_ALL STREQUAL "Unspecified" OR
     NOT CPACK_PRODUCTBUILD_IDENTIFIER STREQUAL "ai.inferencial.inferflux" OR
     NOT CPACK_PACKAGING_INSTALL_PREFIX STREQUAL "/usr/local")
    message(FATAL_ERROR "Invalid CLI productbuild metadata")
  endif()
elseif(DEFINED CPACK_COMPONENTS_ALL OR DEFINED CPACK_PRODUCTBUILD_IDENTIFIER OR
       DEFINED CPACK_PACKAGING_INSTALL_PREFIX)
  message(FATAL_ERROR "Productbuild settings leaked to another generator")
endif()
""")
        for generator in ("productbuild", "TGZ", "DragNDrop", "DEB", "RPM", "WIX"):
            subprocess.run(
                ["cmake", f"-DCPACK_GENERATOR={generator}", "-P", str(probe)],
                check=True,
            )
        cmake = (ROOT / "CMakeLists.txt").read_text()
        self.assertIn(
            'if(APPLE)\n  set(CPACK_PROJECT_CONFIG_FILE\n      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPackMacOS.cmake")\nendif()\ninclude(CPack)',
            cmake,
        )

    def test_macos_installs_pkg_and_detaches_dmg_on_smoke_failure(self):
        self.artifacts(
            ["inferflux-1-Darwin.tar.gz", "inferflux-1.pkg", "inferflux-1.dmg"]
        )
        with patch.object(smoke, "run") as run, patch.object(
            smoke, "smoke_archive"
        ), patch.object(
            smoke, "validate_pkg_distribution"
        ) as distribution, patch.object(
            smoke.shutil, "copytree"
        ), patch.object(
            smoke, "smoke_binaries", side_effect=[None, smoke.SmokeError("bad")]
        ):
            with self.assertRaises(smoke.SmokeError):
                smoke.smoke_macos(self.packages, self.scratch)
        self.assertEqual(run.call_args_list[0].args[:2], ("pkgutil", "--expand"))
        distribution.assert_called_once_with(self.scratch / "pkg-expanded")
        self.assertEqual(run.call_args_list[1].args[:3], ("sudo", "installer", "-pkg"))
        self.assertEqual(run.call_args_list[-1].args[:2], ("hdiutil", "detach"))

    def test_invalid_pkg_stops_before_native_installer(self):
        self.artifacts(
            ["inferflux-1-Darwin.tar.gz", "inferflux-1.pkg", "inferflux-1.dmg"]
        )
        with patch.object(smoke, "run") as run, patch.object(
            smoke, "smoke_archive"
        ), patch.object(
            smoke, "validate_pkg_distribution", side_effect=smoke.SmokeError("empty")
        ):
            with self.assertRaises(smoke.SmokeError):
                smoke.smoke_macos(self.packages, self.scratch)
        self.assertEqual(
            [call.args[:2] for call in run.call_args_list], [("pkgutil", "--expand")]
        )

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
