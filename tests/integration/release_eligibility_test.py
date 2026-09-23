"""Offline negative-path release authorization and manifest regressions."""

import base64
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "eligibility", ROOT / "scripts/release_eligibility.py"
)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
SHA = "a" * 40
MAIN = "b" * 40
REPO = "anvai-labs/inferflux"


class FakeAPI:
    def __init__(self):
        self.run = {
            "id": 10,
            "run_attempt": 1,
            "head_sha": SHA,
            "head_branch": "v0.3.0",
            "head_repository": {"full_name": REPO},
            "event": "push",
            "status": "completed",
            "conclusion": "success",
            "path": ".github/workflows/ci.yml",
        }
        self.source = {
            "event": "push",
            "sha": SHA,
            "repository": REPO,
            "run_id": "10",
            "run_attempt": "1",
            "ref": "refs/tags/v0.3.0",
        }
        self.gpu = {
            **self.run,
            "id": 20,
            "head_branch": "main",
            "path": ".github/workflows/gpu-gates.yml",
        }
        self.gpu_jobs = [
            {
                "name": name,
                "conclusion": "success",
                "steps": [{"name": step, "conclusion": "success"}] if step else [],
            }
            for name, step in gate.GPU_JOBS.items()
        ]
        self.artifacts = [
            {"name": f"{backend}-gate-{SHA}", "expired": False, "size_in_bytes": 100}
            for backend in ("cuda", "rocm")
        ]
        self.tag_object = {"type": "commit", "sha": SHA}
        self.tags = {}
        self.comparison = "ahead"
        self.cmake = "project(InferFlux VERSION 0.3.0 LANGUAGES CXX)"
        self.calls = []
        self.provenance_success = True

    def get(self, path):
        self.calls.append(path)
        if path == "/actions/runs/10":
            return self.run
        if path == "/git/ref/tags/v0.3.0":
            return {"object": self.tag_object}
        if path.startswith("/git/tags/"):
            return {"object": self.tags[path.rsplit("/", 1)[1]]}
        if path == "/git/ref/heads/main":
            return {"object": {"sha": MAIN}}
        if path == f"/compare/{SHA}...{MAIN}":
            return {"status": self.comparison}
        if path == f"/contents/CMakeLists.txt?ref={SHA}":
            return {"content": base64.b64encode(self.cmake.encode()).decode()}
        raise AssertionError(path)

    def pages(self, path, key):
        self.calls.append(path)
        if path == "/actions/runs/10/attempts/1/jobs":
            return [
                {
                    "name": "Release source provenance",
                    "conclusion": "success" if self.provenance_success else "skipped",
                }
            ]
        if path.startswith("/actions/workflows/gpu-gates.yml/runs?"):
            return [self.gpu]
        if path == "/actions/runs/20/attempts/1/jobs":
            return self.gpu_jobs
        if path == "/actions/runs/20/artifacts":
            return self.artifacts
        raise AssertionError(path)

    def provenance(self, _run):
        return self.source


class EligibilityTests(unittest.TestCase):
    def setUp(self):
        self.api = FakeAPI()

    def evaluate(self):
        return gate.evaluate(
            {"workflow_run": copy.deepcopy(self.api.run)}, REPO, self.api
        )

    def rejected(self, pattern):
        with self.assertRaisesRegex(gate.EligibilityError, pattern):
            self.evaluate()

    def test_real_tag_and_main_advance_are_allowed(self):
        result = self.evaluate()
        self.assertEqual(
            result,
            {
                "package": "true",
                "publish": "true",
                "sha": SHA,
                "tag": "v0.3.0",
                "version": "0.3.0",
            },
        )

    def test_annotated_tag_is_peeled(self):
        self.api.tag_object = {"type": "tag", "sha": "c" * 40}
        self.api.tags["c" * 40] = {"type": "commit", "sha": SHA}
        self.assertEqual(self.evaluate()["publish"], "true")

    def test_main_packages_without_gpu_evidence(self):
        self.api.run["head_branch"] = "main"
        self.api.source["ref"] = "refs/heads/main"
        result = self.evaluate()
        self.assertEqual(result["package"], "true")
        self.assertEqual(result["publish"], "false")
        self.assertFalse(any("gpu-gates" in path for path in self.api.calls))

    def test_pr_fork_failed_and_schedule_runs_never_enter_release(self):
        for key, value in (
            ("event", "pull_request"),
            ("event", "schedule"),
            ("conclusion", "failure"),
            ("head_repository", {"full_name": "attacker/inferflux"}),
        ):
            with self.subTest(key=key, value=value):
                self.api = FakeAPI()
                self.api.run[key] = value
                self.assertEqual(self.evaluate()["package"], "false")
                self.assertEqual(self.api.calls, [])

    def test_branch_named_like_real_tag_cannot_publish(self):
        self.api.source["ref"] = "refs/heads/v0.3.0"
        self.rejected("did not run for upstream")

    def test_missing_tag_fails_without_fallback(self):
        with patch.object(
            self.api,
            "get",
            side_effect=urllib.error.HTTPError("url", 404, "Not Found", {}, None),
        ):
            with self.assertRaises(urllib.error.HTTPError):
                gate.peeled_tag(self.api, "v0.3.0")

    def test_tag_cannot_point_at_another_revision(self):
        self.api.tag_object["sha"] = MAIN
        self.rejected("Tag does not resolve")

    def test_develop_only_sha_rejected_before_reading_candidate_metadata(self):
        for status in ("behind", "diverged"):
            with self.subTest(status=status):
                self.api.comparison = status
                self.rejected("not on main ancestry")
                self.assertFalse(any("/contents/" in path for path in self.api.calls))

    def test_version_and_branch_name_mismatch_fail(self):
        self.api.cmake = "project(InferFlux VERSION 0.2.0 LANGUAGES CXX)"
        self.rejected("version differ")
        self.api = FakeAPI()
        self.api.run["head_branch"] = "v0.4.0"
        self.rejected("provenance does not match")

    def test_four_component_cmake_version_is_not_a_three_component_release(self):
        self.api.cmake = "project(InferFlux VERSION 0.3.0.1 LANGUAGES CXX)"
        self.rejected("Invalid CMake project version")

    def test_commented_project_version_cannot_mask_actual_version(self):
        self.api.cmake = (
            "# project(InferFlux VERSION 0.3.0 LANGUAGES CXX)\n"
            "project(InferFlux VERSION 0.2.0 LANGUAGES CXX)"
        )
        self.rejected("version differ")

    def test_provenance_identity_and_old_attempt_fail(self):
        for key in ("event", "sha", "repository", "run_id", "run_attempt"):
            with self.subTest(key=key):
                self.api = FakeAPI()
                self.api.source[key] = "wrong"
                self.rejected("CI provenance mismatch")
        self.api = FakeAPI()
        self.api.provenance_success = False
        self.rejected("provenance job did not succeed")

    def test_disabled_gpu_and_skipped_runtime_or_model_step_fail(self):
        for name in gate.GPU_JOBS:
            with self.subTest(name=name):
                self.api = FakeAPI()
                next(job for job in self.api.gpu_jobs if job["name"] == name)[
                    "conclusion"
                ] = "skipped"
                self.rejected("dual-GPU evidence")
        self.api = FakeAPI()
        self.api.gpu_jobs[0]["steps"][0]["conclusion"] = "skipped"
        self.rejected("dual-GPU evidence")

    def test_gpu_wrong_sha_untrusted_ref_and_fork_fail(self):
        for key, value in (
            ("head_sha", MAIN),
            ("head_branch", "develop"),
            ("event", "pull_request"),
            ("head_repository", {"full_name": "fork/repo"}),
        ):
            with self.subTest(key=key):
                self.api = FakeAPI()
                self.api.gpu[key] = value
                self.rejected("dual-GPU evidence")

    def test_manual_main_gpu_evidence_is_valid(self):
        self.api.gpu["event"] = "workflow_dispatch"
        self.assertEqual(self.evaluate()["publish"], "true")

    def test_expired_missing_duplicate_and_wrong_sha_artifacts_fail(self):
        changes = (
            lambda items: items.pop(),
            lambda items: items[0].update(expired=True),
            lambda items: items[0].update(name="cuda-gate-" + MAIN),
            lambda items: items.append(dict(items[0])),
        )
        for change in changes:
            self.api = FakeAPI()
            change(self.api.artifacts)
            self.rejected("dual-GPU evidence")


class TransportTests(unittest.TestCase):
    def test_source_run_attempt_change_during_validation_fails(self):
        api = FakeAPI()
        event = {"workflow_run": copy.deepcopy(api.run)}
        api.run["run_attempt"] = 2
        with self.assertRaisesRegex(gate.EligibilityError, "Source run changed"):
            gate.evaluate(event, REPO, api)

    def test_pagination_does_not_drop_later_evidence(self):
        api = gate.GitHub(REPO)
        with patch.object(
            api, "get", side_effect=[{"jobs": list(range(100))}, {"jobs": [100]}]
        ) as get:
            self.assertEqual(
                list(api.pages("/jobs?filter=all", "jobs")), list(range(101))
            )
        self.assertEqual(get.call_args.args[0], "/jobs?filter=all&per_page=100&page=2")

    def test_artifact_is_from_exact_run_and_attempt(self):
        api = gate.GitHub(REPO)
        run = FakeAPI().run
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            archive.writestr("source.json", json.dumps(FakeAPI().source))
        artifact = {"id": 99, "name": f"ci-release-source-{SHA}-1", "expired": False}
        with patch.object(api, "pages", return_value=[artifact]) as pages, patch.object(
            api, "read", return_value=buf.getvalue()
        ):
            self.assertEqual(api.provenance(run)["sha"], SHA)
            self.assertEqual(pages.call_args.args[0], "/actions/runs/10/artifacts")
            run["run_attempt"] = 2
            with self.assertRaisesRegex(gate.EligibilityError, "provenance artifact"):
                api.provenance(run)

    def test_redirect_does_not_leak_api_token(self):
        request = urllib.request.Request(
            "https://api.github.com/artifact",
            headers={"Authorization": "Bearer secret"},
        )
        redirected = gate.SafeRedirect().redirect_request(
            request, None, 302, "found", {}, "https://blob.example/artifact"
        )
        self.assertIsNone(redirected.get_header("Authorization"))


class ManifestTests(unittest.TestCase):
    def test_tag_manifests_use_correct_repository_pinned_git_and_actual_msi_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packages = root / "packages"
            packages.mkdir()
            (packages / "inferflux-0.3.0-Windows-AMD64.msi").write_bytes(b"package")
            gate.render_manifests(REPO, SHA, "0.3.0", "v0.3.0", packages, root / "out")
            formula = (root / "out/homebrew/inferflux.rb").read_text()
            winget = (root / "out/winget/inferencial.inferflux.yaml").read_text()
            self.assertIn(
                f'url "https://github.com/{REPO}.git", using: :git, revision: "{SHA}"',
                formula,
            )
            self.assertNotIn("/archive/", formula)
            self.assertIn("-DBUILD_SHARED_LIBS=OFF", formula)
            self.assertIn("/inferfluxd --help", formula)
            self.assertNotIn("inferencial/InferFlux", formula + winget)
            self.assertIn(
                "releases/download/v0.3.0/inferflux-0.3.0-Windows-AMD64.msi", winget
            )
            self.assertIn(gate.hashlib.sha256(b"package").hexdigest(), winget)
            gate.render_manifests(REPO, SHA, "0.3.0", "", packages, root / "nightly")
            self.assertFalse((root / "nightly/winget").exists())

    def test_missing_or_duplicate_msi_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for count in (0, 2):
                for index in range(count):
                    (root / f"{index}.msi").write_bytes(b"package")
                with self.assertRaisesRegex(gate.EligibilityError, "exactly one"):
                    gate.render_manifests(
                        REPO, SHA, "0.3.0", "v0.3.0", root, root / "out"
                    )


class WorkflowWiringTests(unittest.TestCase):
    def test_only_validated_outputs_authorize_candidate_checkout_and_publication(self):
        workflow = (ROOT / ".github/workflows/release.yml").read_text()
        self.assertNotIn("startsWith", workflow)
        self.assertNotIn("inferencial/InferFlux", workflow)
        self.assertIn("ref: ${{ github.sha }}", workflow)
        self.assertIn("persist-credentials: false", workflow)
        self.assertEqual(
            workflow.count("RELEASE_REF: ${{ needs.eligibility.outputs.sha }}"), 5
        )
        self.assertIn("if: needs.eligibility.outputs.publish == 'true'", workflow)
        self.assertIn("tag_name: ${{ needs.eligibility.outputs.tag }}", workflow)
        self.assertIn(
            "target_commitish: ${{ needs.eligibility.outputs.sha }}", workflow
        )
        self.assertIn("actions: read", workflow)

    def test_publisher_rechecks_actual_tag_without_candidate_checkout(self):
        workflow = (ROOT / ".github/workflows/release.yml").read_text()
        publisher = workflow.split("  create-release:\n", 1)[1]
        self.assertNotIn("actions/checkout", publisher)
        check, publish = publisher.split("      - name: Create GitHub release", 1)
        check = check.split("      - name: Revalidate immutable release tag", 1)[1]
        self.assertNotIn("      - name:", check)
        self.assertIn("RELEASE_TAG: ${{ needs.eligibility.outputs.tag }}", check)
        self.assertIn("RELEASE_SHA: ${{ needs.eligibility.outputs.sha }}", check)
        self.assertIn("await github.rest.git.getRef", check)
        self.assertIn("ref: `tags/${tag}`", check)
        self.assertIn("await github.rest.git.getTag", check)
        self.assertIn("object.type !== 'commit' || object.sha !== expected", check)
        self.assertIn("throw new Error", check)
        self.assertNotIn("catch", check)
        self.assertIn("uses: softprops/action-gh-release", publish)

    def test_ci_records_actual_push_provenance_and_runs_offline_regressions(self):
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        self.assertIn("if: github.event_name == 'push'", workflow)
        self.assertIn(
            "ci-release-source-${{ github.sha }}-${{ github.run_attempt }}", workflow
        )
        self.assertIn("python3 tests/integration/release_eligibility_test.py", workflow)


if __name__ == "__main__":
    unittest.main()
