import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_common import REPO_ROOT
from benchmark_standard import _source_revision


class BenchmarkEnvironmentTest(unittest.TestCase):
    def test_source_revision_prefers_environment_override(self) -> None:
        with patch.dict(os.environ, {"TERRAE_SOURCE_REVISION": "artifact-revision"}, clear=False):
            self.assertEqual(_source_revision(Path("/does/not/matter")), "artifact-revision")

    def test_source_revision_is_unknown_without_git_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"TERRAE_SOURCE_REVISION": ""}, clear=False):
                self.assertEqual(_source_revision(Path(directory)), "unknown")

    @unittest.skipUnless(shutil.which("git"), "Git is required for the nested-repository test")
    def test_source_revision_ignores_enclosing_git_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outer_repo = Path(directory)
            subprocess.run(
                ["git", "init", "--quiet", str(outer_repo)],
                check=True,
                capture_output=True,
                text=True,
            )
            artifact_root = outer_repo / "artifact"
            artifact_root.mkdir()
            with patch.dict(os.environ, {"TERRAE_SOURCE_REVISION": ""}, clear=False):
                self.assertEqual(_source_revision(artifact_root), "unknown")

    @unittest.skipUnless(
        shutil.which("git") and (REPO_ROOT / ".git").exists(),
        "A Git checkout is required for the project-repository test",
    )
    def test_source_revision_reads_project_git_root(self) -> None:
        expected = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        with patch.dict(os.environ, {"TERRAE_SOURCE_REVISION": ""}, clear=False):
            self.assertEqual(_source_revision(REPO_ROOT), expected)


if __name__ == "__main__":
    unittest.main()
