#!/usr/bin/env python3

import argparse
import os
import subprocess
from pathlib import Path


def _git_text(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def source_revision(repo_root: Path) -> str:
    override = os.environ.get("TERRAE_SOURCE_REVISION", "").strip()
    if override:
        return override

    root = repo_root.expanduser().resolve()
    if not (root / ".git").exists():
        return "unknown"

    try:
        top_level = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve()
        if top_level != root:
            return "unknown"
        revision = _git_text(root, "rev-parse", "--short", "HEAD")
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return "unknown"
    return revision or "unknown"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print the Terrae source revision")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser.parse_args()


if __name__ == "__main__":
    print(source_revision(_parse_args().repo_root))
