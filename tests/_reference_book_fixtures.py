"""Shared source text for reference-book balance tests.

Keep these fixtures outside test modules so tests can reuse the same book shape
without creating a test-to-test import dependency.
"""

from __future__ import annotations

import pathlib
import subprocess

from ouroboros.reference_books import BOOK_ENTRYPOINTS

CHAPTER = "docs/architecture/01-one.md"
ENTRY = "# Architecture\n\nThe map.\n\n## Chapters\n\n- [One](architecture/01-one.md)\n"
BODY = "# One\n\nThe first chapter.\n\nP1 honest and long enough to shorten.\n"


def _git(root: pathlib.Path, *args: str) -> str:
    return subprocess.check_output(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                                   cwd=root).decode("utf-8").strip()


def _book_repo(root: pathlib.Path, *, official: bool = True) -> pathlib.Path:
    for path, text in {BOOK_ENTRYPOINTS["architecture"]: ENTRY, CHAPTER: BODY,
                       "ouroboros/reference_books.py": "# the book contract\n", "ouroboros/loop.py": "x = 1\n",
                       "notes.md": "notes\n"}.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text, encoding="utf-8", newline="\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    if official:
        _git(root, "remote", "add", "canonical", "https://github.com/razzant/ouroboros.git")
        _git(root, "update-ref", "refs/remotes/canonical/ouroboros", "HEAD")
    return root


def _registry(tmp_path: pathlib.Path, *, external: bool):
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    system = _book_repo(tmp_path / "system")
    drive = tmp_path / "drive"
    drive.mkdir()
    registry = ToolRegistry(repo_dir=system, drive_root=drive)
    if external:  # a user's own project that happens to carry the same docs layout
        project = _book_repo(tmp_path / "project")
        registry.set_context(ToolContext(repo_dir=system, system_repo_dir=system, drive_root=drive,
                                         workspace_root=project, workspace_mode="external"))
    return registry
