"""Shared fixtures and helpers for the headless task/CLI suites.

Split out of ``tests/test_headless_cli.py`` when that module was divided by
theme; the definitions are verbatim so every sibling suite keeps the exact
fixture semantics it was written against. ``_managed_worker_pool_available``
is autouse, so importing it into a test module re-applies it there.
"""
from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _managed_worker_pool_available(monkeypatch):
    """HTTP task tests model a ready server unless a case overrides the pool."""
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "WORKERS", {0: SimpleNamespace()})
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")


def _init_repo_with_file(repo, name="tracked.txt", content="old\n"):
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / name).write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", name], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=T", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
