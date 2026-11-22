"""Shared fixtures and helpers for the headless task/CLI suites.

Split out of ``tests/test_headless_cli.py`` when that module was divided by
theme: the shared worker-pool fixture, and the one-commit repository holding
``tracked.txt`` the suites that use this seed start from, now copied from a
world of the run. ``_managed_worker_pool_available`` is autouse, so importing it into a
test module re-applies it there.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests._workspace_git import git_repository, world_git
from tests._worlds import clone_repository, world


@pytest.fixture(autouse=True)
def _managed_worker_pool_available(monkeypatch):
    """HTTP task tests model a ready server unless a case overrides the pool."""
    import supervisor.workers as workers

    # An idle slot: the liveness readers (settlement probe, live ownership) read these facts.
    monkeypatch.setattr(workers, "WORKERS", {0: SimpleNamespace(busy_task_id=None, reaping=False)})
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")


def _lay_repo_with_file(folder):
    repo = git_repository(folder / "repo", unborn=True)
    (repo / "tracked.txt").write_text("old\n", encoding="utf-8")
    world_git(repo, "add", "tracked.txt")
    world_git(repo, "-c", "user.email=t@example.com", "-c", "user.name=T", "commit", "-m", "init")


def _init_repo_with_file(repo):
    """One commit ``init`` holding ``tracked.txt`` (``old``), copied from the run's world of it."""
    clone_repository(world("headless-cli-tracked-file", _lay_repo_with_file) / "repo", repo)
