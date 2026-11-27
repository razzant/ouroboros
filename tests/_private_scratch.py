"""Shared synthetic-HOME fixtures for the private in-place scratch tests (#4093); no real HOME, socket or temp family."""
from __future__ import annotations

import os
import pathlib

import pytest

from ouroboros import test_environment

REAL_PRIVATE_HOME = test_environment.private_scratch_home  # captured before any test's fixture replaces it


def qualifying_home(tmp_path, monkeypatch, *, limit: int = 4096) -> pathlib.Path:
    """A physical 0700 HOME below ``tmp_path`` that the real decision accepts. Doubled on purpose: what the OS keeps above
    ``tmp_path`` (the chain judgment, and the real OS-managed family ``tmp_path`` itself sits in) and the length budget
    (``limit``); the decision, the allocator and the files are real."""
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    home.chmod(0o700)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(test_environment, "private_scratch_home", REAL_PRIVATE_HOME)  # a suite fixture's None undone
    monkeypatch.setattr(test_environment, "SUN_PATH", limit)
    monkeypatch.setattr(test_environment, "unsafe_ancestor", lambda _path, **_kw: "")
    monkeypatch.setattr(test_environment, "ephemeral_ancestor", lambda _path: "")
    return home.resolve()


def private_children(home: pathlib.Path) -> list[str]:
    return sorted(entry.name for entry in os.scandir(home) if entry.name.startswith(test_environment.PRIVATE_SCRATCH_PREFIX))


@pytest.fixture(autouse=True)
def _synthetic_temp_parent(tmp_path, monkeypatch):
    """The shared synthetic temp parent of the scratch suites: every ordinary case stays below the safe_test-owned
    pytest root, and no real HOME is ever named. Imported (not redefined) by each suite that needs it."""
    parent = tmp_path / "temp"
    parent.mkdir()
    monkeypatch.setattr(test_environment, "session_scratch_parent", lambda: parent)
    monkeypatch.setattr(test_environment, "private_scratch_home", lambda: None)
