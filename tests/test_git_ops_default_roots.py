"""D13 pin: supervisor/git_ops pre-init roots follow the OUROBOROS_* env.

The defect class (issue #455): ``git_ops.REPO_DIR`` / ``DRIVE_ROOT`` were
hardcoded ``~/Ouroboros`` defaults, resolved once and blind to the env, so a
fully env-isolated process still wrote ``managed_update_stash_restored
(context=test)`` into the LIVE ``data/logs/supervisor.jsonl`` through
``update_merge._log_supervisor``. Pre-init the roots now resolve PER CALL from
the env via the config path SSOT (module ``__getattr__``); ``init()``, worker
rebinds and test monkeypatches pin them as real module attributes.
"""

import os
import pathlib
import subprocess
import sys

import pytest

import supervisor.git_ops as git_ops
import supervisor.update_merge as update_merge

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

_UNSET = object()


def _unpin(request, name: str) -> None:
    """Remove a pinned root attribute for the test, restoring it afterwards.

    ``monkeypatch.delattr`` cannot do this: module ``__getattr__`` makes
    ``hasattr`` true while the attribute is absent from ``__dict__``.
    """
    saved = git_ops.__dict__.get(name, _UNSET)
    git_ops.__dict__.pop(name, None)

    def _restore():
        git_ops.__dict__.pop(name, None)
        if saved is not _UNSET:
            setattr(git_ops, name, saved)

    request.addfinalizer(_restore)


def test_unpinned_roots_follow_env_per_call(request, monkeypatch, tmp_path):
    _unpin(request, "DRIVE_ROOT")
    _unpin(request, "REPO_DIR")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "iso-data"))
    monkeypatch.setenv("OUROBOROS_REPO_DIR", str(tmp_path / "iso-repo"))
    assert git_ops.DRIVE_ROOT == tmp_path / "iso-data"
    assert git_ops.REPO_DIR == tmp_path / "iso-repo"
    # PER CALL: a later env change is followed, not cached.
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "late-data"))
    assert git_ops.DRIVE_ROOT == tmp_path / "late-data"
    assert git_ops.current_drive_root() == tmp_path / "late-data"


def test_pinned_roots_win_over_env(monkeypatch, tmp_path):
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "pinned")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "env-root"))
    assert git_ops.DRIVE_ROOT == tmp_path / "pinned"
    assert git_ops.current_drive_root() == tmp_path / "pinned"


def test_log_supervisor_writes_into_env_root_when_unpinned(request, monkeypatch, tmp_path):
    """The issue #455 writer path itself: _log_supervisor lands in the env root."""
    _unpin(request, "DRIVE_ROOT")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "iso"))
    update_merge._log_supervisor({"type": "hermetic_probe", "context": "test"})
    probe = tmp_path / "iso" / "logs" / "supervisor.jsonl"
    assert probe.is_file()
    assert "hermetic_probe" in probe.read_text(encoding="utf-8")


@pytest.mark.serial
def test_pre_isolation_import_still_writes_isolated_root(tmp_path):
    """Red on the pre-fix tree: all four OUROBOROS_* set, module imported, and
    the write STILL went to the (fake) home live root instead of the isolated
    one — the exact issue #455 reproduction, hermetic via a throwaway HOME."""
    fake_home = tmp_path / "home"
    iso = tmp_path / "iso"
    fake_home.mkdir()
    env = {**os.environ,
           "HOME": str(fake_home),
           "OUROBOROS_APP_ROOT": str(iso),
           "OUROBOROS_REPO_DIR": str(iso / "repo"),
           "OUROBOROS_DATA_DIR": str(iso / "data"),
           "OUROBOROS_SETTINGS_PATH": str(iso / "data" / "settings.json")}
    env.pop("OUROBOROS_PYTEST_ACTIVE", None)
    env.pop("OUROBOROS_TEST_LIVE_DATA_ROOT", None)
    script = (
        "import supervisor.update_merge as um\n"
        "um._log_supervisor({'type': 'hermetic_probe', 'context': 'test'})\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], cwd=str(REPO_ROOT), env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    live_shaped = fake_home / "Ouroboros" / "data"
    leaked = [str(p) for p in live_shaped.rglob("*") if p.is_file()] if live_shaped.exists() else []
    assert not leaked, f"write leaked into the live-shaped root: {leaked}"
    assert (iso / "data" / "logs" / "supervisor.jsonl").is_file()
