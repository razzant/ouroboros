"""The wake clock's Panic reader: only a proven absence of ``state/panic_stop.flag`` permits a wake.

A dangling link, an unreadable flag or an absence no directory ancestor proves is a kept Panic,
the same unknown ``confirm_absent`` gives the other readers (``test_state_absence.py``).
Synthetic entries exercise these branches on Windows without symlink permission."""
from __future__ import annotations

import errno
import os
import pathlib
import stat
from types import SimpleNamespace

import pytest

from supervisor import state, state_initialization
from tests.test_state_authority import _prior, _windows_shaped_not_found, root as root

BARRED = (False, "panic_stop", "Background consciousness stays disabled while Panic controls await persistence.",
          "consciousness_disabled_or_unknown")
PERMITTED = (True, "not_due", "Background consciousness is already enabled.", "admission_failed")
LINK = os.stat_result((stat.S_IFLNK | 0o777, 0, 0, 1, 0, 0, 1, 0, 0, 0))


def _decisions(root):
    """The clock's constructor, ``tick`` and ``start``, then the lane's wake admission, on one root."""
    from ouroboros.consciousness import BackgroundConsciousness
    from supervisor import worker_chat_lane

    clock = BackgroundConsciousness(root, root, lambda: 7, now=0)
    enabled = clock.enabled
    return enabled, clock.tick(now=0), clock.start(), worker_chat_lane.handle_wake_direct(7, "wake", {})["reason"]


def _only_for(target, error, real):
    def call(candidate, *args, **kwargs):
        if pathlib.Path(candidate) == target:
            raise error
        return real(candidate, *args, **kwargs)
    return call


@pytest.mark.parametrize("errors", ["native", "windows"])
@pytest.mark.parametrize("flag_entry", ["dangling", "native_link", "unreadable", "unproven_ancestor"])
def test_every_wake_reader_needs_a_proven_absent_panic_flag(root, monkeypatch, flag_entry, errors):
    from supervisor import worker_chat_lane

    state.save_state(_prior())  # healthy: consciousness enabled, witness complete
    assert state_initialization.read_witness(root)[1]["phase"] == "complete"
    admitted: list = []
    monkeypatch.setattr(worker_chat_lane, "_pool", lambda: SimpleNamespace(DRIVE_ROOT=root))
    monkeypatch.setattr(worker_chat_lane, "wake_gate_refusal", lambda: "")
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 100)
    monkeypatch.setattr(worker_chat_lane, "_admit_chat_task", lambda *a, **kw: admitted.append(a))
    if errors == "windows":
        _windows_shaped_not_found(monkeypatch)
    flag = root / "state" / "panic_stop.flag"
    with monkeypatch.context() as entry:
        if flag_entry == "native_link":
            try:
                os.symlink(root / "state" / "gone", flag)
            except OSError as exc:
                pytest.skip(f"symlinks unavailable: {exc}")
        elif flag_entry == "dangling":  # every read and stat of the target still sees ENOENT
            real_lstat = os.lstat
            entry.setattr(os, "lstat", lambda p, *a, **k: LINK if pathlib.Path(p) == flag else real_lstat(p, *a, **k))
        elif flag_entry == "unreadable":
            denied = PermissionError(errno.EACCES, "cannot examine", str(flag))
            for owner, name in ((os, "lstat"), (os, "stat"), (pathlib.Path, "lstat"), (pathlib.Path, "stat")):
                entry.setattr(owner, name, _only_for(flag, denied, getattr(owner, name)))
        else:  # the flag's lookup says not-found, but its directory cannot be examined
            entry.setattr(os, "stat", _only_for(root / "state", OSError(errno.EIO, "cannot examine"), os.stat))
        assert state.control_is(state.load_state(), "bg_consciousness_enabled", True)
        assert _decisions(root) == BARRED and admitted == []
    if flag_entry == "native_link":
        flag.unlink()
    assert not os.path.lexists(flag)
    assert _decisions(root) == PERMITTED and len(admitted) == 1  # proven absence: the healthy control wakes


@pytest.mark.parametrize("errors", ["native", "windows"])
def test_a_file_where_the_state_directory_belongs_keeps_the_wake_barred(tmp_path, monkeypatch, errors):
    from ouroboros.consciousness import panic_blocks_wake

    if errors == "windows":
        _windows_shaped_not_found(monkeypatch)
    (tmp_path / "state").write_bytes(b"obstruction")
    assert panic_blocks_wake(tmp_path)
    assert (tmp_path / "state").read_bytes() == b"obstruction"
    (tmp_path / "state").unlink()
    assert not panic_blocks_wake(tmp_path)  # a genuinely missing directory
    (tmp_path / "state").mkdir()
    assert not panic_blocks_wake(tmp_path)  # a genuinely missing leaf below a directory
    (tmp_path / "state" / "panic_stop.flag").write_bytes(b"panic")
    assert panic_blocks_wake(tmp_path)
