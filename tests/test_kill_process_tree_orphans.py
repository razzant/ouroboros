"""Regression test for the kill_process_tree orphan-reaping fix.

A grandchild spawned in its own session/process group survives ``os.killpg`` of
the parent's group. ``kill_process_tree`` must collect descendants BEFORE
killing (parent death reparents children and loses the ppid links) and then
SIGKILL the escaped descendants by PID, or timed-out subprocess trees (for
example a pytest preflight run whose tests spawn children via
``subprocess_new_group_kwargs``) leak runaway orphan processes.
"""
import pytest


def test_kill_process_tree_sweeps_escaped_descendants(monkeypatch):
    import signal as _signal
    import ouroboros.platform_layer as pl

    if pl.IS_WINDOWS:
        pytest.skip("POSIX process-group sweep path")

    order = []
    escaped = [4101, 4102]
    killpg_calls = []
    kill_calls = []

    def fake_snapshot():
        order.append(("collect", 4242))
        return {4242: escaped}

    def fake_killpg(pgid, sig):
        order.append(("killpg", pgid))
        killpg_calls.append((pgid, sig))

    monkeypatch.setattr(pl, "_process_children", fake_snapshot)
    monkeypatch.setattr(pl.os, "getpgid", lambda pid: 7777)
    monkeypatch.setattr(pl.os, "killpg", fake_killpg)
    monkeypatch.setattr(pl.os, "kill", lambda pid, sig: kill_calls.append((pid, sig)))

    class _FakeProc:
        pid = 4242

    pl.kill_process_tree(_FakeProc())

    # Descendants collected BEFORE the group kill (reparenting can't hide them).
    assert order[0] == ("collect", 4242)
    assert order.index(("collect", 4242)) < order.index(("killpg", 7777))
    # Process group SIGKILLed.
    assert (7777, _signal.SIGKILL) in killpg_calls
    # Escaped descendants and the parent itself SIGKILLed by PID.
    killed = {pid for pid, _ in kill_calls}
    assert escaped[0] in killed and escaped[1] in killed
    assert 4242 in killed
    assert all(sig == _signal.SIGKILL for _, sig in kill_calls)
