"""Regression tests for kill_process_tree: orphan reaping, and never signalling the caller's own group.

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
    monkeypatch.setattr(pl.os, "getpgrp", lambda: 1)  # the caller is in another group
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


@pytest.mark.parametrize("caller_group, group_killed", [(7777, False), (1234, True)])
def test_the_group_kill_skips_only_the_callers_own_group(monkeypatch, caller_group, group_killed):
    """A child left in the caller's group dies by PID; any other group is still killed whole (#2353)."""
    import signal as _signal
    import ouroboros.platform_layer as pl

    if pl.IS_WINDOWS:
        pytest.skip("POSIX process-group sweep path")
    killpg_calls, kill_calls = [], []
    monkeypatch.setattr(pl, "_process_children", lambda: {4242: [4101]})
    monkeypatch.setattr(pl.os, "getpgid", lambda pid: 7777)
    monkeypatch.setattr(pl.os, "getpgrp", lambda: caller_group)
    monkeypatch.setattr(pl.os, "killpg", lambda pgid, sig: killpg_calls.append((pgid, sig)))
    monkeypatch.setattr(pl.os, "kill", lambda pid, sig: kill_calls.append((pid, sig)))

    class _FakeProc:
        pid = 4242

    pl.kill_process_tree(_FakeProc())

    assert killpg_calls == ([(7777, _signal.SIGKILL)] if group_killed else [])
    assert {pid for pid, _ in kill_calls} >= {4242, 4101}


def test_a_child_in_the_callers_group_dies_and_the_caller_lives():
    """The #2353 shape on real processes: a caller in its own session kills a child it left in that group."""
    import subprocess
    import sys
    from pathlib import Path

    import ouroboros.platform_layer as pl

    if pl.IS_WINDOWS:
        pytest.skip("POSIX process groups")
    caller = (
        "import subprocess, sys\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})\n"
        "from ouroboros import platform_layer as pl\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "pl.kill_process_tree(child)\n"
        "print('child', child.wait(timeout=10), flush=True)\n"
    )
    done = subprocess.run([sys.executable, "-c", caller], start_new_session=True,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, (done.returncode, done.stdout, done.stderr)
    assert done.stdout.strip() == "child -9"


@pytest.mark.parametrize("caller_group, group_signalled", [(7777, False), (1234, True)])
def test_terminate_process_tree_spares_only_the_callers_own_group(monkeypatch, caller_group, group_signalled):
    """A child in the caller's group gets SIGTERM alone; any other group still gets it whole (#2353)."""
    import signal as _signal
    import ouroboros.platform_layer as pl

    if pl.IS_WINDOWS:
        pytest.skip("POSIX process groups")
    killpg_calls, terminated = [], []
    monkeypatch.setattr(pl.os, "getpgid", lambda pid: 7777)
    monkeypatch.setattr(pl.os, "getpgrp", lambda: caller_group)
    monkeypatch.setattr(pl.os, "killpg", lambda pgid, sig: killpg_calls.append((pgid, sig)))

    class _FakeProc:
        pid = 4242

        def terminate(self):
            terminated.append(self.pid)

    pl.terminate_process_tree(_FakeProc())

    assert killpg_calls == ([(7777, _signal.SIGTERM)] if group_signalled else [])
    assert terminated == ([] if group_signalled else [4242])


def test_the_local_runtime_install_leads_its_own_group(monkeypatch):
    """Stopping a llama-cpp-python install signals the install's own tree, never the server's group (#2353)."""
    import os
    import sys

    import ouroboros.platform_layer as pl
    from ouroboros import local_model

    if pl.IS_WINDOWS:
        pytest.skip("POSIX process groups")
    probe = "import os; print('leads its own group:', os.getpgid(0) == os.getpid())"
    monkeypatch.setattr(local_model, "_get_install_command", lambda: [sys.executable, "-c", probe])
    monkeypatch.setattr(local_model, "_get_install_env", lambda: dict(os.environ))
    manager = local_model.LocalModelManager()
    monkeypatch.setattr(manager, "check_runtime", lambda: True)

    manager._run_install()

    assert manager._runtime_status == "install_ok", manager._runtime_install_log
    assert "leads its own group: True" in manager._runtime_install_log
