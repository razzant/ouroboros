"""Launcher Quit against a server start in flight, and Quit against Panic.

``start_agent`` begins (refused after the shutdown latch), publishes the Popen it got, and on
Windows decides the resume against a stop under the agent lock; ``stop_agent`` shares one grace
between that publication and the drain and forces a never-resumed child without one. Quit latches
on the UI thread and runs its close body once, off it, so the indicator and its Panic stay until the
exit, which an accepted Panic overrides. POSIX cases use one real child this test starts; Windows
orderings use fake PIDs behind ``native_guard`` (no Job, CREATE_SUSPENDED or resume is native here).
"""
from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros.config import PANIC_EXIT_CODE
from tests.test_launcher_handoff_stop import native_guard as native_guard

pytestmark = pytest.mark.serial  # real children and the launcher's module-global agent owner

READS_QUIT = "import sys; sys.stdin.readline(); sys.exit(0)"  # the server's stdin watcher, nothing else
# The server's drain after its quit line; like ``server.executable_main`` it reports a caught signal as 128+N.
DRAINS = ("import signal, sys, time; caught = []; signal.signal(signal.SIGTERM, lambda *_: caught.append(1)); "
          "sys.stdin.readline(); time.sleep(3); sys.exit(143 if caught else 0)")


def _wait(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


@pytest.fixture
def agent(monkeypatch, tmp_path):
    """The real start/stop owners around an isolated root; ``popen`` replaces only the server argv."""
    import launcher

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "server.py").write_text("", encoding="utf-8")
    (tmp_path / "data" / "state").mkdir(parents=True)
    shutdown = threading.Event()
    for name, value in {
        "IS_WINDOWS": False, "DATA_DIR": tmp_path / "data", "REPO_DIR": repo, "EMBEDDED_PYTHON": sys.executable,
        "_load_settings": lambda: {}, "_apply_settings_to_env": lambda settings: None,
        "_shell_facts": lambda: {"bundle_version": ""},  # the loaded checkout's facts read Git, unrelated here
        "_shutdown_event": shutdown, "_agent_proc": None, "_agent_job": None, "_agent_start": None,
        "_retained_shared_daemon_pids": lambda: set(),
    }.items():
        monkeypatch.setattr(launcher, name, value)
    return SimpleNamespace(launcher=launcher, shutdown=shutdown, data=tmp_path / "data")


def _held_server(release, created, code=READS_QUIT):
    """A real child that ends on the launcher's ``quit`` line, returned only once ``release`` is set."""
    from ouroboros.platform_layer import subprocess_new_group_kwargs

    def popen(_cmd, **kwargs):
        proc = subprocess.Popen([sys.executable, "-c", code], stdin=kwargs["stdin"], stdout=kwargs["stdout"],
                                stderr=kwargs["stderr"], **subprocess_new_group_kwargs())
        created.append(proc)
        assert release.wait(20), "the test never released the publication"
        return proc

    return popen


def _reap(created):
    for proc in created:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)


def test_a_start_after_the_latch_is_refused_before_popen(agent, monkeypatch):
    launcher = agent.launcher
    monkeypatch.setattr(launcher, "_hidden_popen", lambda *a, **kw: pytest.fail("no server after the latch"))
    agent.shutdown.set()
    assert launcher.start_agent(port=1) is None
    assert launcher._agent_start is None and launcher._agent_proc is None
    assert launcher.stop_agent() is True, "nothing published and nothing starting: nothing to confirm"


@pytest.mark.skipif(sys.platform == "win32", reason="a POSIX child reading its stdin; Windows orderings below")
def test_quit_during_popen_waits_within_its_grace_and_stops_the_published_child(agent, monkeypatch):
    launcher = agent.launcher
    release, created, stopped = threading.Event(), [], []
    monkeypatch.setattr(launcher, "_hidden_popen", _held_server(release, created))
    starter = threading.Thread(target=launcher.start_agent, kwargs={"port": 1}, name="agent-start", daemon=True)
    try:
        starter.start()
        _wait(lambda: created and launcher._agent_start is launcher._SPAWNING)
        agent.shutdown.set()  # what Quit latches before its close body stops the agent
        began = time.monotonic()
        stop = threading.Thread(target=lambda: stopped.append(launcher.stop_agent()), name="quit-stop")
        stop.start()
        time.sleep(0.3)
        assert stop.is_alive() and launcher._agent_proc is None, "the stop waits for this start, never returns early"
        release.set()
        stop.join(15)
        starter.join(15)
        assert stopped == [True] and created[0].returncode == 0, "published, then asked to quit and confirmed"
        assert time.monotonic() - began < launcher.LAUNCHER_STOP_GRACE_SEC, "one grace, not a second one"
        assert launcher._agent_proc is None and launcher._agent_start is None
    finally:
        release.set()
        _reap(created)


@pytest.mark.skipif(sys.platform == "win32", reason="a POSIX child reading its stdin")
def test_a_start_still_inside_popen_at_the_grace_is_unconfirmed_and_stopped_once_published(agent, monkeypatch):
    launcher = agent.launcher
    monkeypatch.setattr(launcher, "LAUNCHER_STOP_GRACE_SEC", 0.3)
    release, created = threading.Event(), []
    monkeypatch.setattr(launcher, "_hidden_popen", _held_server(release, created))
    starter = threading.Thread(target=launcher.start_agent, kwargs={"port": 1}, name="agent-start", daemon=True)
    try:
        starter.start()
        _wait(lambda: created)
        agent.shutdown.set()
        assert launcher.stop_agent() is False, "a start still inside Popen at the grace is unconfirmed, never done"
        release.set()
        starter.join(15)
        assert launcher._agent_proc is created[0]
        from ouroboros.runtime_limits import LAUNCHER_STOP_GRACE_SEC

        monkeypatch.setattr(launcher, "LAUNCHER_STOP_GRACE_SEC", LAUNCHER_STOP_GRACE_SEC)
        assert launcher.stop_agent() is True  # the lifecycle loop's own stop after a late publication
        assert created[0].returncode == 0
    finally:
        release.set()
        _reap(created)


@pytest.mark.skipif(sys.platform == "win32", reason="a POSIX child draining after its quit line")
def test_the_loops_own_stop_after_a_quit_during_start_only_awaits_the_drain(agent, monkeypatch, caplog):
    """Native s1: the close body's stop waited for this start, asked the child to quit and closed its pipe;
    then the lifecycle loop's own ``stop_agent`` after ``start_agent`` met the closed pipe and sent SIGTERM
    into the drain. One request per child: the later stop only awaits that exit, holding no lock meanwhile."""
    launcher = agent.launcher
    release, created, stopped = threading.Event(), [], []
    monkeypatch.setattr(launcher, "_hidden_popen", _held_server(release, created, DRAINS))
    starter = threading.Thread(target=launcher.start_agent, kwargs={"port": 1}, name="agent-lifecycle", daemon=True)
    try:
        starter.start()
        _wait(lambda: created and launcher._agent_start is launcher._SPAWNING)
        agent.shutdown.set()
        began = time.monotonic()
        close = threading.Thread(target=lambda: stopped.append(launcher.stop_agent()), name="ouroboros-quit")
        close.start()
        release.set()
        starter.join(15)
        _wait(lambda: created[0].stdin.closed)  # the close body's quit is written and its pipe closed
        with caplog.at_level(logging.WARNING, logger="launcher"):
            loop = threading.Thread(target=lambda: stopped.append(launcher.stop_agent()), name="loop-stop")
            loop.start()
            time.sleep(0.3)
            assert created[0].poll() is None and launcher._agent_proc is created[0], "the drain still runs"
            assert launcher._agent_lock.acquire(timeout=1), "a waiting stop holds no lock Panic or the UI reads"
            launcher._agent_lock.release()
            loop.join(15)
            close.join(15)
        assert stopped == [True, True] and created[0].returncode == 0, "drained once, never signalled"
        assert "stop pipe unavailable" not in caplog.text
        assert time.monotonic() - began < launcher.LAUNCHER_STOP_GRACE_SEC, "one grace, not a second one"
        assert launcher._agent_proc is None and launcher._agent_stopping is None
    finally:
        release.set()
        _reap(created)


class _FakeServer:
    """A Windows server handle that records the launcher's requests; nothing native happens."""

    def __init__(self, calls, pid=4100):
        self.pid, self.calls, self.returncode = pid, calls, None
        self.stdout = SimpleNamespace(readline=lambda *_a: b"", read=lambda *_a: b"", close=lambda: None)
        self.stdin = SimpleNamespace(write=lambda data: calls.append(("write", data)),
                                     flush=lambda: None, close=lambda: calls.append("close stdin"))

    def wait(self, timeout=None):
        self.calls.append(("wait", timeout))
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self):
        self.calls.append("kill")


@pytest.fixture
def windows(agent, monkeypatch, native_guard):
    launcher, calls = agent.launcher, []
    monkeypatch.setattr(launcher, "IS_WINDOWS", True)
    monkeypatch.setattr(launcher, "subprocess_new_group_kwargs", lambda **kw: {})
    monkeypatch.setattr(launcher, "_write_server_process_record", lambda *a, **kw: calls.append("record"))
    monkeypatch.setattr(launcher, "_cleanup_recorded_server_group_for_pid", lambda pid, reason: calls.append(
        ("retire record", reason)) or True)
    monkeypatch.setattr(launcher, "kill_process_tree", lambda proc, **kw: calls.append("kill tree"))
    monkeypatch.setattr(launcher, "terminate_job", lambda job: calls.append(("terminate", job)))
    monkeypatch.setattr(launcher, "close_job", lambda job: calls.append(("close", job)))
    monkeypatch.setattr(launcher, "assign_pid_to_job", lambda job, pid: calls.append(("assign", job)) or True)
    monkeypatch.setattr(launcher, "resume_process", lambda pid: calls.append("resume") or True)
    monkeypatch.setattr(launcher, "create_kill_on_close_job", lambda **kw: calls.append("create job") or "job")
    monkeypatch.setattr("ouroboros.process_logging.copy_capped_output", lambda *a, **kw: None)
    server = _FakeServer(calls)
    monkeypatch.setattr(launcher, "_hidden_popen", lambda cmd, **kw: server)
    return SimpleNamespace(launcher=launcher, calls=calls, server=server, shutdown=agent.shutdown)


def test_windows_a_stop_before_the_resume_forces_the_suspended_child_without_a_drain(windows, monkeypatch):
    launcher, calls = windows.launcher, windows.calls
    stopped, decided = threading.Event(), []
    create = launcher.create_kill_on_close_job

    def job_after_the_stop(**kwargs):
        assert stopped.wait(10)
        return create(**kwargs)

    monkeypatch.setattr(launcher, "create_kill_on_close_job", job_after_the_stop)
    starter = threading.Thread(target=lambda: decided.append(launcher.start_agent(port=1)), daemon=True)
    starter.start()
    _wait(lambda: launcher._agent_start is windows.server)  # published, no resume committed
    windows.shutdown.set()
    assert launcher.stop_agent() is True
    stopped.set()
    starter.join(10)
    assert "resume" not in calls and ("write", b"quit\n") not in calls, "it ran no server code: nothing to drain"
    assert calls.index("kill tree") < calls.index(("wait", 5)), "forced at once, then its exit confirmed"
    assert calls[calls.index("create job"):] == ["create job", ("assign", "job"), ("terminate", "job"),
                                                 ("close", "job"), "kill"], "its private Job ends with it"
    assert decided == [windows.server] and launcher._agent_proc is None and launcher._agent_job is None


def test_windows_a_resume_committed_first_keeps_the_ordinary_drain(windows):
    launcher, calls = windows.launcher, windows.calls
    assert launcher.start_agent(port=1) is windows.server
    assert calls[:4] == ["create job", ("assign", "job"), "resume", "record"]
    assert launcher._agent_job == "job" and launcher._agent_start is None
    assert launcher.stop_agent() is True
    assert ("write", b"quit\n") in calls and ("wait", launcher.LAUNCHER_STOP_GRACE_SEC) in calls
    assert calls.count(("close", "job")) == 1 and "kill tree" not in calls


def test_windows_a_failed_resume_leaves_its_published_job_to_one_closer(windows, monkeypatch):
    launcher, calls = windows.launcher, windows.calls
    monkeypatch.setattr(launcher, "resume_process", lambda pid: calls.append("resume") or False)
    assert launcher.start_agent(port=1) is windows.server
    assert calls[-3:] == ["resume", ("terminate", "job"), "kill"]
    assert ("close", "job") not in calls and launcher._agent_job == "job", "published: its retirement closes it"
    launcher._retire_agent(windows.server, "agent_exit")
    assert calls.count(("close", "job")) == 1 and launcher._agent_job is None


def test_windows_without_a_job_the_child_never_runs_and_a_stop_has_nothing_to_drain(windows, monkeypatch):
    launcher, calls = windows.launcher, windows.calls
    monkeypatch.setattr(launcher, "create_kill_on_close_job", lambda **kw: calls.append("create job") or None)
    assert launcher.start_agent(port=1) is windows.server
    assert "resume" not in calls and calls[-1] == "kill" and launcher._agent_start is windows.server
    assert launcher.stop_agent() is True
    assert ("write", b"quit\n") not in calls and ("wait", 5) in calls


def test_a_recycle_stop_never_closes_the_next_generations_admission(windows):
    launcher = windows.launcher
    assert launcher.start_agent(port=1) is windows.server
    assert launcher.stop_agent() is True and not windows.shutdown.is_set()
    windows.server.returncode = None
    assert launcher.start_agent(port=2) is windows.server, "only the shutdown latch refuses a start"


# Quit and Panic on the launcher's Background owner.
@pytest.fixture
def owner(monkeypatch):
    from ouroboros import launcher_background as lb
    from ouroboros import launcher_bootstrap, server_control

    monkeypatch.setattr(launcher_bootstrap, "source_handoff", launcher_bootstrap._SourceHandoff())  # Panic claims it
    monkeypatch.setattr(lb, "indicator_class", lambda: None)
    monkeypatch.setattr(lb, "_active", None)
    exits, panics = [], []
    monkeypatch.setattr(lb.os, "_exit", lambda code: exits.append(code))
    monkeypatch.setattr(server_control, "execute_panic_stop", lambda *a, **kw: panics.append(kw["panic_exit_code"]))
    monkeypatch.setattr("ouroboros.panic_source.record_source", lambda *a, **kw: None)
    monkeypatch.setattr("ouroboros.launcher_bootstrap.stop_bootstrap_commands", lambda **kw: [])
    body = SimpleNamespace(release=threading.Event(), threads=[], runs=0)

    def close_body():
        body.runs += 1
        body.threads.append(threading.current_thread())
        assert body.release.wait(10)
        background.leave(lambda: body.threads.append("released lock"))

    background = lb.Background(close_body, lambda: 1, threading.Event())
    yield SimpleNamespace(background=background, body=body, exits=exits, panics=panics)
    body.release.set()
    for thread in threading.enumerate():
        if thread.name in ("ouroboros-quit", "ouroboros-indicator-panic"):
            thread.join(timeout=5)


def test_quit_latches_at_once_and_runs_its_body_once_off_the_calling_thread(owner):
    background, body = owner.background, owner.body
    started = time.monotonic()
    background.request_quit()
    assert time.monotonic() - started < 0.5 and background.shutdown.is_set() and background.quit.is_set()
    _wait(lambda: body.runs == 1)
    assert body.threads[0] is not threading.current_thread() and body.threads[0].daemon is False
    background.request_quit()
    assert background.closing() is False and background.exit_launcher() is False
    assert body.runs == 1, "a repeated quit while the body runs starts no second one"
    body.release.set()
    _wait(lambda: owner.exits == [0])
    _wait(lambda: not background._closing)
    body.release.clear()
    background.exit_launcher()  # a body that returned (here after a stubbed exit) may run again
    _wait(lambda: body.runs == 2)


def test_panic_on_the_same_ui_thread_during_a_blocked_quit_owns_the_exit(owner):
    background, body = owner.background, owner.body
    background.request_quit()
    _wait(lambda: body.runs == 1)
    started = time.monotonic()
    background.request_panic()  # the same UI thread that took the Quit: it was never blocked by it
    assert time.monotonic() - started < 0.5 and background.panic.is_set()
    _wait(lambda: owner.panics == [PANIC_EXIT_CODE])
    body.release.set()
    _wait(lambda: "released lock" in body.threads)
    _wait(lambda: not background._closing)
    assert owner.exits == [], "an ordinary exit(0) never overtakes an accepted Panic"


def test_an_exit_in_progress_and_a_panic_are_decided_under_one_lock(owner, monkeypatch):
    from ouroboros import launcher_background as lb

    background, inside, finish = owner.background, threading.Event(), threading.Event()

    def exiting(code):
        inside.set()
        assert finish.wait(10)
        owner.exits.append(code)

    monkeypatch.setattr(lb.os, "_exit", exiting)
    leaving = threading.Thread(target=background.leave, args=(lambda: None,), daemon=True)
    leaving.start()
    assert inside.wait(5)
    panic = threading.Thread(target=background.request_panic, daemon=True)
    panic.start()
    time.sleep(0.2)
    assert not background.panic.is_set(), "a Panic cannot be accepted between the exit's check and the exit"
    finish.set()
    leaving.join(5)
    panic.join(5)
    assert owner.exits == [0] and background.panic.is_set()


def test_windows_tray_quit_keeps_its_icon_and_panic_until_the_exit_removes_it(monkeypatch):
    from ouroboros import launcher_background as lb
    from ouroboros import launcher_tray
    from tests.test_desktop_notifications import StaForms

    forms = StaForms(monkeypatch)
    items = []
    forms_module = sys.modules["System.Windows.Forms"]
    menu_item = forms_module.ToolStripMenuItem

    def recorded(text=""):
        item = menu_item(text)
        items.append(item)
        return item

    monkeypatch.setattr(forms_module, "ToolStripMenuItem", recorded)
    monkeypatch.setattr(lb, "_active", None)
    events = []
    background = SimpleNamespace(shutdown=threading.Event(), window=None, opened=[], exit_launcher=lambda: None)
    background.request_quit = lambda: (background.shutdown.set(), events.append("quit"))
    background.request_panic = lambda: (events.append("panic"), lb.request_tray_cleanup())
    background.show_window = lambda: None
    background.open_notification = lambda token: None
    tray = launcher_tray.WindowsTray(background)
    background._indicator_up = lambda: tray.start()
    try:
        assert tray.start()
        _wait(tray.ready.is_set)
        lb._active = tray
        by_text = {item.Text: item for item in items}
        forms.post(tray._icon.sta, lambda: by_text["Quit Ouroboros"].Click.fire(None))
        _wait(lambda: events == ["quit"])
        forms.settle()  # timer ticks after the shutdown latch
        icon = tray._icon
        assert icon is not None and icon.Visible and not icon.disposed, "Quit keeps the icon and its Panic"
        forms.post(icon.sta, lambda: by_text["Panic"].Click.fire(None))
        _wait(lambda: events == ["quit", "panic"])
        _wait(lambda: icon.disposed and not icon.Visible)  # Panic's cleanup request removes it on the STA
    finally:
        tray.stop(wait=2.0)
        forms.shutdown()
