"""Only an owner's Quit ends this installation's own Claudexor engine, after its starters.

Recycle, Restart, a crash, hiding, a lost window or indicator and a signal before the headless handler
keep the engine; a Quit stops it once, through a private manager, so a Panic accepted meanwhile still
finds no manager in the launcher and takes its immediate ledgered engine stop
(``server_control.execute_panic_stop``). Every Windows retirement drains its generation's Job under the
launcher's publication lock, so a concurrent or later Quit meets no starter; POSIX workers leave their
server's session, and their exit is disclosed as unconfirmed, never inferred. The native Windows
holder/rename witness is ``scripts/claudexor_quit_witness.py``.
"""
import atexit
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest

from ouroboros import claudexor_daemon, launcher_server_reaper as reaper
from tests.test_launcher_handoff_stop import native_guard as native_guard

_THREAD = threading.Thread


@pytest.fixture
def stops(monkeypatch):
    """Every engine stop, by the manager instance that made it; the singleton must stay absent."""
    calls = []
    monkeypatch.setattr(claudexor_daemon, "_MANAGER", None)
    monkeypatch.setattr(claudexor_daemon.OwnedClaudexorDaemon, "stop_outcome",
                        lambda self: calls.append(self) or "stopped")
    return calls


class _Job:
    """Kill-on-close Job semantics in miniature: members end some time after terminate or close."""

    def __init__(self, events, die_after):
        self.events, self.die_after, self.running = events, die_after, True

    def _end(self, cause):
        self.events.append(cause)
        if self.die_after is not None and self.running:
            threading.Timer(self.die_after, self._gone).start()

    def _gone(self):
        if self.running:
            self.running = False
            self.events.append("member gone")

    def terminate(self, job):
        self._end("terminate")
        return ""

    def close(self, job):
        self._end("close")
        return ""

    def query(self, job):
        return self.running


def test_without_quit_the_engine_is_never_stopped(stops):
    quit_ = reaper.QuitEngineStop()
    quit_.retire(None)  # Restart, recycle, crash: a POSIX generation has no Job
    assert quit_.stop_engine(server_stopped=True) == "not_requested"
    assert stops == []


@pytest.mark.parametrize("quitting", [False, True])
def test_every_windows_retirement_empties_its_job_before_closing_it(stops, monkeypatch, quitting):
    events = []
    job = _Job(events, die_after=0.05)
    quit_ = reaper.QuitEngineStop()
    quit_.windows = True
    if quitting:
        quit_.requested.set()
    monkeypatch.setattr(reaper._pl, "job_has_running_member", job.query)
    quit_.retire("job", terminate=job.terminate, close=job.close)
    assert events == ["terminate", "member gone", "close"] and not quit_._gaps
    assert quit_.stop_engine(server_stopped=True) == ("stopped" if quitting else "not_requested")
    assert quit_.disclosed == ([] if quitting else None)


@pytest.mark.parametrize("failure", ["never", "unreadable", "close"])
def test_an_unconfirmed_job_drain_still_closes_and_is_disclosed(stops, monkeypatch, caplog, failure):
    events = []
    job = _Job(events, die_after=None)
    quit_ = reaper.QuitEngineStop()
    quit_.windows, quit_.CONFIRM_SEC = True, 0.2
    if failure == "unreadable":
        monkeypatch.setattr(reaper._pl, "job_has_running_member", lambda job: 1 / 0)
    else:
        monkeypatch.setattr(reaper._pl, "job_has_running_member", job.query)
    close = (lambda handle: 1 / 0) if failure == "close" else job.close
    quit_.retire("job", terminate=job.terminate, close=close)
    assert events[0] == "terminate" and (failure == "close" or events[-1] == "close")
    assert quit_._gaps, "an unconfirmed drain is a gap, never a silent success"
    quit_.requested.set()
    with caplog.at_level(logging.INFO, logger=reaper.__name__):
        assert quit_.stop_engine(server_stopped=True) == "stopped"
    assert quit_.disclosed == quit_._gaps and "starters unconfirmed" in caplog.text
    assert "starters UNCONFIRMED" in caplog.text


def test_an_unconfirmed_drain_is_logged_when_it_happens_even_without_a_quit(stops, monkeypatch, caplog):
    """Restart and recycle retire generations too: their unconfirmed drain is disclosed at once, not only by a Quit
    that may never come; a repeated gap is disclosed once in that Quit's list."""
    job = _Job([], die_after=None)
    quit_ = reaper.QuitEngineStop()
    quit_.windows, quit_.CONFIRM_SEC = True, 0.1
    monkeypatch.setattr(reaper._pl, "job_has_running_member", job.query)
    with caplog.at_level(logging.WARNING, logger=reaper.__name__):
        quit_.retire("job", terminate=job.terminate, close=job.close)
        quit_.retire("job", terminate=job.terminate, close=job.close)
    assert caplog.text.count("did not confirm its exit") == 2 and "Generation Job retirement" in caplog.text
    assert quit_.stop_engine(server_stopped=True) == "not_requested" and stops == []
    assert quit_._gaps == ["a generation Job member did not confirm its exit"]


def test_an_engine_module_that_cannot_import_is_an_unconfirmed_stop_never_a_raise(monkeypatch, caplog):
    """The close body must still reach the PID lock's release: even the stop's own import fails into a disclosed
    ``unconfirmed`` outcome."""
    import builtins

    real_import = builtins.__import__

    def refusing(name, *args, **kwargs):
        if name == "ouroboros.claudexor_daemon":
            raise ImportError("engine module unavailable")
        return real_import(name, *args, **kwargs)

    quit_ = reaper.QuitEngineStop()
    quit_.requested.set()
    monkeypatch.setattr(builtins, "__import__", refusing)
    with caplog.at_level(logging.INFO, logger=reaper.__name__):
        assert quit_.stop_engine(server_stopped=True) == "unconfirmed"
    assert "custody is unconfirmed" in caplog.text


def test_a_generation_that_exited_by_panic_is_retired_without_waiting(windows_launcher, monkeypatch):
    """Panic's exit owns the stop: its retirement terminates and closes the Job, never polls it."""
    world = windows_launcher
    events = world.events
    job = _Job(events, die_after=None)  # a member that would never confirm its exit
    world.install(job)
    monkeypatch.setattr(reaper._pl, "job_has_running_member", lambda handle: pytest.fail("Panic waited on its Job"))
    world.proc.returncode = world.launcher.PANIC_EXIT_CODE
    started = time.monotonic()
    assert world.launcher._retire_agent(world.proc, "agent_exit") is True
    assert events == ["terminate", "close"] and time.monotonic() - started < 1


def test_quit_uses_a_private_manager_once_so_panic_keeps_its_immediate_ledger_stop(monkeypatch):
    """Negative control (evidence): the same probe against a ``get_owned_daemon()`` stop finds the singleton."""
    entered, release, calls = threading.Event(), threading.Event(), []

    def blocked(self):
        calls.append(self)
        entered.set()
        assert release.wait(5)
        return "stopped"

    monkeypatch.setattr(claudexor_daemon, "_MANAGER", None)
    monkeypatch.setattr(claudexor_daemon.OwnedClaudexorDaemon, "stop_outcome", blocked)
    quit_ = reaper.QuitEngineStop()
    quit_.requested.set()
    results = []
    first = threading.Thread(target=lambda: results.append(quit_.stop_engine(server_stopped=True)))
    second = threading.Thread(target=lambda: results.append(quit_.stop_engine(server_stopped=True)))
    first.start()
    try:
        assert entered.wait(5)
        second.start()
        # What execute_panic_stop reads to choose its immediate retained-daemon ledger request.
        assert claudexor_daemon.get_owned_daemon(create=False) is None
    finally:
        release.set()
        first.join(5)
        second.join(5)
    assert results == ["stopped", "stopped"] and len(calls) == 1, "one stop per launcher; a second caller waits"
    assert claudexor_daemon._MANAGER is None and calls[0] is not None


_SERVER = """
import subprocess, sys
worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
print(worker.pid, flush=True)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX: a worker leaves its server's session")
def test_a_posix_quit_never_reports_a_worker_in_its_own_session_as_gone(monkeypatch, caplog):
    """Regression: a product worker calls ``setsid`` (``supervisor/worker_process.py``), so its server's former
    process group is empty while it lives. The former stop waited on that group and logged the starters
    "confirmed gone" beside this live worker (restoring that wait fails this test). Now nothing
    is inferred from the group: the stop does not wait for it, and the worker's exit is disclosed unconfirmed."""
    server = subprocess.Popen([sys.executable, "-c", _SERVER], start_new_session=True, stdout=subprocess.PIPE,
                              text=True)
    worker = int(server.stdout.readline())
    server.wait(30)
    observed = []
    monkeypatch.setattr(claudexor_daemon, "_MANAGER", None)
    monkeypatch.setattr(claudexor_daemon.OwnedClaudexorDaemon, "stop_outcome",
                        lambda self: observed.append(_alive(worker))
                        or "stopped")
    try:
        assert os.getsid(worker) == worker != server.pid, "an independent session, as the product's worker"
        quit_ = reaper.QuitEngineStop()
        quit_.windows = False
        quit_.retire(None)  # what the launcher's POSIX retirement passes: no Job
        quit_.requested.set()
        started = time.monotonic()
        with caplog.at_level(logging.INFO, logger=reaper.__name__):
            assert quit_.stop_engine(server_stopped=True) == "stopped"
        assert time.monotonic() - started < 1, "no wait on a process group that proves nothing"
        assert observed == [True], "the worker was alive at the stop"
        assert quit_.disclosed == [reaper.POSIX_STARTERS] and reaper.POSIX_STARTERS in caplog.text
        assert "confirmed" not in caplog.text.replace("not confirmed", "")
    finally:
        if _alive(worker):
            os.kill(worker, signal.SIGKILL)  # this test's own grandchild, by its measured PID


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class _Exited:
    """A captured server that has already exited (its retirer reaped it)."""

    pid, returncode, stdin = 4242, 0, None

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0


@pytest.fixture
def windows_launcher(monkeypatch):
    """The real ``_retire_agent``/``stop_agent`` over a published generation whose Job is ``_Job``."""
    import launcher

    events, proc, quit_ = [], _Exited(), reaper.QuitEngineStop()
    quit_.windows = True
    for name, value in (("IS_WINDOWS", True), ("_quit", quit_), ("_agent_proc", proc), ("_agent_job", "job"),
                        ("_agent_start", None), ("_agent_stopping", None),
                        ("_cleanup_recorded_server_group_for_pid", lambda *a: True)):
        monkeypatch.setattr(launcher, name, value)
    monkeypatch.setattr(claudexor_daemon, "_MANAGER", None)
    monkeypatch.setattr(claudexor_daemon.OwnedClaudexorDaemon, "stop_outcome",
                        lambda self: events.append("engine stop") or "stopped")

    def install(job):
        monkeypatch.setattr(launcher, "terminate_job", job.terminate)
        monkeypatch.setattr(launcher, "close_job", job.close)
        monkeypatch.setattr(reaper._pl, "job_has_running_member", job.query)

    return SimpleNamespace(launcher=launcher, events=events, proc=proc, quit=quit_, install=install)


def test_a_quit_behind_a_concurrent_retirement_meets_no_terminating_starter(windows_launcher):
    """Regression: the lifecycle retires the generation while the closer's stop runs; the closer used to see an
    empty publication and stop the engine beside a Job member still terminating (no lifecycle join relied on)."""
    world = windows_launcher
    launcher, events = world.launcher, world.events
    entered, job = threading.Event(), _Job(events, die_after=0.3)
    terminate = job.terminate
    job.terminate = lambda handle: (entered.set(), terminate(handle))[1]
    world.install(job)
    lifecycle = threading.Thread(target=launcher._retire_agent, args=(world.proc, "agent_exit"))
    lifecycle.start()
    assert entered.wait(5)
    world.quit.requested.set()  # the close body, while the lifecycle thread is mid-retirement
    stopped = launcher.stop_agent()
    assert world.quit.stop_engine(server_stopped=stopped) == "stopped"
    lifecycle.join(5)
    assert stopped is True and world.quit.disclosed == []
    assert events.index("member gone") < events.index("close") < events.index("engine stop")


@pytest.mark.parametrize("die_after", [0.1, None])
def test_a_late_quit_after_an_ordinary_retirement(windows_launcher, caplog, die_after):
    world = windows_launcher
    events = world.events
    world.quit.CONFIRM_SEC = 0.3
    world.install(_Job(events, die_after=die_after))
    assert world.launcher._retire_agent(world.proc, "agent_exit") is True  # a crash or recycle, not a Quit
    world.quit.requested.set()
    with caplog.at_level(logging.INFO, logger=reaper.__name__):
        assert world.quit.stop_engine(server_stopped=world.launcher.stop_agent()) == "stopped"
    if die_after is not None:
        assert events == ["terminate", "member gone", "close", "engine stop"]
        assert world.quit.disclosed == []
    else:
        assert events[:2] == ["terminate", "close"] and events[-1] == "engine stop"
        assert world.quit.disclosed and "starters unconfirmed" in caplog.text


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows kill-on-close Job semantics")
@pytest.mark.parametrize("concurrent", [False, True])
def test_windows_retirement_returns_only_after_its_real_job_members_are_gone(monkeypatch, concurrent):
    import launcher
    from ouroboros import platform_layer as pl

    job = pl.create_kill_on_close_job(allow_breakaway=True)
    assert job is not None
    member = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                              **pl.subprocess_new_group_kwargs(suspended=True))
    server = subprocess.Popen([sys.executable, "-c", "pass"])
    server.wait(30)
    try:
        assert pl.assign_pid_to_job(job, member.pid) and pl.resume_process(member.pid)
        quit_ = reaper.QuitEngineStop()
        for name, value in (("_quit", quit_), ("_agent_proc", server), ("_agent_job", job), ("_agent_start", None),
                            ("_agent_stopping", None), ("_cleanup_recorded_server_group_for_pid", lambda *a: True)):
            monkeypatch.setattr(launcher, name, value)
        if concurrent:
            retirer = threading.Thread(target=launcher._retire_agent, args=(server, "agent_exit"))
            retirer.start()
            assert launcher.stop_agent() is True  # waits on the publication lock behind that retirement
        else:
            assert launcher._retire_agent(server, "agent_exit") is True
        assert member.poll() is not None, "the retirement returned beside a live Job member"
        assert not quit_._gaps
        if concurrent:
            retirer.join(30)
    finally:
        if member.poll() is None:
            member.kill()
        member.wait(30)


def _headless(monkeypatch, *, signal_first):
    import launcher

    events, quit_ = [], reaper.QuitEngineStop()
    monkeypatch.setattr(claudexor_daemon.OwnedClaudexorDaemon, "stop_outcome",
                        lambda self: events.append("engine stop") or "stopped")
    for name, value in (("_quit", quit_), ("_shutdown_event", threading.Event()), ("_external_ui", True),
                        ("_external_host_update", None), ("stop_agent", lambda: events.append("stop agent") or True),
                        ("_kill_orphaned_children", lambda reason: events.append(reason))):
        monkeypatch.setattr(launcher, name, value)
    lifecycle = threading.Thread(target=lambda: None)
    lifecycle.start()
    lifecycle.join()
    if signal_first:
        launcher._headless_signal_handler(15, None)
    with pytest.raises(SystemExit) as exited:
        launcher._run_headless_main("http://127.0.0.1:1", 1, lifecycle)
    return events, exited.value.code


def test_a_headless_signal_is_a_quit_that_ends_the_engine_after_its_server(monkeypatch, capsys):
    events, code = _headless(monkeypatch, signal_first=True)
    assert code == 0 and events == ["stop agent", "headless_shutdown", "engine stop"]


def test_a_crash_fuse_exit_keeps_the_engine(monkeypatch, capsys):
    events, code = _headless(monkeypatch, signal_first=False)
    assert code == 1 and events == ["stop agent", "crash_fuse"]


@pytest.fixture
def headless_main(native_guard, monkeypatch, tmp_path):
    """The real ``main(["--no-ui"])``: real signal registration, real ``stop_agent`` and record cleanup."""
    import launcher
    from ouroboros import platform_layer

    handlers, stopped = {}, []

    class Registry:
        """``platform_layer``'s own ``signal`` name: registrations are recorded, pytest's handlers untouched."""

        def signal(self, signum, handler):
            handlers[signum] = handler

        def __getattr__(self, name):
            return getattr(signal, name)

    monkeypatch.setattr(platform_layer, "signal", Registry())
    monkeypatch.setattr(atexit, "register", lambda *a: None)
    monkeypatch.setattr(claudexor_daemon, "_MANAGER", None)
    monkeypatch.setattr(claudexor_daemon.OwnedClaudexorDaemon, "stop_outcome",
                        lambda self: stopped.append(self) or "stopped")
    monkeypatch.setattr(launcher.threading, "Thread", lambda **kw: SimpleNamespace(
        start=lambda: None, join=lambda **kw: None, is_alive=lambda: False))
    for name in ("_headless", "_external_ui", "_external_host_update", "_external_seed_bundle", "_launch_argv",
                 "_launch_intent", "_runtime_venv", "EMBEDDED_PYTHON"):
        monkeypatch.setattr(launcher, name, getattr(launcher, name))
    for name, value in {
        "DATA_DIR": tmp_path, "_quit": reaper.QuitEngineStop(0.1), "_shutdown_event": threading.Event(),
        "_agent_proc": None, "_agent_job": None, "_agent_start": None, "_agent_stopping": None,
        "LAUNCHER_STOP_GRACE_SEC": 0.2, "_server_process_record_path": lambda: tmp_path / "server_process.json",
        "acquire_pid_lock": lambda: True, "automatic_launch_allowed": lambda *a: True,
        "remember_loaded_checkout": lambda *a, **kw: "", "check_git": lambda: True, "bootstrap_repo": lambda: True,
        "launcher_sources_changed": lambda *a, **kw: False, "_prepare_first_run_settings": lambda: ({}, False),
        "_requested_server_port": lambda: 0,
    }.items():
        monkeypatch.setattr(launcher, name, value)

    def accept():
        """The OS delivers the signal: only the registered handler runs, as in the signal frame."""
        handlers[signal.SIGINT if sys.platform == "win32" else signal.SIGTERM](signal.SIGTERM, None)

    return SimpleNamespace(launcher=launcher, handlers=handlers, stopped=stopped, accept=accept)


@pytest.mark.parametrize("starting", [False, True])
def test_real_main_routes_an_early_accepted_signal_to_one_cooperative_stop(headless_main, monkeypatch, caplog,
                                                                          starting):
    """proc=None ends the engine with nothing observed unconfirmed; a start still inside Popen past the grace
    does too, once, but the disclosure says its server's exit is unconfirmed: that start may still publish."""
    launcher = headless_main.launcher

    def awaited(port, abort_event, lifecycle_thread):
        if starting:
            launcher._agent_start = launcher._SPAWNING
        headless_main.accept()
        assert abort_event.is_set() and launcher._quit.requested.is_set()
        return False, port

    monkeypatch.setattr(launcher, "_await_server_ready", awaited)
    with caplog.at_level(logging.INFO), pytest.raises(SystemExit) as exited:
        launcher.main(["--no-ui"])
    assert exited.value.code == 0 and len(headless_main.stopped) == 1
    assert bool(set(launcher._quit.disclosed) - {reaper.POSIX_STARTERS}) is starting
    assert ("starters unconfirmed" in caplog.text) is starting
    assert claudexor_daemon._MANAGER is None


def test_real_main_never_reclassifies_an_interrupt_before_its_handler_as_quit(headless_main, monkeypatch):
    launcher = headless_main.launcher

    def interrupted():
        raise KeyboardInterrupt  # SIGINT's default disposition before the headless handler exists

    monkeypatch.setattr(launcher, "bootstrap_repo", interrupted)
    with pytest.raises(KeyboardInterrupt):
        launcher.main(["--no-ui"])
    assert headless_main.handlers == {} and headless_main.stopped == []
    assert not launcher._quit.requested.is_set()


def test_real_main_ends_the_engine_for_a_signal_accepted_during_a_startup_failure(headless_main, monkeypatch,
                                                                                 capsys):
    """The server never became ready and the signal arrives after main's shutdown check: still one stop."""
    launcher = headless_main.launcher
    monkeypatch.setattr(launcher, "_await_server_ready", lambda port, abort_event, lifecycle_thread: (False, port))
    monkeypatch.setattr(launcher, "stop_agent", lambda: (headless_main.accept(), True)[1])
    with pytest.raises(SystemExit) as exited:
        launcher.main(["--no-ui"])
    assert exited.value.code == 1 and len(headless_main.stopped) == 1


def test_real_main_startup_failure_without_quit_keeps_the_engine(headless_main, monkeypatch, capsys):
    launcher = headless_main.launcher
    monkeypatch.setattr(launcher, "_await_server_ready", lambda port, abort_event, lifecycle_thread: (False, port))
    with pytest.raises(SystemExit) as exited:
        launcher.main(["--no-ui"])
    assert exited.value.code == 1 and headless_main.stopped == []


class _Captured(Exception):
    pass


@pytest.fixture
def close_body(native_guard, monkeypatch, tmp_path):
    """main()'s real desktop close body, captured at its server wait, with no native effects."""
    import launcher
    from ouroboros import launcher_background as background

    events, owners = [], []
    monkeypatch.setattr(background, "indicator_class", lambda: None)
    monkeypatch.setattr(background.Background, "listen", lambda *a: None)
    monkeypatch.setattr("ouroboros.launcher_bootstrap.stop_bootstrap_commands", lambda **kw: [])
    monkeypatch.setattr(atexit, "register", lambda *a: None)
    monkeypatch.setattr(atexit, "unregister", lambda *a: None)
    monkeypatch.setitem(sys.modules, "webview", ModuleType("webview"))
    monkeypatch.setattr(claudexor_daemon.OwnedClaudexorDaemon, "stop_outcome",
                        lambda self: events.append("engine stop") or "stopped")
    monkeypatch.setattr(launcher.threading, "Thread", lambda **kw: SimpleNamespace(
        start=lambda: None, join=lambda **kw: None, is_alive=lambda: False))

    def owner(*args, **kwargs):
        owners.append(background.Background(*args, **kwargs))
        return owners[-1]

    def captured(*args):
        raise _Captured

    for name in ("_external_ui", "_external_host_update", "_external_seed_bundle", "_launch_argv", "_launch_intent",
                 "_runtime_venv", "EMBEDDED_PYTHON", "_webview_window"):
        monkeypatch.setattr(launcher, name, getattr(launcher, name))
    for name, value in {
        "IS_WINDOWS": False, "IS_MACOS": False, "_headless": False, "_quit": reaper.QuitEngineStop(),
        "_shutdown_event": threading.Event(), "DATA_DIR": tmp_path, "Background": owner,
        "_agent_proc": None, "_agent_job": None, "_agent_start": None, "_agent_stopping": None,
        "_detect_headless": lambda: None, "acquire_pid_lock": lambda: True,
        "release_pid_lock": lambda: events.append("release lock"), "automatic_launch_allowed": lambda *a: True,
        "stop_tray_before_exit": lambda release, **kw: release(), "remember_loaded_checkout": lambda *a, **kw: "",
        "check_git": lambda: True, "bootstrap_repo": lambda: True, "launcher_sources_changed": lambda *a, **kw: False,
        "_prepare_first_run_settings": lambda: ({}, False), "_requested_server_port": lambda: 0,
        "_await_server_ready": captured, "stop_agent": lambda: events.append("stop agent") or True,
        "_kill_orphaned_children": lambda: events.append("reap orphans"),
    }.items():
        monkeypatch.setattr(launcher, name, value)
    with pytest.raises(_Captured):
        launcher.main([])
    monkeypatch.setattr(launcher.threading, "Thread", _THREAD)  # the closer thread exit_launcher starts
    monkeypatch.setattr(launcher, "_webview_window", object())  # the main window exists: the ordinary close
    monkeypatch.setattr(background.os, "_exit", lambda code: events.append(("exit", code)))
    yield owners[0], events
    for thread in threading.enumerate():  # a closer still running must meet the fake exit, never the real one
        if thread.name == "ouroboros-quit":
            thread.join(timeout=10)
            if thread.is_alive():
                owners[0].panic.set()  # an accepted Panic owns the exit: the body never reaches os._exit
            assert not thread.is_alive(), "the close body never ended"


def _closed(events):
    deadline = time.monotonic() + 5
    while ("exit", 0) not in events:
        assert time.monotonic() < deadline, events
        time.sleep(0.01)
    return events


def test_the_desktop_close_body_ends_the_engine_after_starters_and_before_the_lock(close_body):
    background, events = close_body
    background.exit_launcher()  # a window close, the indicator's Quit, Cmd+Q or sign-out
    assert _closed(events) == ["stop agent", "reap orphans", "engine stop", "release lock", ("exit", 0)]


def test_panic_during_the_server_stop_owns_the_exit_without_an_engine_quit(close_body, monkeypatch):
    """An owner's Quit marked, then Panic accepted while ``stop_agent`` runs: Panic owns persistence and the exit,
    so the body starts no engine Quit, never releases the PID lock and never exits 0 over it."""
    import launcher

    background, events = close_body
    monkeypatch.setattr(launcher, "stop_agent", lambda: background.panic.set() or events.append("stop agent"))
    background.owner_exit.set()  # what exit_launcher(owner=True) marks before it starts this body
    background._close()
    assert launcher._quit.requested.is_set() and events == ["stop agent"]


def test_a_lost_window_or_indicator_exits_without_an_engine_quit(close_body):
    """``exit_launcher(owner=False)``: the indicator went away or the window could not be shown again."""
    background, events = close_body
    background.exit_launcher(owner=False)
    assert _closed(events) == ["stop agent", "reap orphans", "release lock", ("exit", 0)]


@pytest.mark.parametrize("first,second,stops", [
    ("lost", "owner", True),  # an owner's Quit arriving while a lost UI's exit is still stopping the server
    ("owner", "lost", True),  # a later UI loss never withdraws an owner's Quit
    ("lost", "lost", False),
])
def test_an_owner_quit_during_an_involuntary_exit_still_ends_the_engine(close_body, monkeypatch, first, second,
                                                                        stops):
    import launcher

    background, events = close_body
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(launcher, "stop_agent", lambda: (entered.set(), release.wait(5), events.append("stop agent"))
                        and True)
    background.exit_launcher(owner=first == "owner")
    assert entered.wait(5)
    if second == "owner":
        background.request_quit()  # the indicator's Quit, Cmd+Q: the body already runs, so it only marks it
    else:
        assert background.exit_launcher(owner=False) is False
    release.set()
    assert ("engine stop" in _closed(events)) is stops


def _quit_after_its_decision(monkeypatch, launcher, background, events):
    """The owner's Quit (the indicator's, Cmd+Q) lands just after the body decided Quit's engine stop: none."""
    decide = launcher._quit.stop_engine

    def decided(**kwargs):
        events.append(decide(**kwargs))
        if events.count("not_requested") == 1 and events[-1] == "not_requested":
            background.request_quit()  # the body still runs: this only marks it
        return events[-1]

    monkeypatch.setattr(launcher._quit, "stop_engine", decided)


def test_a_server_stop_left_unconfirmed_stays_disclosed_when_a_late_quit_decides_again(stops):
    quit_ = reaper.QuitEngineStop()
    quit_.windows = True
    assert quit_.stop_engine(server_stopped=False) == "not_requested"  # the involuntary exit's own decision
    quit_.requested.set()
    assert quit_.stop_engine(server_stopped=True) == "stopped" and len(stops) == 1
    assert quit_.disclosed == [reaper.SERVER_GAP], "the earlier unconfirmed server exit is not forgotten"


def test_an_owner_quit_after_an_involuntary_exits_decision_still_ends_the_engine_before_the_lock(close_body,
                                                                                               monkeypatch):
    """Mark and admission share the closer lock, so ``leave`` decides again before its exit: never a lost Quit."""
    import launcher

    background, events = close_body
    _quit_after_its_decision(monkeypatch, launcher, background, events)
    background.exit_launcher(owner=False)  # a lost indicator or a failed hide: no Quit yet
    assert _closed(events) == ["stop agent", "reap orphans", "not_requested", "engine stop", "stopped",
                               "release lock", ("exit", 0)]


def _tray_stop_runs(monkeypatch, background, events, during):
    """The icon's removal, the I/O between the body's last Quit check and its PID lock: ``during`` runs inside it."""
    from ouroboros import launcher_background

    monkeypatch.setattr(launcher_background, "_active", SimpleNamespace(
        stop=lambda wait: (events.append(("tray stop", wait)), events.count(("tray stop", wait)) == 1 and during())))


def test_an_owner_quit_while_the_icon_is_removed_still_ends_the_engine_before_the_lock(close_body, monkeypatch):
    """A lost indicator's exit: Cmd+Q or the window's close lands while the icon is being removed."""
    import launcher

    background, events = close_body
    decide = launcher._quit.stop_engine
    monkeypatch.setattr(launcher._quit, "stop_engine", lambda **kw: events.append(decide(**kw)) or events[-1])
    _tray_stop_runs(monkeypatch, background, events, background.request_quit)
    background.exit_launcher(owner=False)
    assert _closed(events) == ["stop agent", "reap orphans", "not_requested", ("tray stop", 0.5), "engine stop",
                               "stopped", "release lock", ("exit", 0)]


def test_an_owner_quit_during_an_involuntary_exits_decision_is_decided_while_the_icon_stays(close_body, monkeypatch):
    import launcher

    background, events = close_body
    _quit_after_its_decision(monkeypatch, launcher, background, events)
    _tray_stop_runs(monkeypatch, background, events, lambda: None)
    background.exit_launcher(owner=False)
    assert _closed(events) == ["stop agent", "reap orphans", "not_requested", "engine stop", "stopped",
                               ("tray stop", 0.5), "release lock", ("exit", 0)]


def test_a_panic_accepted_after_an_involuntary_exits_decision_leaves_the_owed_quit_undecided(close_body, monkeypatch):
    """A lost indicator's exit decided ``not_requested``; then the owner's Quit is marked and a Panic accepted before
    ``leave`` looks again. That owed decision would stop the engine, but Panic owns persistence and the exit: one
    decision, no engine stop, no exit(0). Negative control (evidence): with ``leave``'s ``while not
    self.panic.is_set()`` as ``while True`` this case decides again and stops the engine, and every other case here
    still passes."""
    import launcher
    from ouroboros import server_control
    from ouroboros.config import PANIC_EXIT_CODE

    background, events = close_body
    decide, panics = launcher._quit.stop_engine, []
    monkeypatch.setattr(server_control, "execute_panic_stop", lambda *a, **kw: panics.append(kw["panic_exit_code"]))
    monkeypatch.setattr("ouroboros.panic_source.record_source", lambda *a, **kw: None)

    def decided(**kwargs):
        events.append(decide(**kwargs))
        if events.count("not_requested") == 1 and events[-1] == "not_requested":
            background.request_quit()  # the body still runs: this only marks the owner's Quit, owed a decision
            background.request_panic()  # accepted before that decision is taken
        return events[-1]

    monkeypatch.setattr(launcher._quit, "stop_engine", decided)
    background.exit_launcher(owner=False)  # a lost indicator or a failed hide: no Quit yet
    deadline = time.monotonic() + 5
    while panics != [PANIC_EXIT_CODE]:  # the non-daemon Panic owner's stop
        assert time.monotonic() < deadline, (events, panics)
        time.sleep(0.01)
    for thread in threading.enumerate():
        if thread.name in ("ouroboros-quit", "ouroboros-indicator-panic"):
            thread.join(timeout=5)
    assert launcher._quit.requested.is_set(), "the owner's Quit was marked: a second decision would stop the engine"
    assert events == ["stop agent", "reap orphans", "not_requested", "release lock"], "one decision, no exit(0)"


def test_the_icon_and_its_panic_stay_through_a_quits_engine_stop(close_body, monkeypatch):
    """The engine stop (bounded, tens of seconds) runs before the icon's removal: an owner's Quit meanwhile is marked,
    never refused, and adds no second stop; the indicator's Panic is accepted at once, removes the icon without
    waiting and owns the exit."""
    from ouroboros import launcher_background, server_control
    from ouroboros.config import PANIC_EXIT_CODE

    background, events = close_body
    entered, release, panics = threading.Event(), threading.Event(), []
    monkeypatch.setattr(launcher_background, "_active", SimpleNamespace(stop=lambda wait: events.append(("tray stop", wait))))
    monkeypatch.setattr(claudexor_daemon.OwnedClaudexorDaemon, "stop_outcome",
                        lambda self: (events.append("engine stop"), entered.set(), release.wait(10)) and "stopped")
    monkeypatch.setattr(server_control, "execute_panic_stop", lambda *a, **kw: panics.append(kw["panic_exit_code"]))
    monkeypatch.setattr("ouroboros.panic_source.record_source", lambda *a, **kw: None)
    background.exit_launcher()
    try:
        assert entered.wait(5)
        assert events == ["stop agent", "reap orphans", "engine stop"], "the icon is still up during the stop"
        assert background.exit_launcher() is False and background._owner_late and not background._sealed
        started = time.monotonic()
        background.request_panic()
        assert time.monotonic() - started < 0.5 and background.panic.is_set() and events[-1] == ("tray stop", 0)
        deadline = time.monotonic() + 5
        while panics != [PANIC_EXIT_CODE]:  # the non-daemon Panic owner's stop, never behind the engine stop
            assert time.monotonic() < deadline, panics
            time.sleep(0.01)
    finally:
        release.set()
    for thread in threading.enumerate():
        if thread.name == "ouroboros-quit":
            thread.join(timeout=5)
    assert events == ["stop agent", "reap orphans", "engine stop", ("tray stop", 0), ("tray stop", 0.5),
                      "release lock"], "one stop, and no exit(0) over the accepted Panic"


def test_a_quit_after_the_exits_final_decision_is_refused_and_repeats_cannot_extend_it(close_body, monkeypatch):
    """The final decision closes admission in the same step: a later owner gesture is not taken as a Quit (the exit
    is already leaving), and owner callbacks repeating during every decision end it after the one real stop."""
    import launcher

    background, events = close_body
    decide, refused = launcher._quit.stop_engine, []

    def decided(**kwargs):
        events.append(decide(**kwargs))
        background.request_quit()  # an owner gesture again during every decision
        return events[-1]

    monkeypatch.setattr(launcher._quit, "stop_engine", decided)
    monkeypatch.setattr(launcher, "release_pid_lock", lambda: (events.append("release lock"), refused.append(
        (background.exit_launcher(), background._owner_late))))
    _tray_stop_runs(monkeypatch, background, events, lambda: None)
    background.exit_launcher(owner=False)
    assert _closed(events) == ["stop agent", "reap orphans", "not_requested", "engine stop", "stopped",
                               ("tray stop", 0.5), "release lock", ("exit", 0)]
    assert refused == [(False, False)], "after the final decision a gesture is neither marked nor owed"


def test_repeated_owner_exits_bound_the_final_decision_to_two_even_if_none_stops(close_body, monkeypatch):
    """The bound is structural, not borrowed from the stop: a decision that never stops and an owner gesture repeated
    during each one still leave after the second decision."""
    import launcher

    background, events = close_body
    refused = []
    monkeypatch.setattr(launcher._quit, "stop_engine", lambda **kw: (
        events.append("not_requested"), background.request_quit()) and "not_requested")
    monkeypatch.setattr(launcher, "release_pid_lock", lambda: (events.append("release lock"), refused.append(
        (background.exit_launcher(), background._owner_late))))
    _tray_stop_runs(monkeypatch, background, events, lambda: None)
    background.exit_launcher(owner=False)
    assert _closed(events)[events.index("reap orphans") + 1:] == ["not_requested", "not_requested",
                                                                  ("tray stop", 0.5), "release lock", ("exit", 0)]
    assert refused == [(False, False)], "the second decision closed admission too"


def test_an_owner_quit_after_a_startup_closes_decision_runs_that_body_again(close_body, monkeypatch):
    """A body that returns (a startup close) runs again for the owner's Quit it met; then a later quit may."""
    import launcher

    background, events = close_body
    monkeypatch.setattr(launcher, "_webview_window", None)  # Git, setup or boot: no main window yet
    monkeypatch.setattr(sys.modules["webview"], "windows", [], raising=False)
    _quit_after_its_decision(monkeypatch, launcher, background, events)
    background.exit_launcher(owner=False)
    for thread in threading.enumerate():
        if thread.name == "ouroboros-quit":
            thread.join(timeout=5)
    assert events == ["not_requested", "engine stop", "stopped"] and not background._closing


def test_recycle_and_restart_keep_the_engine(monkeypatch, stops):
    """The settings/onboarding recycle sends the same quit pipe; it is never a Quit."""
    import launcher

    quit_ = reaper.QuitEngineStop()
    proc = SimpleNamespace(pid=4242, stdin=None, poll=lambda: 0, terminate=lambda: None, wait=lambda **kw: 0)
    for name, value in (("_quit", quit_), ("_agent_proc", proc), ("_agent_job", None), ("_agent_start", None),
                        ("_agent_stopping", None), ("_agent_restart_requested", threading.Event()),
                        ("_cleanup_recorded_server_group_for_pid", lambda *a: True)):
        monkeypatch.setattr(launcher, name, value)
    launcher._request_agent_restart()
    assert launcher._agent_restart_requested.is_set() and not quit_.requested.is_set()
    assert quit_.stop_engine(server_stopped=True) == "not_requested" and stops == []


def test_a_real_panic_during_a_quit_stop_requests_the_ledgered_engine_at_once(monkeypatch, tmp_path):
    """The real ``execute_panic_stop`` while the Quit's own cooperative stop is still running.

    Its immediate ``retained-daemon`` request (``timeout_sec=0``) must not wait for the Quit stop's lock or its
    ``_stopping`` latch, and the Quit's manager is never marked ``_panic_requested``. Negative control (evidence):
    a Quit that stopped through ``get_owned_daemon()`` makes Panic skip that request.
    """
    from ouroboros import config, platform_layer, process_custody as custody, server_control

    for path in ("ouroboros.startup_historical_audit.audit.stop", "ouroboros.maintenance.child.stop",
                 "ouroboros.storage.count.stop", "ouroboros.storage_free.free.stop"):
        monkeypatch.setattr(path, lambda: None)
    for path in ("ouroboros.tools.shell.kill_all_tracked_subprocesses", "ouroboros.workspace_executor.kill_all_foreground",
                 "ouroboros.tools.services.kill_all_services", "ouroboros.extension_companion.panic_kill_all"):
        monkeypatch.setattr(path, lambda *a, **kw: [])
    monkeypatch.setattr("ouroboros.local_model.get_manager", lambda **kw: None)
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    monkeypatch.setattr("ouroboros.process_scope.panic_scope_census", lambda root: None)
    monkeypatch.setattr(server_control, "stop_restart_successor", lambda: [])
    monkeypatch.setattr(server_control, "_persist_panic_controls", lambda root: None)
    monkeypatch.setattr(server_control, "_restart_stop_requested", False)
    exited = threading.Event()
    monkeypatch.setattr(server_control.os, "_exit", lambda code: exited.set())
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(claudexor_daemon, "_MANAGER", None)
    monkeypatch.setattr(claudexor_daemon, "verify_owned_home", lambda **kw: "")
    row = {"pid": 444, "purpose": claudexor_daemon.CUSTODY_PURPOSE}
    entered, release, managers, ledger = threading.Event(), threading.Event(), [], []

    def classify(self, **kwargs):
        if threading.current_thread().name == "quit-closer":
            entered.set()
            assert release.wait(5)
        return None, claudexor_daemon._TRANSPORT_UNREACHABLE, ""

    original = claudexor_daemon.OwnedClaudexorDaemon.stop_outcome

    def recorded(self):
        managers.append((threading.current_thread().name, self))
        return original(self)

    def stop(root, purposes, *, unconfirmed=None, expected_entries=None, timeout_sec=None):
        ledger.append({"thread": threading.current_thread().name, "timeout": timeout_sec,
                       "quit_still_stopping": not release.is_set()})
        return [444]

    monkeypatch.setattr(claudexor_daemon.OwnedClaudexorDaemon, "_classify_liveness", classify)
    monkeypatch.setattr(claudexor_daemon.OwnedClaudexorDaemon, "_settle_exited_child", lambda self: None)
    monkeypatch.setattr(claudexor_daemon.OwnedClaudexorDaemon, "_terminate_child", lambda self: False)
    monkeypatch.setattr(claudexor_daemon.OwnedClaudexorDaemon, "stop_outcome", recorded)
    monkeypatch.setattr(custody, "process_stop_snapshot", lambda root, purposes: [row])
    monkeypatch.setattr(custody, "pending_process_stops", lambda root, purposes: [])
    monkeypatch.setattr(custody, "stop_ledgered_processes", stop)
    monkeypatch.setattr(platform_layer, "collect_descendant_pids", lambda pid: set())
    quit_ = reaper.QuitEngineStop()
    quit_.requested.set()
    closer = threading.Thread(target=quit_.stop_engine, kwargs={"server_stopped": True}, name="quit-closer")
    closer.start()
    before = set(threading.enumerate())
    try:
        assert entered.wait(5)
        server_control.execute_panic_stop(None, None, data_dir=tmp_path, panic_exit_code=99,
                                          log=SimpleNamespace(critical=lambda *a: None))
        assert exited.is_set()
        assert {"thread": "panic-retained-daemon", "timeout": 0, "quit_still_stopping": True} in ledger, ledger
    finally:
        release.set()
        closer.join(5)
        for thread in set(threading.enumerate()) - before:
            if thread.name.startswith("panic-"):
                thread.join(timeout=5)
    quit_manager = next(manager for name, manager in managers if name == "quit-closer")
    assert quit_manager._panic_requested is False, "no synthetic Panic latch on the Quit's own manager"
    assert claudexor_daemon._MANAGER is not quit_manager


def test_an_unreadable_lifecycle_thread_is_a_disclosed_gap_never_a_raise(stops, caplog):
    """The close body runs this before the PID lock's release: a broken join must not escape it."""
    quit_ = reaper.QuitEngineStop()
    quit_.lifecycle = SimpleNamespace(join=lambda **kw: None)  # no is_alive
    quit_.requested.set()
    with caplog.at_level(logging.INFO, logger=reaper.__name__):
        assert quit_.stop_engine(server_stopped=True) == "stopped"
    assert "the lifecycle thread unreadable (AttributeError)" in quit_.disclosed and len(stops) == 1
