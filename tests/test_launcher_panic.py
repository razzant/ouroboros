"""Tray Panic races and installation custody, with all process/HTTP effects replaced."""
from __future__ import annotations

import threading
from types import ModuleType, SimpleNamespace

import pytest

from ouroboros import launcher_background as background
from tests.test_launcher_background import settings as settings  # explicit shared pytest fixture


@pytest.fixture(autouse=True)
def no_native_process_effects(monkeypatch):
    """Fake PIDs must never reach signals, a Windows handle or a subprocess."""
    import os
    import signal
    import subprocess
    import sys
    from ouroboros import platform_layer

    calls = []

    def refused(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("native process operation in a fake-PID test")

    for name in ("kill", "killpg", "_exit"):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, refused)
    if hasattr(signal, "pidfd_send_signal"):
        monkeypatch.setattr(signal, "pidfd_send_signal", refused)
    monkeypatch.setattr(subprocess.Popen, "__init__", refused)
    monkeypatch.setattr(platform_layer, "terminate_job", refused)
    winapi = ModuleType("_winapi")
    winapi.TerminateProcess = refused
    monkeypatch.setitem(sys.modules, "_winapi", winapi)
    # No bootstrap owner exists in this fixture, independently of prior tests.
    monkeypatch.setattr("ouroboros.launcher_bootstrap.stop_bootstrap_commands", lambda **kwargs: [])
    yield
    assert calls == [], f"unexpected process operations (all stubbed): {calls}"


def disarm_emergency_owners(monkeypatch, events):
    """Keep the real emergency coordinator and persistence; replace native stop effects."""
    from ouroboros import server_control

    monkeypatch.setattr(server_control, "_restart_stop_requested", False)
    for path in ("ouroboros.startup_historical_audit.audit.stop", "ouroboros.maintenance.child.stop",
                 "ouroboros.storage.count.stop", "ouroboros.storage_free.free.stop"):
        monkeypatch.setattr(path, lambda: None)
    for path in ("ouroboros.tools.shell.kill_all_tracked_subprocesses", "ouroboros.workspace_executor.kill_all_foreground",
                 "ouroboros.tools.services.kill_all_services", "ouroboros.extension_companion.panic_kill_all"):
        monkeypatch.setattr(path, lambda *a, **kw: [])
    monkeypatch.setattr("ouroboros.local_model.get_manager", lambda **kw: None)
    monkeypatch.setattr("ouroboros.claudexor_daemon.get_owned_daemon", lambda **kw: None if kw.get("create") is False
                        else SimpleNamespace(stop_outcome=lambda: "nothing_to_stop"))
    monkeypatch.setattr("ouroboros.process_custody.stop_ledgered_processes", lambda *a, **kw: [])
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    monkeypatch.setattr("ouroboros.process_scope.panic_scope_census", lambda root: events.append("census"))
    monkeypatch.setattr(server_control, "stop_restart_successor", lambda: [])
    monkeypatch.setattr(server_control, "_persist_panic_controls", lambda root: None)


@pytest.fixture
def stopped_world(monkeypatch, tmp_path):
    from ouroboros import config, server_control

    events, exited = [], threading.Event()
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(background, "indicator_class", lambda: None)
    monkeypatch.setattr(background, "_active", None)
    disarm_emergency_owners(monkeypatch, events)
    monkeypatch.setattr(server_control.os, "_exit", lambda code: (events.append(("exit", code)), exited.set()))
    bg = background.Background(lambda: (events.append("ordinary exit"), exited.set()), lambda: 1, threading.Event())
    yield bg, events, exited
    for thread in threading.enumerate():
        if thread.name.startswith(("panic-", "ouroboros-indicator-panic")):
            thread.join(timeout=5)
            assert not thread.is_alive()


def test_panic_latches_before_starting_any_post_thread(stopped_world, monkeypatch):
    bg, _, _ = stopped_world
    threads = []
    monkeypatch.setattr(background.threading, "Thread", lambda **kw: SimpleNamespace(start=lambda: threads.append(kw)))
    bg.request_panic()
    assert bg.shutdown.is_set() and bg.quit.is_set(), "click must stop admission before the persistence thread runs"
    assert len(threads) == 1 and threads[0]["daemon"] is False


def test_early_panic_never_posts_to_the_stale_or_default_port(stopped_world, monkeypatch):
    bg, events, exited = stopped_world
    posts = []

    class AcceptingForeignServer:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(background.urllib.request, "urlopen",
                        lambda request, **kw: posts.append(request.full_url) or AcceptingForeignServer())
    bg.request_panic()
    assert exited.wait(5), "a foreign server accepting /panic cannot keep this bootstrap alive"
    assert posts == [], "a port file/default is not this generation's endpoint authority"
    assert "census" in events and "ordinary exit" not in events


def test_early_panic_stops_retained_daemon_custody_and_its_work(stopped_world, monkeypatch, tmp_path):
    from ouroboros import claudexor_daemon as daemon, process_custody as custody, platform_layer

    bg, events, exited = stopped_world
    manager = daemon.OwnedClaudexorDaemon()
    retained = {"daemon": True, "work": True}
    written, settlement = [], threading.Event()
    row = {"pid": 444, "purpose": daemon.CUSTODY_PURPOSE}
    monkeypatch.setattr(daemon, "get_owned_daemon", lambda **kw: None if kw.get("create") is False else manager)
    monkeypatch.setattr(daemon, "verify_owned_home", lambda **kw: "")
    def classify(**kwargs):
        settlement.wait(3)  # asynchronous settlement must not be the first physical stop
        return None, daemon._TRANSPORT_UNREACHABLE, ""

    monkeypatch.setattr(manager, "_classify_liveness", classify)
    monkeypatch.setattr(manager, "_settle_exited_child", lambda: None)
    monkeypatch.setattr(manager, "_terminate_child", lambda: False)
    monkeypatch.setattr(custody, "process_stop_snapshot", lambda root, purposes: [row])
    monkeypatch.setattr(custody, "pending_process_stops", lambda root, purposes: [] if not any(retained.values()) else [row])
    monkeypatch.setattr(platform_layer, "collect_descendant_pids", lambda pid: {445})

    def stop(root, purposes, *, unconfirmed=None, expected_entries=None, timeout_sec=None):
        assert root == tmp_path and purposes == {daemon.CUSTODY_PURPOSE}
        assert expected_entries == [row] or timeout_sec == 0
        retained.update(daemon=False, work=False)
        events.append("retained custody stopped")
        return [444]

    monkeypatch.setattr(custody, "stop_ledgered_processes", stop)
    # Persistence is fast: a stop cannot rely on disk writes incidentally giving
    # the asynchronously started daemon settlement time to reach its kill.
    from ouroboros import server_control
    write_flag = server_control._write_panic_flag

    def persist(root):
        written.append(dict(retained))
        settlement.set()
        write_flag(root)

    monkeypatch.setattr(server_control, "_write_panic_flag", persist)
    bg.request_panic()
    assert exited.wait(5)
    assert retained == {"daemon": False, "work": False}, "ordinary launcher teardown deliberately retains these"
    assert written == [{"daemon": False, "work": False}], "request retained roots before persistence, not during async settlement"
    assert "retained custody stopped" in events and "census" in events and "ordinary exit" not in events
    assert (tmp_path / "state/panic_stop.flag").read_text() == "panic"


@pytest.mark.parametrize("windows,has_job", [(False, False), (True, False), (True, True)])
def test_local_early_panic_requests_the_captured_child_stop_before_persistence(
        stopped_world, monkeypatch, windows, has_job):
    from ouroboros import platform_layer, server_control

    bg, events, exited = stopped_world
    proc, job = SimpleNamespace(pid=41, poll=lambda: None), object() if has_job else None
    bg.panic_child = lambda: (proc, job)
    monkeypatch.setattr(platform_layer, "IS_WINDOWS", windows)
    monkeypatch.setattr(platform_layer, "kill_process_tree",
                        lambda value: events.append(("captured tree", value, None)))
    monkeypatch.setattr(platform_layer, "request_process_tree_kill",
                        lambda value, **kw: events.append(("native request", value, kw["job_handle"])))
    expected = ("native request" if windows else "captured tree", proc, job)
    write_flag = server_control._write_panic_flag

    def persist(root):
        assert events[0] == expected
        write_flag(root)

    monkeypatch.setattr(server_control, "_write_panic_flag", persist)
    bg.request_panic()
    assert exited.wait(5)
    assert events[0] == expected


def test_windows_job_error_still_requests_the_captured_process(stopped_world, monkeypatch):
    import sys
    from ouroboros import platform_layer

    bg, events, exited = stopped_world
    proc = SimpleNamespace(pid=41, poll=lambda: None, _handle=object())
    job = object()
    bg.panic_child = lambda: (proc, job)
    monkeypatch.setattr(platform_layer, "IS_WINDOWS", True)
    monkeypatch.setattr(platform_layer, "terminate_job",
                        lambda handle: events.append(("job", handle)) or "fixture Job failure")
    monkeypatch.setattr(sys.modules["_winapi"], "TerminateProcess",
                        lambda handle, code: events.append(("process", handle, code)))
    bg.request_panic()
    assert exited.wait(5)
    assert events[:2] == [("job", job), ("process", proc._handle, 1)]


def test_window_close_after_panic_keeps_the_local_owner_until_flag(settings, monkeypatch):
    """The actual main() close callback cannot replace the pending emergency owner."""
    import atexit
    import launcher
    import sys, types
    from ouroboros import server_control, platform_layer

    class Booted(Exception):
        pass

    owners, events = [], []
    waiting, finish = threading.Event(), threading.Event()
    flag = settings.parent / "state" / "panic_stop.flag"
    monkeypatch.setattr(launcher, "DATA_DIR", settings.parent)
    for name in ("_external_host_update", "_external_host_result", "_external_seed_bundle", "_launch_argv", "_launch_intent"):
        monkeypatch.setattr(launcher, name, getattr(launcher, name))
    original_background = background.Background

    def owner(*args, **kwargs):
        bg = original_background(*args, **kwargs)
        owners.append(bg)
        return bg

    proc = SimpleNamespace(pid=41, poll=lambda: None)
    monkeypatch.setattr(background, "indicator_class", lambda: None)
    monkeypatch.setattr(background.Background, "listen", lambda *args: None)
    monkeypatch.setattr(atexit, "register", lambda *args: None)
    monkeypatch.setattr(atexit, "unregister", lambda *args: None)
    monkeypatch.setitem(sys.modules, "webview", types.ModuleType("webview"))
    for name, value in (("IS_WINDOWS", False), ("_headless", False), ("_external_ui", False),
                        ("_detect_headless", lambda: None), ("_shutdown_event", threading.Event()),
                        ("_agent_proc", proc), ("_agent_job", object()), ("_webview_window", None),
                        ("acquire_pid_lock", lambda: True), ("automatic_launch_allowed", lambda *a: True),
                        ("Background", owner), ("check_git", lambda: (_ for _ in ()).throw(Booted())),
                        ("stop_agent", lambda: events.append("ordinary stop")),
                        ("_kill_orphaned_children", lambda: events.append("ordinary reap"))):
        monkeypatch.setattr(launcher, name, value)
    with pytest.raises(Booted):
        launcher.main([])
    bg = owners[0]
    def execute(*args, **kwargs):
        waiting.set()
        assert finish.wait(3), "test did not release the emergency owner"
        server_control._write_panic_flag(kwargs['data_dir'])
        events.append("local Panic persisted")

    monkeypatch.setattr(platform_layer, "request_process_tree_kill", lambda *a, **kw: {"requested": True})
    monkeypatch.setattr(platform_layer, "kill_process_tree", lambda *a, **kw: None)
    monkeypatch.setattr(server_control, "execute_panic_stop", execute)
    monkeypatch.setattr(background.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("no HTTP Panic"))
    api = background.DesktopApi()
    api._background = bg
    assert api.panic_stop() == {"status": "stopping"}
    try:
        assert waiting.wait(2)
        assert not flag.exists()
        assert bg.closing() is False  # the window stays; the quit's closer yields to the accepted Panic
        closer = next((t for t in threading.enumerate() if t.name == "ouroboros-quit"), None)
        if closer is not None:
            closer.join(timeout=3)
        thread = next(t for t in threading.enumerate() if t.name == "ouroboros-indicator-panic")
        assert thread.is_alive() and thread.daemon is False
        assert events == []
    finally:
        finish.set()
        thread = next((t for t in threading.enumerate() if t.name == "ouroboros-indicator-panic"), None)
        if thread is not None:
            thread.join(timeout=3)
    assert events == ["local Panic persisted"]
    assert flag.read_text() == "panic"


@pytest.mark.parametrize("surface", ["tray", "desktop_bridge"])
def test_delayed_owner_never_reads_a_successor_binding_or_stops_its_child(stopped_world, monkeypatch, surface):
    import launcher
    from ouroboros import platform_layer
    bg, events, exited = stopped_world
    first, successor = SimpleNamespace(pid=41, poll=lambda: None), SimpleNamespace(pid=42, poll=lambda: None)
    stops, pending = [], []
    bg.panic_child = lambda: (first, None)
    bg.read_port = lambda: pytest.fail("Panic cannot read a mutable port")
    monkeypatch.setattr(platform_layer, "request_process_tree_kill", lambda proc, **kw: stops.append(proc))
    monkeypatch.setattr(platform_layer, "kill_process_tree", lambda proc: stops.append(proc))
    monkeypatch.setattr(background.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("no successor HTTP"))
    with monkeypatch.context() as held:
        held.setattr(background.threading, "Thread", lambda **kw: SimpleNamespace(start=lambda: pending.append(kw)))
        if surface == "tray":
            bg.request_panic()
        else:
            api = background.DesktopApi()
            api._background = bg
            assert api.panic_stop() == {"status": "stopping"}
    assert bg.shutdown.is_set() and bg.panic.is_set() and bg.quit.is_set()
    assert stops == [first] and len(pending) == 1  # native stop precedes worker admission
    bg.panic_child = lambda: (successor, None)
    monkeypatch.setattr(launcher, "_shutdown_event", bg.shutdown)
    monkeypatch.setattr(launcher, "start_agent", lambda *a: pytest.fail("successor admission after Panic"))
    launcher.agent_lifecycle_loop()
    pending[0]["target"]()  # arbitrary scheduling delay, with the caller's mutable child replaced
    assert exited.wait(2) and stops == [first]
    assert "ordinary exit" not in events


@pytest.mark.parametrize("fault", ["thread", "source", "child", "bootstrap"])
def test_caller_faults_cannot_veto_panic(stopped_world, monkeypatch, fault):
    from ouroboros import panic_source, launcher_bootstrap
    bg, _, exited = stopped_world
    def refused(*a, **kw):
        raise RuntimeError("fixture refusal")
    if fault == "thread":
        thread = background.threading.Thread
        monkeypatch.setattr(background.threading, "Thread",
                            lambda **kw: refused() if kw.get("name") == "ouroboros-indicator-panic" else thread(**kw))
    elif fault == "source":
        monkeypatch.setattr(panic_source, "record_source", refused)
    elif fault == "child":
        bg.panic_child = refused
    else:
        monkeypatch.setattr(launcher_bootstrap, "stop_bootstrap_commands", refused)
    bg.request_panic()
    assert bg.shutdown.is_set() and exited.wait(3)


def test_stalled_source_writer_cannot_delay_actual_launcher_stop(stopped_world, monkeypatch):
    from ouroboros import utils

    bg, _, exited = stopped_world
    entered, release = threading.Event(), threading.Event()
    append = utils.append_jsonl

    def stalled(path, row, **kwargs):
        if threading.current_thread().name == "panic-source":
            entered.set()
            assert release.wait(5)
        return append(path, row, **kwargs)

    monkeypatch.setattr(utils, "append_jsonl", stalled)
    try:
        bg.request_panic()
        assert entered.wait(3)
        assert exited.wait(3), "diagnostic storage must not gate Panic persistence/exit"
    finally:
        release.set()


def test_tray_quit_closes_the_active_git_setup_loop(settings, monkeypatch):
    import atexit
    import sys
    import types
    import launcher
    owners, events = [], []
    original = background.Background

    def owner(*args, **kwargs):
        bg = original(*args, **kwargs)
        owners.append(bg)
        return bg

    webview = types.ModuleType('webview')
    webview.windows = []

    def destroy():
        events.append('setup destroyed')
        webview.windows.remove(window)

    window = SimpleNamespace(destroy=destroy)

    def create_window(*args, **kwargs):
        webview.windows.append(window)
        return window

    webview.create_window = create_window

    def run_setup(**kwargs):
        owners[0].request_quit()
        closer = next((t for t in threading.enumerate() if t.name == 'ouroboros-quit'), None)
        if closer is not None:
            closer.join(timeout=3)  # its body runs off this UI thread
        assert events == ['setup destroyed'], 'Quit must end the setup window, not just its icon'
        assert webview.windows == [], 'the real window registry must be emptied'

    webview.start = run_setup
    monkeypatch.setitem(sys.modules, 'webview', webview)
    monkeypatch.setattr(background, 'indicator_class', lambda: None)
    monkeypatch.setattr(background.Background, 'listen', lambda *args: None)
    monkeypatch.setattr(atexit, 'register', lambda *args: None)
    monkeypatch.setattr(atexit, 'unregister', lambda *args: None)
    for name in ('_external_host_update', '_external_host_result', '_external_seed_bundle', '_launch_argv', '_launch_intent'):
        monkeypatch.setattr(launcher, name, getattr(launcher, name))
    for name, value in (('IS_WINDOWS', False), ('_headless', False), ('_external_ui', False),
                        ('_detect_headless', lambda: None), ('_shutdown_event', threading.Event()),
                        ('_agent_proc', None), ('_agent_job', None), ('_webview_window', None),
                        ('DATA_DIR', settings.parent), ('acquire_pid_lock', lambda: True),
                        ('automatic_launch_allowed', lambda *a: True), ('Background', owner),
                        ('paint_room', lambda window: None),  # the macOS leaf; this fake window has no events
                        ('check_git', lambda: False), ('stop_agent', lambda: events.append('ordinary stop'))):
        monkeypatch.setattr(launcher, name, value)
    launcher.main([])
    assert events == ['setup destroyed'] and owners[0].shutdown.is_set()
