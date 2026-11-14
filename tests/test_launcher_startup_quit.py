"""Run main's early Quit/abort consumer with stale rows and no native effects."""
import atexit
import json
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from tests.test_launcher_handoff_stop import native_guard as native_guard


def _finish_quit():
    """The quit's close body runs on its own non-daemon thread; this test orders it before main continues."""
    for thread in threading.enumerate():
        if thread.name == "ouroboros-quit":
            thread.join(timeout=5)
            assert not thread.is_alive(), "the startup close body must return"


@pytest.mark.parametrize(("identity", "door"), [("ours", "tray"), ("foreign", "abort"), ("unknown", "setup_tray")])
def test_startup_quit_only_cleans_owned_or_identity_proven_server(native_guard, monkeypatch, tmp_path, door, identity):
    import launcher
    from ouroboros import claudexor_daemon, launcher_background as background, launcher_bootstrap as boot
    from ouroboros import launcher_onboarding, launcher_server_reaper

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    stale = {"old": {"pid": 41, "ports": [8123]}}
    (state_dir / "extension_companions.json").write_text(json.dumps(stale))
    record = state_dir / "server_process.json"
    record.write_text(json.dumps({"pid": 42, "pgid": 0}))
    events, owners = [], []
    shutdown = threading.Event()
    monkeypatch.setattr(background, "indicator_class", lambda: None)
    monkeypatch.setattr(background.Background, "listen", lambda *a: None)
    monkeypatch.setattr(boot, "stop_bootstrap_commands", lambda **kw: events.append("bootstrap stop") or [])
    monkeypatch.setattr(claudexor_daemon.OwnedClaudexorDaemon, "stop_outcome",
                        lambda self: events.append("engine stop") or "stopped")

    def owner(*args, **kwargs):
        instance = background.Background(*args, **kwargs)
        owners.append(instance)
        return instance

    def awaited(*args):
        if door == "setup_tray":
            return True, 8124
        if door == "tray":
            owners[0].request_quit()  # returns at once; its body stops the owned server off this thread
            _finish_quit()
        else:
            shutdown.set()
        return False, 0

    monkeypatch.setattr(atexit, "register", lambda *a: None)
    monkeypatch.setattr(atexit, "unregister", lambda *a: None)
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    webview = ModuleType("webview")
    webview.windows = []

    def create_window(title, **kwargs):
        assert title == "Ouroboros — Setup" and kwargs['url'] == 'http://127.0.0.1:8124/onboarding'
        assert launcher._webview_window is None  # setup owns this window, not the main shell
        def destroy():
            events.append("setup window destroyed")
            webview.windows.remove(window)
        window = SimpleNamespace(destroy=destroy)
        webview.windows.append(window)
        events.append("setup window open")
        return window

    def start(**kwargs):
        assert door == "setup_tray" and len(webview.windows) == 1
        assert kwargs == {'private_mode': False, 'menu': []}
        owners[0].request_quit()  # actual tray callback: it latches, and main's _on_closing runs off this thread
        assert shutdown.is_set() and owners[0].quit.is_set()
        _finish_quit()
        assert webview.windows == [], 'Quit must release setup webview.start without an owner close'
        assert shutdown.is_set() and owners[0].quit.is_set() and not owners[0].panic.is_set()
        events.append("setup loop returned")

    webview.create_window, webview.start = create_window, start
    monkeypatch.setitem(sys.modules, "webview", webview)
    monkeypatch.setattr(launcher_onboarding, 'NativeAppearance', lambda: SimpleNamespace(attach=lambda window: None))
    monkeypatch.setattr(launcher_onboarding, 'paint_room', lambda window: None)  # the macOS leaf; no events here
    monkeypatch.setattr(launcher_onboarding, 'setup_recovery_menu', lambda *a: [])
    real_thread = threading.Thread  # only the lifecycle thread is held back; the quit's closer really runs
    monkeypatch.setattr(launcher.threading, "Thread", lambda **kw: real_thread(**kw) if kw.get("name") == "ouroboros-quit"
                        else SimpleNamespace(start=lambda: None, join=lambda **kw: None, is_alive=lambda: False))
    for name in ("_external_ui", "_external_host_update", "_external_seed_bundle", "_launch_argv", "_launch_intent",
                 "_runtime_venv", "EMBEDDED_PYTHON"):
        monkeypatch.setattr(launcher, name, getattr(launcher, name))
    for name, value in {
        "IS_WINDOWS": True, "IS_MACOS": False, "_headless": False, "_runtime_venv": None,
        "_quit": launcher_server_reaper.QuitEngineStop(),
        "_agent_proc": SimpleNamespace(pid=42), "_webview_window": None, "_shutdown_event": shutdown,
        "DATA_DIR": tmp_path, "Background": owner, "_detect_headless": lambda: None,
        "_prepare_windows_webview_runtime": lambda: (True, ""), "acquire_pid_lock": lambda: True,
        "release_pid_lock": lambda: None, "automatic_launch_allowed": lambda *a: True,
        "stop_tray_before_exit": lambda release: release(),
        "remember_loaded_checkout": lambda *a, **kw: "", "check_git": lambda: True, "bootstrap_repo": lambda: True,
        "launcher_sources_changed": lambda *a, **kw: False,
        "_prepare_first_run_settings": lambda: ({}, door == "setup_tray"),
        "_requested_server_port": lambda: 0, "_await_server_ready": awaited,
        "stop_agent": lambda: events.append("owned server stop"), "_server_process_record_path": lambda: record,
        "_recorded_server_identity": lambda row: identity, "_retained_shared_daemon_pids": lambda: set(),
        "kill_pid_tree": lambda pid, **kw: events.append(("proven server", pid)), "_confirm_gone": lambda pid: True,
    }.items():
        monkeypatch.setattr(launcher, name, value)
    with pytest.raises(SystemExit) as exited:
        launcher.main([])
    assert exited.value.code == 0
    assert "owned server stop" in events
    # An owner's Quit during Git, setup or boot ends the owned engine once, after the server; an abort keeps it.
    # The close body's own stop, not main's later startup teardown, which then finds it decided.
    assert events.count("engine stop") == (door != "abort")
    servers = [index for index, event in enumerate(events) if event == "owned server stop"]
    assert door == "abort" or servers[0] < events.index("engine stop") < servers[1]
    assert [e for e in events if isinstance(e, tuple)] == ([("proven server", 42)] if identity == "ours" else [])
    assert record.exists() == (identity == "unknown")
    assert json.loads((state_dir / "extension_companions.json").read_text()) == stale
    if door in {"tray", "setup_tray"}:
        assert "bootstrap stop" in events
    if door == "setup_tray":
        assert [e for e in events if isinstance(e, str) and e.startswith('setup')] == [
            "setup window open", "setup window destroyed", "setup loop returned"]
    assert not (state_dir / 'panic_stop.flag').exists()


def test_true_panic_keeps_runtime_finalizer(native_guard, monkeypatch, tmp_path):
    import logging
    from ouroboros import launcher_server_reaper, platform_layer

    (tmp_path / "state").mkdir()
    (tmp_path / "state/extension_companions.json").write_text(json.dumps({"companion": {"pid": 41, "ports": [8123]}}))
    events = []
    monkeypatch.setattr(platform_layer, "force_kill_pid", lambda pid: events.append(("pid", pid)))
    monkeypatch.setattr(platform_layer, "kill_process_on_port", lambda port: events.append(("port", port)))
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    launcher_server_reaper.kill_runtime_orphans(tmp_path, lambda reason: events.append(reason),
                                               reason="panic_stop", log=logging.getLogger(__name__))
    assert events == ["panic_stop", ("pid", 41), ("port", 8123)]


@pytest.mark.parametrize(("window", "gesture", "kept", "stops"), [
    ("failed", "close", "", 1), ("failed", "close", "true", 0), ("failed", "panic", "", 0),
    ("failed", "tray_then_close", "true", 1), ("git", "close", "", 1), ("git", "close", "true", 0),
    ("headless", "none", "", 0)])
def test_closing_a_window_that_ends_startup_is_the_owners_quit(native_guard, monkeypatch, tmp_path, window, gesture,
                                                              kept, stops):
    """Closing the Startup Failed or Git-setup window ends the app: a Quit unless keep-running is saved, one engine
    stop after the server stop and before the PID lock's release; an accepted Panic decides nothing and waits for
    nothing; a startup failure with no owner gesture (headless, no signal) keeps the engine."""
    import launcher
    from ouroboros import claudexor_daemon, launcher_background as background, launcher_bootstrap as boot
    from ouroboros import launcher_server_reaper

    events, owners, finalizers, decisions = [], [], [], []
    shutdown = threading.Event()
    quit_ = launcher_server_reaper.QuitEngineStop()
    real_stop_engine = quit_.stop_engine
    quit_.stop_engine = lambda **kw: decisions.append(kw) or real_stop_engine(**kw)
    monkeypatch.setattr(background, "indicator_class", lambda: None)
    monkeypatch.setattr(background.Background, "listen", lambda *a: None)
    monkeypatch.setattr(background, "keep_running_choice", lambda: kept)
    monkeypatch.setattr(boot, "stop_bootstrap_commands", lambda **kw: [])
    monkeypatch.setattr(claudexor_daemon.OwnedClaudexorDaemon, "stop_outcome",
                        lambda self: events.append("engine stop") or "stopped")

    def owner(*args, **kwargs):
        owners.append(background.Background(*args, **kwargs))
        return owners[-1]

    monkeypatch.setattr(atexit, "register", finalizers.append)
    monkeypatch.setattr(atexit, "unregister", lambda *a: None)
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    webview = ModuleType("webview")
    webview.windows = []

    def create_window(title, **kwargs):
        assert title == {"failed": "Ouroboros — Startup Failed", "git": "Ouroboros — Setup Required"}[window]
        closed = SimpleNamespace(destroy=lambda: webview.windows.remove(closed))
        webview.windows.append(closed)
        return closed

    def start(**kwargs):  # the owner's gesture; the loop returns once the window is gone
        if gesture == "panic":
            owners[0].panic.set()  # accepted: its non-daemon owner persists and exits
        if gesture == "tray_then_close":
            owners[0].request_quit()
            _finish_quit()
        events.append("window closed")

    webview.create_window, webview.start = create_window, start
    monkeypatch.setitem(sys.modules, "webview", webview)
    real_thread = threading.Thread
    monkeypatch.setattr(launcher.threading, "Thread", lambda **kw: real_thread(**kw) if kw.get("name") == "ouroboros-quit"
                        else SimpleNamespace(start=lambda: None, join=lambda **kw: None, is_alive=lambda: False))
    for name in ("_external_ui", "_external_host_update", "_external_seed_bundle", "_launch_argv", "_launch_intent",
                 "_runtime_venv", "EMBEDDED_PYTHON"):
        monkeypatch.setattr(launcher, name, getattr(launcher, name))
    for name, value in {
        "IS_WINDOWS": True, "IS_MACOS": False, "_headless": window == "headless", "_quit": quit_,
        "_agent_proc": None, "_agent_start": None, "_webview_window": None, "_shutdown_event": shutdown,
        "DATA_DIR": tmp_path, "Background": owner, "_detect_headless": lambda: None,
        "install_shutdown_signal_handlers": lambda handler: None,
        "_prepare_windows_webview_runtime": lambda: (True, ""), "acquire_pid_lock": lambda: True,
        "release_pid_lock": lambda: events.append("lock released"), "automatic_launch_allowed": lambda *a: True,
        "stop_tray_before_exit": lambda release: release(), "paint_room": lambda window: None,
        "remember_loaded_checkout": lambda *a, **kw: "", "check_git": lambda: window != "git",
        "bootstrap_repo": lambda: True, "launcher_sources_changed": lambda *a, **kw: False,
        "_prepare_first_run_settings": lambda: ({}, False), "_requested_server_port": lambda: 0,
        "_await_server_ready": lambda *a: (False, 0), "_cleanup_recorded_server_process": lambda reason: False,
        "stop_agent": lambda: events.append("owned server stop") or True,
    }.items():
        monkeypatch.setattr(launcher, name, value)
    try:
        launcher.main([])
    except SystemExit as exited:
        assert exited.code == 1
    for finalizer in finalizers:  # atexit, after main
        finalizer()
    assert events.count("engine stop") == stops
    if stops:
        assert events.index("engine stop") < events.index("lock released")
        assert window == "git" or events.index("owned server stop") < events.index("engine stop")
    if gesture == "panic":
        assert decisions == [] and quit_.disclosed is None, "an accepted Panic owns the exit: no new decision or wait"
    assert "lock released" in events


class _Halt(Exception):
    """The flow reached the main window's boot: the setup window ended without any Quit."""


@pytest.mark.parametrize(("window", "gesture", "stops"), [
    ("git", "cmd_q", 1), ("setup", "cmd_q", 1), ("failed", "cmd_q", 1),
    ("git", "panic_then_cmd_q", 0), ("setup", "panic_then_cmd_q", 0), ("failed", "panic_then_cmd_q", 0),
    ("setup", "completed", 0)])
def test_a_macos_app_quit_over_a_window_before_the_main_one_is_the_owners_quit(native_guard, monkeypatch, tmp_path,
                                                                                window, gesture, stops):
    """Cmd+Q, Dock Quit or logout over Git setup, onboarding Setup or Startup Failed reaches AppKit's delegate,
    installed at that window's ``before_show``. It answers ``NSTerminateLater`` (a logout is never cancelled), and
    the closer's exit answers AppKit: server stop, one engine stop, PID lock, exit, with no window destroyed first.
    An accepted Panic decides nothing and waits for nothing; a completed setup (a programmatic destroy) stops none."""
    import launcher
    from ouroboros import claudexor_daemon, launcher_background as background, launcher_bootstrap as boot
    from ouroboros import launcher_onboarding, launcher_server_reaper
    from tests.test_launcher_background import NSApp, Window, appkit_doubles, fire

    app = NSApp()
    mac = appkit_doubles(monkeypatch, app)
    events, owners, finalizers, decisions, created = [], [], [], [], {}
    shutdown = threading.Event()
    quit_ = launcher_server_reaper.QuitEngineStop()
    real_stop_engine = quit_.stop_engine
    quit_.stop_engine = lambda **kw: decisions.append(kw) or real_stop_engine(**kw)
    monkeypatch.setattr(background, "indicator_class", lambda: mac.MacStatusItem)
    monkeypatch.setattr(background.Background, "listen", lambda *a: None)
    monkeypatch.setattr(background, "keep_running_choice", lambda: "true")  # a Quit is never a close's choice
    monkeypatch.setattr(background.os, "_exit", lambda code: events.append(("process exit", code)))
    monkeypatch.setattr(boot, "stop_bootstrap_commands", lambda **kw: [])
    monkeypatch.setattr(claudexor_daemon.OwnedClaudexorDaemon, "stop_outcome",
                        lambda self: events.append("engine stop") or "stopped")

    def owner(*args, **kwargs):
        owners.append(background.Background(*args, **kwargs))
        return owners[-1]

    monkeypatch.setattr(atexit, "register", finalizers.append)
    monkeypatch.setattr(atexit, "unregister", lambda *a: None)
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    webview = ModuleType("webview")
    webview.windows = []

    def create_window(title, **kwargs):
        assert title == {"git": "Ouroboros — Setup Required", "setup": "Ouroboros — Setup",
                         "failed": "Ouroboros — Startup Failed"}[window]
        made = Window()
        made.destroy = lambda: events.append("window destroyed") or webview.windows.remove(made)
        created.update(kwargs, window=made)
        webview.windows.append(made)
        return made

    def start(**kwargs):
        for shown in list(webview.windows):
            fire(shown, "before_show")  # pywebview: on the main thread, its own delegate already set
        if gesture == "completed":
            created["js_api"].onboarding_finished({"ok": True, "restart_required": True})
            return
        if gesture.startswith("panic"):
            owners[0].panic.set()  # accepted: its non-daemon owner persists and exits
        answer = app.delegate().applicationShouldTerminate_(app)
        if answer == 1:
            raise SystemExit("AppKit's YES ended the process: no teardown, no engine stop")
        assert answer == 2, "NSTerminateLater: AppKit (and a logout) waits for this process's exit, never cancelled"
        _finish_quit()  # the closer, on its own thread, while AppKit's main thread waits
        if not gesture.startswith("panic"):
            assert ("process exit", 0) in events, "the pending terminate is answered by the exit"
            raise SystemExit("process exited")  # natively nothing of main runs after the exit

    webview.create_window, webview.start = create_window, start
    monkeypatch.setitem(sys.modules, "webview", webview)
    monkeypatch.setattr(launcher_onboarding, 'NativeAppearance', lambda: SimpleNamespace(attach=lambda window: None))
    monkeypatch.setattr(launcher_onboarding, 'paint_room', lambda window: None)
    monkeypatch.setattr(launcher_onboarding, 'setup_recovery_menu', lambda *a: [])
    awaited = []

    def await_ready(*args):
        awaited.append(args)
        if len(awaited) > 1:
            raise _Halt()  # the recycle's wait after a completed setup: startup goes on
        return (window != "failed", 8124)

    real_thread = threading.Thread
    monkeypatch.setattr(launcher.threading, "Thread", lambda **kw: real_thread(**kw) if kw.get("name") == "ouroboros-quit"
                        else SimpleNamespace(start=lambda: None, join=lambda **kw: None, is_alive=lambda: False))
    for name in ("_external_ui", "_external_host_update", "_external_seed_bundle", "_launch_argv", "_launch_intent",
                 "_runtime_venv", "EMBEDDED_PYTHON"):
        monkeypatch.setattr(launcher, name, getattr(launcher, name))
    for name, value in {
        "IS_WINDOWS": False, "IS_MACOS": True, "_headless": False, "_quit": quit_,
        "_agent_proc": SimpleNamespace(pid=42) if window == "setup" else None, "_agent_start": None,
        "_webview_window": None, "_shutdown_event": shutdown, "DATA_DIR": tmp_path, "Background": owner,
        "_detect_headless": lambda: None, "acquire_pid_lock": lambda: True,
        "release_pid_lock": lambda: events.append("lock released"), "automatic_launch_allowed": lambda *a: True,
        "stop_tray_before_exit": lambda release: release(), "paint_room": lambda window: None,
        "remember_loaded_checkout": lambda *a, **kw: "", "check_git": lambda: window != "git",
        "bootstrap_repo": lambda: True, "launcher_sources_changed": lambda *a, **kw: False,
        "_prepare_first_run_settings": lambda: ({}, window == "setup"), "_requested_server_port": lambda: 0,
        "_await_server_ready": await_ready, "_cleanup_recorded_server_process": lambda reason: False,
        "_request_agent_restart": lambda: events.append("recycle"),
        "stop_agent": lambda: events.append("owned server stop") or True,
    }.items():
        monkeypatch.setattr(launcher, name, value)
    try:
        launcher.main([])
    except SystemExit as exited:
        assert exited.code in (0, 1, "process exited"), exited.code
    except _Halt:
        assert gesture == "completed" and events == ["window destroyed", "recycle"]  # the page's own destroy
    for finalizer in finalizers:  # atexit, after main
        finalizer()
    assert type(app.delegate()).__name__ == "OuroborosAppDelegate", "installed at the setup window's before_show"
    assert events.count("engine stop") == stops
    if stops:
        assert events.index("engine stop") < events.index("lock released") < events.index(("process exit", 0))
        assert window == "git" or events.index("owned server stop") < events.index("engine stop")
        assert "window destroyed" not in events[:events.index(("process exit", 0))], "none closes inside AppKit's wait"
    if gesture.startswith("panic"):
        assert not any(isinstance(event, tuple) for event in events), "Panic owns the exit: none from Quit"
        assert decisions == [] and quit_.disclosed is None, "an accepted Panic owns the exit: no new decision or wait"
    if gesture == "completed":
        assert not quit_.requested.is_set() and not owners[0].quit.is_set(), "a programmatic destroy is no Quit"
