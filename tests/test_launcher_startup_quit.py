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


@pytest.mark.parametrize("door", ["tray", "abort", "setup_tray"])
@pytest.mark.parametrize("identity", ["ours", "foreign", "unknown"])
def test_startup_quit_only_cleans_owned_or_identity_proven_server(native_guard, monkeypatch, tmp_path, door, identity):
    import launcher
    from ouroboros import launcher_background as background, launcher_bootstrap as boot, launcher_onboarding

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
                        else SimpleNamespace(start=lambda: None, join=lambda **kw: None))
    for name in ("_external_ui", "_external_host_update", "_external_seed_bundle", "_launch_argv", "_launch_intent",
                 "_runtime_venv", "EMBEDDED_PYTHON"):
        monkeypatch.setattr(launcher, name, getattr(launcher, name))
    for name, value in {
        "IS_WINDOWS": True, "IS_MACOS": False, "_headless": False, "_runtime_venv": None,
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
