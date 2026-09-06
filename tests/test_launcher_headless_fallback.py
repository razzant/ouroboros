"""Linux headless (no-GUI-backend) launcher browser-fallback tests (#56).

The real pywebview is NOT required: every test fakes the ``webview`` module or
the probe result (CI quick-test has no pywebview installed — same precedent as
test_launcher_sync.py / test_packaged_runtime_and_lifecycle.py which import
``launcher`` directly). All mutations of launcher module state go through
``monkeypatch`` so the tests are parallel-safe and leave no residue.
"""

import sys
import types

import pytest


class _FakeEvent:
    """Deterministic threading.Event stand-in: never blocks, records set()."""

    def __init__(self, initially_set: bool = False):
        self._set = initially_set
        self.waits: list = []

    def is_set(self) -> bool:
        return self._set

    def set(self) -> None:
        self._set = True

    def wait(self, timeout=None) -> bool:
        # The keep-alive loop MUST pass a timeout: an untimed wait() would park
        # forever on a live event and never reach the lifecycle-liveness check.
        assert timeout is not None, "keep-alive wait() must carry a timeout"
        self.waits.append(timeout)
        return self._set


def _install_fake_webview(monkeypatch, *, init_raises,
                          backend_module="webview.platforms.gtk",
                          renderer="webkit2"):
    """Install a fake ``webview`` + ``webview.guilib`` into sys.modules.

    The real ``guilib.initialize()`` RETURNS the chosen backend MODULE — the
    module's ``__name__`` identifies the toolkit (webview.platforms.gtk),
    while ``renderer`` names the web ENGINE and is version-drifting (round
    c11: pinned 5.4 says 'gtkwebkit2', other versions say 'webkit2'), so the
    probe keys on module identity and the fake models the representative
    engine-only renderer value."""
    fake_webview = types.ModuleType("webview")
    fake_guilib = types.ModuleType("webview.guilib")
    fake_backend = types.ModuleType(backend_module)
    fake_backend.renderer = renderer
    calls = []

    def initialize(*args, **kwargs):
        calls.append((args, kwargs))
        if init_raises is not None:
            raise init_raises
        return fake_backend

    fake_guilib.initialize = initialize
    fake_webview.guilib = fake_guilib
    monkeypatch.setitem(sys.modules, "webview", fake_webview)
    monkeypatch.setitem(sys.modules, "webview.guilib", fake_guilib)
    return calls


def _stub_headless_teardown(monkeypatch, launcher, order):
    """Recording stubs for everything _run_headless_main touches.

    ``_open_browser_detached`` is stubbed SYNCHRONOUSLY (not via the
    ``webbrowser`` module): the real helper rides a daemon thread, so a
    module-level patch would make the recorded order timing-dependent and
    could even hit the real browser module after monkeypatch teardown."""
    monkeypatch.setattr(launcher, "stop_agent", lambda: order.append("stop_agent"))
    monkeypatch.setattr(
        launcher,
        "_kill_orphaned_children",
        lambda port, reason="window_close": order.append(("kill_orphans", port, reason)),
    )
    monkeypatch.setattr(
        launcher, "release_pid_lock", lambda: order.append("release_pid_lock")
    )
    monkeypatch.setattr(
        launcher,
        "_open_browser_detached",
        lambda url: order.append(("browser", url)),
    )


# ---------------------------------------------------------------------------
# 1. _webview_ready probe
# ---------------------------------------------------------------------------


def test_webview_ready_reports_backend_failure(monkeypatch):
    """A DISPLAY is pinned so the probe reaches initialize() on every OS —
    without it, a display-less Linux runner (ubuntu CI) answers «no DISPLAY»
    before initialize() ever raises (adversarial round c8: the same
    environment-dependency class as its sibling, closed file-wide here)."""
    import launcher

    _install_fake_webview(
        monkeypatch, init_raises=RuntimeError("no GTK/QT backend found")
    )
    monkeypatch.setenv("DISPLAY", ":0")

    ok, reason = launcher._webview_ready()

    assert ok is False
    assert "RuntimeError" in reason
    assert "no GTK/QT backend found" in reason


def test_webview_ready_non_linux_keeps_the_import_only_answer(monkeypatch):
    """macOS/Windows: a successful initialize() is the whole answer, with no
    DISPLAY/WAYLAND_DISPLAY in sight. IS_LINUX is pinned False explicitly —
    without the pin this test would be environment-dependent on the
    ubuntu-latest CI runners (adversarial round c7), where the display
    refinement makes a bare success impossible."""
    import launcher

    calls = _install_fake_webview(monkeypatch, init_raises=None)
    monkeypatch.setattr(launcher, "IS_LINUX", False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    assert launcher._webview_ready() == (True, "")
    assert calls, "initialize() must actually be probed"


def test_webview_ready_linux_gtk_installed_but_no_display_is_headless(monkeypatch):
    """THE regression (adversarial review of v6.91.0): a Linux host WITH gi/GTK
    bindings but WITHOUT a display used to pass the import probe and then crash
    at the first window — `guilib.initialize()` only imports the backend and
    calls `gtk.Application.new()`; the display is first consumed later, in
    `BrowserView.__init__` via `Gdk.Display.get_default()`. The probe must
    answer NOT ready so the browser fallback engages."""
    import launcher

    calls = _install_fake_webview(monkeypatch, init_raises=None)
    monkeypatch.setattr(launcher, "IS_LINUX", True)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    ok, reason = launcher._webview_ready()

    assert ok is False
    assert "DISPLAY" in reason
    # Round c7: the env check must run BEFORE initialize() — even SELECTING a
    # Qt backend constructs a QGuiApplication, which can ABORT the process on
    # a display-less box, so the probe must not touch pywebview at all here.
    assert calls == [], "initialize() must not be called with no display named"


def test_webview_ready_linux_gtk_with_dead_display_is_headless(monkeypatch):
    """A DISPLAY that is SET but unusable (stale value, dead X server) is not a
    display either: the GTK backend's own handle answers None, and that answer
    — not the environment variable — decides."""
    import launcher

    _install_fake_webview(monkeypatch, init_raises=None)
    monkeypatch.setattr(launcher, "IS_LINUX", True)
    monkeypatch.setenv("DISPLAY", ":0")

    fake_gdk = types.ModuleType("gi.repository.Gdk")
    fake_gdk.Display = types.SimpleNamespace(get_default=lambda: None)
    fake_repo = types.ModuleType("gi.repository")
    fake_repo.Gdk = fake_gdk
    monkeypatch.setitem(sys.modules, "gi", types.ModuleType("gi"))
    monkeypatch.setitem(sys.modules, "gi.repository", fake_repo)
    monkeypatch.setitem(sys.modules, "gi.repository.Gdk", fake_gdk)

    ok, reason = launcher._webview_ready()

    assert ok is False
    assert "no default display" in reason


def test_webview_ready_linux_gtk_with_live_display_stays_gui(monkeypatch):
    """A real desktop session keeps GUI mode: environment names a display AND
    the GTK backend hands back a display object."""
    import launcher

    _install_fake_webview(monkeypatch, init_raises=None)
    monkeypatch.setattr(launcher, "IS_LINUX", True)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")

    fake_gdk = types.ModuleType("gi.repository.Gdk")
    fake_gdk.Display = types.SimpleNamespace(get_default=lambda: object())
    fake_repo = types.ModuleType("gi.repository")
    fake_repo.Gdk = fake_gdk
    monkeypatch.setitem(sys.modules, "gi", types.ModuleType("gi"))
    monkeypatch.setitem(sys.modules, "gi.repository", fake_repo)
    monkeypatch.setitem(sys.modules, "gi.repository.Gdk", fake_gdk)

    assert launcher._webview_ready() == (True, "")


def test_webview_ready_linux_qt_backend_is_judged_on_the_environment(monkeypatch):
    """DISCLOSED RESIDUAL, pinned so it stays deliberate: a Qt backend gets the
    environment check only — constructing a QGuiApplication to ask Qt about the
    display ABORTS the process when the platform plugin cannot load, i.e. it
    would cause the very crash the probe exists to detect. With a display named
    in the environment, Qt is taken at its word."""
    import launcher

    _install_fake_webview(monkeypatch, init_raises=None,
                          backend_module="webview.platforms.qt",
                          renderer="qtwebengine")
    monkeypatch.setattr(launcher, "IS_LINUX", True)
    monkeypatch.setenv("DISPLAY", ":0")

    assert launcher._webview_ready() == (True, "")

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    ok, reason = launcher._webview_ready()
    assert ok is False
    assert "DISPLAY" in reason


# ---------------------------------------------------------------------------
# 2/3. _run_headless_main keep-alive + teardown
# ---------------------------------------------------------------------------


def test_run_headless_main_shuts_down_cleanly_on_event(monkeypatch, capsys):
    import launcher

    order = []
    monkeypatch.setattr(launcher, "_shutdown_event", _FakeEvent(initially_set=True))
    _stub_headless_teardown(monkeypatch, launcher, order)
    alive_thread = types.SimpleNamespace(is_alive=lambda: True)

    with pytest.raises(SystemExit) as excinfo:
        launcher._run_headless_main("http://127.0.0.1:8765", 8765, alive_thread)

    assert excinfo.value.code == 0
    # Round c4: shutdown was ALREADY requested at entry — the launcher must
    # not announce the URL or launch the owner's browser mid-teardown.
    assert order == [
        "stop_agent",
        ("kill_orphans", 8765, "headless_shutdown"),
    ]
    # Single-release contract (adversarial round c1): this path exits via
    # sys.exit, so the atexit-registered release owns the PID lock — an
    # explicit release here would run TWICE and unconditionally unlink a
    # lock file a NEWER launcher may already own.
    assert "release_pid_lock" not in order
    out = capsys.readouterr().out
    assert "http://127.0.0.1:8765" not in out, "no URL announcement mid-shutdown"


def test_run_headless_main_exits_nonzero_when_lifecycle_thread_dies(monkeypatch):
    import launcher

    order = []
    # Crash-fuse shape: agent_lifecycle_loop breaks out of its loop WITHOUT
    # setting _shutdown_event, so only thread liveness reveals the death.
    monkeypatch.setattr(launcher, "_shutdown_event", _FakeEvent(initially_set=False))
    _stub_headless_teardown(monkeypatch, launcher, order)
    dead_thread = types.SimpleNamespace(is_alive=lambda: False)

    with pytest.raises(SystemExit) as excinfo:
        launcher._run_headless_main("http://127.0.0.1:8765", 8765, dead_thread)

    assert excinfo.value.code == 1
    # No shutdown was requested at entry, so the URL/browser announcement DID
    # happen (the positive half of the round-c4 pin).
    assert order[0] == ("browser", "http://127.0.0.1:8765")
    # Teardown still ran in full despite the abnormal exit; the PID-lock
    # release belongs to atexit on this sys.exit path (single release).
    assert order[-2:] == ["stop_agent", ("kill_orphans", 8765, "crash_fuse")]
    assert "release_pid_lock" not in order


def test_run_headless_main_crash_fuse_survives_event_set_during_stop_agent(monkeypatch):
    """A signal landing while stop_agent() runs must not relabel a crash-fuse
    exit as a requested shutdown: reason and exit code come from one read."""
    import launcher

    order = []
    event = _FakeEvent(initially_set=False)
    monkeypatch.setattr(launcher, "_shutdown_event", event)
    _stub_headless_teardown(monkeypatch, launcher, order)
    monkeypatch.setattr(launcher, "stop_agent", lambda: (order.append("stop_agent"), event.set()))
    dead_thread = types.SimpleNamespace(is_alive=lambda: False)

    with pytest.raises(SystemExit) as excinfo:
        launcher._run_headless_main("http://127.0.0.1:8765", 8765, dead_thread)

    assert excinfo.value.code == 1
    assert order[-2:] == ["stop_agent", ("kill_orphans", 8765, "crash_fuse")]


# ---------------------------------------------------------------------------
# 4. Signal handlers: platform-layer registration, install-before-lifecycle
# ---------------------------------------------------------------------------


def test_platform_layer_registers_sigint_and_posix_sigterm(monkeypatch):
    """The signal surface lives in platform_layer (checklist 15): SIGINT
    everywhere, SIGTERM only where POSIX delivers it."""
    import signal as signal_mod

    from ouroboros import platform_layer

    registered = []
    monkeypatch.setattr(
        platform_layer.signal,
        "signal",
        lambda num, handler: registered.append((num, handler)),
    )
    handler = object()

    platform_layer.install_shutdown_signal_handlers(handler)

    expected = {signal_mod.SIGINT}
    if not platform_layer.IS_WINDOWS:
        expected.add(signal_mod.SIGTERM)
    assert {num for num, _h in registered} == expected
    assert all(h is handler for _num, h in registered)


def test_headless_handlers_install_before_lifecycle_thread_starts():
    """Order pin (adversarial round 3): the handlers must be registered BEFORE
    agent_lifecycle_loop spawns server.py in its own process group — a SIGTERM
    in the readiness-polling window (up to ~60s) must reach our teardown, not
    the default disposition that orphans the server. Also pins checklist 15:
    launcher never touches signal.signal directly."""
    import inspect
    import pathlib

    import launcher

    main_src = inspect.getsource(launcher.main)
    install_at = main_src.index("install_shutdown_signal_handlers")
    start_at = main_src.index("lifecycle_thread.start()")
    assert install_at < start_at

    module_src = pathlib.Path(launcher.__file__).read_text(encoding="utf-8")
    assert "signal.signal(" not in module_src


def test_wait_for_server_aborts_promptly_on_shutdown_event():
    """A shutdown signal during startup must cut the readiness wait short:
    the handler only sets the event, so the wait itself owns the early exit."""
    import time as time_mod

    import launcher

    started = time_mod.time()
    ready = launcher._wait_for_server(
        1, timeout=30.0, abort_event=_FakeEvent(initially_set=True)
    )

    assert ready is False
    assert time_mod.time() - started < 2.0


def test_headless_teardown_targets_the_authoritative_bound_port():
    """Synthesis round 3: the server may bind a DIFFERENT port than requested
    (conflict rebind writes actual_port to the port file) — every headless
    teardown must sweep THAT port, or orphans survive on exactly the boxes
    that needed the rebind. Pins all three sites in main(): the keep-alive
    entry, the startup-abort sweep, and the startup-failure sweep (which
    round 3 also found entirely missing on the headless branch)."""
    import inspect

    import launcher

    src = inspect.getsource(launcher.main)
    assert "_run_headless_main(url, actual_port" in src
    assert "_kill_orphaned_children(actual_port" in src
    # the requested-port variants must be GONE from main's headless paths
    assert "_run_headless_main(url, port" not in src
    abort_at = src.index("Shutdown requested during headless startup")
    fail_at = src.index("failed to become healthy")
    for section_start, reason in ((abort_at, "startup_abort"), (fail_at, "startup_failure")):
        section = src[section_start:section_start + 700]
        assert f'_kill_orphaned_children(actual_port, reason="{reason}")' in section


def test_headless_signal_handler_only_sets_event(monkeypatch):
    import launcher

    event = _FakeEvent()
    teardown = []
    monkeypatch.setattr(launcher, "_shutdown_event", event)
    monkeypatch.setattr(launcher, "stop_agent", lambda: teardown.append("stop_agent"))
    monkeypatch.setattr(
        launcher, "release_pid_lock", lambda: teardown.append("release_pid_lock")
    )

    launcher._headless_signal_handler(15, None)

    assert event.is_set() is True
    assert teardown == []  # the handler must ONLY set the event


# ---------------------------------------------------------------------------
# 5. First-run wizard headless path (browser onboarding behavioral pin)
# ---------------------------------------------------------------------------


def test_headless_first_run_onboarding_opens_no_window(monkeypatch, capsys):
    """Browser fallback is UNCHANGED by the server-first reordering: no setup
    window is opened, the owner is told onboarding lives in the browser, and the
    caller continues (the blocking /api/onboarding overlay is the surface)."""
    from ouroboros import launcher_onboarding

    # None sentinel: any `import webview` after this raises ImportError, so a
    # passing test proves the headless path never touches webview.
    monkeypatch.setitem(sys.modules, "webview", None)

    outcome = launcher_onboarding.present_first_run_onboarding({}, 8765, headless=True)

    assert outcome == {"saved": False, "restart_required": False}
    assert "browser" in capsys.readouterr().out


def test_first_run_preparation_reports_whether_onboarding_is_due(monkeypatch):
    from ouroboros import launcher_onboarding

    monkeypatch.setitem(sys.modules, "webview", None)
    monkeypatch.setattr(
        launcher_onboarding,
        "apply_runtime_provider_defaults",
        lambda settings: (dict(settings), False, []),
    )
    monkeypatch.setattr(launcher_onboarding, "load_settings", lambda: {})
    monkeypatch.setattr(launcher_onboarding, "_apply_settings_to_env", lambda settings: None)

    monkeypatch.setattr(launcher_onboarding, "has_startup_ready_provider", lambda settings: True)
    assert launcher_onboarding.prepare_first_run_settings()[1] is False

    monkeypatch.setattr(launcher_onboarding, "has_startup_ready_provider", lambda settings: False)
    assert launcher_onboarding.prepare_first_run_settings()[1] is True


# ---------------------------------------------------------------------------
# 6. Non-Linux non-regression: the probe never runs off-Linux
# ---------------------------------------------------------------------------


def test_detect_headless_probes_only_on_linux(monkeypatch):
    import launcher

    probes = []
    monkeypatch.setattr(launcher, "IS_LINUX", False)
    monkeypatch.setattr(launcher, "_headless", False)
    monkeypatch.setattr(
        launcher, "_webview_ready", lambda: probes.append(1) or (False, "x")
    )

    launcher._detect_headless()

    assert probes == [], "the GUI-backend probe must never run on macOS/Windows"
    assert launcher._headless is False


def test_detect_headless_sets_flag_on_linux_probe_failure(monkeypatch):
    import launcher

    monkeypatch.setattr(launcher, "IS_LINUX", True)
    monkeypatch.setattr(launcher, "_headless", False)
    monkeypatch.setattr(
        launcher,
        "_webview_ready",
        lambda: (False, "WebViewException: You must have either QT or GTK"),
    )

    launcher._detect_headless()

    assert launcher._headless is True


def test_detect_headless_keeps_gui_mode_when_backend_works(monkeypatch):
    import launcher

    monkeypatch.setattr(launcher, "IS_LINUX", True)
    monkeypatch.setattr(launcher, "_headless", False)
    monkeypatch.setattr(launcher, "_webview_ready", lambda: (True, ""))

    launcher._detect_headless()

    assert launcher._headless is False


def test_headless_already_running_opens_browser_at_existing_url(monkeypatch, capsys):
    """Second Open in browser mode must surface the running instance.

    A desktop-icon launch has no visible stderr, so the printed notice alone
    reads as "Open does nothing": the branch must also open the default
    browser at the existing URL, and bound-join the opener thread so the
    process does not exit under it (daemon threads die with the process)."""
    import launcher

    opened: list = []
    joins: list = []

    class _FakeThread:
        def join(self, timeout=None):
            joins.append(timeout)

    monkeypatch.setattr(launcher, "IS_WINDOWS", False)
    monkeypatch.setattr(launcher, "_detect_headless", lambda: None)
    monkeypatch.setattr(launcher, "_headless", True)
    monkeypatch.setattr(launcher, "acquire_pid_lock", lambda: False)
    monkeypatch.setattr(launcher, "_read_port_file", lambda: 8123)
    monkeypatch.setattr(launcher, "_wait_for_server", lambda port, timeout=1.0: True)
    monkeypatch.setattr(
        launcher,
        "_open_browser_detached",
        lambda url: opened.append(url) or _FakeThread(),
    )

    launcher.main()

    assert opened == ["http://127.0.0.1:8123"]
    assert joins and joins[0] is not None, (
        "the opener join must be bounded — an untimed join on a console "
        "browser (GenericBrowser) would hang the notice process forever"
    )
    assert "already running at http://127.0.0.1:8123" in capsys.readouterr().err


def test_headless_already_running_rereads_stale_port_file(monkeypatch):
    """Open during the first launcher's bootstrap must land on the live port.

    The lock loss usually races bootstrap: the port file can be absent or
    stale (an older run's port) at first read. The branch polls server
    health and re-reads the file between probes, so a mid-bootstrap Open
    opens the port the server actually bound instead of a dead one."""
    import launcher

    opened: list = []
    ports = iter([9999, 9999, 8123])  # stale, stale, freshly written
    health: dict = {9999: False, 8123: True}

    class _FakeThread:
        def join(self, timeout=None):
            pass

    monkeypatch.setattr(launcher, "IS_WINDOWS", False)
    monkeypatch.setattr(launcher, "_detect_headless", lambda: None)
    monkeypatch.setattr(launcher, "_headless", True)
    monkeypatch.setattr(launcher, "acquire_pid_lock", lambda: False)
    monkeypatch.setattr(launcher, "_read_port_file", lambda: next(ports))
    monkeypatch.setattr(
        launcher, "_wait_for_server", lambda port, timeout=1.0: health[port]
    )
    monkeypatch.setattr(launcher.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        launcher,
        "_open_browser_detached",
        lambda url: opened.append(url) or _FakeThread(),
    )

    launcher.main()

    assert opened == ["http://127.0.0.1:8123"]


def test_open_browser_detached_returns_joinable_thread(monkeypatch):
    import launcher

    calls: list = []
    monkeypatch.setattr(launcher.webbrowser, "open", lambda url: calls.append(url))

    thread = launcher._open_browser_detached("http://127.0.0.1:9999")
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert calls == ["http://127.0.0.1:9999"]


def test_kill_orphaned_children_records_the_real_reason(monkeypatch, tmp_path):
    """The cleanup journal must name the actual trigger, not always the
    window-close default (issue #153): panic, headless shutdown, the crash
    fuse, and the two startup teardowns each pass their own reason."""
    import launcher

    seen = []
    monkeypatch.setattr(launcher, "_cleanup_recorded_server_process", seen.append)
    monkeypatch.setattr(launcher, "_kill_stale_runtime_ports", lambda port: None)
    monkeypatch.setattr(launcher, "_kill_stale_on_port", lambda port: None)
    monkeypatch.setattr(launcher, "DATA_DIR", tmp_path)
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])

    launcher._kill_orphaned_children(0)
    launcher._kill_orphaned_children(0, reason="crash_fuse")

    assert seen == ["window_close", "crash_fuse"]


def test_teardown_reason_lands_in_the_journal_without_a_record(monkeypatch, tmp_path, caplog):
    """Normal teardowns reach _kill_orphaned_children AFTER stop_agent already
    consumed the server-process record, so the reason must be journalled at
    teardown entry — not only on the (usually absent) recorded-process row."""
    import logging

    import launcher

    monkeypatch.setattr(launcher, "_kill_stale_runtime_ports", lambda port: None)
    monkeypatch.setattr(launcher, "_kill_stale_on_port", lambda port: None)
    monkeypatch.setattr(launcher, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        launcher, "_server_process_record_path", lambda: tmp_path / "absent.json")
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])

    with caplog.at_level(logging.INFO, logger=launcher.log.name):
        launcher._kill_orphaned_children(0, reason="headless_shutdown")

    assert any(
        "Tearing down runtime orphans (headless_shutdown)" in rec.getMessage()
        for rec in caplog.records
    ), "the trigger reason must land in the journal even with no record on disk"


def test_panic_stop_teardown_names_its_reason():
    """The panic branch os._exit()s from the lifecycle thread, so it has no
    behavioural twin here; pin its reason by source. The headless
    shutdown/crash-fuse split is proven by test_run_headless_main_shuts_down_
    cleanly_on_event and ..._exits_nonzero_when_lifecycle_thread_dies above."""
    import inspect

    import launcher

    assert 'reason="panic_stop"' in inspect.getsource(launcher)
