"""Desktop background mode: close vs quit, the one consent question, quiet start, second launch.

Contract scenarios for ``ouroboros/launcher_background.py`` with a stand-in window and
indicator; the consent answer goes through the real owner settings write. The platform
adapters' native halves (WinForms tray, AppKit menu-bar item) are driven with stand-in
modules; their real behaviour is the D2 probe's (macOS) and Windows CI's. Nothing here
starts a GUI, the server or the live install.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import launcher_background as lb
from ouroboros import platform_layer

NODE_BIN = (
    str(Path.home() / ".claudexor" / "node" / "bin" / "node")
    if (Path.home() / ".claudexor" / "node" / "bin" / "node").exists()
    else "node"
)
WEB_ROOT = Path(__file__).resolve().parents[1] / "web"


class Hook:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, fn):
        self.handlers.append(fn)
        return self


class Window:
    def __init__(self, answer=None):
        self.events = SimpleNamespace(closing=Hook(), before_show=Hook(), shown=Hook(), closed=Hook())
        self.localization = {"global.ok": "OK", "global.cancel": "Cancel"}
        self.visible, self.calls, self.answer = True, [], answer

    def hide(self):
        self.visible = False
        self.calls.append("hide")

    def show(self):
        self.visible = True
        self.calls.append("show")

    def create_confirmation_dialog(self, title, message):
        self.calls.append(("ask", message, dict(self.localization)))
        return self.answer


class FakeIndicator(lb.Indicator):
    comes_up = True

    def attach_native(self, window):
        self.attached = window

    def _launch(self):
        if self.comes_up:
            self.ready.set()
        return self.comes_up

    def _dispose(self, wait):
        self.background.events.append(("dispose", wait))
        self._stopped()

    def notify(self, title, body, sound=True):
        self.background.events.append(("banner", title, body))
        self.background.banner_sounds = [*getattr(self.background, "banner_sounds", []), sound]
        return True


SIGN_IN = lb.signin_startup_on  # the real reader; the autouse fixture stands it in


class Answer:
    """A stand-in ``urlopen`` response carrying ``body``."""

    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.body


@pytest.fixture(autouse=True)
def no_real_server(monkeypatch):
    """No test here may reach a real server (the live install listens on 8765): HTTP is refused
    unless a test answers it, and sign-in startup reads "on" unless a test asks the real reader."""
    def refused(*_args, **_kwargs):
        raise OSError("no server in these tests")

    monkeypatch.setattr(lb.urllib.request, "urlopen", refused)
    monkeypatch.setattr(lb, "signin_startup_on", lambda port: True)


@pytest.fixture
def settings(tmp_path, monkeypatch):
    from ouroboros import config as cfg

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path, raising=True)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", tmp_path / "settings.json", raising=True)
    cfg.reset_runtime_mode_baseline_for_tests()
    yield tmp_path / "settings.json"
    cfg.reset_runtime_mode_baseline_for_tests()


def choose(settings_path, value):
    settings_path.write_text(json.dumps({"OUROBOROS_DESKTOP_KEEP_RUNNING": value}), encoding="utf-8")


def stored(settings_path):
    return json.loads(settings_path.read_text(encoding="utf-8")).get("OUROBOROS_DESKTOP_KEEP_RUNNING")


@pytest.fixture
def make(monkeypatch):
    """A Background around a stand-in window; teardown stops every thread it started."""
    created = []
    monkeypatch.setattr(lb, "status_line", lambda port: "Ouroboros: waiting")  # never a real server
    monkeypatch.setattr(lb, "_active", None)

    def factory(*, indicator=FakeIndicator, answer=None):
        monkeypatch.setattr(lb, "indicator_class", lambda: indicator)
        background = lb.Background(lambda: background.events.append("exit"), lambda: 8765, threading.Event())
        background.events = []
        created.append(background.shutdown)
        window = Window(answer)
        background.attach(window)
        fire(window, "before_show")  # synchronous once the native window exists
        fire(window, "shown")  # pywebview sets it for a hidden window too
        return background, window

    yield factory
    for shutdown in created:
        shutdown.set()
    for thread in threading.enumerate():
        if thread.name.startswith("ouroboros-"):
            thread.join(timeout=5)


def fire(window, event):
    for handler in getattr(window.events, event).handlers:
        handler()


def close(background, window):
    return [handler() for handler in window.events.closing.handlers][-1]


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


def test_close_with_background_off_quits(settings, monkeypatch, make):
    choose(settings, "false")
    background, window = make()
    assert close(background, window) is None
    assert background.events == ["exit"]
    assert window.calls == []


def test_first_close_asks_once_and_stores_the_answer(settings, monkeypatch, make):
    assert not settings.exists()
    background, window = make(answer=True)
    assert close(background, window) is False  # this close waits for the answer
    wait_for(lambda: "hide" in window.calls)
    asks = [call for call in window.calls if call[0] == "ask"]
    assert len(asks) == 1 and "tasks, schedules, Telegram" in asks[0][1]
    assert stored(settings) == "true"
    assert close(background, window) is False and window.calls.count("hide") == 2
    assert len([call for call in window.calls if call[0] == "ask"]) == 1, "asked once per install"
    assert background.events == []


@pytest.mark.parametrize("answer", [False, None])  # Quit, or the dialog dismissed
def test_declining_or_dismissing_quits_and_is_remembered(settings, monkeypatch, answer, make):
    background, window = make(answer=answer)
    assert close(background, window) is False
    wait_for(lambda: background.events == ["exit"])
    assert stored(settings) == "false"
    assert "hide" not in window.calls


def test_an_unshowable_question_quits_without_recording_a_choice(settings, monkeypatch, make):
    background, window = make()
    window.create_confirmation_dialog = lambda *_: (_ for _ in ()).throw(RuntimeError("no dialog"))
    close(background, window)
    wait_for(lambda: background.events == ["exit"])
    assert not settings.exists()


def test_close_with_background_on_hides_behind_a_live_indicator(settings, monkeypatch, make):
    choose(settings, "true")
    background, window = make()
    assert not background.indicator.ready.is_set()
    assert close(background, window) is False  # enabled during this run: the icon comes up on demand
    assert window.calls == ["hide"] and background.indicator.hidden and background.indicator.ready.is_set()
    assert background.events == []


def test_the_window_is_hidden_outside_the_indicators_state_lock(settings, monkeypatch, make):
    """On Windows hide() is a synchronous Invoke onto the UI thread, which may be waiting for that lock."""
    choose(settings, "true")
    background, window = make()
    free = []

    def hide():
        lock = background.indicator._state_lock
        free.append(lock.acquire(blocking=False))
        if free[-1]:
            lock.release()
        Window.hide(window)

    window.hide = hide
    assert close(background, window) is False and free == [True]
    assert background.indicator.hidden and window.calls == ["hide"]

    background, window = make()

    def hide_while_the_icon_dies():
        Window.hide(window)
        background.indicator._stopped()  # the pump died between the decision and the commit

    window.hide = hide_while_the_icon_dies
    close(background, window)
    assert background.events == ["exit"] and not background.indicator.hidden, "no hidden window without its icon"


def test_no_indicator_means_close_quits_even_when_background_is_on(settings, monkeypatch, make):
    choose(settings, "true")

    class NoIcon(FakeIndicator):
        comes_up = False

    background, window = make(indicator=NoIcon)
    close(background, window)
    assert background.events == ["exit"] and window.calls == []


def test_without_an_adapter_or_native_hooks_closing_quits_and_never_asks(settings, monkeypatch, make):
    background, window = make(indicator=None)  # Linux
    close(background, window)
    assert background.events == ["exit"] and window.calls == []

    class Broken(FakeIndicator):
        def attach_native(self, window):
            raise RuntimeError("delegate unavailable")

    background, window = make(indicator=Broken)
    close(background, window)
    assert background.events == ["exit"] and window.calls == []


def test_quit_requests_bypass_the_question_and_are_never_cancelled(settings, monkeypatch, make):
    background, window = make(answer=True)
    background.quit.set()  # Cmd+Q / Dock Quit / logout (macOS delegate) or a tray Quit
    assert close(background, window) is None
    assert background.events == ["exit"] and window.calls == []
    assert not settings.exists()


def test_tray_quit_is_a_plain_exit_without_panic(settings, monkeypatch, tmp_path, make):
    def no_http(*_args, **_kwargs):
        raise AssertionError("Quit must not reach the server's Panic")

    monkeypatch.setattr(lb.urllib.request, "urlopen", no_http)
    background, window = make()
    background.request_quit()
    assert background.quit.is_set() and background.events == ["exit"]
    assert not (tmp_path / "state" / "panic_stop.flag").exists()


def test_tray_panic_posts_the_servers_emergency_stop_and_shows_the_window_on_failure(settings, monkeypatch, make):
    sent = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def urlopen(request, timeout):
        sent.append((request.full_url, request.get_method(), json.loads(request.data)))
        if len(sent) > 1:
            raise OSError("server gone")
        return Response()

    monkeypatch.setattr(lb.urllib.request, "urlopen", urlopen)
    background, window = make()
    background.request_panic()
    wait_for(lambda: len(sent) == 1)
    assert sent[0] == ("http://127.0.0.1:8765/api/command", "POST", {"cmd": "/panic"})
    background.request_panic()
    wait_for(lambda: "show" in window.calls)
    assert background.events == []


@pytest.mark.parametrize("answer,hidden", [
    ({"state": "on"}, True),
    ({"state": "off"}, False),
    ({"state": "disabled_by_os"}, False),
    ({"state": "other_copy"}, False),
    ({"state": "unavailable", "reason": "Available only when the host runs the packaged desktop app."}, False),
    (OSError("refused"), False),  # a failed read
    (b"<html>", False),
])
def test_a_quiet_start_needs_sign_in_startup_on_as_the_os_reports_it(settings, monkeypatch, make, answer, hidden):
    asked = []

    def urlopen(url, timeout):
        asked.append(url)
        if isinstance(answer, Exception):
            raise answer
        return Answer(answer if isinstance(answer, bytes) else json.dumps(answer).encode())

    choose(settings, "true")
    background, _window = make()
    monkeypatch.setattr(lb.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(lb, "signin_startup_on", SIGN_IN)
    assert background.start_hidden("owner") is False and asked == [], "a manual launch never asks and is shown"
    assert background.start_hidden("automatic") is hidden
    assert asked == ["http://127.0.0.1:8765/api/desktop/autostart"]  # the server's own autostart_status()
    assert background.indicator.hidden is hidden


def test_quiet_start_needs_an_automatic_launch_background_on_and_a_live_indicator(settings, monkeypatch, make):
    for value, intent, indicator, expected in [  # sign-in startup reads "on" here (fixture)
        ("true", "automatic", FakeIndicator, True),
        ("true", "owner", FakeIndicator, False),  # a manual launch always shows the window
        ("false", "automatic", FakeIndicator, False),
        ("", "automatic", FakeIndicator, False),  # never decided: background is off
        ("true", "automatic", None, False),  # Linux
    ]:
        choose(settings, value)
        background, _window = make(indicator=indicator)
        assert background.start_hidden(intent) is expected, (value, intent, indicator)

    class NoIcon(FakeIndicator):
        comes_up = False

    choose(settings, "true")
    monkeypatch.setattr(lb, "INDICATOR_WAIT_SEC", 0.05)
    background, window = make(indicator=NoIcon)
    assert background.start_hidden("automatic")
    background.run()
    assert window.calls == ["show"] and not background.indicator.hidden

    class SlowIcon(FakeIndicator):
        def _launch(self):  # like the real pumps: visible a moment after start() returns
            threading.Timer(0.02, self.ready.set).start()
            return True

    # The short wait above belongs to the no-icon case only: here the icon comes up a moment
    # after start(), and a loaded CI runner must not turn that moment into the no-icon path.
    monkeypatch.setattr(lb, "INDICATOR_WAIT_SEC", 5.0)
    background, window = make(indicator=SlowIcon)
    assert background.start_hidden("automatic")
    background.run()
    assert window.calls == [] and background.indicator.hidden and background.indicator.ready.is_set()
    assert background._poller is not None and background._poller.is_alive(), "the state line keeps updating"


def test_a_hidden_window_comes_back_when_its_indicator_dies(settings, monkeypatch, make):
    choose(settings, "true")
    background, window = make()
    close(background, window)
    background.indicator._stopped()  # pump died without a stop request
    assert window.calls == ["hide", "show"] and not background.indicator.hidden

    def cannot_show():
        raise RuntimeError("GUI unavailable")

    background, window = make()
    close(background, window)
    window.show = cannot_show
    background.indicator._stopped()
    assert background.events == ["exit"], "never a hidden process without a way back"


def _page_tones(answer: dict) -> int:
    """The real page notifier (node) given this launcher answer: how many tones it plays itself."""
    script = """
import { createNotifier, NOTIFY_PREFS_KEY, DEFAULT_NOTIFY_PREFS } from './modules/notifications.js';
let raw = '';
for await (const chunk of process.stdin) raw += chunk;
const answer = JSON.parse(raw);
let tones = 0;
class Audio {
    constructor() { this.currentTime = 0; this.destination = {}; }
    resume() {}
    createOscillator() { tones += 1; return { connect() {}, start() {}, stop() {}, frequency: {} }; }
    createGain() { return { gain: { value: 0 }, connect() {} }; }
}
const prefs = JSON.stringify({ ...DEFAULT_NOTIFY_PREFS, enabled: true });
const notifier = createNotifier({
    storage: { getItem: (key) => (key === NOTIFY_PREFS_KEY ? prefs : null), setItem() {} },
    notificationCtor: undefined, audioContextCtor: Audio, showToast: () => null,
    documentRef: { addEventListener() {}, removeEventListener() {}, querySelectorAll: () => [] },
    hostApi: { request_attention: () => answer, notify_owner: () => answer },
});
notifier.handleFrame({ role: 'system', system_type: 'task_summary', task_id: 'boundary' }, { kind: 'chat', isMain: true });
await new Promise((resolve) => setTimeout(resolve, 20));
notifier.destroy();
process.stdout.write(JSON.stringify({ tones }));
"""
    result = subprocess.run([NODE_BIN, "--input-type=module", "-e", script], input=json.dumps(answer),
                            text=True, capture_output=True, cwd=WEB_ROOT, timeout=60, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)["tones"]


def test_the_page_plays_no_second_sound_after_the_launchers_banner(settings, monkeypatch, make):
    """Python to JS: the real attention() answer for a queued banner (WindowsTray.notify with a live icon)."""
    choose(settings, "true")
    background, window = make()
    close(background, window)  # hidden on purpose; FakeIndicator.notify queues a banner
    answer = background.attention(True, "Task finished", "")  # the real request_native_attention, sound off
    assert answer == {"ok": True, "status": "background", "banner": True, "sound_played": False}, "never claims it played"
    assert _page_tones(answer) == 0, "the banner owns the sound"
    assert _page_tones({**answer, "banner": False}) == 1, "no banner and no system sound: the page's one tone"


def test_attention_never_raises_a_window_hidden_on_purpose(settings, monkeypatch, make):
    cues = []
    monkeypatch.setattr(lb, "request_native_attention",
                        lambda show, sound=True: cues.append((show, sound)) or {"ok": True, "sound_played": sound})
    choose(settings, "true")
    background, window = make()
    close(background, window)
    result = background.attention(True, "Task finished", "Report ready")
    assert window.calls == ["hide"], "no auto-raise while hidden"
    assert ("banner", "Task finished", "Report ready") in background.events
    assert cues == [(None, False)], "the banner owns its sound; no second one"
    assert result["status"] == "background" and result["ok"] and result["banner"]
    background.attention(False, "Task finished", "")
    assert background.banner_sounds[-2:] == [True, False], "Sound off reaches the banner itself"
    cues.clear()
    assert background.attention(True, "Task finished", "", False)["status"] == "background", \
        "a page with its own browser banner asks first: hidden, the native signal is placed all the same"
    assert cues == [(None, False)] and window.calls == ["hide"]

    background.show_window()  # the banner or the icon was clicked
    cues.clear()
    background.attention(True)
    assert cues == [(window.show, True)], "a visible window keeps today's raise-and-sound cue"
    cues.clear()
    assert background.attention(True, "Task finished", "", False) == {"ok": False, "status": "visible"}
    assert cues == [], "visible and the page shows its own banner: no raise, no second sound"


def test_panic_removes_the_indicator_without_waiting_and_before_the_lock(monkeypatch):
    calls = []

    class Started:
        def __init__(self, name):
            self.name = name

        def stop(self, *, wait):
            calls.append((self.name, wait))

    monkeypatch.setattr(lb, "_active", Started("indicator"))
    monkeypatch.setattr(lb, "_notifier", Started("notifications"))  # Windows' notification icons
    lb.request_tray_cleanup()
    lb.stop_tray_before_exit(lambda: calls.append("release"), wait=0)
    assert calls == [("indicator", 0), ("notifications", 0), ("indicator", 0), ("notifications", 0), "release"]
    calls.clear()
    lb.stop_tray_before_exit(lambda: calls.append("release"))
    assert calls == [("indicator", 0.5), ("notifications", 0.5), "release"]
    monkeypatch.setattr(lb, "_active", None)
    calls.clear()
    lb.stop_tray_before_exit(lambda: calls.append("release"))
    assert calls == [("notifications", 0.5), "release"], "no background icon: the notification icons still go"


def test_turning_background_off_removes_the_icon_only_while_the_window_is_visible(settings, monkeypatch, make):
    class OneTick(threading.Event):
        def wait(self, timeout=None):
            return True  # one poll iteration, then stop

    choose(settings, "true")
    background, window = make()
    close(background, window)  # hidden: the icon is the way back
    background.shutdown.set()
    background._poller.join(timeout=5)
    background.shutdown = OneTick()
    choose(settings, "false")
    background._poll()
    assert ("dispose", 0.0) not in background.events
    background.show_window()
    background._poll()
    assert ("dispose", 0.0) in background.events and not background.indicator.ready.is_set()


READY = {"supervisor_ready": True, "supervisor_error": None, "active_chat_activities_complete": True}


def census(*phases, **facts):
    return {**READY, "active_chat_activities": [{"phase": phase} for phase in phases], **facts}


@pytest.mark.parametrize("body,line", [
    (OSError("refused"), "Ouroboros: not responding"),  # refused, timed out (a busy server is not "stopped")
    (b"<html>", "Ouroboros: not responding"),
    (census(), "Ouroboros: waiting"),
    (census("queued"), "Ouroboros: waiting"),
    (census("working", "finalizing", "thinking", "queued"), "Ouroboros: working on 3 tasks"),
    (census("working"), "Ouroboros: working on 1 task"),
    (census("budget_paused"), "Ouroboros: paused"),
    (census("budget_pausing", "budget_paused"), "Ouroboros: pausing"),  # settling a Pause is not work (page)
    (census("working", "budget_pausing"), "Ouroboros: working on 1 task"),
    # What the server says it does not know is neither "waiting" nor "working" (review r1).
    (census(supervisor_ready=False, supervisor_error="Supervisor init failed: boom"),
     "Ouroboros: error, tasks are not running"),
    (census(supervisor_ready=False), "Ouroboros: starting"),
    (census(active_chat_activities_complete=False), "Ouroboros: activity unconfirmed"),
    (census("unknown", active_chat_activities_complete=False), "Ouroboros: activity unconfirmed"),
    (census("unknown"), "Ouroboros: activity unconfirmed"),
    (census("working", active_chat_activities_complete=False), "Ouroboros: activity unconfirmed"),
])
def test_the_state_line_reads_the_servers_own_activity(monkeypatch, body, line):
    def urlopen(url, timeout):
        assert url == "http://127.0.0.1:8765/api/state"
        if isinstance(body, Exception):
            raise body
        return Answer(body if isinstance(body, bytes) else json.dumps(body).encode())

    monkeypatch.setattr(lb.urllib.request, "urlopen", urlopen)
    assert lb.status_line(8765) == line


def test_windows_form_closing_decides_with_the_close_reason(settings, monkeypatch, make):
    forms = types.ModuleType("System.Windows.Forms")
    forms.CloseReason = SimpleNamespace(UserClosing="user", WindowsShutDown="shutdown")
    for name in ("System", "System.Windows"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, "System.Windows.Forms", forms)
    from ouroboros import launcher_tray

    background, window = make(indicator=launcher_tray.WindowsTray, answer=True)
    window.native = SimpleNamespace(FormClosing=Hook())
    background._attach_native()
    assert background.native_ready and close(background, window) is None, "pywebview's closing defers"
    form_closing = window.native.FormClosing.handlers[-1]

    shutdown = SimpleNamespace(CloseReason="shutdown", Cancel=False)
    form_closing(None, shutdown)  # sign-out with the question never answered
    assert background.events == ["exit"] and shutdown.Cancel is False and window.calls == []

    background.events.clear()
    background.quit.clear()
    user = SimpleNamespace(CloseReason="user", Cancel=False)
    form_closing(None, user)
    assert user.Cancel is True  # the owner's close waits for the one question
    wait_for(lambda: any(call[0] == "ask" for call in window.calls if isinstance(call, tuple)))
    assert "OK keeps it running" in [call for call in window.calls if isinstance(call, tuple)][0][1]
    wait_for(lambda: background.events == ["exit"])  # no tray off Windows: the answer cannot hide, so it quits


def test_macos_quit_requests_are_marked_before_pywebviews_closing_runs(monkeypatch):
    objc = types.ModuleType("objc")
    objc.super = lambda cls, obj: super(cls, obj)
    appkit = types.ModuleType("AppKit")
    appkit.NSObject = object
    monkeypatch.setitem(sys.modules, "objc", objc)
    monkeypatch.setitem(sys.modules, "AppKit", appkit)
    from ouroboros import launcher_tray_macos as mac

    monkeypatch.setattr(mac, "_shared", {})

    class PywebviewDelegate:
        def applicationShouldTerminate_(self, sender):
            return "pywebview decides" if background.quit.is_set() else "asked as a close"

    background = SimpleNamespace(quit=threading.Event(), shown=[])
    background.show_window = lambda: background.shown.append(True)
    mac._shared["background"] = background
    delegate_class, _target = mac._classes(PywebviewDelegate)
    delegate = delegate_class()
    assert delegate.applicationShouldTerminate_(None) == "pywebview decides"
    assert delegate.applicationShouldHandleReopen_hasVisibleWindows_(None, False) is False
    assert background.shown == [True]

    window = Window(answer=True)
    assert mac.MacStatusItem.confirm(None, window, "t", "m") is True
    assert window.calls[-1][2] == {"global.ok": "Keep running", "global.cancel": "Quit"}
    assert window.localization == {"global.ok": "OK", "global.cancel": "Cancel"}, "labels restored"


@pytest.mark.parametrize("intent,answered,expected", [
    ("automatic", True, []),  # a sign-in start of a second copy stays quiet
    ("owner", True, ["activate"]),  # a manual one shows the running window
    ("owner", False, ["activate", "notice window"]),  # nobody answered: the old notice
])
def test_second_launch_shows_the_running_window_only_when_manual(monkeypatch, caplog, intent, answered, expected):
    import launcher

    calls = []
    fake_webview = types.ModuleType("webview")
    fake_webview.create_window = lambda *args, **kwargs: calls.append("notice window")
    fake_webview.start = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "webview", fake_webview)
    monkeypatch.setattr(launcher, "IS_WINDOWS", False)
    monkeypatch.setattr(launcher, "_detect_headless", lambda: None)
    monkeypatch.setattr(launcher, "_headless", False)
    monkeypatch.setattr(launcher, "_external_ui", False)
    monkeypatch.setattr(launcher, "acquire_pid_lock", lambda: False)
    monkeypatch.setattr(launcher, "activate_running_instance",
                        lambda lock: calls.append("activate") or answered)
    launcher.main(["--launch-intent", intent])
    assert calls == expected
    assert "Another instance already running." in caplog.text


def test_a_losing_launcher_leaves_the_holders_pid_readable(tmp_path):
    """A manual second launch signals the PID in the lock file, so the loser must not erase it."""
    lock = tmp_path / "ouroboros.pid"
    assert platform_layer.pid_lock_acquire(str(lock))
    try:
        assert lock.read_text(encoding="utf-8") == str(os.getpid())
        loser = subprocess.run(
            [sys.executable, "-c", "import sys; from ouroboros import platform_layer as p; "
             "sys.exit(0 if p.pid_lock_acquire(sys.argv[1]) else 7)", str(lock)],
            capture_output=True, text=True, timeout=60, cwd=str(os.getcwd()),
            env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [os.getcwd(), os.environ.get("PYTHONPATH")]))},
        )
        assert loser.returncode == 7, loser.stderr
        assert lock.read_text(encoding="utf-8") == str(os.getpid())
    finally:
        platform_layer.pid_lock_release(str(lock))
    assert not lock.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal path")
def test_a_second_launch_signals_sigurg_which_an_older_launcher_survives(tmp_path):
    lock = tmp_path / "ouroboros.pid"
    lock.write_text("not a pid", encoding="utf-8")
    assert not lb.activate_running_instance(lock)
    lock.write_text(str(os.getpid()), encoding="utf-8")
    assert not lb.activate_running_instance(lock), "never signal itself"
    old = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])  # no SIGURG handler
    try:
        time.sleep(0.3)
        lock.write_text(str(old.pid), encoding="utf-8")
        assert lb.activate_running_instance(lock)
        time.sleep(0.3)
        assert old.poll() is None, "the default disposition ignores SIGURG"
    finally:
        old.kill()
        old.wait(timeout=10)


def test_an_open_request_before_the_window_exists_is_kept_and_cancels_the_quiet_start(settings, monkeypatch):
    """A manual second launch while the first copy still boots (review r1): kept, never dropped."""
    monkeypatch.setattr(lb, "indicator_class", lambda: FakeIndicator)
    choose(settings, "true")
    background = lb.Background(lambda: None, lambda: 8765, threading.Event())
    background.show_window()  # the listener's call during bootstrap: there is no window yet
    assert background.start_hidden("automatic") is False, "the owner asked for the window"
    window = Window()
    background.attach(window)
    fire(window, "before_show")
    assert window.calls == []
    fire(window, "shown")
    assert window.calls == ["show"] and not background.indicator.hidden
    background.show_window()  # once shown, a request acts at once
    assert window.calls == ["show", "show"]

    background = lb.Background(lambda: None, lambda: 8765, threading.Event())
    assert background.start_hidden("automatic") is True  # no request yet: a quiet start
    window = Window()
    background.attach(window)
    fire(window, "before_show")
    background.show_window()  # lands after that decision, before pywebview showed the window
    assert window.calls == []
    fire(window, "shown")
    assert window.calls == ["show"] and not background.indicator.hidden


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal; Windows uses its kernel event")
def test_a_real_sigurg_during_the_boot_opens_the_window_once_it_exists(settings, monkeypatch, tmp_path):
    """The signal.signal branch (Linux) on any POSIX host; macOS's MachSignals queues the
    same signal until the run loop starts, after the window exists (MAC_PROBE addendum)."""
    monkeypatch.setattr(lb, "indicator_class", lambda: FakeIndicator)
    monkeypatch.setattr(lb, "sys", SimpleNamespace(platform="linux"))
    choose(settings, "true")
    previous = signal.getsignal(signal.SIGURG)
    try:
        background = lb.Background(lambda: None, lambda: 8765, threading.Event()).listen(tmp_path / "ouroboros.pid")
        signal.raise_signal(signal.SIGURG)  # a second launch while bootstrap and the server start run
        assert background.start_hidden("automatic") is False
        window = Window()
        background.attach(window)
        fire(window, "before_show")
        fire(window, "shown")
        assert window.calls == ["show"]
        signal.raise_signal(signal.SIGURG)
        assert window.calls == ["show", "show"]
    finally:
        signal.signal(signal.SIGURG, previous)


@pytest.mark.parametrize("headless", [False, True])
def test_the_launcher_listens_for_a_second_launch_before_its_long_boot(monkeypatch, headless):
    import launcher

    class Boot(Exception):
        pass

    order = []

    class Recorder:
        def __init__(self, exit_launcher, read_port, shutdown_event):
            order.append("background")

        def listen(self, lock_path):
            order.append(("listen", lock_path))
            return self

    def check_git():
        order.append("boot")
        raise Boot

    monkeypatch.setitem(sys.modules, "webview", types.ModuleType("webview"))
    monkeypatch.setattr(launcher, "IS_WINDOWS", False)
    monkeypatch.setattr(launcher, "_detect_headless", lambda: None)
    monkeypatch.setattr(launcher, "_headless", headless)
    monkeypatch.setattr(launcher, "_external_ui", False)
    monkeypatch.setattr(launcher, "acquire_pid_lock", lambda: True)
    monkeypatch.setattr(launcher, "release_pid_lock", lambda: None)  # atexit keeps this stand-in
    monkeypatch.setattr(launcher, "automatic_launch_allowed", lambda *args: True)
    monkeypatch.setattr(launcher, "Background", Recorder)
    monkeypatch.setattr(launcher, "check_git", check_git)
    with pytest.raises(Boot):
        launcher.main(["--launch-intent", "automatic"])
    assert order == (["boot"] if headless else ["background", ("listen", launcher.PID_FILE), "boot"])


@pytest.mark.skipif(sys.platform != "win32", reason="Windows kernel event contract")
def test_real_windows_kernel_event_is_shared_by_same_install(tmp_path):
    from ouroboros import launcher_tray

    kernel = launcher_tray._kernel()
    name = launcher_tray._event_name(tmp_path / "ouroboros.pid")
    handle = kernel.CreateEventW(None, False, False, name)
    assert handle
    try:
        assert launcher_tray.activate_existing_tray(tmp_path / "ouroboros.pid", timeout=0)
        assert kernel.WaitForSingleObject(handle, 0) == 0
        assert kernel.WaitForSingleObject(handle, 0) == 258  # auto-reset, no stale request
    finally:
        kernel.CloseHandle(handle)
