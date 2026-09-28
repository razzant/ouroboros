"""Windows tray ownership and second-launch activation without a real GUI."""

import ast
import sys
import threading
import types
from types import SimpleNamespace

import pytest

from ouroboros import launcher_tray


class EventHook:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, fn):
        self.handlers.append(fn)
        return self

    def fire(self, *args):
        for handler in self.handlers:
            handler(*args)


class FakeKernel:
    def __init__(self):
        self.event = threading.Event()
        self.open = True
        self.closed = []

    def CreateEventW(self, *_):
        return 42

    def OpenEventW(self, *_):
        return 42 if self.open else None

    def SetEvent(self, handle):
        self.event.set()
        return 1

    def WaitForSingleObject(self, handle, timeout):
        if self.event.is_set():
            self.event.clear()
            return 0
        return 258

    def CloseHandle(self, handle):
        self.closed.append(handle)


def install_forms(monkeypatch, kernel, *, fail_sta=False):
    clr = types.ModuleType("clr")
    clr.AddReference = lambda name: None
    drawing = types.ModuleType("System.Drawing")
    drawing.Icon = object
    drawing.SystemIcons = SimpleNamespace(Application=object())
    dotnet_threading = types.ModuleType("System.Threading")
    dotnet_threading.ApartmentState = SimpleNamespace(STA=object())
    dotnet_threading.ThreadStart = lambda fn: fn

    class Thread:
        def __init__(self, fn):
            self.fn = fn

        def SetApartmentState(self, state):
            if fail_sta:
                raise RuntimeError("no STA")

        def Start(self):
            self.fn()  # deterministic fake pump

    dotnet_threading.Thread = Thread

    class Control:
        def __init__(self, *args):
            self.Click = EventHook()
            self.MouseClick = EventHook()
            self.MouseDoubleClick = EventHook()
            self.Tick = EventHook()
            self.Items = SimpleNamespace(Add=lambda item: None)

        def Start(self):
            pass

        def Stop(self):
            pass

        def Dispose(self):
            pass

    class Icon(Control):
        def __init__(self):
            super().__init__()
            self.Visible = False
            icons.append(self)

        def Dispose(self):
            calls.append("dispose")

    class Application:
        @staticmethod
        def Run(context):
            icon = icons[-1]
            timer = timers[-1]
            timer.Tick.fire(None, None)
            assert icon.Visible
            on_running(icon, timer)

        @staticmethod
        def ExitThread():
            calls.append("exit_thread")

    icons, timers, calls = [], [], []
    on_running = lambda icon, timer: None

    class Timer(Control):
        def __init__(self):
            super().__init__()
            timers.append(self)

    forms = types.ModuleType("System.Windows.Forms")
    forms.Application = Application
    forms.ApplicationContext = Control
    forms.ContextMenuStrip = Control
    forms.MouseButtons = SimpleNamespace(Left=1)
    forms.NotifyIcon = Icon
    forms.Timer = Timer
    forms.ToolStripMenuItem = Control
    system = types.ModuleType("System")
    system.__path__ = []
    for name, module in (("clr", clr), ("System", system), ("System.Drawing", drawing),
                         ("System.Threading", dotnet_threading), ("System.Windows.Forms", forms)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(launcher_tray, "_kernel", lambda: kernel)
    return SimpleNamespace(icons=icons, timers=timers, calls=calls,
                           set_running=lambda fn: setattr(Application, "Run", staticmethod(fn)))


def test_close_falls_back_when_unready_and_hidden_window_restores_after_lost_pump():
    seen = []
    window = SimpleNamespace(hide=lambda: seen.append("hide"), show=lambda: seen.append("show"))
    tray = launcher_tray.WindowsTray(lambda: window, lambda: None, "lock", threading.Event())
    assert not tray.hide_on_close(window)
    tray.ready.set()
    assert tray.hide_on_close(window)
    tray._pump_stopped()
    assert seen == ["hide", "show"]


def test_activation_kernel_event_is_ephemeral_and_failure_falls_back(monkeypatch):
    kernel = FakeKernel()
    monkeypatch.setattr(launcher_tray, "_kernel", lambda: kernel)
    assert launcher_tray.activate_existing_tray("lock", timeout=0)
    assert kernel.event.is_set() and kernel.closed == [42]
    kernel.open = False
    assert not launcher_tray.activate_existing_tray("lock", timeout=0)


def test_exit_disposes_icon_before_process_exit_and_activation_restores(monkeypatch):
    kernel = FakeKernel()
    ui = install_forms(monkeypatch, kernel)
    calls = ui.calls
    window = SimpleNamespace(hide=lambda: calls.append("hide"), show=lambda: calls.append("show"))
    tray = launcher_tray.WindowsTray(lambda: window, lambda: calls.append("process_exit"), "lock", threading.Event())

    def run(context):
        icon, timer = ui.icons[-1], ui.timers[-1]
        timer.Tick.fire(None, None)
        assert tray.ready.is_set()
        assert tray.hide_on_close(window)
        kernel.SetEvent(42)  # second launch asks first to show its hidden window
        timer.Tick.fire(None, None)
        assert calls == ["hide", "show"]
        assert tray.hide_on_close(window)
        # Exit menu invokes callback on the STA; no finally is required for cleanup.
        # Capture menu items below through a replacement ContextMenuStrip.
        menu_items[1].Click.fire(None, None)
        assert calls.index("dispose") < calls.index("process_exit")
        assert icon.Visible is False

    menu_items = []
    class Menu:
        def __init__(self):
            self.Items = SimpleNamespace(Add=menu_items.append)
    monkeypatch.setattr(sys.modules["System.Windows.Forms"], "ContextMenuStrip", Menu)
    ui.set_running(run)
    assert tray.start()
    assert kernel.closed == [42]


def test_panic_cleanup_is_bounded_and_precedes_lock_release(monkeypatch):
    calls = []
    class Tray:
        def stop(self, *, wait):
            calls.append(("stop", wait))
    monkeypatch.setattr(launcher_tray, "_active_tray", Tray())
    launcher_tray.stop_tray_before_exit(lambda: calls.append("release"))
    assert calls == [("stop", 0.5), "release"]
    calls.clear()
    launcher_tray.request_tray_cleanup()
    launcher_tray.stop_tray_before_exit(lambda: calls.append("release"), wait=0)
    assert calls == [("stop", 0.0), ("stop", 0), "release"]


def test_failed_show_keeps_tray_alive_for_next_activation(monkeypatch):
    kernel = FakeKernel()
    ui = install_forms(monkeypatch, kernel)
    attempts = []
    def show():
        attempts.append("show")
        if len(attempts) == 1:
            raise RuntimeError("temporary GUI error")
    window = SimpleNamespace(hide=lambda: None, show=show)
    tray = launcher_tray.WindowsTray(lambda: window, lambda: None, "lock", threading.Event())
    def run(context):
        timer = ui.timers[-1]
        timer.Tick.fire(None, None)
        assert tray.hide_on_close(window)
        kernel.SetEvent(42)
        timer.Tick.fire(None, None)
        assert tray.ready.is_set() and tray._hidden
        kernel.SetEvent(42)
        timer.Tick.fire(None, None)
        assert not tray._hidden and attempts == ["show", "show"]
        tray.stop()
    ui.set_running(run)
    assert tray.start()


def test_sta_failure_keeps_ordinary_close(monkeypatch):
    kernel = FakeKernel()
    install_forms(monkeypatch, kernel, fail_sta=True)
    tray = launcher_tray.WindowsTray(lambda: None, lambda: None, "lock", threading.Event())
    assert not tray.start()
    assert not tray.ready.is_set()
    assert kernel.closed == [42]


def test_automatic_launch_stays_hidden_with_ready_icon(monkeypatch):
    kernel = FakeKernel()
    ui = install_forms(monkeypatch, kernel)
    shown = []
    window = SimpleNamespace(show=lambda: shown.append("show"))
    tray = launcher_tray.WindowsTray(lambda: window, lambda: None, "lock", threading.Event())
    def run(context):
        ui.timers[-1].Tick.fire(None, None)
        assert tray.ready.is_set()
        tray.show_if_unavailable(window)
        assert shown == [] and tray._hidden
        tray.stop()
    ui.set_running(run)
    tray.attach(window, initially_hidden=True)


def test_automatic_launch_shows_window_if_tray_fails(monkeypatch):
    kernel = FakeKernel()
    install_forms(monkeypatch, kernel, fail_sta=True)
    shown = []
    window = SimpleNamespace(show=lambda: shown.append("show"))
    tray = launcher_tray.WindowsTray(lambda: window, lambda: None, "lock", threading.Event())
    tray.attach(window, initially_hidden=True)
    monkeypatch.setattr(tray.ready, "wait", lambda timeout: False)
    tray.show_if_unavailable(window)
    assert shown == ["show"] and not tray._hidden


def test_automatic_fallback_retries_transient_show_failure(monkeypatch):
    shown = []
    def show():
        shown.append("attempt")
        if len(shown) == 1:
            raise RuntimeError("GUI still initializing")
    window = SimpleNamespace(show=show)
    tray = launcher_tray.WindowsTray(lambda: window, lambda: None, "lock", threading.Event())
    tray._hidden = True
    monkeypatch.setattr(tray.ready, "wait", lambda timeout: False)
    tray.show_if_unavailable(window)
    assert shown == ["attempt", "attempt"] and not tray._hidden


def test_automatic_launch_exits_if_icon_and_window_both_fail(monkeypatch):
    exits = []
    def cannot_show():
        raise RuntimeError("GUI unavailable")
    window = SimpleNamespace(show=cannot_show)
    tray = launcher_tray.WindowsTray(lambda: window, lambda: exits.append("exit"),
                                     "lock", threading.Event())
    tray._hidden = True
    monkeypatch.setattr(tray.ready, "wait", lambda timeout: False)
    tray.show_if_unavailable(window)
    assert exits == ["exit"]


def test_launcher_automatic_window_and_gui_callback_are_wired():
    from pathlib import Path
    source = Path(__file__).resolve().parents[1] / "launcher.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    create = next(node for node in calls if isinstance(node.func, ast.Attribute)
                  and node.func.attr == "create_window"
                  and any(keyword.arg == "js_api" for keyword in node.keywords)
                  and any(keyword.arg == "hidden" for keyword in node.keywords))
    hidden = next(keyword.value for keyword in create.keywords if keyword.arg == "hidden")
    assert "options.launch_intent" in ast.unparse(hidden)
    assert any(isinstance(node.func, ast.Attribute) and node.func.attr == "start"
               and any(keyword.arg == "func" and "show_if_unavailable" in ast.unparse(keyword.value)
                       for keyword in node.keywords) for node in calls)


def test_initially_hidden_window_recovers_if_pump_dies():
    shown = []
    window = SimpleNamespace(show=lambda: shown.append("show"))
    tray = launcher_tray.WindowsTray(lambda: window, lambda: None, "lock", threading.Event())
    tray._hidden = True
    tray._pump_stopped()
    assert shown == ["show"] and not tray._hidden


def test_ready_icon_then_pump_death_exits_if_window_cannot_restore():
    calls = []
    def cannot_show():
        calls.append("show")
        raise RuntimeError("GUI unavailable")
    window = SimpleNamespace(show=cannot_show)
    tray = launcher_tray.WindowsTray(lambda: window, lambda: calls.append("exit"),
                                     "lock", threading.Event())
    tray._hidden = True
    tray.ready.set()
    tray._pump_stopped()
    assert calls == ["show", "show", "exit"] and not tray.ready.is_set()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows kernel event contract")
def test_real_windows_kernel_event_is_shared_by_same_install(tmp_path):
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
