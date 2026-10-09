"""The desktop window's caption follows its page's palette: contract checks with stand-ins.

The pywebview window, its WinForms form, the .NET preference event and the three
Win32 calls are stand-ins, so this runs on every CI OS. What it pins is the
contract of ``ouroboros/launcher_appearance.py``: the handle is taken in
``before_show`` at full width; every later write happens in a UI-thread action
posted with ``BeginInvoke`` and repaints the shown frame, whatever was written
before; out-of-order bridge calls cannot let an older palette or an older page
win, even a replaced page's first call arriving after its successor's; High
Contrast clears the attribute and its change re-applies; the window's close
releases the subscription; the Win32 signatures are typed. None of this is a
Windows pixel or a native Windows run.
"""
from __future__ import annotations

import ctypes
import sys
import types

import pytest

from ouroboros import launcher_appearance as la

FULL_WIDTH_HWND = 0x7FFF_0000_1234  # above 32 bits: a ToInt32 handle would not survive
# Each page names itself by when it became the window's page (theme.js: performance.timeOrigin).
PAGE_A = 1_791_100_000_000.0
PAGE_B = PAGE_A + 5_250.4  # the reload of page A, five seconds later


class Hook:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, fn):
        self.handlers.append(fn)
        return self

    def __isub__(self, fn):
        self.handlers.remove(fn)
        return self

    def fire(self, *args):
        for handler in list(self.handlers):
            handler(*args)


class Form:
    """pywebview 5.4's BrowserForm surface the leaf touches: Handle and BeginInvoke."""

    def __init__(self):
        self.Handle = types.SimpleNamespace(ToInt64=lambda: FULL_WIDTH_HWND)
        self.posted = []

    def BeginInvoke(self, action):
        self.posted.append(action)

    def pump(self):
        while self.posted:
            self.posted.pop(0)()


class Window:
    def __init__(self, background_color="#FFFFFF"):
        self.background_color = background_color
        self.events = types.SimpleNamespace(before_show=Hook(), closed=Hook())
        self.native = Form()


@pytest.fixture
def windows(monkeypatch):
    """The Windows path with its native seams replaced by recorders."""
    monkeypatch.setattr(la, "_WINDOWS", True)
    state = types.SimpleNamespace(writes=[], refreshes=[], high_contrast=False, hresult=0)

    def write(hwnd, dark):
        state.writes.append((hwnd, dark))
        return state.hresult

    monkeypatch.setattr(la, "_set_dark_frame", write)
    monkeypatch.setattr(la, "_refresh_frame", state.refreshes.append)
    monkeypatch.setattr(la, "_high_contrast", lambda: state.high_contrast)
    system = types.ModuleType("System")
    system.Action = lambda fn: fn
    events = types.SimpleNamespace(UserPreferenceChanged=Hook())
    win32 = types.ModuleType("Microsoft.Win32")
    win32.SystemEvents = events
    win32.UserPreferenceChangedEventHandler = lambda fn: fn
    microsoft = types.ModuleType("Microsoft")
    microsoft.Win32 = win32
    clr = types.ModuleType("clr")
    clr.AddReference = lambda name: None
    for name, module in (("System", system), ("Microsoft", microsoft), ("Microsoft.Win32", win32), ("clr", clr)):
        monkeypatch.setitem(sys.modules, name, module)
    state.preferences = events.UserPreferenceChanged
    return state


def _shown(window, appearance):
    appearance.attach(window)
    window.events.before_show.fire()
    return window.native


def test_the_frame_starts_with_the_windows_own_background_and_takes_a_full_width_handle(windows):
    main, setup = la.NativeAppearance(), la.NativeAppearance()
    _shown(Window("#0d0b0f"), main)  # the main window paints this dark colour before its page loads
    _shown(Window(), setup)  # pywebview's default white
    assert windows.writes == [(FULL_WIDTH_HWND, True)]  # setup's light frame is the untouched default
    assert windows.refreshes == []  # written before the window is shown: nothing to repaint


def test_only_the_newest_request_of_the_newest_page_paints(windows):
    appearance = la.NativeAppearance()
    form = _shown(Window("#0d0b0f"), appearance)
    assert appearance.request("light", PAGE_A, 2) == {"ok": True, "state": "scheduled"}
    # pywebview runs each bridge call on its own thread: the older call arrives late.
    assert appearance.request("dark", PAGE_A, 1) == {"ok": True, "state": "stale"}
    assert windows.writes == [(FULL_WIDTH_HWND, True)]  # nothing is written off the UI thread
    form.pump()
    assert windows.writes[-1] == (FULL_WIDTH_HWND, False) and windows.refreshes == [FULL_WIDTH_HWND]
    # A reload is a newer page; the replaced page's late call can never win again.
    assert appearance.request("dark", PAGE_B, 1)["state"] == "scheduled"
    assert appearance.request("light", PAGE_A, 9)["state"] == "stale"
    form.pump()
    assert windows.writes[-1] == (FULL_WIDTH_HWND, True)
    # Two queued actions both read the newest palette, so their order cannot matter.
    appearance.request("light", PAGE_B, 2)
    appearance.request("dark", PAGE_B, 3)
    form.pump()
    assert windows.writes[-1] == (FULL_WIDTH_HWND, True) and len(windows.writes) == 3


def test_a_replaced_pages_first_call_arriving_after_its_successors_cannot_win(windows):
    """Page A asked for light just before the reload, but that call's worker thread runs
    only after the reloaded page B's first call. Arrival order is not page order: A was
    never seen before, yet it is older than B, so B keeps the frame and goes on updating it."""
    appearance = la.NativeAppearance()
    form = _shown(Window("#0d0b0f"), appearance)
    assert appearance.request("dark", PAGE_B, 1)["state"] == "scheduled"
    assert appearance.request("light", PAGE_A, 1) == {"ok": True, "state": "stale"}
    form.pump()
    assert windows.writes == [(FULL_WIDTH_HWND, True)]  # B's dark was already the frame
    assert appearance.request("light", PAGE_B, 2)["state"] == "scheduled"
    form.pump()
    assert windows.writes[-1] == (FULL_WIDTH_HWND, False)
    # A page the back-forward cache restores after B speaks from its restore time.
    assert appearance.request("dark", PAGE_B + 60_000.0, 3)["state"] == "scheduled"


def test_a_light_window_already_shown_repaints_on_its_first_dark_write(windows):
    """The setup window starts light, so before_show writes nothing; its first dark write
    comes later, to a window on screen, and must still ask Windows to repaint the frame."""
    appearance = la.NativeAppearance()
    form = _shown(Window(), appearance)
    assert windows.writes == [] and windows.refreshes == []
    appearance.request("dark", PAGE_A, 1)
    form.pump()
    assert windows.writes == [(FULL_WIDTH_HWND, True)] and windows.refreshes == [FULL_WIDTH_HWND]
    appearance.request("light", PAGE_A, 2)
    form.pump()
    assert windows.writes[-1] == (FULL_WIDTH_HWND, False)
    assert windows.refreshes == [FULL_WIDTH_HWND, FULL_WIDTH_HWND]


def test_a_request_before_the_window_exists_is_written_when_it_is_shown(windows):
    appearance, window = la.NativeAppearance(), Window()
    appearance.attach(window)
    assert appearance.request("dark", PAGE_A, 1) == {"ok": True, "state": "pending"}
    assert windows.writes == []
    window.events.before_show.fire()
    assert windows.writes == [(FULL_WIDTH_HWND, True)]
    assert windows.refreshes == []  # before the first paint: nothing on screen to repaint


def test_high_contrast_clears_the_attribute_and_its_change_reapplies_the_page(windows):
    appearance = la.NativeAppearance()
    form = _shown(Window("#0d0b0f"), appearance)
    windows.high_contrast = True
    windows.preferences.fire(None, None)  # Windows announces a preference change; the page does not
    form.pump()
    assert windows.writes[-1] == (FULL_WIDTH_HWND, False)
    windows.high_contrast = False
    windows.preferences.fire(None, None)
    form.pump()
    assert windows.writes[-1] == (FULL_WIDTH_HWND, True)
    windows.preferences.fire(None, None)  # an unrelated preference: nothing to change
    form.pump()
    assert len(windows.writes) == 3


def test_the_windows_close_releases_the_subscription_and_stops_writing(windows):
    appearance, window = la.NativeAppearance(), Window("#0d0b0f")
    form = _shown(window, appearance)
    assert len(windows.preferences.handlers) == 1
    appearance.request("light", PAGE_A, 1)  # queued, then the window closes before it runs
    window.events.closed.fire()
    form.pump()
    assert windows.preferences.handlers == []
    assert windows.writes == [(FULL_WIDTH_HWND, True)]
    assert appearance.request("dark", PAGE_A, 2) == {"ok": False, "state": "closed"}


def test_a_refused_write_is_logged_once_and_claims_nothing(windows, caplog):
    windows.hresult = -2147024809  # E_INVALIDARG
    appearance = la.NativeAppearance()
    form = _shown(Window("#0d0b0f"), appearance)
    appearance.request("light", PAGE_A, 1)
    appearance.request("dark", PAGE_A, 2)
    form.pump()
    refusals = [record for record in caplog.records if "refused the caption tint" in record.getMessage()]
    assert len(refusals) == 1 and "0x80070057" in refusals[0].getMessage()
    assert windows.refreshes == []


def test_unknown_palettes_and_other_platforms_are_refused_typed(monkeypatch):
    appearance = la.NativeAppearance()
    assert appearance.request("sepia", PAGE_A, 1) == {"ok": False, "state": "invalid_theme"}
    monkeypatch.setattr(la, "_WINDOWS", True)
    # A call that cannot say when its page began has no place in the order.
    for page, sequence in (("page-a", 1), (None, 1), (float("nan"), 1), (float("inf"), 1), (PAGE_A, None)):
        assert appearance.request("dark", page, sequence) == {"ok": False, "state": "invalid_order"}
    monkeypatch.setattr(la, "_WINDOWS", False)
    window = Window("#0d0b0f")
    appearance.attach(window)
    assert window.events.before_show.handlers == [] and window.events.closed.handlers == []
    assert appearance.request("dark", PAGE_A, 1) == {"ok": False, "state": "unsupported"}


@pytest.mark.parametrize("color,palette", [("#0d0b0f", "dark"), ("#FFFFFF", "light"), ("fff", "light"),
                                           ("#1a1a2e", "dark"), ("", ""), ("#12", ""), ("#zzzzzz", "")])
def test_the_starting_palette_reads_the_windows_background(color, palette):
    assert la.palette_of_background(color) == palette


class _Function:
    def __init__(self, calls, name, result):
        self.calls, self.name, self.result = calls, name, result

    def __call__(self, *args):
        self.calls.append((self.name, self.argtypes, self.restype, args))
        if self.name == "SystemParametersInfoW":
            info = args[2]._obj  # the HIGHCONTRASTW the caller passed by reference
            assert info.cbSize == ctypes.sizeof(info) == args[1]
            info.dwFlags = la._HCF_HIGHCONTRASTON
        return self.result


def test_the_win32_calls_are_typed_at_full_width(monkeypatch):
    from ctypes import wintypes

    calls = []

    class Library:
        def __init__(self, name):
            self.name = name

        def __getattr__(self, function):
            found = _Function(calls, function, 0 if function == "DwmSetWindowAttribute" else 1)
            setattr(self, function, found)
            return found

    monkeypatch.setattr(ctypes, "WinDLL", Library, raising=False)
    assert la._set_dark_frame(FULL_WIDTH_HWND, True) == 0
    assert la._high_contrast() is True
    la._refresh_frame(FULL_WIDTH_HWND)
    dwm, spi, swp = calls
    assert dwm[1] == (wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD) and dwm[2] is ctypes.c_long
    hwnd, attribute, value, size = dwm[3]
    assert hwnd.value == FULL_WIDTH_HWND and attribute == 20 and size == ctypes.sizeof(wintypes.BOOL)
    assert spi[0] == "SystemParametersInfoW" and spi[3][0] == la._SPI_GETHIGHCONTRAST and spi[2] is wintypes.BOOL
    assert swp[3][0].value == FULL_WIDTH_HWND and swp[3][-1] == la._SWP_REFRESH_FRAME
    assert swp[1][0] is wintypes.HWND and swp[2] is wintypes.BOOL


def test_each_desktop_window_owns_its_appearance_and_bridge(windows, monkeypatch):
    """The main window registers its hooks whether or not a background indicator exists,
    its bridge (``DesktopApi``, inherited by ``launcher.MainApi``) reaches ITS appearance,
    and the first-run setup window carries a separate one through ``OnboardingHostApi``."""
    import threading

    from ouroboros import launcher_background as lb
    from ouroboros import launcher_onboarding

    monkeypatch.setattr(lb, "indicator_class", lambda: None)
    background = lb.Background(lambda: None, lambda: 8765, threading.Event())
    main = Window("#0d0b0f")
    main.events.closing, main.events.shown = Hook(), Hook()
    background.attach(main)
    assert main.events.before_show.handlers == [background.appearance._before_show]  # no indicator hook

    class Api(lb.DesktopApi):
        _background = background

    assert Api().set_native_appearance("light", PAGE_A, 1) == {"ok": True, "state": "pending"}
    main.events.before_show.fire()
    assert windows.writes == []  # the page's light arrived before the first paint: the default stays

    created = {}
    fake = types.ModuleType("webview")
    fake.windows = []

    def create_window(title, url=None, js_api=None, **kwargs):
        created.update(js_api=js_api, window=Window())
        return created["window"]

    fake.create_window, fake.start = create_window, lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "webview", fake)
    launcher_onboarding.present_first_run_onboarding({}, 8765, open_external_url=lambda url: {"ok": False})
    setup = created["window"]
    assert created["js_api"].set_native_appearance("dark", PAGE_A, 1) == {"ok": True, "state": "pending"}
    setup.events.before_show.fire()
    assert windows.writes == [(FULL_WIDTH_HWND, True)]  # the setup window's own handle and request
