"""The macOS main window's integrated title bar: contract checks with AppKit and WebKit stand-ins.

The NSWindow, its views, the WKWebView and the AppKit/WebKit/Foundation modules are stand-ins,
so this runs on every CI OS. What it pins is the contract of
``ouroboros/launcher_window_chrome.py``: the document script is installed before the first load in
its own world; the window starts integrated, only a document that declares keeps it, and only what is
applied and measured is written into that document; an undeclared, script-less or older document
cannot leave the window integrated; full screen hides the toolbar band and reserves nothing; a failed
switch restores the ordinary bar; drag, double-click and appearance use the window's own facts; every
window opens in the room colour of its own appearance; a warm source handoff runs its teardown on the main
thread only in a turn that found every window out of full screen by AppKit's four notifications and the live
style mask, asking a full-screen window once to leave, and refuses unknown, hidden or unconfirmed windows.
Where the native qualification (macOS 26.4) saw AppKit behave otherwise than assumed, the stand-ins
do what it saw. None of this is a native macOS run: pixels, gestures and AppKit's real geometry
belong to that qualification (ARCHITECTURE §3 "Navigation and shared UI contracts").
"""
from __future__ import annotations

import json
import re
import sys
import threading
import types
from pathlib import Path

import pytest

from ouroboros import launcher_appearance
from ouroboros import launcher_window_chrome as wc

ROOT = Path(__file__).resolve().parents[1]
DOC_A = 1_791_100_000_000.25
DOC_B = DOC_A + 5_250.5  # the next document of the same window
TITLE_BAR, TOOLBAR_BAND = 28.0, 52.0  # the stand-in AppKit's bar heights (a native run measures real ones)


class Hook:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, fn):
        self.handlers.append(fn)
        return self

    def fire(self):
        for handler in list(self.handlers):
            handler()


class Rect:
    def __init__(self, x, y, width, height):
        self.origin = types.SimpleNamespace(x=x, y=y)
        self.size = types.SimpleNamespace(width=width, height=height)

    def __eq__(self, other):
        return (self.origin, self.size) == (other.origin, other.size)


class View:
    def __init__(self, frame=None, children=()):
        self._frame, self.children, self.colors = frame, list(children), []
        self.parent = None

    def frame(self):
        return self._frame()

    def superview(self):
        return self.parent

    def subviews(self):
        return types.SimpleNamespace(lastObject=lambda: self.children[-1])

    def setBackgroundColor_(self, color):
        self.colors.append(color)


class TitleBar:
    """AppKit's title bar view (the traffic lights' superview). Natively, a press handed to its ``mouseDown:``
    under the transparent bar went back into the web view, which reported it again until release (383
    messages for two double-clicks), and the window never zoomed: nothing may be handed to it."""

    def __init__(self):
        self.presses = []

    def mouseDown_(self, event):
        self.presses.append(event)


class Button:
    def __init__(self, container, x, title_bar):
        self.container, self.x, self.hidden_flag, self.title_bar = container, x, False, title_bar

    def isDescendantOf_(self, view):
        return view is self.container

    def superview(self):
        return self.title_bar

    def isHidden(self):
        return self.hidden_flag

    def bounds(self):
        return Rect(0, 0, 14, 16)

    def convertRect_toView_(self, rect, view):
        assert view is None  # window coordinates
        return Rect(self.x, 18, rect.size.width, rect.size.height)


class Toolbar:
    def __init__(self, identifier):
        self.identifier, self.visible = identifier, True

    def setVisible_(self, visible):
        self.visible = visible


class NSWindow:
    """The NSWindow surface the leaf touches; its geometry follows the real rules in miniature."""

    def __init__(self, fill_holds_buttons=True):
        self.mask = 0b1111  # titled, closable, miniaturizable, resizable
        self._frame = Rect(100, 100, 1100, 750)
        self.transparent, self.title_visibility, self._toolbar = False, 0, None
        self.toolbar_style, self.separator, self.tabbing = 0, 0, 0
        self.appearances, self.calls, self.backgrounds = [], [], []
        self.exiting_full_screen = False
        self.visible = True
        self.fill = View()
        self.content = View(lambda: Rect(0, 0, self._frame.size.width, self.content_height()))
        theme = View(children=[self.content, self.fill])
        self.content.parent = theme
        self.title_bar = TitleBar()
        self.buttons = [Button(self.fill if fill_holds_buttons else View(), x, self.title_bar) for x in (19, 39, 59)]
        self.fail_on = None

    def _maybe_fail(self, name):
        if self.fail_on == name:
            raise RuntimeError(f"{name} refused")

    def bar(self):
        return TOOLBAR_BAND if self._toolbar is not None and self._toolbar.visible else TITLE_BAR

    def content_height(self):
        full = bool(self.mask & wc._FULL_SIZE_CONTENT)
        return self._frame.size.height - (0 if full else self.bar())

    def contentLayoutRect(self):
        self._maybe_fail("layout")
        return Rect(0, 0, self._frame.size.width, self._frame.size.height - self.bar())

    def contentView(self):
        return self.content

    def frame(self):
        return Rect(self._frame.origin.x, self._frame.origin.y, self._frame.size.width, self._frame.size.height)

    def setFrame_display_(self, frame, display):
        self._frame = frame
        self.calls.append("frame")

    def styleMask(self):
        return self.mask

    def setStyleMask_(self, mask):
        self._maybe_fail("mask")
        change = (mask & wc._FULL_SIZE_CONTENT) - (self.mask & wc._FULL_SIZE_CONTENT)
        if change:  # AppKit keeps the content size, so the frame loses or regains the bar
            self._frame = Rect(100, 100, 1100, self._frame.size.height - TITLE_BAR * (1 if change > 0 else -1))
        self.mask = mask
        self.calls.append("mask")

    def setTitlebarAppearsTransparent_(self, value):
        self.transparent = value

    def setTitleVisibility_(self, value):
        self.title_visibility = value

    def toolbar(self):
        return self._toolbar

    def setToolbar_(self, toolbar):
        self._maybe_fail("toolbar")
        if toolbar is not None and self.exiting_full_screen:
            toolbar.visible = False  # natively, a toolbar set during DidExit read hidden afterwards
        self._toolbar = toolbar

    def toolbarStyle(self):
        return self.toolbar_style

    def setToolbarStyle_(self, value):
        self.toolbar_style = value

    def titlebarSeparatorStyle(self):
        return self.separator

    def setTitlebarSeparatorStyle_(self, value):
        self.separator = value

    def tabbingMode(self):
        return self.tabbing

    def setTabbingMode_(self, value):
        self.tabbing = value

    def standardWindowButton_(self, kind):
        return self.buttons[kind]

    def setAppearance_(self, appearance):
        self.appearances.append(appearance)

    def setBackgroundColor_(self, color):
        self.backgrounds.append(color)

    def performWindowDragWithEvent_(self, event):
        self.calls.append(("drag", event))

    def performMiniaturize_(self, sender):
        self.calls.append("miniaturize")

    def performZoom_(self, sender):
        self.calls.append("zoom")

    def isVisible(self):
        return self.visible

    def toggleFullScreen_(self, sender):
        self.calls.append("toggle full screen")  # AppKit then posts the exit's notifications; the test fires them


class WebView:
    def __init__(self):
        self.scripts, self.handlers, self.evaluated, self.zoom = [], [], [], 1.0
        controller = types.SimpleNamespace(
            addScriptMessageHandler_contentWorld_name_=lambda handler, world, name: self.handlers.append((handler, world, name)),
            addUserScript_=self.scripts.append)
        self._configuration = types.SimpleNamespace(userContentController=lambda: controller)

    def configuration(self):
        return self._configuration

    def evaluateJavaScript_inFrame_inContentWorld_completionHandler_(self, script, frame, world, handler):
        assert frame is None and handler is None and world == ("world", wc.WORLD)
        self.evaluated.append(json.loads(re.search(r"__ouroWindowChrome\((\{.*\})\)$", script).group(1)))

    def pageZoom(self):
        return self.zoom

    def magnification(self):
        return 1.0


class Window:
    def __init__(self, native):
        self.native, self.uid = native, "main"
        self.events = types.SimpleNamespace(before_show=Hook(), before_load=Hook(), closed=Hook())


class MainThread:
    """The main run loop in miniature: ``callAfter`` queues a turn; the test thread takes the turns in order."""

    def __init__(self):
        self.turns, self.posted = [], threading.Condition()

    def callAfter(self, fn, *args):
        with self.posted:
            self.turns.append((fn, args))
            self.posted.notify_all()

    def run(self, *, first=False):
        """Every queued turn, including those queued meanwhile; ``first`` waits for the worker's own post."""
        with self.posted:
            assert not first or self.posted.wait_for(lambda: self.turns, timeout=5), "the worker never posted"
        while True:
            with self.posted:
                if not self.turns:
                    return
                fn, args = self.turns.pop(0)
            fn(*args)


@pytest.fixture
def mac(monkeypatch):
    """The macOS path with AppKit, WebKit, Foundation, PyObjC's AppHelper and pywebview's window list replaced."""
    state = types.SimpleNamespace(observers={}, removed=[], event=None, webview=WebView(), defaults={},
                                  main=MainThread(), pywebview=types.ModuleType("webview"))
    state.pywebview.windows = []
    monkeypatch.setattr(wc, "_MACOS", True)
    monkeypatch.setattr(wc, "_classes", {})
    monkeypatch.setattr(wc, "_tracked", [], raising=False)
    monkeypatch.setattr(wc, "_transfer", None, raising=False)
    monkeypatch.setattr(wc, "_webview_of", lambda window: state.webview)

    foundation = types.ModuleType("Foundation")

    class NSObject:
        @classmethod
        def alloc(cls):
            return cls()

        def init(self):
            return self

    foundation.NSObject = NSObject
    foundation.NSUserDefaults = types.SimpleNamespace(standardUserDefaults=lambda: types.SimpleNamespace(
        stringForKey_=state.defaults.get))  # the app's then the global domain: a missing key is None
    webkit = types.ModuleType("WebKit")
    webkit.WKUserScriptInjectionTimeAtDocumentStart = "document-start"
    webkit.WKContentWorld = types.SimpleNamespace(worldWithName_=lambda name: ("world", name))

    class WKUserScript:
        @classmethod
        def alloc(cls):
            return cls()

        def initWithSource_injectionTime_forMainFrameOnly_inContentWorld_(self, source, time, main_only, world):
            self.source, self.time, self.main_only, self.world = source, time, main_only, world
            return self

    webkit.WKUserScript = WKUserScript

    class Center:
        def addObserverForName_object_queue_usingBlock_(self, name, obj, queue, block):
            assert queue is None
            state.observers[name] = block
            return name

        def removeObserver_(self, token):
            state.removed.append(token)

    center = Center()
    appkit = types.ModuleType("AppKit")
    appkit.NSColor = types.SimpleNamespace(
        clearColor=lambda: "clear", windowBackgroundColor=lambda: "window-background",
        colorWithSRGBRed_green_blue_alpha_=lambda red, green, blue, alpha: (
            "srgb", round(red * 255), round(green * 255), round(blue * 255), alpha),
        colorWithName_dynamicProvider_=lambda name, provider: ("dynamic", name, provider))
    appkit.NSToolbar = types.SimpleNamespace(alloc=lambda: types.SimpleNamespace(initWithIdentifier_=Toolbar))
    appkit.NSNotificationCenter = types.SimpleNamespace(defaultCenter=lambda: center)
    # A value no hand-typed number equals: the style must be the binding's own name.
    appkit.NSWindowToolbarStyleUnified = ("toolbar-style", "NSWindowToolbarStyleUnified")
    # No setAppearance_ on the application: a global override would raise here.
    appkit.NSApplication = types.SimpleNamespace(
        sharedApplication=lambda: types.SimpleNamespace(currentEvent=lambda: state.event))
    appkit.NSAppearance = types.SimpleNamespace(appearanceNamed_=lambda name: ("appearance", name))
    appkit.NSAppearanceNameAqua, appkit.NSAppearanceNameDarkAqua = "Aqua", "DarkAqua"
    pyobjc_tools = types.ModuleType("PyObjCTools")
    pyobjc_tools.AppHelper = state.main
    for name, module in (("AppKit", appkit), ("WebKit", webkit), ("Foundation", foundation),
                         ("PyObjCTools", pyobjc_tools), ("webview", state.pywebview)):
        monkeypatch.setitem(sys.modules, name, module)
    state.appkit = appkit
    return state


def shown(native=None):
    native = native or NSWindow()
    chrome = wc.WindowChrome()
    window = Window(native)
    chrome.attach(window)
    window.events.before_show.fire()
    return chrome, window, native


def declare(chrome, doc, integrated=True, choice="system", theme="dark"):
    chrome.receive({"type": "declare", "doc": doc, "chrome": wc.INTEGRATED if integrated else "",
                    "choice": choice, "theme": theme})


def leave_full_screen(mac, native):
    """AppKit's exit as the native qualification saw it in every round trip: WillExit showed the toolbar,
    yet the window ended the exit with it hidden, and a toolbar set during DidExit read hidden too."""
    mac.observers["NSWindowWillExitFullScreenNotification"](None)
    native.exiting_full_screen = True
    if native.toolbar() is not None:
        native.toolbar().visible = False
    mac.observers["NSWindowDidExitFullScreenNotification"](None)
    native.exiting_full_screen = False


def assert_ordinary(native):
    assert not native.mask & wc._FULL_SIZE_CONTENT and native.mask == 0b1111
    assert (native.transparent, native.title_visibility, native.toolbar()) == (False, 0, None)
    assert (native.toolbar_style, native.separator, native.tabbing) == (0, 0, 0)


def test_the_leaf_is_inert_off_macos(monkeypatch):
    monkeypatch.setattr(wc, "_MACOS", False)
    monkeypatch.setattr(wc, "_LINUX", False)  # GTK's one before_show: test_on_gtk_...
    monkeypatch.setattr(wc, "host_palette", lambda: "")  # the room off macOS: test_launcher_appearance
    window = Window(NSWindow())
    wc.WindowChrome().attach(window)
    assert not (window.events.before_show.handlers or window.events.before_load.handlers or window.events.closed.handlers)


def test_the_document_script_is_installed_in_its_own_world_before_the_first_load(mac):
    chrome, window, native = shown()
    (handler, world, name), = mac.webview.handlers
    script, = mac.webview.scripts
    assert (world, name) == (("world", wc.WORLD), wc.HANDLER) and handler.chrome is chrome
    assert (script.source, script.time, script.main_only, script.world) == (
        wc.DOCUMENT_SCRIPT, "document-start", True, ("world", wc.WORLD))
    assert mac.webview.evaluated == [], "no document yet: nothing to tell, nothing reserved"


def test_the_window_starts_integrated_and_an_undeclaring_first_document_returns_it_unseen(mac):
    """pywebview shows the web view only after the first document finished loading, and that document
    declares while it parses: the bar never flashes at launch, and an older or custom first UI is ordinary
    again before it appears."""
    chrome, window, native = shown()
    assert native.mask & wc._FULL_SIZE_CONTENT and native.transparent and native.toolbar() is not None
    assert native.frame() == Rect(100, 100, 1100, 750)
    declare(chrome, DOC_A, integrated=False)
    assert_ordinary(native)
    assert mac.webview.evaluated == [{"doc": DOC_A, "top": 0, "start": 0}]


def test_a_declaring_document_gets_the_integrated_bar_and_only_its_measured_geometry(mac):
    chrome, window, native = shown()
    declare(chrome, DOC_A)
    assert native.mask == 0b1111 | wc._FULL_SIZE_CONTENT
    assert (native.transparent, native.title_visibility) == (True, wc._TITLE_HIDDEN)
    assert native.fill.colors == ["clear"]  # exactly the view pywebview painted
    assert native.toolbar().identifier == wc.TOOLBAR
    assert native.toolbar_style == mac.appkit.NSWindowToolbarStyleUnified, "AppKit's Unified style, by name"
    assert (native.separator, native.tabbing) == (wc._SEPARATOR_NONE, wc._TABBING_DISALLOWED)
    assert native.frame() == Rect(100, 100, 1100, 750), "the window keeps the size the launcher asked for"
    assert all(not button.hidden_flag for button in native.buttons)  # the real traffic lights stay
    assert mac.webview.evaluated == [{"doc": DOC_A, "top": TOOLBAR_BAND, "start": 73.0}]


def test_page_zoom_turns_points_into_css_pixels(mac):
    mac.webview.zoom = 1.25
    chrome, _window, _native = shown()
    declare(chrome, DOC_A)
    assert mac.webview.evaluated == [{"doc": DOC_A, "top": round(TOOLBAR_BAND / 1.25, 2), "start": round(73 / 1.25, 2)}]


def test_an_undeclared_document_gets_the_ordinary_bar_back_and_reserves_nothing(mac):
    chrome, _window, native = shown()
    declare(chrome, DOC_A)
    declare(chrome, DOC_B, integrated=False)  # an older, custom or replaced UI
    assert_ordinary(native)
    assert native.fill.colors == ["clear", "window-background"]  # pywebview's own fill again
    assert native.frame() == Rect(100, 100, 1100, 750)
    assert mac.webview.evaluated[-1] == {"doc": DOC_B, "top": 0, "start": 0}


def test_a_reload_of_a_declaring_page_tells_the_new_document_without_touching_the_window(mac):
    chrome, _window, native = shown()
    declare(chrome, DOC_A)
    calls = list(native.calls)
    declare(chrome, DOC_B)
    assert native.calls == calls
    assert mac.webview.evaluated[-1] == {"doc": DOC_B, "top": TOOLBAR_BAND, "start": 73.0}


def test_an_older_documents_late_messages_never_undo_a_newer_one(mac):
    chrome, _window, native = shown()
    declare(chrome, DOC_B)
    declare(chrome, DOC_A, integrated=False, choice="light")
    chrome.receive({"type": "appearance", "doc": DOC_A, "choice": "dark", "theme": "dark"})
    mac.event = types.SimpleNamespace(type=lambda: wc._LEFT_MOUSE_DOWN)
    chrome.receive({"type": "drag", "doc": DOC_A})
    assert native.mask & wc._FULL_SIZE_CONTENT and native.appearances == []
    assert ("drag", mac.event) not in native.calls
    for malformed in ({"type": "declare"}, {"type": "declare", "doc": "nan"}, "not a message"):
        chrome.receive(malformed)
    assert native.mask & wc._FULL_SIZE_CONTENT


def test_a_document_that_never_ran_the_script_is_ordinary(mac):
    chrome, window, native = shown()
    declare(chrome, DOC_A, choice="light")
    window.events.before_load.fire()  # that document finished loading: it had declared
    assert native.mask & wc._FULL_SIZE_CONTENT and native.appearances == [("appearance", "Aqua")]
    window.events.before_load.fire()  # a later navigation finished without a word from its document
    assert_ordinary(native)
    assert native.appearances[-1] is None, "an unthemed document inherits the system appearance"


def test_full_screen_hides_the_toolbar_band_reserves_nothing_and_defers_the_switches(mac):
    chrome, _window, native = shown()
    declare(chrome, DOC_A)
    mac.observers["NSWindowWillEnterFullScreenNotification"](None)
    assert native.toolbar().visible is False, "no permanent band over a full-screen page"
    assert mac.webview.evaluated[-1] == {"doc": DOC_A, "top": 0, "start": 0}
    declare(chrome, DOC_B, integrated=False)  # a replaced UI arrives in full screen
    assert native.mask & wc._FULL_SIZE_CONTENT, "the style mask never changes in full screen"
    assert mac.webview.evaluated[-1] == {"doc": DOC_B, "top": 0, "start": 0}
    mac.observers["NSWindowWillExitFullScreenNotification"](None)
    assert native.toolbar().visible is True
    mac.observers["NSWindowDidExitFullScreenNotification"](None)
    assert_ordinary(native)  # the deferred decision lands once the window is out


def test_the_toolbar_band_is_back_after_full_screen_though_appkit_ends_the_exit_with_it_hidden(mac):
    """Natively every exit left the toolbar hidden: the band shrank to the bare title bar until relaunch,
    after a round trip and when a declaring UI that arrived in full screen was integrated at DidExit."""
    chrome, _window, native = shown()
    declare(chrome, DOC_A)
    mac.observers["NSWindowWillEnterFullScreenNotification"](None)
    leave_full_screen(mac, native)
    assert native.toolbar().visible is True
    assert mac.webview.evaluated[-1] == {"doc": DOC_A, "top": TOOLBAR_BAND, "start": 73.0}

    declare(chrome, DOC_B, integrated=False)  # an older UI, then the new one arrives in full screen
    mac.observers["NSWindowWillEnterFullScreenNotification"](None)
    declare(chrome, DOC_B + 1)
    assert_ordinary(native)
    leave_full_screen(mac, native)
    assert native.mask & wc._FULL_SIZE_CONTENT and native.toolbar().visible is True
    assert mac.webview.evaluated[-1] == {"doc": DOC_B + 1, "top": TOOLBAR_BAND, "start": 73.0}
    assert all(state["top"] in (0, TOOLBAR_BAND) for state in mac.webview.evaluated), \
        "the page is never told to reserve the bare title bar under the integrated one"


def test_a_failed_switch_restores_the_ordinary_bar_and_is_not_retried(mac):
    native = NSWindow()
    native.fail_on = "toolbar"
    chrome, _window, native = shown(native)
    declare(chrome, DOC_A)
    native.fail_on = None
    assert_ordinary(native)
    assert mac.webview.evaluated == [{"doc": DOC_A, "top": 0, "start": 0}]
    declare(chrome, DOC_B)
    assert_ordinary(native)


def test_a_fill_that_does_not_hold_the_traffic_lights_is_never_cleared(mac):
    chrome, _window, native = shown(NSWindow(fill_holds_buttons=False))
    declare(chrome, DOC_A, choice="dark")
    assert_ordinary(native)
    assert native.fill.colors == []
    assert mac.webview.evaluated == [{"doc": DOC_A, "top": 0, "start": 0}]
    assert native.appearances == [("appearance", "DarkAqua")], "the palette still follows the page"


def test_a_press_drags_with_appkits_own_window_drag_only_while_it_is_live(mac):
    chrome, _window, native = shown()
    press = types.SimpleNamespace(type=lambda: wc._LEFT_MOUSE_DOWN)
    mac.event = press
    chrome.receive({"type": "drag", "doc": DOC_A})  # no document has declared: not the window's page
    declare(chrome, DOC_A, integrated=False)
    chrome.receive({"type": "drag", "doc": DOC_A})
    assert ("drag", press) not in native.calls, "an ordinary title bar drags itself"
    declare(chrome, DOC_B)
    chrome.receive({"type": "drag", "doc": DOC_B})
    assert native.calls[-1] == ("drag", press)
    mac.event = types.SimpleNamespace(type=lambda: 2)  # the button is already up
    chrome.receive({"type": "drag", "doc": DOC_B})
    assert native.calls[-1] == ("drag", press)
    mac.event = press
    mac.observers["NSWindowWillEnterFullScreenNotification"](None)
    chrome.receive({"type": "drag", "doc": DOC_B})
    assert native.calls.count(("drag", press)) == 1


def press(kind=wc._LEFT_MOUSE_DOWN, clicks=1):
    return types.SimpleNamespace(type=lambda: kind, clickCount=lambda: clicks)


@pytest.mark.parametrize("setting, action", [
    (None, "zoom"),  # never chosen: Zoom, as on the native run whose ordinary bar zoomed
    ("Minimize", "miniaturize"),
    ("None", None),  # Do Nothing
    ("Fill", "fill"),  # a distinct native action, never substituted by Zoom
    ("Tile Left", None),  # a word this leaf does not know
])
def test_a_double_click_in_the_band_does_what_the_macs_title_bar_setting_says(mac, setting, action):
    chrome, _window, native = shown()
    native.respondsToSelector_ = lambda selector: selector == "_zoomFill:"
    native._zoomFill_ = lambda sender: native.calls.append("fill")
    declare(chrome, DOC_A)
    if setting is not None:
        mac.defaults["AppleActionOnDoubleClick"] = setting
    before = len(native.calls)
    mac.event = press(clicks=2)
    chrome.receive({"type": "double-click", "doc": DOC_A})
    assert native.calls[before:] == ([action] if action else [])
    assert native.title_bar.presses == [], "a press handed to the title bar view comes back to the page"


def test_missing_native_fill_never_substitutes_zoom(mac, caplog):
    chrome, _window, native = shown()
    declare(chrome, DOC_A)
    mac.defaults[wc._DOUBLE_CLICK_SETTING] = "Fill"
    mac.event = press(clicks=2)
    native.respondsToSelector_ = lambda selector: False
    before = list(native.calls)
    with caplog.at_level("INFO", logger="launcher.window_chrome"):
        chrome.receive({"type": "double-click", "doc": DOC_A})
    assert native.calls == before
    assert "unavailable" in caplog.text
    assert native.title_bar.presses == []


def test_only_a_live_double_click_of_an_integrated_window_acts_once(mac):
    chrome, _window, native = shown()
    mac.event = press(clicks=2)
    chrome.receive({"type": "double-click", "doc": DOC_A})  # no document has declared
    declare(chrome, DOC_A)
    for gone in (press(clicks=1), press(clicks=3), press(kind=wc._LEFT_MOUSE_DRAGGED, clicks=2),
                 press(kind=5, clicks=0), None):
        mac.event = gone  # a single press, a triple, a drag, a later mouse move, no event: no double-click
        chrome.receive({"type": "double-click", "doc": DOC_A})
    assert "zoom" not in native.calls
    mac.event = press(kind=wc._LEFT_MOUSE_UP, clicks=2)  # the message came after the release
    chrome.receive({"type": "double-click", "doc": DOC_A})
    assert native.calls.count("zoom") == 1
    mac.defaults["AppleActionOnDoubleClick"] = "Minimize"  # changed in System Settings meanwhile
    mac.event = press(clicks=2)
    chrome.receive({"type": "double-click", "doc": DOC_A})
    assert native.calls[-1] == "miniaturize"
    mac.observers["NSWindowWillEnterFullScreenNotification"](None)
    chrome.receive({"type": "double-click", "doc": DOC_A})
    declare(chrome, DOC_B, integrated=False)
    leave_full_screen(mac, native)
    chrome.receive({"type": "double-click", "doc": DOC_B})  # the ordinary bar handles its own clicks
    assert (native.calls.count("zoom"), native.calls.count("miniaturize")) == (1, 1)
    assert native.title_bar.presses == []


def test_unreadable_geometry_returns_the_ordinary_bar_instead_of_reserving_nothing(mac):
    """Zero reserve under a still-transparent bar would let the bar cover the page's controls."""
    chrome, _window, native = shown()
    native.fail_on = "layout"
    declare(chrome, DOC_A)
    native.fail_on = None
    assert_ordinary(native)
    assert mac.webview.evaluated == [{"doc": DOC_A, "top": 0, "start": 0}]
    declare(chrome, DOC_B)  # for this window's lifetime, as after a refused switch
    assert_ordinary(native)


def test_unreadable_geometry_at_the_end_of_full_screen_returns_the_ordinary_bar(mac):
    chrome, _window, native = shown()
    declare(chrome, DOC_A)
    mac.observers["NSWindowWillEnterFullScreenNotification"](None)
    native.fail_on = "layout"
    mac.observers["NSWindowWillExitFullScreenNotification"](None)
    assert native.mask & wc._FULL_SIZE_CONTENT, "nothing is measured, and the mask is kept, in full screen"
    mac.observers["NSWindowDidExitFullScreenNotification"](None)
    native.fail_on = None
    assert_ordinary(native)
    assert mac.webview.evaluated[-1] == {"doc": DOC_A, "top": 0, "start": 0}


def test_a_state_webkit_refused_is_told_again(mac):
    """A push that raised was never delivered, so the same state is not a duplicate next time."""
    chrome, _window, native = shown()
    original = mac.webview.evaluateJavaScript_inFrame_inContentWorld_completionHandler_
    refusals = []

    def refuse_once(*args):
        if not refusals:
            refusals.append(args)
            raise RuntimeError("web view busy")
        original(*args)

    mac.webview.evaluateJavaScript_inFrame_inContentWorld_completionHandler_ = refuse_once
    declare(chrome, DOC_A)
    assert refusals and mac.webview.evaluated == []
    mac.observers["NSWindowWillExitFullScreenNotification"](None)  # any later change publishes again
    mac.observers["NSWindowDidExitFullScreenNotification"](None)
    assert mac.webview.evaluated == [{"doc": DOC_A, "top": TOOLBAR_BAND, "start": 73.0}]


def test_the_window_appearance_follows_the_painted_choice_and_system_inherits(mac):
    chrome, _window, native = shown()
    declare(chrome, DOC_A, choice="light", theme="light")
    chrome.receive({"type": "appearance", "doc": DOC_A, "choice": "dark", "theme": "dark"})
    chrome.receive({"type": "appearance", "doc": DOC_A, "choice": "dark", "theme": "dark"})  # unchanged: no write
    # System must inherit: the window's appearance is also the page's prefers-color-scheme.
    chrome.receive({"type": "appearance", "doc": DOC_A, "choice": "system", "theme": "light"})
    chrome.receive({"type": "appearance", "doc": DOC_A, "choice": "", "theme": "light"})  # a page before System
    declare(chrome, DOC_B, integrated=False, choice="", theme="")  # an unthemed UI
    assert native.appearances == [("appearance", "Aqua"), ("appearance", "DarkAqua"), None,
                                  ("appearance", "Aqua"), None]


class Appearance:
    """An NSAppearance as the room colour's provider meets it: AppKit's own best match among the names."""

    def __init__(self, best):
        self.best, self.asked = best, []

    def bestMatchFromAppearancesWithNames_(self, names):
        self.asked.append(list(names))
        return self.best


LIGHT_ROOM, DARK_ROOM = ("srgb", 255, 255, 255, 1.0), ("srgb", 15, 15, 17, 1.0)


def room(native):
    """The one dynamic colour the window opened with: its provider, as AppKit calls it at each draw."""
    (kind, name, provider), = native.backgrounds
    assert (kind, name) == ("dynamic", None)
    return provider


def test_the_window_opens_in_the_room_colour_of_its_own_appearance(mac):
    """Before the window is shown, pywebview's fixed colour becomes one dynamic colour AppKit resolves for the
    appearance it draws in: the Mac's until a page forces Light or Dark (owner decision on #1111: A)."""
    _chrome, _window, native = shown()
    provider = room(native)
    aqua, dark_aqua = Appearance("Aqua"), Appearance("DarkAqua")
    assert provider(aqua) == LIGHT_ROOM and provider(dark_aqua) == DARK_ROOM
    assert aqua.asked == [["Aqua", "DarkAqua"]]  # AppKit's names: a High Contrast variant matches its base
    assert provider(Appearance(None)) == DARK_ROOM  # nothing matched: the room this window always had


def test_a_window_shown_before_the_server_gets_the_room_and_keeps_its_ordinary_bar(mac):
    window = Window(NSWindow())
    window.html = "<!DOCTYPE html>"  # no caption owner here: macOS has none to give it
    wc.paint_room(window)
    assert len(window.events.before_show.handlers) == 1 and window.events.before_load.handlers == []
    window.events.before_show.fire()
    assert room(window.native)(Appearance("Aqua")) == LIGHT_ROOM
    assert_ordinary(window.native)
    assert window.native.appearances == [], "the Mac's appearance: no page can say otherwise"


def test_off_macos_the_room_is_the_colour_pywebview_paints_first_only_for_a_known_host_palette(mac, monkeypatch):
    pending = Window(None)
    wc.paint_room(pending)
    pending.events.before_show.fire()  # no NSWindow to paint, and nothing raised into pywebview
    monkeypatch.setattr(wc, "_MACOS", False)
    monkeypatch.setattr(wc, "_LINUX", False)
    for palette, colour in (("dark", "#0f0f11"), ("light", "#ffffff"), ("", "#123456")):
        monkeypatch.setattr(wc, "host_palette", lambda palette=palette: palette)
        other = Window(NSWindow())
        other.background_color = "#123456"  # what the call site gave pywebview
        wc.paint_room(other)
        assert other.background_color == colour and other.events.before_show.handlers == []
    monkeypatch.setattr(launcher_appearance, "_WINDOWS", False)
    other_os = Window(NSWindow())
    other_os.html = "<!DOCTYPE html>"
    wc.paint_room(other_os)
    assert other_os.events.before_show.handlers == [] and other_os.events.closed.handlers == []  # the caption is Windows'
    wc.paint_room(None)


class GtkRgba:
    """``Gdk.RGBA`` as ``_paint_gtk_room`` and pywebview read and parse it."""

    def __init__(self, red=0.0, green=0.0, blue=0.0):
        self.red, self.green, self.blue, self.parsed = red, green, blue, None

    def parse(self, text):
        self.parsed = text
        return True


def gtk_window(theme=(0x35 / 255, 0x35 / 255, 0x35 / 255), known=True, renderer="gtkwebkit2"):
    """A window as pywebview 5.4's GTK backend leaves it at ``before_show``: its toplevel, its module, its web view."""
    window = Window(types.SimpleNamespace(get_style_context=lambda: types.SimpleNamespace(
        lookup_color=lambda name: (known and name == "theme_bg_color", GtkRgba(*theme)))))
    window.uid = "master"
    window.painted = []
    view = types.SimpleNamespace(set_background_color=window.painted.append)
    window.gui = types.SimpleNamespace(
        renderer=renderer, Gdk=types.SimpleNamespace(RGBA=GtkRgba),
        BrowserView=types.SimpleNamespace(instances={"master": types.SimpleNamespace(webview=view)}))
    return window


@pytest.fixture
def gtk(monkeypatch):
    """The Linux path. No ``gi`` exists here: the leaf only uses what pywebview's backend module already holds."""
    monkeypatch.setattr(wc, "_MACOS", False)
    monkeypatch.setattr(wc, "_LINUX", True)


@pytest.mark.parametrize("theme,room", [
    ((0x35 / 255, 0x35 / 255, 0x35 / 255), "#0f0f11"),  # #353535: Adwaita:dark, and GTK's prefer-dark setting
    ((0xf6 / 255, 0xf5 / 255, 0xf4 / 255), "#ffffff"),  # #f6f5f4: Adwaita
])
def test_on_gtk_the_web_view_opens_in_the_room_of_the_palette_its_theme_resolved(gtk, theme, room):
    """The surface run 41575443436 saw white on every host while a ``url=`` response was held. Painted in
    ``before_show``, which pywebview fires just before the first load; nothing happens before it."""
    window = gtk_window(theme)
    window.background_color = "#123456"
    wc.paint_room(window)
    assert window.painted == [] and len(window.events.before_show.handlers) == 1
    window.events.before_show.fire()
    (colour,) = window.painted
    assert colour.parsed == room
    assert window.background_color == "#123456"  # pywebview's own GtkWindow rule is not this surface


def test_on_gtk_each_seam_paints_once_and_a_preserver_gets_no_windows_caption(gtk):
    main, setup, preserver = gtk_window(), gtk_window(), gtk_window()
    preserver.html = "<!DOCTYPE html>"
    wc.WindowChrome().attach(main)  # launcher_background: the window's caption owner, then this
    wc.paint_room(setup)  # launcher_onboarding, after its caption owner
    wc.paint_room(preserver)  # launcher.py's three windows shown before the server
    for window in (main, setup, preserver):
        assert len(window.events.before_show.handlers) == 1 and window.events.closed.handlers == []
        window.events.before_show.fire()
        assert [colour.parsed for colour in window.painted] == ["#0f0f11"]
    wc.paint_room(None)


@pytest.mark.parametrize("window", [gtk_window(known=False), gtk_window(renderer="qtwebengine")],
                         ids=["a theme without the colour", "pywebview's Qt backend"])
def test_on_gtk_an_unknown_palette_or_another_backend_leaves_webkits_white(gtk, window):
    wc.paint_room(window)
    window.events.before_show.fire()
    assert window.painted == []


def test_on_gtk_a_refused_colour_is_said_and_never_reaches_pywebview(gtk, caplog):
    window = gtk_window()

    def refuse(colour):
        raise RuntimeError("no such surface")

    window.gui.BrowserView.instances["master"].webview.set_background_color = refuse
    wc.paint_room(window)
    window.events.before_show.fire()
    assert "keeps WebKit's white" in caplog.text


def test_a_refused_room_colour_keeps_pywebviews_and_the_title_bar_still_integrates(mac, caplog):
    def refuse(name, provider):
        raise RuntimeError("no dynamic colours here")

    mac.appkit.NSColor.colorWithName_dynamicProvider_ = refuse
    chrome, _window, native = shown()
    assert native.backgrounds == [] and "keeps its fixed background colour" in caplog.text
    declare(chrome, DOC_A)
    assert native.mask & wc._FULL_SIZE_CONTENT


def test_close_releases_the_full_screen_observers(mac):
    chrome, window, native = shown()
    names = sorted(mac.observers)
    assert names == sorted(name for name, _phase in wc._PHASES) == sorted([
        "NSWindowWillEnterFullScreenNotification", "NSWindowDidEnterFullScreenNotification",
        "NSWindowWillExitFullScreenNotification", "NSWindowDidExitFullScreenNotification"])
    assert wc._tracked == [chrome] and chrome.windowed(object()) is True
    window.events.closed.fire()
    assert sorted(mac.removed) == names
    assert wc._tracked == [] and chrome.windowed(object()) is False, "a closed window is never vouched for"
    calls = list(native.calls)
    declare(chrome, DOC_A, integrated=False, choice="dark")
    assert native.calls == calls and native.appearances == [] and mac.webview.evaluated == []


# ------------------------------------------------------------- a warm source handoff's full-screen barrier ---

def full_screen(mac, *names):
    for name in names:
        mac.observers[f"NSWindow{name}FullScreenNotification"](None)


def hand_over(final=lambda: None):
    """``transfer_when_windowed`` on its worker thread (the lifecycle loop's); the test thread is the main thread."""
    outcome = {}

    def work():
        try:
            outcome["ran"] = wc.transfer_when_windowed(final)
        except BaseException as error:  # the worker is where ``final``'s exception must arrive
            outcome["error"] = error

    worker = threading.Thread(target=work, name="lifecycle", daemon=True)
    worker.start()
    return worker, outcome


def finished(worker, outcome):
    worker.join(5)
    assert not worker.is_alive(), "the worker is never left waiting"
    return outcome


@pytest.mark.parametrize("script", [True, False], ids=["integrated", "without the document script"])
@pytest.mark.parametrize("start", [("WillEnter", "DidEnter"), ("WillEnter",), ("WillEnter", "DidEnter", "WillExit")],
                         ids=["full screen", "entering", "exiting"])
def test_a_full_screen_window_is_asked_once_to_leave_and_the_handoff_runs_after_did_exit(mac, monkeypatch, script,
                                                                                         start):
    """Entering waits for DidEnter before asking, exiting asks nothing; the teardown runs on the main thread in the
    turn after DidExit that found the live mask windowed, never on the worker, and nothing raises a window."""
    if not script:
        monkeypatch.setattr(wc, "_webview_of", lambda window: (_ for _ in ()).throw(KeyError(window.uid)))
    chrome, window, native = shown()
    assert chrome._phase == "windowed" and (chrome._native is native) is script, "observed either way"
    mac.pywebview.windows.append(window)
    full_screen(mac, *start)
    main, seen = threading.current_thread(), []

    def final():
        seen.append((threading.current_thread() is main, chrome._phase, bool(native.mask & wc._FULL_SCREEN)))

    worker, outcome = hand_over(final)
    mac.main.run(first=True)
    if start == ("WillEnter",):
        assert "toggle full screen" not in native.calls, "a transition under way is never toggled"
        full_screen(mac, "DidEnter")
        mac.main.run()
    expected = 0 if start[-1] == "WillExit" else 1
    assert native.calls.count("toggle full screen") == expected and seen == []
    if expected:
        full_screen(mac, "WillExit")
        mac.main.run()
    assert seen == [] and worker.is_alive(), "exiting still waits"
    full_screen(mac, "DidExit")
    mac.main.run()
    assert seen == [(True, "windowed", False)], "on the main thread, in the turn that found it windowed"
    assert finished(worker, outcome) == {"ran": True}
    assert native.calls.count("toggle full screen") == expected, "asked once"
    assert not any(call in ("show", "makeKeyAndOrderFront", "orderFront") for call in native.calls)


def test_a_windowed_shell_hands_over_in_its_first_main_thread_turn(mac):
    chrome, window, native = shown()
    mac.pywebview.windows.append(window)
    main, seen = threading.current_thread(), []
    worker, outcome = hand_over(lambda: seen.append(threading.current_thread() is main))
    mac.main.run(first=True)
    assert seen == [True] and finished(worker, outcome) == {"ran": True}
    assert "toggle full screen" not in native.calls


@pytest.mark.parametrize("unknown", ["setup appears", "hidden in full screen", "mask disagrees", "unobserved",
                                     "unreadable", "none left"])
def test_a_window_that_cannot_be_confirmed_refuses_before_any_teardown(mac, unknown):
    """The main thread judges again: a setup window that appeared after the handoff was posted has no observer, and
    nothing is asked to leave when it refuses; only showing a hidden full-screen window could end its stint; the live
    mask outranks a missed notification; a window gone by the turn leaves nothing to vouch for."""
    chrome, window, native = shown()
    mac.pywebview.windows.append(window)
    if unknown == "setup appears":
        full_screen(mac, "WillEnter", "DidEnter")
    elif unknown == "hidden in full screen":
        full_screen(mac, "WillEnter", "DidEnter")
        native.visible = False
    elif unknown == "mask disagrees":
        native.mask |= wc._FULL_SCREEN
    elif unknown == "unobserved":
        chrome._phase = None  # an observer that could not be registered
    elif unknown == "unreadable":
        full_screen(mac, "WillEnter", "DidEnter")
        native.isVisible = lambda: (_ for _ in ()).throw(RuntimeError("window gone"))
    seen = []
    worker, outcome = hand_over(lambda: seen.append("final"))
    if unknown in ("setup appears", "none left"):  # after the handoff was posted, before its turn
        with mac.main.posted:
            assert mac.main.posted.wait_for(lambda: mac.main.turns, timeout=5)
        if unknown == "none left":
            mac.pywebview.windows.clear()  # closed: never an exec over no window
        else:
            mac.pywebview.windows.insert(0, Window(NSWindow()))  # unobserved, listed before the main window
    mac.main.run(first=True)
    assert finished(worker, outcome) == {"ran": False} and seen == []
    assert "toggle full screen" not in native.calls
    full_screen(mac, "WillExit", "DidExit")  # a late word changes nothing for a refused handoff
    mac.main.run()
    assert seen == [] and mac.main.turns == []


def test_no_confirmation_by_the_deadline_abandons_and_a_later_handoff_asks_again(mac, monkeypatch):
    """The deadline delays only a refusal: no main-thread turn, or AppKit ignoring the request. A turn that comes
    afterwards does nothing, and the next handoff asks again rather than trusting the earlier request."""
    monkeypatch.setattr(wc, "FULL_SCREEN_EXIT_SEC", 0.5)  # the test thread takes each posted turn well within it
    chrome, window, native = shown()
    mac.pywebview.windows.append(window)
    seen = []
    worker, outcome = hand_over(lambda: seen.append("late"))
    assert finished(worker, outcome) == {"ran": False}, "no run loop took the turn"
    mac.main.run()
    assert seen == []
    full_screen(mac, "WillEnter", "DidEnter")
    for attempt in (1, 2):
        worker, outcome = hand_over(lambda: seen.append("ran"))
        mac.main.run(first=True)
        assert finished(worker, outcome) == {"ran": False} and native.calls.count("toggle full screen") == attempt
    full_screen(mac, "WillExit", "DidExit")
    mac.main.run()
    assert seen == [], "an abandoned handoff never runs, however late its window leaves"


@pytest.mark.parametrize("pywebview", ["no window yet", "not loaded", "no window list", "created, not yet shown",
                                       "setup only", "setup beside the main window"])
def test_a_handoff_before_any_window_refuses_at_once_and_queues_nothing(mac, monkeypatch, pywebview):
    """An exit 42 during the first boot, before the setup or main window exists or is observed: pywebview lists a
    window from ``create_window`` on, but only its ``before_show`` registers an observer, and the setup window never
    has one. Without an observed window for each listed one no main-thread turn could vouch for them all, so the
    handoff refuses at once instead of waiting out its deadline, asks AppKit nothing and leaves no attempt that a
    later window could run."""
    monkeypatch.setattr(wc, "FULL_SCREEN_EXIT_SEC", 3600.0)  # a wait would outlast the join below by far
    main, native = None, None
    if pywebview == "not loaded":
        monkeypatch.delitem(sys.modules, "webview")
    elif pywebview == "no window list":
        del mac.pywebview.windows
    elif pywebview == "created, not yet shown":  # create_window listed it; webview.start has not run before_show
        main = Window(NSWindow())
        wc.WindowChrome().attach(main)
        mac.pywebview.windows.append(main)
    elif pywebview == "setup only":
        mac.pywebview.windows.append(Window(NSWindow()))
    elif pywebview == "setup beside the main window":
        _chrome, main, native = shown()
        full_screen(mac, "WillEnter", "DidEnter")
        mac.pywebview.windows.extend([Window(NSWindow()), main])
    seen = []
    worker, outcome = hand_over(lambda: seen.append("final"))
    try:
        worker.join(5)
        assert not worker.is_alive(), "the handoff waited for a turn no run loop will take"
    finally:
        if worker.is_alive():  # a regression: release its wait so the test ends
            wc._transfer.settle("refused")
    assert outcome == {"ran": False} and seen == []
    assert mac.main.turns == [] and wc._transfer is None, "nothing queued, no attempt left for a later window"
    if native is not None:  # the observed main window, already in full screen: the refusal asked AppKit nothing
        assert native.calls.count("toggle full screen") == 0
        full_screen(mac, "WillExit", "DidExit")
    else:  # the main window is then shown (observed from its before_show) and goes through full screen
        if main is None:
            _chrome, main, native = shown()
        else:
            main.events.before_show.fire()
            native = main.native
        full_screen(mac, "WillEnter", "DidEnter", "WillExit", "DidExit")
    mac.main.run()
    assert seen == [] and "toggle full screen" not in native.calls, "nothing ran, and nothing was asked of AppKit"


def test_a_claimed_handoff_is_never_abandoned_and_its_exception_reaches_the_worker(mac, monkeypatch):
    """Once the main thread runs ``final`` the deadline cannot refuse it; an exec failure is reported to the worker,
    which owns the recovery that may wait (the lock and the menu-bar item)."""
    monkeypatch.setattr(wc, "FULL_SCREEN_EXIT_SEC", 0.05)
    _chrome, window, _native = shown()
    mac.pywebview.windows.append(window)
    abandoning, settle = threading.Event(), wc._Transfer.settle

    def observed(transfer, state):
        if state == "abandoned":
            abandoning.set()
        return settle(transfer, state)

    def final():
        assert abandoning.wait(5), "the worker's deadline passes while the main thread is inside final"
        raise PermissionError("entry refused")

    monkeypatch.setattr(wc._Transfer, "settle", observed)
    worker, outcome = hand_over(final)
    mac.main.run(first=True)
    result = finished(worker, outcome)
    assert isinstance(result.get("error"), PermissionError) and "ran" not in result


def test_without_its_webview_the_window_stays_ordinary(mac, monkeypatch):
    def missing(window):
        raise KeyError(window.uid)

    monkeypatch.setattr(wc, "_webview_of", missing)
    chrome, _window, native = shown()
    declare(chrome, DOC_A)
    assert_ordinary(native)
    assert native.appearances == []


def test_only_the_main_window_gets_the_integrated_bar(mac, monkeypatch):
    """``Background`` owns the main window; the first-run setup window keeps its ordinary bar and, like every
    window, opens in the room colour of the Mac's appearance (its page then paints the saved choice)."""
    from ouroboros import launcher_background as lb
    from ouroboros import launcher_onboarding

    monkeypatch.setattr(lb, "indicator_class", lambda: None)
    background = lb.Background(lambda: None, lambda: 1, threading.Event())
    main = Window(NSWindow())
    main.events.closing, main.events.shown = Hook(), Hook()
    background.attach(main)
    assert background.chrome._before_show in main.events.before_show.handlers
    assert main.events.before_load.handlers == [background.chrome._navigation_finished]

    created = {}
    fake = types.ModuleType("webview")
    fake.windows = []
    menu = types.ModuleType("webview.menu")
    menu.Menu = lambda title, items: types.SimpleNamespace(title=title, items=items)
    menu.MenuAction = lambda title, function: types.SimpleNamespace(title=title, function=function)

    def create_window(title, url=None, js_api=None, **kwargs):
        created["window"] = Window(NSWindow())
        fake.windows.append(created["window"])
        return created["window"]

    fake.create_window, fake.start = create_window, lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "webview", fake)
    monkeypatch.setitem(sys.modules, "webview.menu", menu)
    launcher_onboarding.present_first_run_onboarding({}, 1, open_external_url=lambda url: {"ok": False})
    setup = created["window"]
    assert setup.events.before_load.handlers == []
    setup.events.before_show.fire()
    assert room(setup.native)(Appearance("DarkAqua")) == DARK_ROOM
    assert room(setup.native)(Appearance("Aqua")) == LIGHT_ROOM
    assert_ordinary(setup.native)
    assert setup.native.appearances == [], "no page can force the setup window's appearance"


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _selector_list(text):
    """Split selector lists, keeping commas in :is/:where, attributes and strings."""
    start, depth, quote, escaped = 0, 0, "", False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif quote:
            if char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "," and depth == 0:
            yield text[start:index].strip()
            start = index + 1
    yield text[start:].strip()


def _assert_measured_geometry_is_scoped(css):
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    readers = [m for m in re.finditer(r"([^{}]*)\{([^{}]*var\(--window-chrome-[^{}]*)\}", css)]
    assert readers, "style.css consumes the measured title bar geometry"
    for rule in readers:
        for selector in _selector_list(rule.group(1)):
            assert selector.startswith(':root[data-window-chrome="integrated"]'), selector


def test_the_page_reserves_room_only_under_the_measured_state():
    """No phantom spacer: every rule reading the shell's geometry is scoped to the attribute the shell
    writes after measuring, so a browser, another OS or an older shell lays out exactly as before."""
    css = _read("web/style.css")
    _assert_measured_geometry_is_scoped(css)
    with pytest.raises(AssertionError):
        _assert_measured_geometry_is_scoped(css.replace(':root[data-window-chrome="integrated"]', ':root'))
    assert "window-chrome" not in re.sub(r"/\*.*?\*/", "", _read("web/ui.css"), flags=re.S)


@pytest.mark.parametrize("selector", [
    ':root[data-window-chrome="integrated"] :is(.chat, .project), .unguarded',
    '.unguarded, :root[data-window-chrome="integrated"] :where(.chat, .project)',
    ':is(:root[data-window-chrome="integrated"] .chat, .unguarded)',
])
def test_a_functional_selector_does_not_hide_an_unguarded_geometry_reader(selector):
    with pytest.raises(AssertionError):
        _assert_measured_geometry_is_scoped(selector + ' { height: var(--window-chrome-top); }')


def test_the_shell_writes_exactly_the_names_the_page_reads():
    script = wc.DOCUMENT_SCRIPT
    for name in ("--window-chrome-top", "--window-chrome-start"):
        assert f"setProperty('{name}'" in script and f"removeProperty('{name}')" in script
        assert f"var({name})" in _read("web/style.css")
    assert "root.dataset.windowChrome = 'integrated'" in script
    assert 'meta[name="ouroboros-window-chrome"]' in script and f"messageHandlers.{wc.HANDLER}" in script
