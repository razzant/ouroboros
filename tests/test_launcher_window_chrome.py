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

from ouroboros import launcher_appearance, launcher_background
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


class PywebviewDelegate:
    """pywebview 5.4's window delegate in miniature: it answers close and resize, never AppKit's full-screen failures."""

    def __init__(self):
        self.messages = []

    def respondsToSelector_(self, selector):
        return selector in ("windowShouldClose:", "windowWillClose:", "windowDidResize:")

    def windowWillClose_(self, notification):
        self.messages.append(("windowWillClose:", notification))


class NSWindow:
    """The NSWindow surface the leaf touches; its geometry follows the real rules in miniature."""

    def __init__(self, fill_holds_buttons=True):
        self.mask = 0b1111  # titled, closable, miniaturizable, resizable
        self._frame = Rect(100, 100, 1100, 750)
        self.transparent, self.title_visibility, self._toolbar = False, 0, None
        self.toolbar_style, self.separator, self.tabbing = 0, 0, 0
        self.appearances, self.calls, self.backgrounds = [], [], []
        self.exiting_full_screen = False
        self.notifying = None  # the notification AppKit is posting now (``Center``)
        self.visible = True
        self.fill = View()
        self.content = View(lambda: Rect(0, 0, self._frame.size.width, self.content_height()))
        theme = View(children=[self.content, self.fill])
        self.content.parent = theme
        self.title_bar = TitleBar()
        self.buttons = [Button(self.fill if fill_holds_buttons else View(), x, self.title_bar) for x in (19, 39, 59)]
        self.fail_on = None
        self._delegate = PywebviewDelegate()

    def delegate(self):
        return self._delegate

    def setDelegate_(self, delegate):
        self._maybe_fail("delegate")
        self._delegate = delegate

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
        if self.notifying == "NSWindowDidEnterFullScreenNotification":
            self.calls.append("ignored toggle")  # native QA: sent inside DidEnter, no WillExit came (4 of 4)
            return
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

    def hide(self):  # pywebview's: orderOut: in a main-thread turn
        self.native.calls.append("hide")

    def show(self):  # pywebview's: makeKeyAndOrderFront: in a main-thread turn, queued in call order like ``hide``
        self.native.calls.append("show")

    def evaluate_js(self, script):
        """pywebview's: a main-thread turn queued after those already queued (the real one then blocks its caller)."""
        from PyObjCTools import AppHelper

        AppHelper.callAfter(self.native.calls.append, ("js", script))


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

        def respondsToSelector_(self, selector):  # NSObject's own answer: the methods its class defines
            return callable(getattr(type(self), selector.replace(":", "_"), None))

        def forwardingTargetForSelector_(self, selector):
            return None

    foundation.NSObject = NSObject
    objc = types.ModuleType("objc")
    objc.super = super  # PyObjC's own super for its classes; the stand-in NSObject is a plain class
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

            def post(note):  # AppKit runs the block inside its posting of ``name``
                previous, obj.notifying = getattr(obj, "notifying", None), name
                try:
                    return block(note)
                finally:
                    obj.notifying = previous

            state.observers[name] = post
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
                         ("PyObjCTools", pyobjc_tools), ("webview", state.pywebview), ("objc", objc)):
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
    assert wc._tracked == [chrome] and chrome.windowed() is True
    window.events.closed.fire()
    assert sorted(mac.removed) == names
    assert wc._tracked == [] and chrome.windowed() is False, "a closed window is never vouched for"
    calls = list(native.calls)
    declare(chrome, DOC_A, integrated=False, choice="dark")
    assert native.calls == calls and native.appearances == [] and mac.webview.evaluated == []


# ------------------------------------------------------------- a warm source handoff's full-screen barrier ---

def full_screen(mac, *names):
    for name in names:
        mac.observers[f"NSWindow{name}FullScreenNotification"](None)


def hand_over(final=lambda: None, cancelled=None):
    """``transfer_when_windowed`` on its worker thread (the lifecycle loop's); the test thread is the main thread."""
    outcome = {}

    def work():
        try:
            outcome["ran"] = (wc.transfer_when_windowed(final) if cancelled is None
                              else wc.transfer_when_windowed(final, cancelled))
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
    monkeypatch.setattr(wc, "MACOS_HANDOFF_FULLSCREEN_WAIT_SEC", 0.5)  # the test thread takes each posted turn well within it
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
    monkeypatch.setattr(wc, "MACOS_HANDOFF_FULLSCREEN_WAIT_SEC", 3600.0)  # a wait would outlast the join below by far
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
    monkeypatch.setattr(wc, "MACOS_HANDOFF_FULLSCREEN_WAIT_SEC", 0.05)
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


def released(worker, outcome):
    """``finished`` for a handoff whose deadline is far away: a regression that leaves it waiting is released."""
    try:
        worker.join(5)
        assert not worker.is_alive(), "the handoff waited out its deadline instead of looking again"
    finally:
        if worker.is_alive():
            wc._transfer.settle("refused")
            worker.join(5)
    return outcome


def unreadable_mask_once(native):
    """The next ``styleMask`` read raises (a window AppKit is tearing down); later reads see the mask again."""
    def unreadable():
        del native.styleMask
        raise RuntimeError("window gone")

    native.styleMask = unreadable


@pytest.mark.parametrize("mask", ["still full screen", "unreadable"])
def test_a_failed_exit_keeps_full_screen_and_refuses_the_waiting_handoff_at_once(mac, monkeypatch, mask):
    """AppKit's DidFail (a delegate message, no notification) the live mask confirms, or cannot deny: the window
    stays in full screen with its full-screen chrome, the handoff refuses in that turn without asking again or waiting
    out its deadline, and a later one asks."""
    monkeypatch.setattr(wc, "MACOS_HANDOFF_FULLSCREEN_WAIT_SEC", 3600.0)
    chrome, window, native = shown()
    declare(chrome, DOC_A)
    mac.pywebview.windows.append(window)
    full_screen(mac, "WillEnter", "DidEnter")
    seen = []
    worker, outcome = hand_over(lambda: seen.append("final"))
    mac.main.run(first=True)
    full_screen(mac, "WillExit")
    assert native.toolbar().visible
    native.mask |= wc._FULL_SCREEN
    if mask == "unreadable":
        unreadable_mask_once(native)
    native.delegate().windowDidFailToExitFullScreen_(native)
    native.mask &= ~wc._FULL_SCREEN
    mac.main.run()
    assert released(worker, outcome) == {"ran": False} and seen == []
    assert chrome._phase == "full" and not native.toolbar().visible, "the full-screen chrome, no band"
    assert mac.webview.evaluated[-1]["top"] == 0, "nothing reserved in full screen"
    assert native.calls.count("toggle full screen") == 1, "asked once, never again for this handoff"
    worker, outcome = hand_over(lambda: seen.append("later"))
    mac.main.run(first=True)
    assert native.calls.count("toggle full screen") == 2, "a later handoff asks anew"
    leave_full_screen(mac, native)
    mac.main.run()
    assert released(worker, outcome) == {"ran": True} and seen == ["later"]


@pytest.mark.parametrize("then", ["DidExit", "an extra toggle re-enters"])
def test_a_failed_exit_said_with_the_mask_already_out_waits_for_appkits_last_word(mac, monkeypatch, caplog, then):
    """macOS 26.4 also says an exit failed in its exiting phase, the mask already out of full screen, 1-31 ms before
    the DidExit of an exit that succeeds: nothing changes, so DidExit hands over in its turn, and a re-entry asks
    anew once DidEnter, as #3198 did. launcher.log keeps the message as #3134's trail."""
    monkeypatch.setattr(wc, "MACOS_HANDOFF_FULLSCREEN_WAIT_SEC", 3600.0)
    chrome, window, native = shown()
    declare(chrome, DOC_A)
    mac.pywebview.windows.append(window)
    full_screen(mac, "WillEnter", "DidEnter")
    main, seen = threading.current_thread(), []
    worker, outcome = hand_over(lambda: seen.append((threading.current_thread() is main, chrome._phase)))
    mac.main.run(first=True)
    full_screen(mac, "WillExit")
    published = len(mac.webview.evaluated)
    assert not native.mask & wc._FULL_SCREEN and native.calls.count("toggle full screen") == 1
    with caplog.at_level("INFO", logger="launcher.window_chrome"):
        native.delegate().windowDidFailToExitFullScreen_(native)
    mac.main.run()
    assert seen == [] and worker.is_alive() and wc._transfer.state == "pending", "still waiting, not refused"
    assert chrome._phase == "exiting" and native.toolbar().visible and len(mac.webview.evaluated) == published
    assert "a handoff waits for DidExit" in caplog.text
    if then != "DidExit":
        full_screen(mac, "WillEnter", "DidEnter")
        mac.main.run()
        assert native.calls.count("toggle full screen") == 2, "the stint that began asks anew"
        full_screen(mac, "WillExit")
    full_screen(mac, "DidExit")
    mac.main.run()
    assert seen == [(True, "windowed")] and released(worker, outcome) == {"ran": True}


@pytest.mark.parametrize("mask", ["already full screen", "unreadable"])
def test_a_failed_entry_the_mask_does_not_deny_changes_nothing_until_did_enter(mac, monkeypatch, mask):
    """A failed entry whose live mask already holds full screen, or cannot be read, is no windowed window: the
    phase, band and page stay as WillEnter left them, nothing runs, and DidEnter asks the window to leave."""
    monkeypatch.setattr(wc, "MACOS_HANDOFF_FULLSCREEN_WAIT_SEC", 3600.0)
    chrome, window, native = shown()
    declare(chrome, DOC_A)
    mac.pywebview.windows.append(window)
    full_screen(mac, "WillEnter")
    seen = []
    worker, outcome = hand_over(lambda: seen.append("final"))
    mac.main.run(first=True)
    published = len(mac.webview.evaluated)
    native.mask |= wc._FULL_SCREEN
    if mask == "unreadable":
        unreadable_mask_once(native)
        native.mask &= ~wc._FULL_SCREEN  # unread, so never the reason to call the window windowed
    native.delegate().windowDidFailToEnterFullScreen_(native)
    mac.main.run()
    assert chrome._phase == "entering" and not native.toolbar().visible and len(mac.webview.evaluated) == published
    assert seen == [] and worker.is_alive() and wc._transfer.state == "pending", "the handoff waits for DidEnter"
    assert "toggle full screen" not in native.calls
    full_screen(mac, "DidEnter")
    mac.main.run()
    assert native.calls.count("toggle full screen") == 1 and seen == []
    native.mask &= ~wc._FULL_SCREEN
    leave_full_screen(mac, native)
    mac.main.run()
    assert seen == ["final"] and released(worker, outcome) == {"ran": True}


def test_a_failed_entry_restores_the_windowed_title_bar_and_the_waiting_handoff_runs(mac, monkeypatch):
    """A window that could not enter full screen is windowed: its band, drag and double-click come back, and the
    handoff that waited for the transition runs in the next turn instead of at its deadline."""
    monkeypatch.setattr(wc, "MACOS_HANDOFF_FULLSCREEN_WAIT_SEC", 3600.0)
    chrome, window, native = shown()
    declare(chrome, DOC_A)
    mac.pywebview.windows.append(window)
    full_screen(mac, "WillEnter")
    assert not native.toolbar().visible
    main, seen = threading.current_thread(), []
    worker, outcome = hand_over(lambda: seen.append(threading.current_thread() is main))
    mac.main.run(first=True)
    assert seen == [] and "toggle full screen" not in native.calls, "a transition under way is never toggled"
    assert not native.mask & wc._FULL_SCREEN, "the live mask denies full screen"
    native.delegate().windowDidFailToEnterFullScreen_(native)
    assert chrome._phase == "windowed" and native.toolbar().visible
    assert mac.webview.evaluated[-1]["top"] == TOOLBAR_BAND, "the band is reserved again"
    mac.event = press()
    chrome.receive({"type": "drag", "doc": DOC_A})
    assert native.calls[-1] == ("drag", mac.event), "the title bar drags again"
    mac.main.run()
    assert seen == [True] and released(worker, outcome) == {"ran": True}
    assert "toggle full screen" not in native.calls


@pytest.mark.parametrize("when", ["never in full screen", "after DidExit"])
def test_a_late_or_repeated_failed_exit_of_a_windowed_window_changes_nothing(mac, when):
    """A failed exit said once the window is windowed, its mask out of full screen (late, or said again), leaves the
    ordinary window as it is: windowed, its band shown, nothing published."""
    chrome, window, native = shown()
    declare(chrome, DOC_A)
    if when == "after DidExit":
        full_screen(mac, "WillEnter", "DidEnter")
        leave_full_screen(mac, native)
    published = len(mac.webview.evaluated)
    assert chrome._phase == "windowed" and native.toolbar().visible
    for _ in range(2):
        native.delegate().windowDidFailToExitFullScreen_(native)
    assert chrome._phase == "windowed" and native.toolbar().visible and len(mac.webview.evaluated) == published


def test_a_failed_exit_the_mask_contradicts_without_did_exit_ends_at_the_deadline(mac, monkeypatch):
    """The remainder #3198 shares: a failed exit said while the mask is already out, then no DidExit. The waiting
    handoff is abandoned at its deadline, nothing torn down; a transition still under way is never toggled, so the
    next handoff asks nothing and is abandoned too; once AppKit settles the window in full screen, a handoff asks."""
    monkeypatch.setattr(wc, "MACOS_HANDOFF_FULLSCREEN_WAIT_SEC", 0.3)
    chrome, window, native = shown()
    mac.pywebview.windows.append(window)
    full_screen(mac, "WillEnter", "DidEnter")
    seen = []
    worker, outcome = hand_over(lambda: seen.append("first"))
    mac.main.run(first=True)
    full_screen(mac, "WillExit")
    first = wc._transfer
    native.delegate().windowDidFailToExitFullScreen_(native)
    mac.main.run()
    assert finished(worker, outcome) == {"ran": False} and first.state == "abandoned", "the deadline, not a refusal"
    worker, outcome = hand_over(lambda: seen.append("second"))
    mac.main.run(first=True)
    assert finished(worker, outcome) == {"ran": False} and wc._transfer.state == "abandoned"
    assert chrome._phase == "exiting" and native.calls.count("toggle full screen") == 1, "never toggled mid-exit"
    full_screen(mac, "WillEnter", "DidEnter")
    monkeypatch.setattr(wc, "MACOS_HANDOFF_FULLSCREEN_WAIT_SEC", 3600.0)
    worker, outcome = hand_over(lambda: seen.append("third"))
    mac.main.run(first=True)
    assert native.calls.count("toggle full screen") == 2, "a stint AppKit confirmed is asked anew"
    leave_full_screen(mac, native)
    mac.main.run()
    assert released(worker, outcome) == {"ran": True} and seen == ["third"]


def test_a_failed_entry_judged_windowed_refuses_once_the_mask_turns_full_screen(mac, monkeypatch):
    """A failed entry with the mask out is windowed, yet the handoff's turn reads the live mask again: one that took
    the full-screen bit meanwhile is unknown, so the handoff refuses and nothing runs."""
    monkeypatch.setattr(wc, "MACOS_HANDOFF_FULLSCREEN_WAIT_SEC", 3600.0)
    chrome, window, native = shown()
    mac.pywebview.windows.append(window)
    full_screen(mac, "WillEnter")
    seen = []
    worker, outcome = hand_over(lambda: seen.append("final"))
    mac.main.run(first=True)
    native.delegate().windowDidFailToEnterFullScreen_(native)
    assert chrome._phase == "windowed" and mac.main.turns, "a turn to look again is posted"
    native.mask |= wc._FULL_SCREEN
    mac.main.run()
    assert released(worker, outcome) == {"ran": False} and seen == []
    assert "toggle full screen" not in native.calls


def test_a_failed_exit_the_mask_contradicts_in_one_of_two_windows_settles_nothing(mac, monkeypatch):
    """Two observed windows asked to leave: one says its exit failed with the mask already out. The handoff still
    waits for both DidExit and runs only in the turn that found both windowed."""
    monkeypatch.setattr(wc, "MACOS_HANDOFF_FULLSCREEN_WAIT_SEC", 3600.0)
    first, first_window, first_native = shown()
    full_screen(mac, "WillEnter", "DidEnter")
    first_observers = dict(mac.observers)  # the stand-in center keeps only the last window's observers
    second, second_window, second_native = shown()
    full_screen(mac, "WillEnter", "DidEnter")
    mac.pywebview.windows.extend([first_window, second_window])
    seen = []
    worker, outcome = hand_over(lambda: seen.append((first._phase, second._phase)))
    mac.main.run(first=True)
    assert first_native.calls.count("toggle full screen") == second_native.calls.count("toggle full screen") == 1
    first_observers["NSWindowWillExitFullScreenNotification"](None)
    first_native.delegate().windowDidFailToExitFullScreen_(first_native)
    mac.main.run()
    assert seen == [] and worker.is_alive() and wc._transfer.state == "pending"
    first_observers["NSWindowDidExitFullScreenNotification"](None)
    mac.main.run()
    assert seen == [] and wc._transfer.state == "pending", "the other window is still in full screen"
    leave_full_screen(mac, second_native)
    mac.main.run()
    assert seen == [("windowed", "windowed")] and released(worker, outcome) == {"ran": True}


class AnsweringDelegate(PywebviewDelegate):
    """A previous delegate that also answers AppKit's full-screen failures."""

    def respondsToSelector_(self, selector):
        return super().respondsToSelector_(selector) or selector.startswith("windowDidFailTo")

    def windowDidFailToExitFullScreen_(self, window):
        self.messages.append(("windowDidFailToExitFullScreen:", window))

    def windowDidFailToEnterFullScreen_(self, window):
        self.messages.append(("windowDidFailToEnterFullScreen:", window))


@pytest.mark.parametrize("previous", ["pywebview", "answering", "none"])
def test_the_failure_delegate_forwards_the_previous_ones_messages_and_gives_it_back_on_close(mac, previous):
    """NSWindow has one weak delegate: this one stands in front of pywebview's (close, resize...), answers exactly
    what that one answers plus the two failures, forwards them, holds both strongly, and after close the main thread
    restores the previous one, unless another delegate replaced this one meanwhile."""
    native = NSWindow()
    native._delegate = {"pywebview": PywebviewDelegate, "answering": AnsweringDelegate, "none": lambda: None}[previous]()
    own = native._delegate
    chrome, window, native = shown(native)
    delegate = native.delegate()
    assert delegate is not own and delegate.previous is own and delegate.chrome is chrome
    assert chrome._delegate == (native, delegate), "held strongly: the window's reference is weak"
    for selector in (wc._FAILED_ENTER, wc._FAILED_EXIT):
        assert delegate.respondsToSelector_(selector)
    for selector in ("windowShouldClose:", "windowWillClose:", "windowDidResize:"):
        assert delegate.respondsToSelector_(selector) is (own is not None)
        assert delegate.forwardingTargetForSelector_(selector) is own
    assert not delegate.respondsToSelector_("windowDidMiniaturize:"), "nothing the previous one does not answer"
    assert delegate.forwardingTargetForSelector_("windowDidMiniaturize:") is None
    full_screen(mac, "WillEnter")
    delegate.windowDidFailToEnterFullScreen_(native)
    full_screen(mac, "WillEnter", "DidEnter", "WillExit")
    native.mask |= wc._FULL_SCREEN  # a real failure: the mask stays in full screen
    delegate.windowDidFailToExitFullScreen_(native)
    if previous == "answering":
        assert own.messages == [(wc._FAILED_ENTER, native), (wc._FAILED_EXIT, native)], "forwarded as received"
    assert chrome._phase == "full"
    window.events.closed.fire()  # pywebview's own thread: the window is touched only in a main-thread turn
    assert native.delegate() is delegate
    mac.main.run()
    assert native.delegate() is own and chrome._delegate is None
    delegate.windowDidFailToEnterFullScreen_(native)  # a late word after close changes nothing
    assert chrome._phase == "full"


def test_a_replaced_delegate_is_kept_and_a_refused_install_keeps_the_handoff(mac, caplog):
    chrome, window, native = shown()
    theirs = PywebviewDelegate()
    native.setDelegate_(theirs)
    window.events.closed.fire()
    mac.main.run()
    assert native.delegate() is theirs, "only this delegate is ever taken back"
    native = NSWindow()
    native.fail_on = "delegate"
    own = native.delegate()
    chrome, window, native = shown(native)
    assert native.delegate() is own and chrome._delegate is None and "Full-screen failures unheard" in caplog.text
    mac.pywebview.windows.append(window)
    seen = []
    worker, outcome = hand_over(lambda: seen.append("final"))
    mac.main.run(first=True)
    assert finished(worker, outcome) == {"ran": True} and seen == ["final"], "#3198's barrier, unchanged"
    window.events.closed.fire()
    mac.main.run()
    assert native.delegate() is own


@pytest.mark.parametrize("left", ["a windowed window", "none"])
def test_closing_the_full_screen_window_wakes_the_waiting_handoff(mac, monkeypatch, left):
    """A window closed while it was asked to leave full screen sends no DidExit: its close is what the handoff waits
    for. The windows left decide in the next turn; with none left it refuses, never an exec over no window."""
    monkeypatch.setattr(wc, "MACOS_HANDOFF_FULLSCREEN_WAIT_SEC", 3600.0)
    if left != "none":
        _other, other, _native = shown()
        mac.pywebview.windows.append(other)
    _chrome, window, native = shown()  # the stand-in center keeps the last window's observers
    mac.pywebview.windows.append(window)
    full_screen(mac, "WillEnter", "DidEnter")
    seen = []
    worker, outcome = hand_over(lambda: seen.append("final"))
    mac.main.run(first=True)
    assert native.calls.count("toggle full screen") == 1 and seen == []
    mac.pywebview.windows.remove(window)  # pywebview's windowWillClose:, then its closed event
    window.events.closed.fire()
    mac.main.run()
    expected = {"ran": False} if left == "none" else {"ran": True}
    assert released(worker, outcome) == expected and seen == (["final"] if expected["ran"] else [])


def test_one_window_that_refuses_leaves_the_others_in_full_screen(mac, monkeypatch):
    """Every window is judged before any is asked to leave: a hidden full-screen window listed after a visible one
    refuses the handoff, and the visible one keeps its full screen."""
    first, first_window, first_native = shown()
    full_screen(mac, "WillEnter", "DidEnter")
    _second, second_window, second_native = shown()
    full_screen(mac, "WillEnter", "DidEnter")
    second_native.visible = False
    mac.pywebview.windows.extend([first_window, second_window])
    seen = []
    worker, outcome = hand_over(lambda: seen.append("final"))
    mac.main.run(first=True)
    assert finished(worker, outcome) == {"ran": False} and seen == []
    assert "toggle full screen" not in first_native.calls + second_native.calls
    assert first._phase == "full"


@pytest.mark.parametrize("when", ["before the first turn", "while the window leaves"])
def test_a_stop_before_the_windows_agree_asks_no_window_to_leave_and_runs_nothing(mac, monkeypatch, when):
    monkeypatch.setattr(wc, "MACOS_HANDOFF_FULLSCREEN_WAIT_SEC", 3600.0)
    _chrome, window, native = shown()
    mac.pywebview.windows.append(window)
    full_screen(mac, "WillEnter", "DidEnter")
    stopped, seen = threading.Event(), []
    if when == "before the first turn":
        stopped.set()
    worker, outcome = hand_over(lambda: seen.append("final"), stopped.is_set)
    mac.main.run(first=True)
    if when == "before the first turn":
        assert "toggle full screen" not in native.calls, "a Stop asks no window to leave"
    else:
        assert native.calls.count("toggle full screen") == 1
        stopped.set()
        leave_full_screen(mac, native)
        mac.main.run()
    assert released(worker, outcome) == {"ran": False} and seen == []


@pytest.mark.parametrize("cause", ["stop", "window refuses"])
def test_the_refusal_log_names_a_stop_apart_from_a_window_that_cannot_leave(mac, caplog, cause):
    """launcher.log is the evidence trail of a refused handoff: a Stop names itself, never a window state it checked
    not, and a window that cannot be confirmed windowed keeps the generic line."""
    _chrome, window, native = shown()
    mac.pywebview.windows.append(window)
    full_screen(mac, "WillEnter", "DidEnter")
    if cause == "window refuses":
        native.visible = False  # hidden in full screen: only showing it could end its stint
    with caplog.at_level("WARNING", logger="launcher.window_chrome"):
        worker, outcome = hand_over(cancelled=lambda: cause == "stop")
        mac.main.run(first=True)
        assert finished(worker, outcome) == {"ran": False}
    stop, generic = "A Stop came before the windows agreed", "cannot be confirmed out of full screen"
    assert (stop in caplog.text, generic in caplog.text) == ((True, False) if cause == "stop" else (False, True))


# ---------------------------------------------------- a background close of a window in full screen (#3134) ---

def closed(mac, native, chrome, closes=1, *, mask_full=True):
    """The owner's background close(s) on the main thread; ``hidden`` records Background's ``later``."""
    hidden = []
    if mask_full:
        native.mask |= wc._FULL_SCREEN
    answers = [chrome.hide_when_windowed(lambda: hidden.append("hidden")) for _ in range(closes)]
    return hidden, answers


def exited(mac, native):
    native.mask &= ~wc._FULL_SCREEN
    leave_full_screen(mac, native)


def test_a_windowed_or_unobserved_window_hides_at_once_as_before(mac, caplog):
    """(a)/(k) The caller hides now: no toggle, nothing waits. A windowed phase whose live mask still says full screen
    (or cannot be read) is not confirmed out: it stays open, one line says why, nothing waits for it."""
    chrome, _window, native = shown()
    hidden, answers = closed(mac, native, chrome, mask_full=False)
    assert answers == [True] and hidden == [] and chrome._hiding is None
    assert wc.WindowChrome().hide_when_windowed(lambda: hidden.append("never")) is True, "unobserved: as before"
    with caplog.at_level("WARNING", logger="launcher.window_chrome"):
        hidden, answers = closed(mac, native, chrome)
    assert answers == [False] and hidden == [] and chrome._hiding is None
    assert "does not confirm it out of full screen" in caplog.text
    assert "toggle full screen" not in native.calls


def test_a_full_screen_window_leaves_once_and_hides_only_at_did_exit(mac):
    """(b)/(i) One toggle however often the owner closes meanwhile; the window stays on screen until AppKit's
    DidExit, then hides once."""
    chrome, _window, native = shown()
    full_screen(mac, "WillEnter", "DidEnter")
    hidden, answers = closed(mac, native, chrome, closes=3)
    assert answers == [False] * 3 and native.calls.count("toggle full screen") == 1 and hidden == []
    full_screen(mac, "WillExit")
    assert hidden == [], "the exit is not confirmed yet"
    exited(mac, native)
    assert hidden == ["hidden"] and chrome._hiding is None
    assert native.calls.count("toggle full screen") == 1


def test_an_entering_window_is_asked_to_leave_only_after_did_enter_returns(mac):
    """(c) A transition under way is never toggled, and AppKit ignores a toggle sent inside DidEnter (native QA, 4 of 4:
    the window stayed in full screen, the close pending): the close asks in its own main-thread turn after DidEnter,
    and hides at DidExit."""
    chrome, _window, native = shown()
    full_screen(mac, "WillEnter")
    hidden, _ = closed(mac, native, chrome, closes=2)
    assert "toggle full screen" not in native.calls and hidden == []
    full_screen(mac, "DidEnter")
    assert "toggle full screen" not in native.calls and hidden == []
    mac.main.run()
    assert native.calls.count("toggle full screen") == 1 and "ignored toggle" not in native.calls and hidden == []
    exited(mac, native)
    assert hidden == ["hidden"] and native.calls.count("toggle full screen") == 1


@pytest.mark.parametrize("then", ["a second close", "a handoff"])
def test_a_close_during_entry_is_not_stuck_behind_the_toggle_appkit_ignores_in_did_enter(mac, monkeypatch, then):
    """Native QA's regression (``cand-bgc-entering``): a close at WillEnter left the window full and the close pending,
    so a second close started nothing and a handoff was abandoned. Now one real toggle follows DidEnter; the second
    close or the handoff asks nothing more and both follow DidExit."""
    monkeypatch.setattr(wc, "MACOS_HANDOFF_FULLSCREEN_WAIT_SEC", 3600.0)
    chrome, window, native = shown()
    mac.pywebview.windows.append(window)
    full_screen(mac, "WillEnter")
    order, worker = [], None
    hidden, _ = closed(mac, native, chrome)
    full_screen(mac, "DidEnter")
    mac.main.run()
    try:
        if then == "a second close":
            assert chrome.hide_when_windowed(lambda: order.append("twice")) is False
        else:
            worker, outcome = hand_over(lambda: order.append("final"))
            mac.main.run(first=True)
        assert native.calls.count("toggle full screen") == 1 and "ignored toggle" not in native.calls
        full_screen(mac, "WillExit")
        mac.main.run()
        exited(mac, native)
        mac.main.run()
        assert hidden == ["hidden"] and order == ([] if then == "a second close" else ["final"])
        if worker is not None:
            assert released(worker, outcome) == {"ran": True}
    finally:
        if worker is not None and worker.is_alive():
            wc._transfer.settle("refused")
            worker.join(5)


@pytest.mark.parametrize("before_the_turn", ["an open", "the window closed"])
def test_a_close_settled_between_did_enter_and_its_turn_asks_and_hides_nothing(mac, caplog, before_the_turn):
    """The close's turn queued at DidEnter is judged by the close's identity: a reopen in between (any thread) or the
    window's own close leaves it nothing to toggle, and a later exit of the owner's own hides nothing."""
    chrome, window, native = shown()
    full_screen(mac, "WillEnter")
    hidden, _ = closed(mac, native, chrome)
    full_screen(mac, "DidEnter")
    if before_the_turn == "an open":
        opener = threading.Thread(target=chrome.keep)
        opener.start()
        opener.join(5)
    else:
        window.events.closed.fire()
    mac.main.run()
    assert "toggle full screen" not in native.calls and "ignored toggle" not in native.calls and hidden == []
    exited(mac, native)
    assert hidden == [] and chrome._hiding is None


def test_an_exiting_window_is_not_toggled_and_hides_at_its_did_exit(mac):
    """(d) The owner's own exit is under way: nothing is asked, the hide waits for its DidExit."""
    chrome, _window, native = shown()
    full_screen(mac, "WillEnter", "DidEnter", "WillExit")
    hidden, _ = closed(mac, native, chrome)
    assert hidden == [] and "toggle full screen" not in native.calls
    native.mask &= ~wc._FULL_SCREEN
    full_screen(mac, "DidExit")
    assert hidden == ["hidden"] and "toggle full screen" not in native.calls


def test_a_failed_entry_with_a_clean_mask_is_windowed_and_hides(mac):
    """An entry that failed with the mask out is windowed (``_failed``): no DidExit will come, so it hides there."""
    chrome, _window, native = shown()
    full_screen(mac, "WillEnter")
    hidden, _ = closed(mac, native, chrome, mask_full=False)
    assert hidden == []
    native.delegate().windowDidFailToEnterFullScreen_(native)
    assert chrome._phase == "windowed" and hidden == ["hidden"] and "toggle full screen" not in native.calls


@pytest.mark.parametrize("closed_too", [False, True], ids=["handoff", "handoff and close"])
def test_a_failed_entry_said_while_exiting_waits_for_did_exit(mac, monkeypatch, closed_too):
    """Opus's probe: the exit clears the bit too (12 of 12 native observations), so a failed entry said with a clean
    mask while exiting is not windowed: the handoff (and a close) still wait for DidExit."""
    monkeypatch.setattr(wc, "MACOS_HANDOFF_FULLSCREEN_WAIT_SEC", 3600.0)
    chrome, window, native = shown()
    mac.pywebview.windows.append(window)
    full_screen(mac, "WillEnter", "DidEnter")
    order = []
    worker, outcome = hand_over(lambda: order.append("final"))
    mac.main.run(first=True)
    assert native.calls.count("toggle full screen") == 1
    full_screen(mac, "WillExit")
    if closed_too:
        assert chrome.hide_when_windowed(lambda: order.append("hidden")) is False
    native.mask &= ~wc._FULL_SCREEN
    native.delegate().windowDidFailToEnterFullScreen_(native)
    mac.main.run()
    assert chrome._phase == "exiting" and order == [] and worker.is_alive() and wc._transfer.state == "pending"
    full_screen(mac, "DidExit")
    mac.main.run()
    assert order == ["hidden", "final"][not closed_too:] and released(worker, outcome) == {"ran": True}


@pytest.mark.parametrize("mask", ["still full screen", "unreadable"])
def test_a_real_failed_exit_keeps_the_window_open_and_a_new_close_tries_again(mac, caplog, mask):
    """(e)/(7) The wait is dropped with one line in launcher.log; the window stays; the next close asks anew."""
    chrome, _window, native = shown()
    full_screen(mac, "WillEnter", "DidEnter")
    hidden, _ = closed(mac, native, chrome)
    full_screen(mac, "WillExit")
    if mask == "unreadable":
        unreadable_mask_once(native)
    with caplog.at_level("INFO", logger="launcher.window_chrome"):
        native.delegate().windowDidFailToExitFullScreen_(native)
    assert hidden == [] and chrome._hiding is None and chrome._phase == "full"
    assert caplog.text.count("could not take the closed window out of full screen") == 1
    full_screen(mac, "WillExit", "DidExit")  # a later exit of the owner's own hides nothing
    assert hidden == []
    full_screen(mac, "WillEnter", "DidEnter")
    hidden, _ = closed(mac, native, chrome)
    assert native.calls.count("toggle full screen") == 2, "the new close asks anew"
    exited(mac, native)
    assert hidden == ["hidden"]


def test_a_failed_exit_said_with_the_mask_already_out_still_hides_at_did_exit(mac):
    """(f) macOS's false failure (mask already out, 1-31 ms before DidExit) drops nothing."""
    chrome, _window, native = shown()
    full_screen(mac, "WillEnter", "DidEnter")
    hidden, _ = closed(mac, native, chrome)
    full_screen(mac, "WillExit")
    native.mask &= ~wc._FULL_SCREEN
    native.delegate().windowDidFailToExitFullScreen_(native)
    assert hidden == [] and chrome._hiding is not None
    full_screen(mac, "DidExit")
    assert hidden == ["hidden"]


@pytest.mark.parametrize("phase", ["full", "entering"])
def test_opening_the_window_again_drops_the_waiting_close(mac, caplog, phase):
    """(g)/(7) A reopen (any thread) keeps the window: DidExit hides nothing and an entering window is not asked.
    Closing again re-arms."""
    chrome, _window, native = shown()
    full_screen(mac, "WillEnter", *(["DidEnter"] if phase == "full" else []))
    hidden, _ = closed(mac, native, chrome)
    with caplog.at_level("INFO", logger="launcher.window_chrome"):
        opener = threading.Thread(target=chrome.keep)
        opener.start()
        opener.join(5)
    assert caplog.text.count("opened again before it left full screen") == 1
    if phase == "entering":
        full_screen(mac, "DidEnter")
        assert "toggle full screen" not in native.calls, "nobody asks for the exit"
        hidden, _ = closed(mac, native, chrome)  # closed again
        assert native.calls.count("toggle full screen") == 1
    exited(mac, native)
    assert hidden == ([] if phase == "full" else ["hidden"])


def test_a_new_entry_drops_the_waiting_close_and_a_new_close_rearms(mac, caplog):
    """(h)/(7) AppKit re-entering full screen (WillEnter after the close) keeps the window open in it."""
    chrome, _window, native = shown()
    full_screen(mac, "WillEnter", "DidEnter")
    hidden, _ = closed(mac, native, chrome)
    full_screen(mac, "WillExit")
    with caplog.at_level("INFO", logger="launcher.window_chrome"):
        full_screen(mac, "WillEnter", "DidEnter")
    assert chrome._hiding is None and caplog.text.count("new full-screen entry keeps the closed window open") == 1
    assert native.calls.count("toggle full screen") == 1, "DidEnter asks nothing for a dropped close"
    hidden, _ = closed(mac, native, chrome)
    assert native.calls.count("toggle full screen") == 2
    exited(mac, native)
    assert hidden == ["hidden"]


@pytest.mark.parametrize("phase", ["full", "windowed"])
def test_the_consent_threads_close_is_decided_in_a_main_thread_turn(mac, phase):
    """The first close asks on its own thread (``launcher_background._ask``): the phase is read and the toggle sent
    only on the main thread."""
    chrome, _window, native = shown()
    if phase == "full":
        full_screen(mac, "WillEnter", "DidEnter")
        native.mask |= wc._FULL_SCREEN
    hidden, answers = [], []
    asker = threading.Thread(target=lambda: answers.append(chrome.hide_when_windowed(lambda: hidden.append("hidden"))))
    asker.start()
    asker.join(5)
    assert answers == [False] and hidden == [] and "toggle full screen" not in native.calls
    mac.main.run()
    if phase == "windowed":
        assert hidden == ["hidden"]
        return
    assert native.calls.count("toggle full screen") == 1 and hidden == []
    exited(mac, native)
    assert hidden == ["hidden"]


@pytest.mark.parametrize("phase", ["full", "entering", "windowed"])
def test_an_open_before_the_consent_threads_turn_cancels_that_close(mac, caplog, phase):
    """Carbon's case 1: the consent thread's close is armed before its main-thread turn is queued, so an open in
    between drops it; the turn then hides nothing, toggles nothing and arms nothing. A later close works as ever."""
    chrome, _window, native = shown()
    if phase != "windowed":
        full_screen(mac, "WillEnter", *(["DidEnter"] if phase == "full" else []))
        native.mask |= wc._FULL_SCREEN
    hidden = []
    asker = threading.Thread(target=lambda: chrome.hide_when_windowed(lambda: hidden.append("cancelled")))
    asker.start()
    asker.join(5)
    with caplog.at_level("INFO", logger="launcher.window_chrome"):
        opener = threading.Thread(target=chrome.keep)
        opener.start()
        opener.join(5)
    mac.main.run()
    assert hidden == [] and "toggle full screen" not in native.calls and chrome._hiding is None
    assert caplog.text.count("opened again before it left full screen") == 1
    if phase == "windowed":
        assert chrome.hide_when_windowed(lambda: hidden.append("hidden")) is True, "the next close hides at once"
        return
    if phase == "entering":
        full_screen(mac, "DidEnter")
        assert "toggle full screen" not in native.calls, "nobody asks for a cancelled close"
    hidden, _ = closed(mac, native, chrome)
    assert native.calls.count("toggle full screen") == 1 and chrome.hiding, "an alert now treats it as leaving"
    exited(mac, native)
    assert hidden == ["hidden"] and not chrome.hiding


@pytest.mark.parametrize("before_the_turn", ["DidExit", "a new entry"])
def test_the_consent_threads_close_settled_before_its_turn_acts_once(mac, before_the_turn):
    """The window leaves full screen (the close hides there, once) or enters it again (the close is dropped) before
    the queued turn runs: the turn finds nothing of its own to do."""
    chrome, _window, native = shown()
    full_screen(mac, "WillEnter", "DidEnter")
    native.mask |= wc._FULL_SCREEN
    hidden = []
    asker = threading.Thread(target=lambda: chrome.hide_when_windowed(lambda: hidden.append("hidden")))
    asker.start()
    asker.join(5)
    if before_the_turn == "DidExit":
        exited(mac, native)  # the owner's own exit
    else:
        full_screen(mac, "WillExit", "WillEnter", "DidEnter")
    mac.main.run()
    assert hidden == (["hidden"] if before_the_turn == "DidExit" else []) and chrome._hiding is None
    assert "toggle full screen" not in native.calls


@pytest.mark.parametrize("next_request", ["close", "handoff", "close and handoff"])
def test_a_close_dropped_after_an_ignored_toggle_lets_the_next_request_ask(mac, monkeypatch, next_request):
    """Carbon's case 3: AppKit ignored the close's toggle (no WillExit, the phase stays full) and the owner opened the
    window again. The dropped close no longer counts as asked, so the next close or handoff asks again, once between
    them; no clock is involved. While the close still waits, nothing asks twice (the cases above)."""
    monkeypatch.setattr(wc, "MACOS_HANDOFF_FULLSCREEN_WAIT_SEC", 3600.0)
    chrome, window, native = shown()
    mac.pywebview.windows.append(window)
    full_screen(mac, "WillEnter", "DidEnter")
    first, _ = closed(mac, native, chrome, closes=2)
    assert native.calls.count("toggle full screen") == 1
    chrome.keep()
    assert chrome._phase == "full" and chrome._hiding is None
    order = []
    if "close" in next_request:
        assert chrome.hide_when_windowed(lambda: order.append("hidden")) is False
        assert chrome.hide_when_windowed(lambda: order.append("twice")) is False
    if "handoff" in next_request:
        worker, outcome = hand_over(lambda: order.append("final"))
        mac.main.run(first=True)
    assert native.calls.count("toggle full screen") == 2
    full_screen(mac, "WillExit")
    mac.main.run()
    exited(mac, native)
    mac.main.run()
    expected = {"close": ["hidden"], "handoff": ["final"], "close and handoff": ["hidden", "final"]}[next_request]
    assert first == [] and order == expected
    if "handoff" in next_request:
        assert released(worker, outcome) == {"ran": True}
    assert native.calls.count("toggle full screen") == 2


@pytest.mark.parametrize("first", ["close", "handoff"])
@pytest.mark.parametrize("phase", ["full", "entering"])
def test_a_close_and_a_handoff_together_ask_once_and_both_follow_did_exit(mac, monkeypatch, first, phase):
    """(l)/4 One toggle per stint whichever asked first; at DidExit the window hides, then the handoff's turn finds it
    windowed and runs ``final``."""
    monkeypatch.setattr(wc, "MACOS_HANDOFF_FULLSCREEN_WAIT_SEC", 3600.0)
    chrome, window, native = shown()
    mac.pywebview.windows.append(window)
    full_screen(mac, "WillEnter", *(["DidEnter"] if phase == "full" else []))
    order = []

    def close():
        native.mask |= wc._FULL_SCREEN
        assert chrome.hide_when_windowed(lambda: order.append("hidden")) is False

    if first == "close":
        close()
    worker, outcome = hand_over(lambda: order.append("final"))
    mac.main.run(first=True)
    if first == "handoff":
        close()
    if phase == "entering":
        full_screen(mac, "DidEnter")
        mac.main.run()
    assert native.calls.count("toggle full screen") == 1 and order == []
    full_screen(mac, "WillExit")
    mac.main.run()
    exited(mac, native)
    mac.main.run()
    assert order == ["hidden", "final"] and released(worker, outcome) == {"ran": True}
    assert native.calls.count("toggle full screen") == 1


def test_a_handoff_after_the_close_finds_the_hidden_window_windowed(mac):
    """(l) The close completed: the window is hidden in its windowed phase, so the handoff runs at once."""
    chrome, window, native = shown()
    mac.pywebview.windows.append(window)
    full_screen(mac, "WillEnter", "DidEnter")
    hidden, _ = closed(mac, native, chrome)
    exited(mac, native)
    native.visible = False  # pywebview's orderOut:
    assert hidden == ["hidden"]
    seen = []
    worker, outcome = hand_over(lambda: seen.append("final"))
    mac.main.run(first=True)
    assert seen == ["final"] and finished(worker, outcome) == {"ran": True}


class Icon(launcher_background.Indicator):
    """The platform-neutral half of the indicator over a live icon."""

    def _launch(self):
        self.ready.set()
        return True


@pytest.mark.parametrize("meanwhile", ["nothing", "the icon goes away", "a quit"])
def test_the_indicator_hides_a_full_screen_window_after_did_exit_only_behind_a_live_icon(mac, meanwhile):
    """Background's half: the close is taken (the window stays), and at DidExit the window hides and counts as hidden
    only while the icon is live and no quit runs; otherwise it stays open and nothing hangs."""
    chrome, window, native = shown()
    background = types.SimpleNamespace(chrome=chrome, shutdown=threading.Event(), window=window)
    icon = Icon(background)
    assert icon.start() and icon.ready.is_set()
    full_screen(mac, "WillEnter", "DidEnter")
    native.mask |= wc._FULL_SCREEN
    assert icon.hide_on_close(window) is True and not icon.hidden and "hide" not in native.calls
    if meanwhile == "the icon goes away":
        icon.stop()
    elif meanwhile == "a quit":
        background.shutdown.set()
    exited(mac, native)
    expected = meanwhile == "nothing"
    assert ("hide" in native.calls) is expected and icon.hidden is expected and chrome._hiding is None


def macos_background(mac, monkeypatch):
    """``Background`` with its live icon over this window's real chrome, on macOS, with the window shown."""
    monkeypatch.setattr(launcher_background, "_MACOS", True, raising=False)
    monkeypatch.setattr(launcher_background, "indicator_class", lambda: Icon)
    background = launcher_background.Background(lambda: None, lambda: 1, threading.Event())
    chrome, window, native = shown()
    background.chrome, background.window, background._can_show = chrome, window, True
    assert background.indicator.start()
    return background, chrome, window, native


@pytest.mark.parametrize("boundary", ["after DidExit took the close", "between the open's show and its flag"])
def test_a_notification_click_racing_the_deferred_hide_leaves_the_window_open(mac, monkeypatch, boundary):
    """All three final critics' case: a notification click opens the window from its own thread
    (``desktop_notifications``: ``ouroboros-notification-click``). Held at the hide's boundaries, it was lost: (a) DidExit
    had taken the close, so the reopen's ``keep`` dropped nothing and the hide that followed won; (b) its show was queued
    before the hide and its flag cleared after it, a hidden window counted visible. The whole open is a main-thread turn
    now, before or after the hide: the window ends shown, counted visible, and the page hears the token. As in pywebview
    5.4 on Cocoa, this window's ``show`` and ``hide`` are queued main-thread turns and ``evaluate_js`` blocks its caller
    until its own turn ran, so the token may run before the queued show: only the end state is pinned."""
    from ouroboros.desktop_notifications import NativeNotifier

    background, chrome, window, native = macos_background(mac, monkeypatch)
    icon = background.indicator
    full_screen(mac, "WillEnter", "DidEnter")
    native.mask |= wc._FULL_SCREEN
    assert icon.hide_on_close(window) is True and native.calls.count("toggle full screen") == 1

    visible = [True]
    reached, resume, waiting, outcome = threading.Event(), threading.Event(), threading.Event(), {}

    def queued(name, shows):
        def turn():
            visible[0] = shows
            native.calls.append(name)
        return lambda: mac.main.callAfter(turn)

    show = queued("show", True)

    def held_show():
        show()  # pywebview queued the show
        if boundary.startswith("between") and threading.current_thread() is not threading.main_thread():
            reached.set()
            assert resume.wait(5)

    def evaluate_js(script):  # pywebview's: its turn queued after those already queued, the caller blocked until it ran
        ran = threading.Event()

        def turn():
            native.calls.append(("js", script))
            ran.set()

        mac.main.callAfter(turn)
        waiting.set()
        reached.set()
        assert ran.wait(5), "the main thread never ran the page's turn"

    window.show, window.hide, window.evaluate_js = held_show, queued("hide", False), evaluate_js

    def on_click(token):  # the click thread
        outcome["thread"] = threading.current_thread()
        try:
            background.open_notification(token)
        except BaseException as error:
            outcome["error"] = error

    hide_later = icon._hide_later

    def racing_hide(win):  # DidExit's turn on the main thread
        NativeNotifier(on_click)._clicked("task:7")
        assert reached.wait(5), "the click thread reached its boundary"
        hide_later(win)
        resume.set()

    icon._hide_later = racing_hide
    exited(mac, native)
    assert waiting.wait(5), "the click thread queued the page's turn"
    mac.main.run()
    outcome["thread"].join(5)
    assert not outcome["thread"].is_alive() and "error" not in outcome, outcome
    seen = [call for call in native.calls if call in ("hide", "show") or isinstance(call, tuple) and call[0] == "js"]
    assert "hide" in seen and [call[1] for call in seen if isinstance(call, tuple)] == [
        "window.ouroNotifications && window.ouroNotifications.activate(\"task:7\")"], seen
    assert visible == [True] and not icon.hidden and chrome._hiding is None, seen


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
