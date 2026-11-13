"""The macOS main window as one surface: its page runs under a transparent title bar, the real traffic lights on top.

pywebview 5.4 offers a titled window or ``frameless``, and frameless hides the system traffic
lights and turns the page into a drag handle. This leaf keeps the titled NSWindow and flips its
public switches instead: full-size content view, transparent title bar, hidden title (still the
window's accessible name), an empty unified toolbar that gives the bar the height of the page's
header row, no separator, no window tabs. One step is pywebview's own private detail undone:
pywebview paints its theme frame's last subview, the title bar container, with the window colour;
integration clears exactly that view and refuses when the view it painted does not hold the close
button. Any failed step restores the ordinary title bar for the window's lifetime.

The web UI is replaced independently of this shell, and a replacement may never speak to it: an
older UI after Restore, a custom ``OUROBOROS_UI_DIR`` page, and every page until pywebview injects its
bridge, which it does only after loading, when the first page is already on screen. So the opt-in is a
static declaration (``<meta name="ouroboros-window-chrome" content="integrated">``) the window reads
itself: one script, registered in ``before_show`` and injected at document start into the main
frame, posts it here on the main thread. It runs in its own content world, so a replaced page's
scripts can neither call nor overwrite it and no names collide. A document without the declaration
gets the ordinary title bar back, and one that never ran the script (``before_load`` fires at each
finished navigation) does too; messages name their document by ``performance.timeOrigin``, so an
older document never undoes a newer one. The window starts integrated: pywebview adds the web view
only after the first document loaded and declared, so the bar never flashes. The page reserves room
only for what is applied: this leaf measures the overlap (content height above ``contentLayoutRect``)
and the traffic lights' far edge and writes them into that document (``data-window-chrome``,
``--window-chrome-top``, ``--window-chrome-start``). A browser, another OS, an older shell, full
screen or a refused apply writes nothing, so nothing is reserved there; geometry that cannot be
measured returns the window to the ordinary bar rather than leave a transparent one over the page.

WKWebView takes the clicks under a transparent bar, so AppKit's title bar never sees them: the script
reports a plain left press inside the measured band, off any control and any text the page declares
selectable (``cursor: text``). A first press goes back to
AppKit's ``performWindowDragWithEvent:``. A second cannot go back: natively the title bar view hands a
press under a transparent bar to the web view again, never zooming. So this leaf performs the Mac's
"Double-click a window's title bar" choice, read at each double-click as Chromium reads it: Zoom
(stored as ``Maximize``, and the default) and Minimize by their public calls, Do Nothing by nothing.
Fill uses AppKit's private ``_zoomFill:`` only when present, as Chromium does; it is not Zoom.
An isolated macOS 26.4 probe matched the ordinary bar; other versions remain unqualified.
A visible toolbar stays as an opaque band over a full-screen page, so it hides on entering full screen. AppKit
ends the exit with the toolbar hidden again, one set during the exit too, so it is shown again once
the window is out; the style mask never changes in full screen.

The window's full-screen phase is AppKit's four notifications, observed from ``before_show`` with or without the
document script, and its two failures, which AppKit sends only to the window's delegate: a small delegate stands in
front of pywebview's, forwarding what that one answers, and gives it back on close. A failure is a hint the live style
mask checks (macOS also says a succeeding exit failed). A warm source handoff replaces this process only from the main
thread, in the turn that found every window out of full screen (``transfer_when_windowed``): once no window refuses, one
in full screen is asked once to leave, and the handoff follows its DidExit (isolated macOS 26.4 probes saw an exec in
full screen leave its Space briefly windowless, one after DidExit not). No window, or one no ``before_show`` has
registered (not yet shown, or the setup window), refuses at once, before anything is queued; one hidden in full screen,
a failed exit, a Stop before the windows agree, or no confirmation within ``MACOS_HANDOFF_FULLSCREEN_WAIT_SEC`` also
keeps the running shell: nothing is torn down.

The window's appearance follows the document's painted ``data-theme-choice`` (``data-theme`` for a
page without one): Light and Dark force Aqua or DarkAqua on this window only, System and an unthemed
page inherit. The window's appearance is also its page's ``prefers-color-scheme``, so forcing the
palette a System page painted would pin System to it; the Windows caption tint stays the bridge's
(``launcher_appearance``), which cannot say System.

Until its first document has loaded, a window shows only its own background, which pywebview paints
from a fixed ``background_color`` (the main window's was always the dark room). ``paint_room`` swaps
it, before the window is shown, for one dynamic colour AppKit resolves against the window's effective
appearance: the room colour (``web/ui.css`` ``--bg-primary``) of the Mac's Light or Dark, so the first
frame follows macOS and nothing waits for the page (owner decision on #1111: A). A page whose choice
differs forces its appearance when it declares, and the same colour then follows it, live resizes
included. The windows shown before the server have no page choice to read and keep the Mac's.
On Windows ``paint_room`` sets that fixed colour to the room of the app mode instead; it stays put
after the page paints. On GTK that colour reaches nothing seen: pywebview 5.4's ``GtkWindow`` style rule
showed no effect, so the window keeps its theme's colour, and from the first load the web view's own
surface, opaque white by default, covers it on every host until a page paints (hosted run on GTK 3.24.33
and WebKitGTK 2.50.4; a head-blocked ``html=`` document still showed the theme's colour). ``paint_room``
gives that surface the room of the palette the theme resolved; a subsequent hosted run observed it
before the held URL's page painted.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import threading
from typing import Any, Callable, Optional

from ouroboros.launcher_appearance import NativeAppearance, host_palette, palette_of_background
from ouroboros.runtime_limits import MACOS_HANDOFF_FULLSCREEN_WAIT_SEC

log = logging.getLogger("launcher.window_chrome")

_MACOS = sys.platform == "darwin"
_LINUX = sys.platform.startswith("linux")
HANDLER = "ouroborosWindowChrome"
WORLD = "ouroboros-window"
TOOLBAR = "ouroboros-window-chrome"
INTEGRATED = "integrated"
APPEARANCES = {"light": "NSAppearanceNameAqua", "dark": "NSAppearanceNameDarkAqua"}
# The room colour of each palette, web/ui.css --bg-primary (tests/test_appearance_static.py keeps them equal).
ROOM = {"light": "#ffffff", "dark": "#0f0f11"}
# AppKit enum values (macOS 11+ for the separator style; Ouroboros needs macOS 12). The toolbar style is read
# from AppKit by name (``_integrate``): a mistyped number is another style with another bar height.
_FULL_SIZE_CONTENT = 1 << 15  # NSWindowStyleMaskFullSizeContentView
_TITLE_VISIBLE, _TITLE_HIDDEN = 0, 1
_SEPARATOR_NONE = 1  # NSTitlebarSeparatorStyleNone
_TABBING_DISALLOWED = 2
_LEFT_MOUSE_DOWN, _LEFT_MOUSE_UP, _LEFT_MOUSE_DRAGGED = 1, 2, 6
_BUTTONS = (0, 1, 2)  # close, miniaturize, zoom
_FULL_SCREEN = 1 << 14  # NSWindowStyleMaskFullScreen
# AppKit's full-screen notifications, each with the phase it begins; its failures are delegate messages only.
_PHASES = (("NSWindowWillEnterFullScreenNotification", "entering"), ("NSWindowDidEnterFullScreenNotification", "full"),
           ("NSWindowWillExitFullScreenNotification", "exiting"), ("NSWindowDidExitFullScreenNotification", "windowed"))
_FAILED_ENTER, _FAILED_EXIT = "windowDidFailToEnterFullScreen:", "windowDidFailToExitFullScreen:"
# System Settings → Desktop & Dock → "Double-click a window's title bar to", in the global defaults domain.
# Undocumented by Apple; Chromium's native_widget_mac_nswindow.mm uses these words and guarded
# _zoomFill: (source inspected 2026-10-17). Selector presence cannot guarantee future semantics.
_DOUBLE_CLICK_SETTING = "AppleActionOnDoubleClick"
_ZOOM, _MINIMIZE, _NOTHING = "Maximize", "Minimize", "None"  # unset is Zoom; "Fill" has no public call

# Runs in every main-frame document of the window, in the shell's content world: the page cannot
# call it, and it needs nothing from the page beyond the declaration and the painted theme.
DOCUMENT_SCRIPT = r"""(() => {
    'use strict';
    const port = window.webkit && window.webkit.messageHandlers && window.webkit.messageHandlers.%(handler)s;
    if (!port) return;
    const clock = window.performance;
    let doc = clock.timeOrigin;
    let declared = false;
    let painting = null;
    let top = 0;
    const post = (message) => { try { port.postMessage(Object.assign({ doc }, message)); } catch (_) { /* window gone */ } };
    const painted = () => {
        const data = document.documentElement ? document.documentElement.dataset : {};
        return { choice: data.themeChoice || '', theme: data.theme || '' };
    };
    const declaration = () => document.querySelector('meta[name="ouroboros-window-chrome"]');
    const declare = () => {
        const meta = declaration();
        if (declared || (!meta && !document.body && document.readyState === 'loading')) return declared;
        declared = true;
        post(Object.assign({ type: 'declare', chrome: meta && meta.content === '%(integrated)s' ? '%(integrated)s' : '' }, painted()));
        if (!painting && document.documentElement) {
            painting = new MutationObserver(() => post(Object.assign({ type: 'appearance' }, painted())));
            painting.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'data-theme-choice'] });
        }
        return true;
    };
    if (!declare()) {
        const parsing = new MutationObserver(() => { if (declare()) parsing.disconnect(); });
        parsing.observe(document, { childList: true, subtree: true });
        document.addEventListener('DOMContentLoaded', () => { parsing.disconnect(); declare(); }, { once: true });
    }
    window.addEventListener('pageshow', (event) => {
        if (!event.persisted) return;
        doc = clock.timeOrigin + clock.now();  // back from the page cache: the window's page again
        declared = false;
        declare();
    });
    window.__ouroWindowChrome = (state) => {
        const root = document.documentElement;
        if (!state || state.doc !== doc || !root) return;
        top = state.top > 0 ? state.top : 0;
        if (top) {
            root.dataset.windowChrome = '%(integrated)s';
            root.style.setProperty('--window-chrome-top', `${top}px`);
            root.style.setProperty('--window-chrome-start', `${state.start}px`);
        } else {
            delete root.dataset.windowChrome;
            root.style.removeProperty('--window-chrome-top');
            root.style.removeProperty('--window-chrome-start');
        }
    };
    const CONTROLS = 'a[href], button, input, select, textarea, summary, label, iframe, video, audio, '
        + '[contenteditable]:not([contenteditable="false"]), [tabindex]:not([tabindex="-1"]), [draggable="true"], '
        + '[role="button"], [role="link"], [role="tab"], [role="menuitem"], [role="option"], [role="checkbox"], '
        + '[role="radio"], [role="switch"], [role="slider"], [role="textbox"]';
    // Any other cursor announces the page's own interaction. Text computes `auto` and drags like a title;
    // `cursor: text` declares text the page keeps selectable (the document reader's name).
    const PLAIN = ['auto', 'default'];
    // On the window, after every element's handler: a press the page took (a prevented mousedown, or a
    // cancelled pointerdown, which suppresses mousedown) stays the page's.
    window.addEventListener('mousedown', (event) => {
        if (!top || event.button !== 0 || event.defaultPrevented || event.clientY >= top) return;
        if (event.metaKey || event.ctrlKey || event.altKey || event.shiftKey) return;
        const target = event.target instanceof Element ? event.target : null;
        if (!target || target.closest(CONTROLS) || !PLAIN.includes(getComputedStyle(target).cursor)) return;
        event.preventDefault();  // no text selection or focus move: this press belongs to the title bar
        post({ type: event.detail >= 2 ? 'double-click' : 'drag' });
    });
})();
""" % {"handler": HANDLER, "integrated": INTEGRATED}

_classes: dict = {}  # Objective-C class names are process-wide: the handler class is defined once
_tracked: list = []  # WindowChrome observing its window's full-screen phase: from before_show until closed
_transfer: Any = None  # the warm source handoff waiting for the windows, if any


def _handler_class():
    if "handler" not in _classes:
        import Foundation

        class OuroborosWindowChromeHandler(Foundation.NSObject):
            def userContentController_didReceiveScriptMessage_(self, controller, message):
                self.chrome.receive(message.body())

        _classes["handler"] = OuroborosWindowChromeHandler
    return _classes["handler"]


def _delegate_class():
    """The window delegate that hears AppKit's two full-screen failures. NSWindow keeps one weak delegate, so this one
    stands in front of the previous (pywebview's: close, resize, move...), answers what it answers and forwards those
    messages to it unchanged; ``chrome`` and ``previous`` are its strong references."""
    if "delegate" not in _classes:
        import Foundation
        import objc

        def failed(delegate: Any, selector: str, window: Any, entering: bool) -> None:
            """AppKit passes the NSWindow itself to these two (NSWindow.h), not a notification."""
            previous = getattr(delegate, "previous", None)
            try:
                if previous is not None and previous.respondsToSelector_(selector):
                    getattr(previous, selector.replace(":", "_"))(window)
            except Exception:
                log.warning("The window's own delegate did not take %s.", selector, exc_info=True)
            chrome = getattr(delegate, "chrome", None)
            if chrome is not None:
                chrome._failed(entering)

        class OuroborosWindowDelegate(Foundation.NSObject):
            def respondsToSelector_(self, selector):
                previous = getattr(self, "previous", None)
                return bool(objc.super(OuroborosWindowDelegate, self).respondsToSelector_(selector)
                            or (previous is not None and previous.respondsToSelector_(selector)))

            def forwardingTargetForSelector_(self, selector):
                previous = getattr(self, "previous", None)
                if previous is not None and previous.respondsToSelector_(selector):
                    return previous
                return objc.super(OuroborosWindowDelegate, self).forwardingTargetForSelector_(selector)

            def windowDidFailToEnterFullScreen_(self, window):
                failed(self, _FAILED_ENTER, window, True)

            def windowDidFailToExitFullScreen_(self, window):
                failed(self, _FAILED_EXIT, window, False)

        _classes["delegate"] = OuroborosWindowDelegate
    return _classes["delegate"]


def _give_back_delegate(native: Any, delegate: Any) -> None:
    """Main thread, after close: the previous delegate again, unless another one replaced this one meanwhile."""
    try:
        if native.delegate() is delegate:
            native.setDelegate_(delegate.previous)
    except Exception:
        log.debug("The window's previous delegate was not restored.", exc_info=True)
    delegate.chrome = None


def _webview_of(window: Any) -> Any:
    """pywebview 5.4's WKWebView of ``window``: it exists in ``before_show`` but is no subview yet."""
    from webview.platforms.cocoa import BrowserView

    return BrowserView.instances[window.uid].webview


def paint_room(window: Any) -> None:
    """Before ``webview.start``: ``window`` opens in the room colour of its host's appearance.

    macOS: one dynamic colour, set in ``before_show``. Windows: the room of the app mode a System page
    will paint (``launcher_appearance.host_palette``) becomes pywebview's ``background_color``, the form's,
    its WebView2's default and the caption's first colour (a dark host's WebView2 then held a ``url=``
    response over its own #121212), and a window made from ``html=`` has no origin, so no page choice or
    bridge will speak for it: its caption stays with its host. GTK: ``_paint_gtk_room``, set in
    ``before_show``. Elsewhere, or with the palette unknown, the colour does not change."""
    if window is None:
        return
    if _MACOS:
        window.events.before_show += lambda: _paint_room(getattr(window, "native", None))
        return
    if _LINUX:
        window.events.before_show += lambda: _paint_gtk_room(window)
        return
    if room := ROOM.get(host_palette()):
        window.background_color = room
    if getattr(window, "html", None):
        NativeAppearance().attach(window)  # inert off Windows


def _paint_room(native: Any) -> None:
    """pywebview ``before_show`` (main thread, before the window is ordered front): one dynamic room colour."""
    if native is None:
        return
    try:
        import AppKit

        def srgb(value: str) -> Any:  # as pywebview's own fixed colour is made, so the page's sRGB room matches
            red, green, blue = (int(value[index:index + 2], 16) / 255.0 for index in (1, 3, 5))
            return AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(red, green, blue, 1.0)

        light, dark = srgb(ROOM["light"]), srgb(ROOM["dark"])
        names = [AppKit.NSAppearanceNameAqua, AppKit.NSAppearanceNameDarkAqua]

        def resolve(appearance: Any) -> Any:  # AppKit asks at each draw; a High Contrast variant matches its base
            return light if appearance.bestMatchFromAppearancesWithNames_(names) == names[0] else dark

        native.setBackgroundColor_(AppKit.NSColor.colorWithName_dynamicProvider_(None, resolve))
    except Exception:
        log.warning("The window keeps its fixed background colour until its page paints.", exc_info=True)


def _paint_gtk_room(window: Any) -> None:
    """pywebview ``before_show`` on GTK (main thread, just before the first load, so WebKit carries the colour
    into the page it creates): the web view's own background becomes the room of the palette its window's GTK
    theme resolved. The host's meaning is that theme's ``theme_bg_color`` as GTK answers it, by luminance and
    never by theme name; a page's own background still paints over it. A theme without that colour leaves
    WebKit's white, as does any failure here. Another backend (Qt on Linux) is not this surface."""
    gui = getattr(window, "gui", None)
    if getattr(gui, "renderer", "") != "gtkwebkit2":  # pywebview's name for this backend, whose module it is
        return
    try:
        known, theme = window.native.get_style_context().lookup_color("theme_bg_color")
        room = ROOM.get(palette_of_background("#%02x%02x%02x" % tuple(
            round(part * 255) for part in (theme.red, theme.green, theme.blue)))) if known else None
        if room is None:
            log.info("The GTK theme gives no readable window colour; the web view keeps WebKit's white.")
            return
        colour = gui.Gdk.RGBA()
        colour.parse(room)
        gui.BrowserView.instances[window.uid].webview.set_background_color(colour)
    except Exception:
        log.warning("The web view keeps WebKit's white until its page paints.", exc_info=True)


class WindowChrome:
    """One window's title bar and appearance, decided per document on the main thread."""

    def __init__(self) -> None:
        self._window: Any = None
        self._native: Any = None
        self._webview: Any = None
        self._world: Any = None
        self._handler: Any = None
        self._fill: Any = None  # the title bar container pywebview painted; None: never integrate
        self._ordinary: dict = {}  # the switches integration changed, as they were before
        self._observers: list = []
        self._delegate: Any = None  # (window, failure delegate in front of its own) while installed
        self._doc = -math.inf  # the window's current document (its time origin)
        self._declared = False  # that document's script spoke since the last finished navigation
        self._wanted = False
        self._integrated = False
        self._refused = False
        self._phase: Optional[str] = None  # AppKit's full-screen phase (_PHASES); None: not observed
        self._leaving: Any = None  # the handoff, or the close's ``later``, that asked this full-screen stint to end
        self._hiding: Optional[Callable[[], None]] = None  # the background close waiting for its turn or DidExit
        self._hiding_lock = threading.Lock()  # a reopen may drop that close from another thread
        self._appearance: Optional[str] = None  # None: inherited
        self._sent: tuple = ()
        self._closed = False

    @property
    def _fullscreen(self) -> bool:
        """From WillEnter until DidExit: the title bar's switches wait and nothing is reserved."""
        return self._phase not in (None, "windowed")

    def attach(self, window: Any) -> None:
        """Register on ``window`` before ``webview.start``: its room colour everywhere, the rest on macOS only."""
        paint_room(window)
        if not _MACOS or window is None:
            return
        self._window = window
        window.events.before_show += self._before_show
        window.events.before_load += self._navigation_finished
        window.events.closed += self._close

    def _before_show(self) -> None:
        """pywebview ``before_show`` (synchronous, main thread, before the first load)."""
        native = getattr(self._window, "native", None)
        if native is None:
            return
        self._observe_full_screen(native)  # a source handoff needs the phase even without the document script
        try:
            import WebKit

            webview = _webview_of(self._window)
            world = WebKit.WKContentWorld.worldWithName_(WORLD)
            handler = _handler_class().alloc().init()
            handler.chrome = self
            controller = webview.configuration().userContentController()
            controller.addScriptMessageHandler_contentWorld_name_(handler, world, HANDLER)
            controller.addUserScript_(WebKit.WKUserScript.alloc().initWithSource_injectionTime_forMainFrameOnly_inContentWorld_(
                DOCUMENT_SCRIPT, WebKit.WKUserScriptInjectionTimeAtDocumentStart, True, world))
        except Exception:
            log.warning("The window keeps its ordinary title bar and appearance: no document script.", exc_info=True)
            return
        self._native, self._webview, self._world, self._handler = native, webview, world, handler
        try:
            fill = native.contentView().superview().subviews().lastObject()  # the view pywebview just painted
            self._fill = fill if native.standardWindowButton_(_BUTTONS[0]).isDescendantOf_(fill) else None
        except Exception:
            self._fill = None
        if self._fill is None:
            log.warning("pywebview's title bar fill is not the traffic lights' container; the title bar stays ordinary.")
        # Nothing is on screen to collide with yet: pywebview adds the web view to the window only once the first
        # document has finished loading, after it declared, so the bar never flashes at launch and an undeclaring
        # first document is ordinary again before it is seen.
        self._wanted = True
        self._reconcile()

    def receive(self, body: Any) -> None:
        """A message of a document's shell script (main thread, in the order its documents sent them)."""
        if self._closed or self._native is None:
            return
        try:
            message = dict(body)
            kind, doc = str(message.get("type") or ""), float(message.get("doc"))
        except (TypeError, ValueError):
            return
        if not math.isfinite(doc):
            return
        if kind == "declare":
            if doc < self._doc:
                return  # an older document's late word
            self._doc, self._declared = doc, True
            self._wanted = str(message.get("chrome") or "") == INTEGRATED
            self._set_appearance(message.get("choice"), message.get("theme"))
            self._reconcile()
        elif doc != self._doc:
            return
        elif kind == "appearance":
            self._set_appearance(message.get("choice"), message.get("theme"))
        elif kind == "drag":
            self._drag()
        elif kind == "double-click":
            self._double_click()

    def _navigation_finished(self) -> None:
        """pywebview ``before_load`` (synchronous, main thread): a document that never declared is ordinary."""
        if self._native is not None and not self._declared:
            self._wanted = False
            self._set_appearance("", "")
            self._reconcile()
        self._declared = False

    def _reconcile(self) -> None:
        self._switch()
        self._publish()

    def _switch(self) -> None:
        want = self._wanted and self._fill is not None and not self._refused
        if not self._fullscreen and want != self._integrated:
            if want:
                self._integrate()
            else:
                self._restore()

    def _integrate(self) -> None:
        import AppKit

        native = self._native
        try:
            frame = native.frame()
            self._ordinary = {
                "mask": native.styleMask(), "tabbing": native.tabbingMode(),
                "toolbar_style": native.toolbarStyle(), "separator": native.titlebarSeparatorStyle(),
            }
            self._integrated = True  # from here a failure must restore
            native.setStyleMask_(self._ordinary["mask"] | _FULL_SIZE_CONTENT)
            native.setTitlebarAppearsTransparent_(True)
            native.setTitleVisibility_(_TITLE_HIDDEN)
            self._fill.setBackgroundColor_(AppKit.NSColor.clearColor())
            native.setToolbar_(AppKit.NSToolbar.alloc().initWithIdentifier_(TOOLBAR))
            native.setToolbarStyle_(AppKit.NSWindowToolbarStyleUnified)  # lights centred on the header row
            native.setTitlebarSeparatorStyle_(_SEPARATOR_NONE)
            native.setTabbingMode_(_TABBING_DISALLOWED)
            native.setFrame_display_(frame, True)  # the switches resize the frame; the window keeps its size
        except Exception:
            log.warning("The integrated title bar could not be applied; the window keeps the ordinary one.", exc_info=True)
            self._refused = True
            self._restore()

    def _restore(self) -> None:
        """Every switch back, each on its own: one refusal never keeps the rest integrated."""
        import AppKit

        native, ordinary = self._native, self._ordinary
        self._integrated = False
        frame = native.frame()
        steps = [
            lambda: native.setToolbar_(None),
            lambda: native.setStyleMask_(ordinary.get("mask", native.styleMask() & ~_FULL_SIZE_CONTENT)),
            lambda: native.setTitlebarAppearsTransparent_(False),
            lambda: native.setTitleVisibility_(_TITLE_VISIBLE),
            lambda: self._fill.setBackgroundColor_(AppKit.NSColor.windowBackgroundColor()),  # pywebview's own fill
        ]
        if ordinary:
            steps += [lambda: native.setToolbarStyle_(ordinary["toolbar_style"]),
                      lambda: native.setTitlebarSeparatorStyle_(ordinary["separator"]),
                      lambda: native.setTabbingMode_(ordinary["tabbing"])]
        steps.append(lambda: native.setFrame_display_(frame, True))
        for step in steps:
            try:
                step()
            except Exception:
                log.warning("A title bar switch could not be restored.", exc_info=True)

    def zoom_changed(self) -> None:
        """The window's page zoom changed (main thread, ``launcher_page_zoom``): the band, in CSS px, changes with it."""
        if not self._closed and self._webview is not None:
            self._publish()

    def _publish(self) -> None:
        """Tell the current document what the window actually covers; the page reserves nothing else."""
        if not math.isfinite(self._doc) or self._webview is None:
            return
        try:
            state = self._measure()
        except Exception:
            # A transparent bar over a page told to reserve nothing could cover its controls: the ordinary
            # bar returns for this window's lifetime, now or as full screen ends (its mask never changes there).
            log.warning("Title bar geometry unreadable; the window returns to the ordinary title bar.", exc_info=True)
            state = {"top": 0, "start": 0}
            self._refused = True
            if self._integrated and not self._fullscreen:
                self._restore()
        sent = (self._doc, state["top"], state["start"])
        if sent == self._sent:
            return
        script = f"window.__ouroWindowChrome && window.__ouroWindowChrome({json.dumps({'doc': self._doc, **state})})"
        try:
            self._webview.evaluateJavaScript_inFrame_inContentWorld_completionHandler_(script, None, self._world, None)
        except Exception:
            log.warning("The page could not be told the title bar geometry; the next change tells it again.", exc_info=True)
            return  # not delivered: the next publish of the same state is not a duplicate
        self._sent = sent
        log.info("Title bar for document %.3f: %s (band %s, lights to %s CSS px; full screen %s)", self._doc,
                 INTEGRATED if self._integrated else "ordinary", state["top"], state["start"], self._fullscreen)

    def _measure(self) -> dict:
        """CSS pixels the title bar covers: the band above ``contentLayoutRect`` and the traffic lights' far edge."""
        native = self._native
        if self._fullscreen or not native.styleMask() & _FULL_SIZE_CONTENT:
            return {"top": 0, "start": 0}  # full screen hides the bar until the pointer reveals it
        layout = native.contentLayoutRect()
        top = native.contentView().frame().size.height - (layout.origin.y + layout.size.height)
        start = 0.0
        for kind in _BUTTONS:
            button = native.standardWindowButton_(kind)
            if button is not None and not button.isHidden():
                rect = button.convertRect_toView_(button.bounds(), None)
                start = max(start, rect.origin.x + rect.size.width)
        try:
            zoom = float(self._webview.pageZoom()) * float(self._webview.magnification())
        except Exception:
            zoom = 1.0
        zoom = zoom if zoom > 0 else 1.0
        top = round(max(0.0, top) / zoom, 2)
        return {"top": top, "start": round(start / zoom, 2) if top else 0}

    def _observe_full_screen(self, native: Any) -> None:
        """The phase is known only once all four notifications are observed; a new window is not in full screen."""
        try:
            import AppKit

            center = AppKit.NSNotificationCenter.defaultCenter()
            for name, phase in _PHASES:
                # No queue: the block runs where AppKit posts, the main thread, around the transition.
                self._observers.append(center.addObserverForName_object_queue_usingBlock_(
                    name, native, None, lambda _note, phase=phase: self._full_screen(phase)))
        except Exception:
            log.warning("Full-screen notifications unavailable; a source handoff keeps this shell.", exc_info=True)
            return
        self._phase = "windowed"
        _tracked.append(self)
        try:
            delegate = _delegate_class().alloc().init()
            delegate.previous, delegate.chrome = native.delegate(), self
            native.setDelegate_(delegate)
            self._delegate = (native, delegate)
        except Exception:
            log.warning("Full-screen failures unheard; a failed exit keeps this shell after the wait.", exc_info=True)

    def _full_screen(self, phase: str) -> None:
        """An observed notification (main thread): the window's phase, its title bar, then a waiting handoff."""
        self._phase = phase
        if phase == "entering":
            self._show_toolbar(False)
            self._publish()
        elif phase == "exiting":
            self._show_toolbar(True)
        elif phase == "windowed":
            self._switch()  # a document that arrived in full screen gets its title bar now
            # AppKit ends the exit with the toolbar hidden, after WillExit showed it, and a toolbar set here starts
            # hidden too: shown now, before the band is measured, or the page reserves only the bare title bar.
            self._show_toolbar(True)
            self._publish()
        if phase in ("full", "windowed"):
            self._leaving = None  # a stint began or ended: a handoff still waiting asks this one anew
        if phase == "entering":
            self._drop_hiding("A new full-screen entry keeps the closed window open.")
        elif phase == "full" and self._hiding is not None:
            self._queue_close(self._hiding)  # AppKit ignores a toggle sent inside DidEnter: ask in the next turn
        elif phase == "windowed":
            self._hide_now()
        _resume()

    def _failed(self, entering: bool) -> None:
        """AppKit's delegate call (main thread), a hint the live mask checks. A failed entry the mask confirms (bit
        clear) while entering is windowed, and a handoff looks again; a failed exit the mask does not deny (unreadable
        too) stays full screen, refusing a handoff at once, unasked again. Anything else changes nothing."""
        if self._closed or self._phase is None:
            return
        try:
            full: Optional[bool] = bool(self._window.native.styleMask() & _FULL_SCREEN)
        except Exception:
            log.debug("The window's style mask is unreadable after a full-screen failure.", exc_info=True)
            full = None  # unknown: never the reading that makes an exec likelier
        if entering:
            if full is False and self._phase == "entering":  # exiting clears the bit too: never windowed before DidExit
                self._full_screen("windowed")
            return
        if full is False:
            log.info("macOS said a full-screen exit failed with the window's mask already out; a handoff waits for "
                     "DidExit.")
            return
        self._phase, self._leaving = "full", None
        self._drop_hiding("macOS could not take the closed window out of full screen; it stays open.", logging.WARNING)
        self._show_toolbar(False)  # WillExit showed it
        self._publish()
        transfer = _transfer
        if transfer is not None and transfer.settle("refused"):
            log.warning("macOS could not take a window out of full screen; this shell keeps running.")

    def windowed(self) -> Optional[bool]:
        """Main thread, for a source handoff: True when AppKit's last word and the live style mask both say this window
        is out of full screen; None while it is in or moving through full screen; False when unknown or hidden in full
        screen, where only showing it could end the stint."""
        native = getattr(self._window, "native", None)
        if self._phase is None or self._closed or native is None:
            return False
        if self._phase == "windowed":
            return not native.styleMask() & _FULL_SCREEN  # the mask disagreeing is unknown, not windowed
        return None if native.isVisible() else False

    def leave(self, request: Any) -> None:
        """Ask once per stint, in a main-thread turn after DidEnter (never inside it): a transition under way is never
        toggled. A handoff or the background close (its ``later``) that asked still waits; one that gave up (abandoned,
        refused, a close dropped by a reopen, failure or entry) leaves a later request to ask, as when AppKit ignored the
        toggle and nothing began."""
        leaving = self._leaving
        if self._phase != "full" or leaving is request or (leaving is not None and (
                leaving is self._hiding or getattr(leaving, "state", None) == "pending")):
            return
        self._leaving = request
        self._window.native.toggleFullScreen_(None)

    def hide_when_windowed(self, later: Callable[[], None]) -> bool:
        """A background close (``launcher_background``): True when the caller hides the window now, as before (off macOS,
        not observed, or out of full screen by AppKit's last word and the live mask). Else False: ``later`` hides it on the
        main thread at the DidExit that ends full screen, asked once and only in a turn after DidEnter; off the main
        thread it is taken at once and judged in a main-thread turn. A real failed exit, a new entry or ``keep`` drops it with one
        line in launcher.log, the window stays open and a new close, or a handoff, asks again.
        No timer: unconfirmed, a window is never hidden in full screen."""
        if self._phase is None or self._closed:
            return True
        main = threading.current_thread() is threading.main_thread()
        if main and self._phase == "windowed":
            return self._out()
        with self._hiding_lock:
            if self._hiding is not None:
                return False  # a close already waits: one exit, one toggle
            self._hiding = later  # armed before any turn: a reopen from now on drops it (``keep``)
        if not main:  # the consent question's thread: the phase and the toggle are the main thread's
            self._queue_close(later, reraise=True)
        elif self._phase == "full":
            self._ask_to_hide()
        return False

    def _queue_close(self, later: Callable[[], None], *, reraise: bool = False) -> None:
        """The close's own later main-thread turn (``_close_turn``): from another thread, or from inside DidEnter, where
        AppKit ignored the toggle (4 of 4 native runs). A reopen, failure, entry or closed window before it leaves it
        nothing to do."""
        try:
            from PyObjCTools import AppHelper

            AppHelper.callAfter(self._close_turn, later)
        except Exception:
            self._drop_hiding("The main thread cannot take the close; the window stays open.", logging.WARNING)
            if reraise:
                raise
            log.debug("callAfter refused the close's turn.", exc_info=True)

    @property
    def hiding(self) -> bool:
        """A background close waits: the window is leaving, so an alert treats it as backgrounded."""
        return self._hiding is not None

    def _out(self) -> bool:
        try:
            out = not self._window.native.styleMask() & _FULL_SCREEN
        except Exception:
            out = False
        if not out:
            log.warning("The closed window's style mask does not confirm it out of full screen; it stays open.")
        return out

    def keep(self) -> None:
        """The window is opened again (any thread): a close waiting for DidExit hides nothing."""
        self._drop_hiding("The closed window was opened again before it left full screen; it stays open.")

    def _close_turn(self, later: Callable[[], None]) -> None:
        """Main thread: the consent thread's close, unless a reopen, entry, failure or DidExit settled it meanwhile."""
        if self._hiding is not later or self._closed:
            return
        if self._phase == "windowed":
            with self._hiding_lock:
                if self._hiding is not later:
                    return
                self._hiding = None
            if self._out():
                self._run(later)
        elif self._phase == "full":
            self._ask_to_hide()

    def _ask_to_hide(self) -> None:
        later = self._hiding
        if later is None:  # a reopen on another thread dropped it just now
            return
        try:
            self.leave(later)
        except Exception:
            log.debug("toggleFullScreen: refused.", exc_info=True)
            self._leaving = None
            self._drop_hiding("macOS refused to take the closed window out of full screen; it stays open.",
                              logging.WARNING)

    def _hide_now(self) -> None:
        with self._hiding_lock:
            later, self._hiding = self._hiding, None
        if later is not None:
            self._run(later)

    def _drop_hiding(self, line: str, level: int = logging.INFO) -> None:
        with self._hiding_lock:
            waiting, self._hiding = self._hiding, None
        if waiting is not None:
            log.log(level, line)

    @staticmethod
    def _run(later: Callable[[], None]) -> None:
        try:
            later()
        except Exception:
            log.warning("The closed window could not be hidden; it stays open.", exc_info=True)

    def _show_toolbar(self, visible: bool) -> None:
        toolbar = self._native.toolbar() if self._integrated else None
        if toolbar is not None:
            toolbar.setVisible_(visible)

    def _live_press(self, kinds: tuple) -> Any:
        """The left-button event of ``kinds`` a band message came from while AppKit still processes it; None once
        AppKit moved past it."""
        if not self._integrated or self._fullscreen:
            return None
        import AppKit

        event = AppKit.NSApplication.sharedApplication().currentEvent()
        return event if event is not None and event.type() in kinds else None

    def _drag(self) -> None:
        """The title bar's press: AppKit's own window drag, from the event the press is still in."""
        press = self._live_press((_LEFT_MOUSE_DOWN, _LEFT_MOUSE_DRAGGED))
        if press is not None:
            self._native.performWindowDragWithEvent_(press)

    def _double_click(self) -> None:
        """The Mac's title-bar double-click choice, read now so a changed setting applies at once. The second
        press counts while held, or by its release when the message arrives late (AppKit's own bar acts on the
        release); a third press is no double-click. One message, one action: no press is handed to a view."""
        press = self._live_press((_LEFT_MOUSE_DOWN, _LEFT_MOUSE_UP))
        if press is None or press.clickCount() != 2:
            return
        import Foundation

        action = Foundation.NSUserDefaults.standardUserDefaults().stringForKey_(_DOUBLE_CLICK_SETTING)
        if action is None or action == _ZOOM:
            self._native.performZoom_(None)
        elif action == _MINIMIZE:
            self._native.performMiniaturize_(None)
        elif action == "Fill" and self._native.respondsToSelector_("_zoomFill:"):
            self._native._zoomFill_(None)
        elif action != _NOTHING:
            log.info("Title bar double-click action %r is unavailable; nothing done.", str(action))

    def _set_appearance(self, choice: Any, theme: Any) -> None:
        choice, theme = str(choice or ""), str(theme or "")
        name = APPEARANCES.get(choice) if choice else APPEARANCES.get(theme)
        if name == self._appearance:
            return
        try:
            import AppKit

            self._native.setAppearance_(AppKit.NSAppearance.appearanceNamed_(getattr(AppKit, name)) if name else None)
            self._appearance = name
        except Exception:
            log.warning("The window's appearance could not follow the page.", exc_info=True)

    def _close(self) -> None:
        """pywebview ``closed`` (its own thread): no further writes, the notification center lets go of this window, the
        main thread gives the previous delegate back, and a waiting handoff looks again at the windows left."""
        self._closed = True
        self._hiding = None
        if self in _tracked:
            _tracked.remove(self)
        installed, self._delegate = self._delegate, None
        try:
            if installed is not None:
                from PyObjCTools import AppHelper

                AppHelper.callAfter(_give_back_delegate, *installed)
            _resume()
        except Exception:
            log.debug("The closed window's delegate or waiting handoff was not reached.", exc_info=True)
        observers, self._observers = self._observers, []
        if not observers:
            return
        try:
            import AppKit

            center = AppKit.NSNotificationCenter.defaultCenter()
            for token in observers:
                center.removeObserver_(token)
        except Exception:
            log.debug("Full-screen observers not released.", exc_info=True)


class _Transfer:
    """One warm source handoff: pending until the main thread claims it to run ``final``, refuses it, or its worker's
    deadline abandons it. The lock guards only that one transition, never animation, I/O or ``final``."""

    def __init__(self, final: Callable[[], None], cancelled: Callable[[], bool]) -> None:
        self.final, self.cancelled, self.state, self.error = final, cancelled, "pending", None
        self.lock, self.done = threading.Lock(), threading.Event()

    def settle(self, state: str) -> bool:
        with self.lock:
            if self.state != "pending":
                return False
            self.state = state
        if state != "claimed":
            self.done.set()
        return True


def transfer_when_windowed(final: Callable[[], None], cancelled: Callable[[], bool] = lambda: False) -> bool:
    """A warm macOS source handoff, from its worker thread: ``final`` (teardown and exec) runs on the main thread in
    the very turn that found every window of this process out of full screen, so no window input comes between.

    False: refused, ``final`` never ran (at once, before anything is queued, while no window exists or one is not
    observed: not yet shown, or the setup window; one hidden in full screen or kept by a failed exit;
    ``cancelled`` in a turn, so a Stop asks no window to leave; no main-thread turn or AppKit confirmation within
    ``MACOS_HANDOFF_FULLSCREEN_WAIT_SEC``). True: ``final`` returned; its exception is raised on the worker, where
    the caller's recovery may wait. The deadline abandons only an attempt the main thread has not claimed; a later
    turn of an abandoned one does nothing."""
    global _transfer
    observers = _observers()
    if not observers or None in observers:  # an exit 42 before the setup or main window exists or is observed
        log.warning("No window is observed yet, so none can be confirmed out of full screen; this shell keeps running.")
        return False
    transfer = _transfer = _Transfer(final, cancelled)
    try:
        from PyObjCTools import AppHelper

        AppHelper.callAfter(_step, transfer)
    except Exception:
        transfer.settle("refused")
        log.warning("The main thread cannot take the source handoff; this shell keeps running.", exc_info=True)
        return False
    if not transfer.done.wait(MACOS_HANDOFF_FULLSCREEN_WAIT_SEC) and transfer.settle("abandoned"):
        log.warning("No main-thread turn found the windows out of full screen within %.0fs; this shell keeps running.",
                    MACOS_HANDOFF_FULLSCREEN_WAIT_SEC)
        return False
    transfer.done.wait()  # refused at once, or claimed: ``final`` ends in exec or reports back
    if transfer.error is not None:
        raise transfer.error
    return transfer.state == "claimed"


def _observers() -> list:
    """Each pywebview window's observing WindowChrome, None where no ``before_show`` registered one; empty without a
    window. Python state only, no native read: the main thread's turn judges again, and natively."""
    tracked = list(_tracked)  # pywebview runs ``closed`` handlers on their own thread
    return [next((chrome for chrome in tracked if chrome._window is window), None)
            for window in list(getattr(sys.modules.get("webview"), "windows", None) or ())]


def _resume() -> None:
    """A full-screen phase changed (main thread): a waiting handoff looks again in the next turn."""
    transfer = _transfer
    if transfer is not None and transfer.state == "pending":
        from PyObjCTools import AppHelper

        AppHelper.callAfter(_step, transfer)


def _step(transfer: _Transfer) -> None:
    """One main-thread turn of a handoff: refuse, wait for a notification, or claim and run ``final``; never block.
    Every window is judged before any is asked to leave full screen, so one refusing leaves the others as they are."""
    if transfer.state != "pending":
        return
    try:
        if transfer.cancelled():  # a Stop came first: no window is asked to leave
            if transfer.settle("refused"):
                log.warning("A Stop came before the windows agreed; nothing was torn down.")
            return
        chromes = _observers()
        # An unobserved window, or none left at all, refuses before any other is asked to leave full screen.
        states = [False] if None in chromes or not chromes else [chrome.windowed() for chrome in chromes]
        if False not in states:
            for chrome in chromes:
                chrome.leave(transfer)
    except Exception:
        log.warning("A window's full-screen state is unreadable.", exc_info=True)
        states = [False]
    if False in states:
        if transfer.settle("refused"):
            log.warning("A window cannot be confirmed out of full screen; this shell keeps running.")
        return
    if None in states or not transfer.settle("claimed"):
        return  # a notification of the moving window looks again
    try:
        transfer.final()
    except BaseException as error:  # reported to the worker: the main thread never runs recovery that may wait
        transfer.error = error
    finally:
        transfer.done.set()
