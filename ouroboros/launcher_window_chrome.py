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
"""

from __future__ import annotations

import json
import logging
import math
import sys
from typing import Any, Optional

log = logging.getLogger("launcher.window_chrome")

_MACOS = sys.platform == "darwin"
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


def _handler_class():
    if "handler" not in _classes:
        import Foundation

        class OuroborosWindowChromeHandler(Foundation.NSObject):
            def userContentController_didReceiveScriptMessage_(self, controller, message):
                self.chrome.receive(message.body())

        _classes["handler"] = OuroborosWindowChromeHandler
    return _classes["handler"]


def _webview_of(window: Any) -> Any:
    """pywebview 5.4's WKWebView of ``window``: it exists in ``before_show`` but is no subview yet."""
    from webview.platforms.cocoa import BrowserView

    return BrowserView.instances[window.uid].webview


def paint_room(window: Any) -> None:
    """Register ``window`` to open in the room colour of its own appearance; inert off macOS."""
    if _MACOS and window is not None:
        window.events.before_show += lambda: _paint_room(getattr(window, "native", None))


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
        self._doc = -math.inf  # the window's current document (its time origin)
        self._declared = False  # that document's script spoke since the last finished navigation
        self._wanted = False
        self._integrated = False
        self._refused = False
        self._fullscreen = False
        self._appearance: Optional[str] = None  # None: inherited
        self._sent: tuple = ()
        self._closed = False

    def attach(self, window: Any) -> None:
        """Register on ``window`` before ``webview.start``; inert off macOS."""
        if not _MACOS or window is None:
            return
        self._window = window
        paint_room(window)
        window.events.before_show += self._before_show
        window.events.before_load += self._navigation_finished
        window.events.closed += self._close

    def _before_show(self) -> None:
        """pywebview ``before_show`` (synchronous, main thread, before the first load)."""
        native = getattr(self._window, "native", None)
        if native is None:
            return
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
        self._observe_full_screen()
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

    def _observe_full_screen(self) -> None:
        import AppKit

        center = AppKit.NSNotificationCenter.defaultCenter()
        for name, handler in (("NSWindowWillEnterFullScreenNotification", self._entering_full_screen),
                              ("NSWindowWillExitFullScreenNotification", self._leaving_full_screen),
                              ("NSWindowDidExitFullScreenNotification", self._left_full_screen)):
            # No queue: the block runs where AppKit posts, the main thread, before the transition.
            self._observers.append(center.addObserverForName_object_queue_usingBlock_(
                name, self._native, None, lambda _note, handler=handler: handler()))

    def _entering_full_screen(self) -> None:
        self._fullscreen = True
        self._show_toolbar(False)
        self._publish()

    def _leaving_full_screen(self) -> None:
        self._show_toolbar(True)

    def _left_full_screen(self) -> None:
        self._fullscreen = False
        self._switch()  # a document that arrived in full screen gets its title bar now
        # AppKit ends the exit with the toolbar hidden, after WillExit showed it, and a toolbar set here starts
        # hidden too: shown now, before the band is measured, or the page reserves only the bare title bar.
        self._show_toolbar(True)
        self._publish()

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
        """pywebview ``closed``: no further writes, and the notification center lets go of this window."""
        self._closed = True
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
