"""The desktop window's page zoom: the owner's Interface size for this window (Settings → Appearance, ⌘/Ctrl + − 0).

The engine zooms the page the way a browser's Cmd+ does: the page is laid out in a narrower or wider viewport of CSS
pixels, so its media queries, viewport units and measured lengths stay in one coordinate space. A real WKWebView at
``pageZoom`` 1.25 reports innerWidth 880 of an 1100 pt window and devicePixelRatio 2.5, nothing overflowing; CSS
``zoom`` on the page instead made it taller than its window. macOS: WKWebView ``pageZoom`` (public since macOS 11) on
the main thread, then the title band is measured again (``launcher_window_chrome`` divides by ``pageZoom``).
Windows: the WebView2 control's ``ZoomFactor`` on the form's UI thread (a host value applies across navigations).
Linux: WebKitGTK's zoom level on the GTK main loop. Another backend reports no capability, and the page says the
window cannot change its size. On macOS the View menu carries Actual Size, Zoom In and Zoom Out (⌘0, ⌘+, ⌘−) as a
browser's does; each runs the page's own step (``window.ouroInterfaceSize``), so one setter serves the menu, the keys
and the slider.

The app keeps the size in ``state/desktop_page_zoom.json`` so the next window opens at it before its first paint
(``before_show``): here the app's value is the setting, and the page reads it through ``shell_info`` and changes it
through the bridge. Bridge calls arrive on pywebview's worker threads, possibly out of order, so each carries when its
document became the window's page and its count there, as the caption tint's do (``launcher_appearance``): only a
request newer by that pair is kept, and what is stored and applied is always the newest size: two calls that overlap
on their threads cannot leave the older one in the file or on the screen.
"""

from __future__ import annotations

import json
import logging
import math
import pathlib
import sys
import threading
from typing import Any, Callable, Dict, Optional

log = logging.getLogger("launcher.page_zoom")

PERCENT_MIN, PERCENT_MAX, PERCENT_STEP = 50, 200, 5
STATE_NAME = "desktop_page_zoom.json"
VIEW_MENU_ITEMS = (("Actual Size", "0", "reset"), ("Zoom In", "+", "zoomIn"), ("Zoom Out", "-", "zoomOut"))


def clamp_percent(value: Any) -> Optional[int]:
    """A size in whole percent inside the window's range, on its 5 % grid; None when it is no number."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    stepped = int(round(number / PERCENT_STEP) * PERCENT_STEP)
    return max(PERCENT_MIN, min(PERCENT_MAX, stepped))


def _backend() -> str:
    if sys.platform == "darwin":
        return "cocoa"
    if sys.platform == "win32":
        return "edgechromium"
    if sys.platform.startswith("linux"):
        return "gtk"
    return ""


class PageZoom:
    """One window's page zoom: stored by the app, applied by its engine, changed by its page."""

    def __init__(self, data_dir: Callable[[], pathlib.Path] | None = None,
                 on_changed: Callable[[], None] | None = None) -> None:
        self._data_dir = data_dir or _default_data_dir
        self._on_changed = on_changed  # macOS: the title band is measured again after a change
        self._lock = threading.Lock()
        self._store_lock = threading.Lock()  # writers queue, and each writes the newest size
        self._window: Any = None
        self._view: Any = None   # the engine's web view, once before_show found it
        self._form: Any = None   # Windows: the form whose UI thread owns the control
        self._kind = ""          # the backend that applies it ("" = no capability)
        self._percent = 100
        self._order: tuple[float, int] = (-math.inf, 0)
        self._closed = False
        self._menu_actions: list[Any] = []  # a menu item does not retain its target

    # -- lifecycle -------------------------------------------------------------------------------------------------
    def attach(self, window: Any) -> None:
        """Register on ``window`` before ``webview.start``."""
        if window is None or not _backend():
            return
        self._window = window
        self._percent = self._load()
        window.events.before_show += self._before_show
        window.events.closed += self._close
        if _backend() == "cocoa" and hasattr(window.events, "shown"):
            window.events.shown += self._shown

    def _before_show(self) -> None:
        """pywebview ``before_show`` (synchronous, the GUI thread, before the first load): find the engine's view and
        give it the stored size, so the first paint is already at it."""
        kind = _backend()
        try:
            view, form = _find_view(kind, self._window)
        except Exception:
            log.info("No page zoom for this window: its web view is not reachable.", exc_info=True)
            return
        if view is None:
            return
        with self._lock:
            if self._closed:
                return
            self._view, self._form, self._kind, percent = view, form, kind, self._percent
        if percent != 100:
            self._apply_now(kind, view)

    def _close(self) -> None:
        with self._lock:
            self._closed = True
            self._view = self._form = None

    # -- the View menu (macOS) -------------------------------------------------------------------------------------
    def _shown(self) -> None:
        """pywebview ``shown`` (a worker thread): pywebview builds the app's menus on the main thread right after it,
        so the size's View items are queued behind that."""
        with self._lock:
            if self._closed or self._kind != "cocoa":
                return
        try:
            from PyObjCTools import AppHelper

            AppHelper.callAfter(self._add_view_items)
        except Exception:
            log.info("The View menu keeps only its own items.", exc_info=True)

    def _add_view_items(self, main: Any = None) -> None:
        """The View menu's size items (main thread), once per app: a later window finds them there."""
        try:
            import AppKit

            menu = view_menu(main if main is not None else AppKit.NSApplication.sharedApplication().mainMenu())
            if menu is None or menu.itemWithTitle_(VIEW_MENU_ITEMS[0][0]) is not None:
                return
            for index, (title, key, command) in enumerate(VIEW_MENU_ITEMS):
                action = _MenuAction(lambda command=command: self._run_page(command))
                item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action.selector, key)
                item.setTarget_(action)
                menu.insertItem_atIndex_(item, index)
                self._menu_actions.append(action)
            menu.insertItem_atIndex_(AppKit.NSMenuItem.separatorItem(), len(VIEW_MENU_ITEMS))
        except Exception:
            log.info("The View menu keeps only its own items.", exc_info=True)

    def _run_page(self, command: str) -> None:
        """A View item (main thread): the page takes the step through its own setter, as its keys do; a page without
        one (an older page, the setup window) ignores it."""
        with self._lock:
            view = None if self._closed else self._view
        if view is not None:
            view.evaluateJavaScript_completionHandler_(
                f"window.ouroInterfaceSize && window.ouroInterfaceSize.{command}()", None)

    # -- the bridge ------------------------------------------------------------------------------------------------
    def info(self) -> Dict[str, Any]:
        """For ``shell_info``: whether this window can change its size, its range and its size now."""
        with self._lock:
            supported = bool(self._kind) and not self._closed
            return {"supported": supported, "min": PERCENT_MIN, "max": PERCENT_MAX, "step": PERCENT_STEP,
                    "percent": self._percent}

    def request(self, percent: Any, page: Any = None, sequence: Any = None) -> Dict[str, Any]:
        """The page's new size for this window: keep the newest request of the newest page, store it, apply it."""
        value = clamp_percent(percent)
        if value is None:
            return {"ok": False, "state": "invalid_size"}
        try:
            order = (float(page), int(sequence))
        except (TypeError, ValueError):
            return {"ok": False, "state": "invalid_order"}
        if not math.isfinite(order[0]):
            return {"ok": False, "state": "invalid_order"}
        with self._lock:
            if self._closed:
                return {"ok": False, "state": "closed"}
            if not self._kind:
                return {"ok": False, "state": "unsupported", "percent": self._percent}
            if order <= self._order:
                return {"ok": True, "state": "stale", "percent": self._percent}
            self._order, self._percent = order, value
            kind, view, form = self._kind, self._view, self._form
        self._store()
        scheduled = self._post(kind, view, form)
        return {"ok": scheduled, "state": "scheduled" if scheduled else "unavailable", "percent": value}

    # -- engines ---------------------------------------------------------------------------------------------------
    def _post(self, kind: str, view: Any, form: Any) -> bool:
        """Run the engine's write on the thread that owns the view."""
        try:
            if kind == "cocoa":
                from PyObjCTools import AppHelper

                AppHelper.callAfter(self._apply_now, kind, view)
            elif kind == "edgechromium":
                from System import Action

                form.BeginInvoke(Action(lambda: self._apply_now(kind, view)))
            elif kind == "gtk":
                from gi.repository import GLib

                GLib.idle_add(lambda: self._apply_now(kind, view) and False)
            else:
                return False
            return True
        except Exception:
            log.warning("The window's size change could not be scheduled.", exc_info=True)
            return False

    def _apply_now(self, kind: str, view: Any) -> bool:
        """The engine's own page zoom (on its GUI thread), at the newest size, whichever request queued this write."""
        with self._lock:
            if self._closed:
                return False
            factor = self._percent / 100
        try:
            if kind == "cocoa":
                view.setPageZoom_(factor)
            elif kind == "edgechromium":
                view.ZoomFactor = factor
            elif kind == "gtk":
                view.set_zoom_level(factor)
        except Exception:
            log.warning("The engine refused the window's size %.2f.", factor, exc_info=True)
            return False
        if kind == "cocoa" and self._on_changed is not None:
            try:
                self._on_changed()
            except Exception:
                log.debug("The title band was not measured again after a size change.", exc_info=True)
        return True

    # -- the stored size -------------------------------------------------------------------------------------------
    def _path(self) -> pathlib.Path:
        return pathlib.Path(self._data_dir()) / "state" / STATE_NAME

    def _load(self) -> int:
        try:
            value = clamp_percent(json.loads(self._path().read_text(encoding="utf-8")).get("percent"))
        except (OSError, ValueError, AttributeError):
            return 100
        return value if value is not None else 100

    def _store(self) -> None:
        with self._store_lock:
            with self._lock:
                percent = self._percent
            try:
                from ouroboros.utils import atomic_write_json

                path = self._path()
                path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_json(path, {"percent": percent})
            except Exception:
                log.warning("The window's size is not kept for the next launch.", exc_info=True)


def view_menu(main: Any) -> Any:
    """The app's View menu: the one pywebview builds around Enter Full Screen (macOS)."""
    for item in (main.itemArray() if main is not None else ()):
        submenu = item.submenu()
        if submenu is not None and any("toggleFullScreen" in str(entry.action()) for entry in submenu.itemArray()):
            return submenu
    return None


class _MenuAction:
    """A View item's target, a plain object with an ``objc.selector`` as pywebview's own menu actions are."""

    def __init__(self, run: Callable[[], None]) -> None:
        from objc import selector

        self._run = run
        self.selector = selector(self._call, signature=b"v@:")

    def _call(self) -> None:
        try:
            self._run()
        except Exception:
            log.info("A View menu size item failed.", exc_info=True)


def _default_data_dir() -> pathlib.Path:
    from ouroboros.config import DATA_DIR

    return pathlib.Path(DATA_DIR)


def _find_view(kind: str, window: Any) -> tuple[Any, Any]:
    """The engine's web view of ``window`` in the bundled pywebview, and the Windows form that owns it."""
    if window is None:
        return None, None
    if kind == "cocoa":
        from webview.platforms.cocoa import BrowserView

        view = BrowserView.instances[window.uid].webview
        return (view if hasattr(view, "setPageZoom_") else None), None
    if kind == "edgechromium":
        form = getattr(window, "native", None)
        view = getattr(form, "webview", None)
        if view is None or not hasattr(view, "ZoomFactor"):  # the legacy MSHTML control has no page zoom
            return None, None
        return view, form
    if kind == "gtk":
        import webview

        guilib = getattr(webview, "guilib", None)  # the Linux backend pywebview chose: GTK has a page zoom, Qt none here
        if not str(getattr(guilib, "__name__", "")).endswith(".gtk"):
            return None, None
        view = guilib.BrowserView.instances[window.uid].webview
        return (view if hasattr(view, "set_zoom_level") else None), None
    return None, None
