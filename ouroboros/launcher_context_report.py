"""An offline WebView2 window for retained context; the ordinary page can only open it."""
from __future__ import annotations

import base64
import logging
import threading
from pathlib import Path

log = logging.getLogger(__name__)
_window_lock = threading.Lock()


class ContextArchivePage:
    """Methods exposed to the private blank-origin window only, after its navigation fence is installed."""

    def __init__(self, root: Path):
        self._root = root
        self._enabled = threading.Event()
        self._presented = threading.Event()
        self._lock = threading.Lock()
        self._lifecycle = threading.Lock()
        self._revoked = False
        self._selection: dict[str, str] = {}

    def _enable(self, closed_event) -> bool:
        with self._lifecycle:
            if self._revoked or closed_event.is_set():
                return False
            self._enabled.set()
            return True

    def _revoke(self) -> None:
        with self._lifecycle:
            self._revoked = True
            self._enabled.clear()
        self._presented.set()  # wake the opener; it must still check whether the window is open

    def _read(self, fn):
        if not self._enabled.is_set() or not self._lock.acquire(blocking=False):
            return {"ok": False, "error": "The local viewer is closed or busy. Try again while it is open."}
        try:
            value = fn()
            return {"ok": True, "value": value} if self._enabled.is_set() else {"ok": False}
        except Exception as exc:
            log.info("Beautiful Context local read unavailable (%s)", type(exc).__name__)
            return {"ok": False, "error": "The saved record could not be read. Try again or choose another request."}
        finally:
            self._lock.release()

    def rooms(self) -> dict:
        if self._enabled.is_set():
            self._presented.set()  # the trusted reader script started, not merely a navigation event
        from ouroboros import context_report_archive

        return self._read(lambda: context_report_archive.rooms(self._root))

    def requests(self, room_id: int) -> dict:
        from ouroboros import context_report_archive

        def read():
            self._selection = {}
            self._selection, rows = context_report_archive.recent_requests(self._root, room_id)
            return rows

        return self._read(read)

    def report(self, attempt_id: str) -> dict:
        from ouroboros import context_report_archive

        return self._read(lambda: context_report_archive.document(self._root, self._selection, attempt_id))


def webview2_renderer(webview) -> bool:
    return getattr(getattr(webview, "guilib", None), "renderer", None) == "edgechromium"


def _load_offline(window, html: str, page: ContextArchivePage) -> None:
    # This adapter is Windows WebView2-specific. Arm on the GUI thread, then
    # navigate once using the host's literal HTML (not a URL or served asset).
    from System import Action

    native = window.native
    control = native.browser.webview
    closed_event = window.events.closed

    def arm():
        if closed_event.is_set():
            raise RuntimeError("Local viewer closed before navigation")
        from ouroboros.launcher_appearance import NativeAppearance
        from ouroboros.launcher_page_zoom import PageZoom
        from ouroboros.launcher_window_chrome import _fit

        try:
            _fit(native)  # same WinForms calculation as fit_to_work_area, after create_window's show
            PageZoom(lambda: page._root).apply_saved(window)
            NativeAppearance().paint_shown(window)
        except Exception:
            log.debug('Beautiful Context kept its default window appearance', exc_info=True)
        core = control.CoreWebView2
        if str(core.Source) != "about:blank":
            raise RuntimeError("Local viewer did not start on its blank document")
        core.Settings.AreDevToolsEnabled = False
        core.Settings.AreDefaultContextMenusEnabled = False
        core.Settings.AreBrowserAcceleratorKeysEnabled = False
        control.AllowExternalDrop = False
        encoded = base64.b64encode(html.encode('utf-8')).decode('ascii')
        pending = True

        def navigate(_sender, event):
            nonlocal pending
            if pending:
                pending = False
                uri = str(event.Uri)
                kind, comma, payload = uri.partition(',')
                document = uri == 'about:blank' or (comma and kind.lower().startswith('data:text/html')
                                                      and kind.lower().endswith(';base64') and payload == encoded)
                if document and page._enable(closed_event):
                    return
                log.warning('Beautiful Context rejected its document navigation')
            event.Cancel = True

        core.NavigationStarting += navigate
        core.NavigateToString(html)

    native.Invoke(Action(arm))


def open_local_report(webview, root: Path, report_path: Path) -> dict:
    if not webview2_renderer(webview):
        return {"state": "unavailable"}
    page = ContextArchivePage(root)
    release_lock = threading.Lock()
    released = False
    if not _window_lock.acquire(blocking=False):
        return {"state": "already_open"}

    def release():
        nonlocal released
        with release_lock:
            if not released:
                released = True
                _window_lock.release()

    def closed():
        page._revoke()
        release()

    window = closed_event = None
    stage = "read_document"
    try:
        html = report_path.read_text(encoding="utf-8").replace('<html lang="en">', '<html lang="en" data-private-archive>')
        if len(html.encode("utf-8")) > 2 * 1024 * 1024 or 'data-private-archive>' not in html:
            raise ValueError("Invalid offline viewer document")
        stage = "blank_window"
        from ouroboros.launcher_window_chrome import ROOM, host_palette

        colour = ROOM.get(host_palette())
        window = webview.create_window("Beautiful Context — Local requests", html="<!doctype html><title></title>",
                                       js_api=page, width=1030, height=760, min_size=(390, 480), text_select=True,
                                       **({'background_color': colour} if colour else {}))
        # `closed` does not replay an event dispatched before subscription.
        closed_event = window.events.closed
        closed_event += closed
        if closed_event.is_set() or not window.events.loaded.wait(10) or closed_event.is_set():
            raise TimeoutError("Offline viewer's blank window did not load")
        stage = "offline_navigation"
        _load_offline(window, html, page)
        if not page._presented.wait(10) or not page._enabled.is_set() or closed_event.is_set():
            raise TimeoutError("Offline viewer did not request its initial list")
        return {"state": "opened"}
    except Exception as exc:
        page._revoke()
        log.warning("Beautiful Context window unavailable at %s (%s)", stage, type(exc).__name__)
        try:
            if window is not None and not (closed_event is not None and closed_event.is_set()):
                window.destroy()
        except Exception as destroy_exc:
            log.warning('Beautiful Context could not close its unavailable window (%s)', type(destroy_exc).__name__)
        finally:
            release()  # no-op if native close arrived first; never release another window's lease
        return {"state": "unavailable"}
