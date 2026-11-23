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
        self._selection: dict[str, str] = {}

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

    def arm():
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
                if uri == 'about:blank' or (comma and kind.lower().startswith('data:text/html')
                                            and kind.lower().endswith(';base64') and payload == encoded):
                    page._enabled.set()
                    return
                log.warning('Beautiful Context rejected its document navigation (%s)', uri.split(':', 1)[0])
            event.Cancel = True

        core.NavigationStarting += navigate
        core.NavigateToString(html)

    native.Invoke(Action(arm))


def open_local_report(webview, root: Path, report_path: Path) -> dict:
    if not webview2_renderer(webview):
        return {"state": "unavailable"}
    if not _window_lock.acquire(blocking=False):
        return {"state": "already_open"}
    page = ContextArchivePage(root)
    window = None
    stage = "read_document"
    try:
        html = report_path.read_text(encoding="utf-8").replace('<html lang="en">', '<html lang="en" data-private-archive>')
        if len(html.encode("utf-8")) > 2 * 1024 * 1024 or 'data-private-archive>' not in html:
            raise ValueError("Invalid offline viewer document")
        stage = "blank_window"
        window = webview.create_window("Beautiful Context — Local requests", html="<!doctype html><title></title>",
                                       js_api=page, width=1030, height=760, min_size=(390, 480), text_select=True)

        def closed():
            page._enabled.clear()
            _window_lock.release()

        window.events.closed += closed
        if not window.events.loaded.wait(10):
            raise TimeoutError("Offline viewer's blank window did not load")
        stage = "offline_navigation"
        _load_offline(window, html, page)
        if not page._presented.wait(10) or not page._enabled.is_set():
            raise TimeoutError("Offline viewer did not request its initial list")
        return {"state": "opened"}
    except Exception as exc:
        page._enabled.clear()
        log.warning("Beautiful Context window unavailable at %s (%s)", stage, type(exc).__name__)
        if window is None:
            _window_lock.release()
        elif not window.events.closed.is_set():
            window.destroy()
        return {"state": "unavailable"}
