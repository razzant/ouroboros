"""Grant the desktop's microphone request only on its own loopback page.

WebKitGTK denies UserMediaPermissionRequest by default. The shell installs its
handler before the first load; the request must ask for audio alone and the
view must still be at the exact scheme/host/port the launcher started. Camera,
display capture, unrelated origins and other permission classes stay denied.
The Windows and macOS WebViews keep their native permission prompts.

GTK's UserMediaPermissionRequest exposes media flags but no requesting-frame
origin. WebKit's Permissions Policy controls subframes; the app must not delegate
microphone permission to foreign embedded content. The view's current URI is the
native origin witness, never a prefix match or a client-provided URL.
"""

from __future__ import annotations

import ipaddress
import logging
import sys
from urllib.parse import urlsplit

log = logging.getLogger("launcher.media")


def loopback_origin(url: str):
    """Keep scheme, host and port because another local service is another origin."""
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        if parsed.scheme not in ("http", "https") or not host or parsed.username or parsed.password:
            return None
        if host != "localhost" and not ipaddress.ip_address(host).is_loopback:
            return None
        return parsed.scheme, host, parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
    except (TypeError, ValueError):
        return None


def handle_permission_request(view, request, *, origin, media_type) -> bool:
    """Resolve every permission explicitly so a later handler cannot broaden it."""
    allowed = False
    try:
        expected = loopback_origin(origin)
        allowed = bool(expected and loopback_origin(view.get_uri()) == expected
                       and isinstance(request, media_type)
                       and request.get_property("is-for-audio-device")
                       and not request.get_property("is-for-video-device"))
        # Display capture exists on newer GTK. Older libraries cannot request it.
        if allowed and request.find_property("is-for-display-device"):
            allowed = not request.get_property("is-for-display-device")
    except Exception:
        allowed = False
        log.debug("Cannot establish microphone permission facts", exc_info=True)
    if allowed:
        request.allow()
    else:
        request.deny()
    return True


def bind_media_view(view, origin, media_type):
    """Return a disposer for one native view's signal registration."""
    signal = view.connect("permission-request", lambda widget, request: handle_permission_request(
        widget, request, origin=origin, media_type=media_type))

    def dispose():
        if view.handler_is_connected(signal):
            view.disconnect(signal)

    return dispose


def attach_media_permissions(window, origin):
    """Attach before show, when GTK's native widget exists but no page has loaded."""
    if sys.platform != "linux" or not loopback_origin(origin):
        return window
    disposers = []

    def before_show():
        # Import the backend that pywebview selected; no second GTK initialization.
        gui = getattr(window, "gui", None)
        if not str(getattr(gui, "__name__", "")).endswith(".gtk"):
            return
        try:
            webkit = gui.webkit
            pending = [window.native]
            while pending:
                widget = pending.pop()
                if isinstance(widget, webkit.WebView):
                    disposers.append(bind_media_view(widget, origin, webkit.UserMediaPermissionRequest))
                else:
                    pending.extend(getattr(widget, "get_children", lambda: [])())
            if not disposers:
                log.warning("No GTK WebView found for microphone permissions")
        except Exception:
            log.warning("GTK microphone permission handler could not be attached", exc_info=True)

    def closed():
        for dispose in disposers:
            try:
                dispose()
            except Exception:
                log.debug("GTK view already destroyed", exc_info=True)
        disposers.clear()
        window.events.before_show -= before_show
        window.events.closed -= closed

    window.events.before_show += before_show
    window.events.closed += closed
    return window


def display_ready(backend) -> tuple[bool, str]:
    """Check GTK's display before creating a window: importing bindings is insufficient.

    Qt keeps the launcher's environment check because probing its display creates
    a QGuiApplication, which can abort the process instead of reporting failure.
    """
    if str(getattr(backend, "__name__", "")).endswith(".gtk"):
        try:
            from gi.repository import Gdk
        except Exception as exc:  # gi vanished between import and probe
            return False, f"GTK backend without gi.repository.Gdk: {type(exc).__name__}: {exc}"
        if Gdk.Display.get_default() is None:
            return False, "GTK backend has no default display (DISPLAY/WAYLAND_DISPLAY is unusable)"
    return True, ""
